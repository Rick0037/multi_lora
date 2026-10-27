#!/usr/bin/env python3
"""基于已有的 rank-64 LoRA checkpoint,降秩造出 rank 8/16/32 的独立 checkpoint.

两种降秩:
  slice - 切 A 前 r 行 / B 前 r 列,秒级,但训练后的行不按重要性排序,损失大
  svd   - 对 B@A 取 top-r 再分解回 B_r/A_r,精确重构;用 svd_lowrank,
          每个 module 只分解一次,三个 rank 复用,总耗时约 2 分钟
落盘时同步改 config 的 r 和 lora_alpha,保持 scaling 不变。
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = "/root/autodl-tmp/qwen3_4b_r64_lora/v2-20261022-131552/last-checkpoint"
OUT_ROOT = Path("/root/autodl-tmp/qwen3_4b_rank_variants")
TARGET_RANKS = [8, 16, 32]
MODE = "svd"  # "slice" 或 "svd"


def shrink_slice(a: torch.Tensor, b: torch.Tensor, r: int):
    """A [R,in] 切前 r 行,B [out,R] 切前 r 列."""
    return a[:r].contiguous(), b[:, :r].contiguous()


def lowrank_factorize(a: torch.Tensor, b: torch.Tensor, q: int):
    """W = B@A 的 top-q 低秩分解,返回 (u, s, v).

    u [out, q], s [q], v [in, q];满足 b @ a ≈ (u * s) @ v.T。
    用 svd_lowrank 而不是 linalg.svd:只要 top-q,不需要完整谱,
    实测 252 个 module 从 29 分钟降到 76 秒。
    """
    w = b @ a  # [out, in]
    u, s, v = torch.svd_lowrank(w, q=q, niter=4)
    return u, s, v


def shrink_from_factor(u: torch.Tensor, s: torch.Tensor, v: torch.Tensor, r: int):
    """从低秩分解里取 top-r,分解回 B_r/A_r,使 B_r@A_r == W 的 top-r 近似."""
    root = s[:r].sqrt()
    b_r = (u[:, :r] * root).to(u.dtype)
    # v 是 [in, q],转置成 [r, in] 才是 A
    a_r = (v[:, :r] * root).T.to(u.dtype)
    return a_r.contiguous(), b_r.contiguous()


def main() -> int:
    src_dir = Path(SRC)
    cfg = json.loads((src_dir / "adapter_config.json").read_text())
    old_r = int(cfg["r"])
    old_alpha = float(cfg["lora_alpha"])
    use_rslora = bool(cfg.get("use_rslora", False))
    state = load_file(src_dir / "adapter_model.safetensors")

    # 收集所有 lora_A 的 key;同名的 lora_B 一一对应
    a_keys = sorted(k for k in state if k.endswith("lora_A.weight"))
    b_keys = {k: k.replace("lora_A", "lora_B") for k in a_keys}
    missing = [b_keys[k] for k in a_keys if b_keys[k] not in state]
    if missing:
        print(f"ERROR: 缺少配对的 lora_B:{missing[:3]}")
        return 1
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        print(
            "WARN: config 里有 rank_pattern/alpha_pattern,"
            "各模块 rank 不同,本脚本假设统一 rank"
        )

    print(
        f"src={SRC}\nrank={old_r} alpha={old_alpha} rslora={use_rslora} "
        f"modules={len(a_keys)} mode={MODE}"
    )

    # svd 模式:每个 module 只分解一次,所有 rank 复用同一个 (u, s, v)
    # slice 模式没有可复用的中间量,factors 留空
    q = max(TARGET_RANKS) + 8
    factors = {}
    if MODE == "svd":
        print(f"低秩分解 q={q}(一次,{len(a_keys)} 个 module 复用)...")
        for i, k in enumerate(a_keys):
            factors[k] = lowrank_factorize(state[k], state[b_keys[k]], q)
            if (i + 1) % 50 == 0:
                print(f"  factorize {i + 1}/{len(a_keys)}")

    for new_r in TARGET_RANKS:
        if new_r > old_r:
            print(f"skip r{new_r}: > 源 rank {old_r}")
            continue
        new_state = {}
        for k in a_keys:
            if MODE == "svd":
                u, s, v = factors[k]
                a_new, b_new = shrink_from_factor(u, s, v, new_r)
            else:
                a_new, b_new = shrink_slice(state[k], state[b_keys[k]], new_r)
            new_state[k] = a_new
            new_state[b_keys[k]] = b_new

        # 保持 scaling 不变,按公式反推新 alpha
        if use_rslora:
            new_alpha = old_alpha * (new_r / old_r) ** 0.5
        else:
            new_alpha = old_alpha * new_r / old_r

        new_cfg = dict(cfg)
        new_cfg["r"] = new_r
        new_cfg["lora_alpha"] = new_alpha

        out_dir = OUT_ROOT / f"r{new_r}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(new_state, str(out_dir / "adapter_model.safetensors"))
        (out_dir / "adapter_config.json").write_text(json.dumps(new_cfg, indent=2))
        print(f"wrote {out_dir}  r={new_r} alpha={new_alpha:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
