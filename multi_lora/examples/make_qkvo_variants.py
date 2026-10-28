#!/usr/bin/env python3
"""从 rank-64 的 LoRA checkpoint 派生 N 个只含 qkvo、且 rank 互不相同的 adapter.

做法:
  1. 只取 self_attn.{q,k,v,o}_proj 的 lora_A/lora_B,config 的 target_modules 同步收窄
  2. 对 B@A 做一次 top-Q 低秩分解(Q = 目标 rank 的最大值),第 i 个变体取前 r 个分量,
     分解一次全部 rank 复用
  3. 对 A 逐元素乘 (1 + eps * randn),保证同 rank 的副本之间权重也不相同
  4. alpha 按 alpha * r / r_src 反推,使 scaling alpha/r 与原 adapter 一致

Example::

    python multi_lora/examples/make_qkvo_variants.py
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("/root/autodl-tmp/qwen3_4b_r64_lora/v2-20261022-131552/last-checkpoint")
OUT_ROOT = Path("/root/autodl-tmp/qwen3_4b_qkvo_variants")
NUM = 64
RANKS = [8, 16, 32, 64]  # 第 i 个变体的 rank = RANKS[i % len(RANKS)]
EPS = 0.02
SEED = 0


def factorize(a: torch.Tensor, b: torch.Tensor, q: int):
    """W = B@A 的 top-q 分解,返回 (u, s, v);u [out,q], s [q], v [in,q]."""
    return torch.svd_lowrank(b @ a, q=q, niter=4)


def shrink(u, s, v, r: int):
    """取 top-r 再分解回 (A_r, B_r),使 B_r @ A_r == W 的 top-r 近似."""
    root = s[:r].sqrt()
    b_r = (u[:, :r] * root).contiguous()
    a_r = (v[:, :r] * root).T.contiguous()
    return a_r, b_r


def main() -> int:
    cfg = json.loads((SRC / "adapter_config.json").read_text())
    cfg["target_modules"] = [f"{m}_proj" for m in ("q", "k", "v", "o")]
    old_r, old_alpha = int(cfg["r"]), float(cfg["lora_alpha"])

    state = load_file(str(SRC / "adapter_model.safetensors"))
    attn = {k: v for k, v in state.items()
            if any(f"self_attn.{m}_proj." in k for m in ("q", "k", "v", "o"))}
    a_keys = sorted(k for k in attn if k.endswith("lora_A.weight"))
    print(f"src={SRC} rank={old_r} alpha={old_alpha}")
    print(f"kept {len(attn)}/{len(state)} tensors, modules={len(a_keys)}")

    q = max(RANKS)
    print(f"低秩分解 q={q} ...")
    factors = {}
    for i, k in enumerate(a_keys):
        b_key = k.replace("lora_A", "lora_B")
        factors[k] = factorize(attn[k].float(), attn[b_key].float(), q)
        if (i + 1) % 50 == 0:
            print(f"  factorize {i + 1}/{len(a_keys)}")

    gen = torch.Generator().manual_seed(SEED)
    for i in range(NUM):
        r = RANKS[i % len(RANKS)]
        new_cfg = dict(cfg, r=r, lora_alpha=old_alpha * r / old_r)
        new_state = {}
        for k in a_keys:
            a_r, b_r = shrink(*factors[k], r=r)
            b_key = k.replace("lora_A", "lora_B")
            noise = torch.randn(a_r.shape, generator=gen, dtype=torch.float32)
            new_state[k] = (a_r * (1.0 + EPS * noise)).contiguous()
            new_state[b_key] = b_r

        out = OUT_ROOT / f"q{i:02d}"
        out.mkdir(parents=True, exist_ok=True)
        save_file(new_state, str(out / "adapter_model.safetensors"))
        (out / "adapter_config.json").write_text(json.dumps(new_cfg, indent=2))
        if (i + 1) % 8 == 0:
            print(f"  wrote {i + 1}/{NUM} (last rank={r})")

    total = sum(f.stat().st_size for f in OUT_ROOT.rglob("*.safetensors"))
    print(
        f"wrote {NUM} adapters ranks={RANKS} -> {OUT_ROOT}  ({total / 2**30:.1f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
