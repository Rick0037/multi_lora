#!/usr/bin/env python3
"""离线"生产级" multi-LoRA workload 压测(整模型级,合成 Qwen2-7B 规模权重)。

驱动``MultiLoRAManager.forward_with_triton_lora`` 的整模型热路径(pool prepare/evict
+ 每层 ``LinearWithLoRA`` 的 segment GEMM A+B)。

合成模型(Qwen2-7B 近似),
  - hidden = 3584,attention q/k/v/o(GQA,k/v out=512),num_layers=28(默认)
  - 仅含注意力四投影(LoRA 实际作用的层),用合成随机权重,规模与 Qwen2-7B 对齐。
  - 走 MultiLoRAManager 全流程 → 整模型级,但聚焦 LoRA 路径,便于看清 A/B kernel 影响。

生产 workload 维度,
  - adapter 请求分布,uniform / distinct / skewed(Zipf)
  - max_loras_per_batch(一个 batch 内并存的 distinct adapter 上限)
  - prefill(变长 seq)/ decode(seq=1)
  - 混合 rank(不同 adapter 不同 rank)
  - base-only 基线(不挂 LoRA),用于量化 LoRA 额外开销

A 阶段 kernel 对比,原版 ``sgemm_lora_a`` vs 优化版 ``sgemm_lora_a_opti``
(通过 monkeypatch backend 的函数引用切换,两者跑同一批 workload)。

指标,吞吐(req/s、token/s)+ 延迟 mean/p50/p90/p99(每 batch 一次 forward 计时)。
"""


from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

import multi_lora.backend.triton_backend as triton_backend  # noqa: E402
from multi_lora.kernels.sgemm_lora_a import sgemm_lora_a_fwd as A_ORIG  # noqa: E402
from multi_lora.kernels.sgemm_lora_a_opti import (
    sgemm_lora_a_fwd as A_OPTI,
)  # noqa: E402
from multi_lora.manager import MultiLoRAManager  # noqa: E402


# ----------------------------- 合成模型 ----------------------------------- #
class SynthAttn(nn.Module):
    """Qwen2 风格注意力四投影(GQA)。仅触发各投影的 base+LoRA GEMM,不做注意力混合。"""

    def __init__(self, hidden: int, kv_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        _ = self.k_proj(x)  # 触发 LoRA,结果不参与(压测足矣)
        _ = self.v_proj(x)
        return x + self.o_proj(q)  # 残差,保持数值稳定


class SynthDecoderLayer(nn.Module):
    def __init__(self, hidden: int, kv_dim: int) -> None:
        super().__init__()
        self.self_attn = SynthAttn(hidden, kv_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(x)


class SynthModel(nn.Module):
    """整模型壳,28 层注意力投影。forward 接收展平 token 张量 [S, hidden]。"""

    def __init__(self, hidden: int, kv_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SynthDecoderLayer(hidden, kv_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _build_base(hidden: int, kv_dim: int, num_layers: int, device, dtype) -> SynthModel:
    return SynthModel(hidden, kv_dim, num_layers).to(device=device, dtype=dtype)


def _save_rank_adapters(
    ranks: List[int], hidden: int, kv_dim: int, num_layers: int, tmp: Path
) -> Dict[int, str]:
    """为每个 unique rank 在合成 base 上建一个 LoRA 并 save_pretrained,返回 rank->dir。"""
    rank_dir: Dict[int, str] = {}
    for r in ranks:
        torch.manual_seed(1000 + r)
        base = _build_base(hidden, kv_dim, num_layers, "cpu", torch.float32)
        cfg = LoraConfig(
            r=r,
            lora_alpha=2 * r,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=TARGET_MODULES,
        )
        pm = get_peft_model(base, cfg)
        d = tmp / f"rank{r}"
        pm.save_pretrained(str(d))
        rank_dir[r] = str(d)
        del pm, base
        gc.collect()
    return rank_dir


# ----------------------------- workload 生成 ------------------------------ #
def _zipf_probs(n: int, alpha: float) -> np.ndarray:
    w = np.array([alpha ** (-i) for i in range(n)], dtype=np.float64)
    return w / w.sum()


def _pick_active(
    adapter_names: List[str],
    maxlpb: int,
    dist: str,
    alpha: float,
    rng: np.random.Generator,
) -> List[str]:
    """为一个 batch 选出 <= maxlpb 个 distinct adapter(受 pool 槽位上限约束)。"""
    n = len(adapter_names)
    k = min(maxlpb, n)
    if dist == "skewed":
        probs = _zipf_probs(n, alpha)
        idx = rng.choice(n, size=k, replace=False, p=probs)
    else:  # uniform / distinct 都先等概率选活跃集合
        idx = rng.choice(n, size=k, replace=False)
    return [adapter_names[i] for i in idx]


def _assign_requests(
    active: List[str], bs: int, dist: str, alpha: float, rng: np.random.Generator
) -> List[str]:
    """把 bs 个请求分配到活跃 adapter 集合上,按分布决定每请求选谁。"""
    k = len(active)
    if dist == "distinct":
        # 轮询 → 每个 batch 用满 k 个 distinct adapter(最坏 batching 效率)
        return [active[i % k] for i in range(bs)]
    if dist == "skewed":
        probs = _zipf_probs(k, alpha)
        return [active[i] for i in rng.choice(k, size=bs, p=probs)]
    # uniform
    return [active[i] for i in rng.integers(0, k, size=bs)]


def _make_schedule(
    *,
    adapter_names: List[str],
    iters: int,
    bs: int,
    phase: str,
    dist: str,
    maxlpb: int,
    alpha: float,
    seqlen_lo: int,
    seqlen_hi: int,
    base_only: bool,
    seed: int,
) -> List[Tuple[List[Optional[str]], np.ndarray]]:
    """预生成每个 iter 的 (per-segment adapter 名列表, seq_lens)。确定性,供两个 kernel 复用。"""
    rng = np.random.default_rng(seed)
    sched: List[Tuple[List[Optional[str]], np.ndarray]] = []
    for _ in range(iters):
        if base_only:
            names: List[Optional[str]] = [None] * bs
        else:
            active = _pick_active(adapter_names, maxlpb, dist, alpha, rng)
            names = _assign_requests(active, bs, dist, alpha, rng)  # type: ignore
        if phase == "decode":
            seq_lens = np.ones(bs, dtype=np.int64)
        else:
            seq_lens = rng.integers(seqlen_lo, seqlen_hi, size=bs).astype(np.int64)
        sched.append((names, seq_lens))
    return sched


# ----------------------------- 计时 -------------------------------------- #
def _run_scenario(
    mgr: MultiLoRAManager,
    name_to_id: Dict[str, str],
    sched: List[Tuple[List[Optional[str]], np.ndarray]],
    x_pool: torch.Tensor,
    hidden: int,
    warmup: int,
) -> Dict[str, float]:
    device = x_pool.device
    lat_ms: List[float] = []
    total_tokens = 0
    total_segs = 0

    def one_iter(names, seq_lens):
        S = int(seq_lens.sum())
        x = x_pool[:S]
        lora_ids = [name_to_id[n] if n is not None else None for n in names]
        seq_t = torch.from_numpy(seq_lens).to(device=device, dtype=torch.int32)
        return mgr.forward_with_triton_lora(x=x, lora_ids=lora_ids, seq_lens=seq_t)

    # warmup(含 opti 的 autotune 首次开销)
    for i in range(min(warmup, len(sched))):
        names, seq_lens = sched[i % len(sched)]
        one_iter(names, seq_lens)
    torch.cuda.synchronize()

    for names, seq_lens in sched:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        one_iter(names, seq_lens)
        end.record()
        torch.cuda.synchronize()
        lat_ms.append(start.elapsed_time(end))
        total_tokens += int(seq_lens.sum())
        total_segs += len(names)

    lat = np.array(lat_ms)
    total_s = lat.sum() / 1e3
    return {
        "mean_ms": float(lat.mean()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p99_ms": float(np.percentile(lat, 99)),
        "tok_per_s": total_tokens / total_s,
        "req_per_s": total_segs / total_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, default=3584)
    ap.add_argument("--kv-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=28)
    ap.add_argument("--num-adapters", type=int, default=16)
    ap.add_argument("--ranks", type=str, default="16,32,64")
    ap.add_argument("--max-loras-per-batch", type=int, default=4)
    ap.add_argument("--prefill-bs", type=int, default=16)
    ap.add_argument("--decode-bs", type=int, default=64)
    ap.add_argument("--prefill-seqlo", type=int, default=256)
    ap.add_argument("--prefill-seqhi", type=int, default=768)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--zipf-alpha", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return 1
    device = torch.device("cuda")
    dtype = torch.bfloat16
    ranks = [int(r) for r in args.ranks.split(",")]
    max_rank = max(ranks)
    print(f"GPU={torch.cuda.get_device_name(0)} dtype={dtype}")
    print(
        f"model: hidden={args.hidden} kv_dim={args.kv_dim} layers={args.num_layers} "
        f"(q/k/v/o per layer)"
    )
    print(
        f"adapters={args.num_adapters} ranks={ranks} max_loras_per_batch="
        f"{args.max_loras_per_batch}"
    )

    tmp = Path(tempfile.mkdtemp(prefix="mlora_bench_"))
    print(f"building synthetic LoRA adapters in {tmp} ...")
    rank_dir = _save_rank_adapters(
        ranks, args.hidden, args.kv_dim, args.num_layers, tmp
    )

    # num_adapters 个逻辑 adapter,循环复用 rank 目录 → 混合 rank + 多 distinct 槽位
    adapter_names = [f"ad{i}" for i in range(args.num_adapters)]
    adapters = {
        nm: rank_dir[ranks[i % len(ranks)]] for i, nm in enumerate(adapter_names)
    }

    print("loading base + adapters into MultiLoRAManager ...")
    base = _build_base(args.hidden, args.kv_dim, args.num_layers, device, dtype)
    # 池容量取场景中用到的最大 maxlpb(场景内各自再限制活跃集合)
    pool_capacity = max(args.max_loras_per_batch, 8)
    mgr = MultiLoRAManager.from_pretrained(
        base,
        adapters,
        max_loras_per_batch=pool_capacity,
        max_lora_rank=max_rank,
        lora_backend="triton",
        lora_target_suffixes=None,  # 包装全部 q/k/v/o
    )
    name_to_id = {nm: mgr._peft_name_to_id[nm] for nm in adapter_names}
    print(f"wrapped_modules={len(mgr.lora_wrappers)} backend={mgr._lora_backend.name}")

    max_S = max(args.prefill_bs * args.prefill_seqhi, args.decode_bs)
    x_pool = torch.randn(max_S, args.hidden, device=device, dtype=dtype)

    # 压测矩阵,(标签, phase, dist, maxlpb, bs, base_only)
    scen = [
        (
            "prefill/uniform",
            "prefill",
            "uniform",
            args.max_loras_per_batch,
            args.prefill_bs,
            False,
        ),
        (
            "prefill/distinct",
            "prefill",
            "distinct",
            args.max_loras_per_batch,
            args.prefill_bs,
            False,
        ),
        (
            "prefill/skewed",
            "prefill",
            "skewed",
            args.max_loras_per_batch,
            args.prefill_bs,
            False,
        ),
        ("prefill/distinct-mlpb8", "prefill", "distinct", 8, args.prefill_bs, False),
        (
            "prefill/base-only",
            "prefill",
            "uniform",
            args.max_loras_per_batch,
            args.prefill_bs,
            True,
        ),
        (
            "decode/uniform",
            "decode",
            "uniform",
            args.max_loras_per_batch,
            args.decode_bs,
            False,
        ),
        (
            "decode/distinct",
            "decode",
            "distinct",
            args.max_loras_per_batch,
            args.decode_bs,
            False,
        ),
        (
            "decode/skewed",
            "decode",
            "skewed",
            args.max_loras_per_batch,
            args.decode_bs,
            False,
        ),
        (
            "decode/base-only",
            "decode",
            "uniform",
            args.max_loras_per_batch,
            args.decode_bs,
            True,
        ),
    ]

    kernels = [("orig", A_ORIG), ("opti", A_OPTI)]
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for tag, phase, dist, maxlpb, bs, base_only in scen:
        sched = _make_schedule(
            adapter_names=adapter_names,
            iters=args.iters,
            bs=bs,
            phase=phase,
            dist=dist,
            maxlpb=maxlpb,
            alpha=args.zipf_alpha,
            seqlen_lo=args.prefill_seqlo,
            seqlen_hi=args.prefill_seqhi,
            base_only=base_only,
            seed=args.seed,
        )
        results[tag] = {}
        for kname, kfn in kernels:
            triton_backend.sgemm_lora_a_fwd = kfn  # 切换 A 阶段 kernel
            results[tag][kname] = _run_scenario(
                mgr, name_to_id, sched, x_pool, args.hidden, args.warmup
            )

    # ----------------------------- 报告 ----------------------------------- #
    print("\n" + "=" * 100)
    print("RESULT  (整模型级 forward 延迟; A 阶段 orig vs opti)")
    print("=" * 100)
    hdr = f"{'scenario':<26} {'kernel':<6} {'p50(ms)':>9} {'p99(ms)':>9} {'tok/s':>10} {'req/s':>9}"
    print(hdr)
    print("-" * 100)
    for tag, _, _, _, _, _ in scen:
        for kname, _ in kernels:
            m = results[tag][kname]
            print(
                f"{tag:<26} {kname:<6} {m['p50_ms']:>9.3f} {m['p99_ms']:>9.3f} "
                f"{m['tok_per_s']:>10.0f} {m['req_per_s']:>9.1f}"
            )
        o, p = results[tag]["orig"], results[tag]["opti"]
        sp = o["p50_ms"] / p["p50_ms"] if p["p50_ms"] > 0 else float("nan")
        print(f"{'  -> opti/orig p50 speedup':<33} {sp:>5.2f}x")
        print("-" * 100)

    print("\n说明,合成模型仅含注意力 q/k/v/o 投影(LoRA 实际作用层),")
    print("故 A/B kernel 在端到端中占比高于真实 7B(真实 7B 中 LoRA 仅占百分之几)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
