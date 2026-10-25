#!/usr/bin/env python3
"""torch.profiler 剖析 multi_lora 整模型热路径,定位耗时/重复/可优化操作。

复用 bench_multilora_workload 的合成 Qwen2-7B 规模模型,但聚焦在：
  1. timeline(chrome trace)—— 导出 json,可在 chrome://tracing / perfetto 打开。
  2. key_averages 表 —— 分别按 CUDA self time 与 CPU self time 排序,看 kernel vs host 开销。
  3. per-step 操作计数 —— 量化每个 forward step 重复发生的 host 张量构造 / kernel launch 次数。
  4. host-side 元数据重建开销的独立计时 —— 验证 P0-2(per-step 重建 GPU 元数据)。
  5. orig vs opti A kernel 在 timeline 上的对比。

用法：
  cd /home
  python multi_lora/examples/profile_multilora.py --phase prefill
  python multi_lora/examples/profile_multilora.py --phase decode
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

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

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class SynthAttn(nn.Module):
    def __init__(self, hidden: int, kv_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        _ = self.k_proj(x)
        _ = self.v_proj(x)
        return x + self.o_proj(q)


class SynthDecoderLayer(nn.Module):
    def __init__(self, hidden: int, kv_dim: int) -> None:
        super().__init__()
        self.self_attn = SynthAttn(hidden, kv_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(x)


class SynthModel(nn.Module):
    def __init__(self, hidden: int, kv_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SynthDecoderLayer(hidden, kv_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _build_base(hidden, kv_dim, num_layers, device, dtype):
    return SynthModel(hidden, kv_dim, num_layers).to(device=device, dtype=dtype)


def _save_rank_adapters(ranks, hidden, kv_dim, num_layers, tmp):
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


def _make_schedule(
    adapter_names, iters, bs, phase, maxlpb, seqlo, seqhi, seed, warm=False
):
    rng = np.random.default_rng(seed)
    sched = []
    # warm：固定活跃集,所有 iter 复用同一批 adapter → pool 常驻不驱逐(隔离 reload churn)
    fixed_active = list(adapter_names[: min(maxlpb, len(adapter_names))])
    for _ in range(iters):
        k = min(maxlpb, len(adapter_names))
        active = (
            fixed_active
            if warm
            else list(rng.choice(adapter_names, size=k, replace=False))
        )
        names = [
            active[i % k] for i in range(bs)
        ]  # distinct round-robin(最坏 batching)
        if phase == "decode":
            seq_lens = np.ones(bs, dtype=np.int64)
        else:
            seq_lens = rng.integers(seqlo, seqhi, size=bs).astype(np.int64)
        sched.append((names, seq_lens))
    return sched


# ----- 用 record_function 给关键 host 段打标,便于在 timeline 归因 ----- #
def _instrument(mgr: MultiLoRAManager):
    orig_setup = mgr._setup_forward_batch
    orig_build = mgr.build_kernel_input_metadata

    def setup(*a, **k):
        with record_function("## _setup_forward_batch (host meta rebuild)"):
            return orig_setup(*a, **k)

    def build(*a, **k):
        with record_function("## build_kernel_input_metadata"):
            return orig_build(*a, **k)

    mgr._setup_forward_batch = setup
    mgr.build_kernel_input_metadata = build

    # 包装 backend 的 A/B 两阶段,量化双 launch + 中间量
    be = mgr._lora_backend
    oa, ob = be.run_lora_a_sgemm, be.run_lora_b_sgemm

    def ra(*a, **k):
        with record_function("## lora_A_sgemm"):
            return oa(*a, **k)

    def rb(*a, **k):
        with record_function("## lora_B_sgemm"):
            return ob(*a, **k)

    be.run_lora_a_sgemm = ra
    be.run_lora_b_sgemm = rb


def _host_meta_microbench(mgr, name_to_id, names, seq_lens, device, reps=200):
    """单独测 host 侧元数据重建(不含 kernel)的每步耗时。"""
    lora_ids = [name_to_id[n] if n is not None else None for n in names]
    seq_t = torch.from_numpy(seq_lens).to(device=device, dtype=torch.int32)
    # warmup
    for _ in range(20):
        mgr.build_kernel_input_metadata(lora_ids, seq_lens=seq_t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        mgr.build_kernel_input_metadata(lora_ids, seq_lens=seq_t)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps * 1e3
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=3584)
    ap.add_argument("--kv-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=28)
    ap.add_argument("--num-adapters", type=int, default=16)
    ap.add_argument("--ranks", type=str, default="16,32,64")
    ap.add_argument("--max-loras-per-batch", type=int, default=4)
    ap.add_argument(
        "--phase", type=str, default="prefill", choices=["prefill", "decode"]
    )
    ap.add_argument("--bs", type=int, default=0, help="0=按 phase 默认")
    ap.add_argument("--prefill-seqlo", type=int, default=256)
    ap.add_argument("--prefill-seqhi", type=int, default=768)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--kernel", type=str, default="orig", choices=["orig", "opti"])
    ap.add_argument(
        "--outdir", type=str, default="/home/LLMQRT/multi_lora/examples/prof_out"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", action="store_true", help="固定活跃集,pool 常驻不驱逐")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return 1
    device = torch.device("cuda")
    dtype = torch.bfloat16
    ranks = [int(r) for r in args.ranks.split(",")]
    max_rank = max(ranks)
    bs = args.bs or (16 if args.phase == "prefill" else 64)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(
        f"GPU={torch.cuda.get_device_name(0)} dtype={dtype} phase={args.phase} bs={bs} "
        f"kernel={args.kernel} layers={args.num_layers}"
    )

    tmp = Path(tempfile.mkdtemp(prefix="mlora_prof_"))
    rank_dir = _save_rank_adapters(
        ranks, args.hidden, args.kv_dim, args.num_layers, tmp
    )
    adapter_names = [f"ad{i}" for i in range(args.num_adapters)]
    adapters = {
        nm: rank_dir[ranks[i % len(ranks)]] for i, nm in enumerate(adapter_names)
    }

    base = _build_base(args.hidden, args.kv_dim, args.num_layers, device, dtype)
    pool_capacity = max(args.max_loras_per_batch, 8)
    mgr = MultiLoRAManager.from_pretrained(
        base,
        adapters,
        max_loras_per_batch=pool_capacity,
        max_lora_rank=max_rank,
        lora_backend="triton",
        lora_target_suffixes=None,
    )
    name_to_id = {nm: mgr._peft_name_to_id[nm] for nm in adapter_names}
    print(f"wrapped_modules={len(mgr.lora_wrappers)} backend={mgr._lora_backend.name}")

    # 选 kernel
    triton_backend.sgemm_lora_a_fwd = A_OPTI if args.kernel == "opti" else A_ORIG

    sched = _make_schedule(
        adapter_names,
        args.iters,
        bs,
        args.phase,
        args.max_loras_per_batch,
        args.prefill_seqlo,
        args.prefill_seqhi,
        args.seed,
        warm=args.warm,
    )
    max_S = max(int(s.sum()) for _, s in sched)
    x_pool = torch.randn(max_S, args.hidden, device=device, dtype=dtype)

    def one_iter(names, seq_lens):
        S = int(seq_lens.sum())
        x = x_pool[:S]
        lora_ids = [name_to_id[n] if n is not None else None for n in names]
        seq_t = torch.from_numpy(seq_lens).to(device=device, dtype=torch.int32)
        return mgr.forward_with_triton_lora(x=x, lora_ids=lora_ids, seq_lens=seq_t)

    # host-meta microbench(未插桩前)
    nm0, sl0 = sched[0]
    host_ms = _host_meta_microbench(mgr, name_to_id, nm0, sl0, device)

    # warmup
    for i in range(args.warmup):
        names, seq_lens = sched[i % len(sched)]
        one_iter(names, seq_lens)
    torch.cuda.synchronize()

    _instrument(mgr)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for names, seq_lens in sched:
            with record_function("## forward_step"):
                one_iter(names, seq_lens)
            torch.cuda.synchronize()

    trace_path = outdir / f"trace_{args.phase}_{args.kernel}_bs{bs}.json"
    prof.export_chrome_trace(str(trace_path))

    # ---- 报告 ---- #
    print("\n" + "=" * 96)
    print(
        f"[host-side 元数据重建] build_kernel_input_metadata 每步耗时 ≈ {host_ms:.4f} ms "
        f"(纯 host, 不含 forward kernel)"
    )
    print("=" * 96)

    print("\n--- TOP 20 by CUDA self time ---")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    print("\n--- TOP 20 by CPU self time ---")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    # per-step 计数：统计每步发生的关键操作数
    n_steps = len(sched)
    counts: Dict[str, int] = defaultdict(int)
    cuda_kernel_launches = 0
    for evt in prof.key_averages():
        nm = evt.key
        if nm in (
            "aten::zero_",
            "aten::zeros",
            "aten::copy_",
            "aten::contiguous",
            "aten::to",
            "aten::tensor",
            "aten::reshape",
            "aten::cumsum",
            "aten::arange",
            "aten::empty",
        ):
            counts[nm] += int(evt.count)
    print("\n--- 关键 aten op 总调用次数(除以 step 数得到 per-step)---")
    print(f"steps = {n_steps}")
    for k in sorted(counts):
        print(f"  {k:<22} total={counts[k]:>6}  per_step≈{counts[k]/n_steps:.1f}")

    print(f"\nchrome trace 已写出: {trace_path}")
    print("  打开方式 chrome://tracing 或 https://ui.perfetto.dev 载入该 json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
