#!/usr/bin/env python3
"""单跑一档:batch=64 / batch 内 64 个 adapter,产出三个 profiler 窗口。

测量和打印全部复用 fix_batch_test_each_cost.py 里的 run_one_tier + run_graph_tier
+ print_report,本脚本只做三件事:指定输出目录、跑窗口、跑完自动串 analyze_trace.py。

三个窗口:
  prefill_n64   prefill
  plain_n64     真实 decode(每步写 KV),不录 graph
  graph_n64     同一份真实 decode,录成 graph 后 replay
plain 与 graph 只差“有没录 graph”一件事,两者的 GPU 时间应当接近,
墙钟之差就是 CUDA Graph 的净收益。

产物落在 traces/04_decode_n64/v{n}/,每次跑自动 +1(v1/v2/...),不覆盖上一轮:
  <窗口名>.json.gz          原始 chrome trace
  <窗口名>.txt              top-10 kernel 表
  <窗口名>.bucketed.json.gz 分桶上色的派生 trace(Perfetto 打开这个)
  bucket_summary.csv / bucket_detail.csv / kernel_group_summary.csv
02 目录里已有的 trace 和 CSV 一律不动。

Example::

    HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python multi_lora/examples/bench_decode_n64.py
"""

import subprocess
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from multi_lora.examples.fix_batch_test_each_cost import (  # noqa: E402
    ADAPTERS, ADAPTER_RANKS, BASE, BATCH_SIZE, PROMPT, SEED,
    assign_names, print_report, run_graph_tier, run_one_tier,
)
from multi_lora.hf_runner import (  # noqa: E402
    load_causal_lm, load_tokenizer, pick_device, pick_dtype, release_model,
)
from multi_lora.manager import MultiLoRAManager  # noqa: E402

OUT_ROOT = _ROOT / "traces" / "04_decode_n64"  # 历次结果都堆在这下面
NUM_ADAPTERS = 64  # batch 内不同 adapter 的个数(64 行正好一人一个)
# 先跑 prefill 窗口当“挡枪”,吃掉进程首个 profiler 窗口的初始化开销
# (实测:只开 decode 时墙钟 158.97ms,补一个 prefill 窗口后 decode 降到 113.64ms)
# decode 阶段的两个窗口(plain / graph)由 run_graph_tier 产出


def next_run_dir(root: Path) -> Path:
    """看已有目录名 v1/v2...,给出下一个 v{n+1},保证不覆盖历史."""
    used = [int(p.name[1:]) for p in root.glob("v*")
            if p.is_dir() and p.name[1:].isdigit()]
    return root / f"v{max(used, default=0) + 1}"


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required")
        return 1

    torch.manual_seed(SEED)
    run_dir = next_run_dir(OUT_ROOT)
    run_dir.mkdir(parents=True)
    print(f"本次产物 -> {run_dir}")

    device = pick_device()
    dtype = pick_dtype(device)
    enc = load_tokenizer(BASE)(
        [PROMPT] * BATCH_SIZE, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    mgr = MultiLoRAManager.from_pretrained(
        load_causal_lm(BASE, device=device, dtype=dtype),
        ADAPTERS,
        # 槽位数 = 本档 adapter 数;rank 上限 = 变体的实际 rank
        max_loras_per_batch=NUM_ADAPTERS,
        max_lora_rank=max(ADAPTER_RANKS.values()),
        lora_backend="triton",
    )

    try:
        row = run_one_tier(
            mgr, enc, assign_names(NUM_ADAPTERS, list(ADAPTERS)),
            device, run_dir, NUM_ADAPTERS)
        # 真实 decode 的 CUDA Graph 单独一档:static KV + 常驻 buffer
        # 同时产出不录 graph 的对照窗口 plain_*
        graph_rows = run_graph_tier(
            mgr, enc, assign_names(NUM_ADAPTERS, list(ADAPTERS)),
            run_dir, NUM_ADAPTERS)
        row["graph"] = graph_rows["graph"]
        row["plain"] = graph_rows["plain"]
    finally:
        release_model(mgr.model)

    print_report([row])
    print(f"\nTPOT: 非 graph {row['tpot']:.2f} ms -> graph "
          f"{row['graph']['tpot']:.2f} ms  (加速 {row['tpot'] / row['graph']['tpot']:.2f}x)")
    # 分桶纯 CPU,放最后跑(显存已释放),不再手敲第二条命令
    return subprocess.call([
        sys.executable,
        str(_ROOT / "multi_lora" / "examples" / "analyze_trace.py"),
        str(run_dir),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
