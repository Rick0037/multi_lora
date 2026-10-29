#!/usr/bin/env python3
"""单跑一档:batch=64 / batch 内 64 个 adapter / 只 profile decode 阶段。

测量和打印全部复用 fix_batch_test_each_cost.py 里的 run_one_tier + print_report,
本脚本只做三件事:换输出目录、只开 decode 窗口、跑完自动串 analyze_trace.py。

产物都落在 traces/04_decode_n64/,不动 02 目录里已有的 trace 和 CSV:
  decode_n64.json.gz          原始 chrome trace
  decode_n64.txt              top-10 kernel 表
  decode_n64.bucketed.json.gz 分桶上色的派生 trace(Perfetto 打开这个)
  bucket_summary.csv / bucket_detail.csv / kernel_group_summary.csv

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
    assign_names, print_report, run_one_tier,
)
from multi_lora.hf_runner import (  # noqa: E402
    load_causal_lm, load_tokenizer, pick_device, pick_dtype, release_model,
)
from multi_lora.manager import MultiLoRAManager  # noqa: E402

OUT_DIR = _ROOT / "traces" / "04_decode_n64"  # 本脚本专用目录
NUM_ADAPTERS = 64  # batch 内不同 adapter 的个数(64 行正好一人一个)
STAGES = ("prefill", "decode")


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required")
        return 1

    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
            device, OUT_DIR, NUM_ADAPTERS, stages=STAGES)
    finally:
        release_model(mgr.model)

    print_report([row])
    # 分桶纯 CPU,放最后跑(显存已释放),不再手敲第二条命令
    return subprocess.call([
        sys.executable,
        str(_ROOT / "multi_lora" / "examples" / "analyze_trace.py"),
        str(OUT_DIR),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
