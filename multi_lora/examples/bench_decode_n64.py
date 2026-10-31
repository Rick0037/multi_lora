#!/usr/bin/env python3
"""单跑一档:batch=64 / batch 内 64 个 adapter,复用 fix_batch 的测量函数。

窗口产物落在 traces/04_decode_n64/v{n}/(自动 +1 不覆盖),跑完自动串 analyze_trace:
  prefill_k64swap{S}          run_one_tier(prefill 窗口 + 表1 的 TTFT/TPOT)
  graph_/gplain_k64swap{S}    run_graph_tier:每步 graph 外换入 → replay,
                              换入 H2D 全在 graph 外;gplain 是不录 graph 的对照。
                              加不加 graph 的对比 = graph vs gplain 这两个 trace。


    HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    .venv/bin/python multi_lora/examples/bench_decode_n64.py
"""

import gc
import subprocess
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from multi_lora.examples.fix_batch_test_each_cost import (  # noqa: E402
    ADAPTERS,
    ADAPTER_RANKS,
    BASE,
    BATCH_SIZE,
    PROMPT,
    SEED,
    build_rotation,
    print_report,
    run_graph_tier,
    run_one_tier,
)
from multi_lora.hf_runner import (  # noqa: E402
    load_causal_lm,
    load_tokenizer,
    pick_device,
    pick_dtype,
)
from multi_lora.manager import MultiLoRAManager  # noqa: E402

OUT_ROOT = _ROOT / "traces" / "04_decode_n64"
NUM_ADAPTERS = 64
SWAP = 32  # 每步换入=换出个数;0 = 不换入对照
WINDOW_STEPS = 15  # 高 swap 档必须 ~15,50 步的 kineto 事件会顶爆 62 GiB cgroup
# prefill 窗口先跑,吃掉进程首个 profiler 窗口的初始化开销(decode 墙钟 159→114ms)


def next_run_dir(root: Path) -> Path:
    """看已有目录名 v1/v2...,给出下一个 v{n+1},保证不覆盖历史."""
    used = [
        int(p.name[1:]) for p in root.glob("v*") if p.is_dir() and p.name[1:].isdigit()
    ]
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
    enc = load_tokenizer(BASE)([PROMPT] * BATCH_SIZE, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    # 只注册 rotation 会用到的(SWAP=0 → 64 个,SWAP=32 → 96 个)
    label = f"k{NUM_ADAPTERS}swap{SWAP}"
    rotation = build_rotation(NUM_ADAPTERS, SWAP)
    used = {n: ADAPTERS[n] for n in rotation.used_names()}
    mgr = MultiLoRAManager.from_pretrained(
        load_causal_lm(BASE, device=device, dtype=dtype),
        used,
        # 槽位数 = 本档 adapter 数;rank 上限 = 变体的实际 rank
        max_loras_per_batch=NUM_ADAPTERS,
        max_lora_rank=max(ADAPTER_RANKS.values()),
        lora_backend="triton",
    )

    try:
        # plain 窗口跳过:graph 的对照是 gplain(同 KV 背靠背),plain 与它代码同路径
        # 只是 KV 尺寸不同,当不了对照;TTFT/TPOT 走 CUDA event,不依赖 plain 窗口
        row = run_one_tier(
            mgr, enc, rotation, run_dir, label, WINDOW_STEPS, with_plain=False
        )
        graph_rows = run_graph_tier(mgr, enc, rotation, run_dir, label, WINDOW_STEPS)
        row["graph"] = graph_rows["graph"]
        row["gplain"] = graph_rows["gplain"]
    finally:
        del mgr  # 连模型、pool、graph 一起放
        gc.collect()
        torch.cuda.empty_cache()

    print_report([row])
    g = row["graph"]
    print(
        f"\nTPOT: 非 graph {row['tpot']:.2f} ms -> 换入+replay {g['tpot']:.2f} ms"
        f"  (replay-only 基准 {g['tpot_replay_only']:.2f} ms;"
        f"graph 省 launch ≈ {row['gplain']['cpu_ms'] - g['cpu_ms']:.0f} ms/步)"
    )
    # 分桶纯 CPU,放最后
    return subprocess.call(
        [
            sys.executable,
            str(_ROOT / "multi_lora" / "examples" / "analyze_trace.py"),
            str(run_dir),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
