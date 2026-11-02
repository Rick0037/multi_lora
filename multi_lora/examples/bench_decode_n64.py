#!/usr/bin/env python3
"""单跑一档:batch=64 / batch 内 64 个 adapter,复用 fix_batch 的测量函数。

窗口产物落在 traces/04_decode_n64/v{n}/(自动 +1 不覆盖),跑完自动串 analyze_trace:
  prefill_k64swap{S}          run_one_tier(prefill 窗口 + 表1 的 TTFT/TPOT)
  graph_/gplain_k64swap{S}    run_graph_tier:每步 graph 外换入 → replay,
                              换入 H2D 全在 graph 外;gplain 是不录 graph 的对照。
                              加不加 graph 的对比 = graph vs gplain 这两个 trace。

传参数单跑某一段(都自动串 analyze_trace,产物进新 v{n}/):
  `decode`  只跑 Baseline(DynamicCache) 窗口(~7min)
  `graph`   只跑 graph/gplain 两窗口(~10min)—— 优化迭代模式,
            在 V2 基础上改一版跑一次,阶梯表自动追加
  第二个参数 = 每步换入数(默认 32):`graph 0` 跑 k64swap0 对照档,
            `graph 32`(或不传)= k64swap32;两档各跑一次,产物进各自 v{n}/

    HF_HOME=... .venv/bin/python multi_lora/examples/bench_decode_n64.py [decode|graph] [swap]
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
    WARMUP_ITERS,
    build_rotation,
    print_report,
    profile_window,
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

    # 档位开关:第一个参数 = 窗口模式(graph/gplain/decode/all);
    # 第二个 = 每步换入数 H(默认 32;0 = 不换入对照档);
    # 第三个 = batch 内 adapter 数 K(默认 64)。k8swap4 就是 K=8 H=4。
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    swap = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    num_adapters = int(sys.argv[3]) if len(sys.argv) > 3 else NUM_ADAPTERS

    # 只注册 rotation 会用到的(swap=0 → K 个,swap=32 → K+32 个)
    label = f"k{num_adapters}swap{swap}"
    rotation = build_rotation(num_adapters, swap)
    used = {n: ADAPTERS[n] for n in rotation.used_names()}
    mgr = MultiLoRAManager.from_pretrained(
        load_causal_lm(BASE, device=device, dtype=dtype),
        used,
        # 槽位数 = 本档 adapter 数;rank 上限 = 变体的实际 rank
        max_loras_per_batch=num_adapters,
        max_lora_rank=max(ADAPTER_RANKS.values()),
        lora_backend="triton",
    )

    try:
        # `decode` 单跑模式:只跑 Baseline(DynamicCache) 窗口 —— 阶梯表第一行。
        # 1 次 prefill 填 DynamicCache,之后每步真换入 decode;V1/V2 复用已有产物
        if only == "decode":
            seq = int(enc["input_ids"].shape[1])
            out = mgr.forward_with_lora(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                lora_names=rotation.next_names(),
                seq_lens=[seq] * BATCH_SIZE,
                use_cache=True,
            )
            ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            past = out.past_key_values

            def dyn_decode():
                return mgr.forward_with_lora(
                    input_ids=ids,
                    attention_mask=None,
                    seq_lens=[1] * BATCH_SIZE,
                    lora_names=rotation.next_names(),
                    past_key_values=past,
                    use_cache=True,
                )

            for _ in range(WARMUP_ITERS):
                dyn_decode()
            profile_window(dyn_decode, f"decode_{label}", run_dir, WINDOW_STEPS)
            print(f"decode(Baseline) 窗口完成 -> {run_dir}")
        elif only == "graph":
            # `graph` 迭代模式:只跑 graph/gplain 两窗口(V2 基础上改优化时用),
            # 跳过 Baseline 计时;与历史 graph 行对比看优化收益
            graph_rows = run_graph_tier(
                mgr, enc, rotation, run_dir, label, WINDOW_STEPS
            )
            g, p = graph_rows["graph"], graph_rows["gplain"]
            print(
                f"\ngraph TPOT {g['tpot']:.2f} ms "
                f"(replay-only {g['tpot_replay_only']:.2f} ms) | "
                f"CPU: graph {g['cpu_ms']:.1f} vs gplain {p['cpu_ms']:.1f} ms"
            )
        else:
            # plain 窗口已删:StaticCache 不录图的 V1 行由 gplain 担任
            # (与 graph 同 KV 背靠背);decode 窗口(DynamicCache)= 阶梯 Baseline 行
            row = run_one_tier(
                mgr, enc, rotation, run_dir, label, WINDOW_STEPS, with_dyn=True
            )
            graph_rows = run_graph_tier(
                mgr, enc, rotation, run_dir, label, WINDOW_STEPS
            )
            row["graph"] = graph_rows["graph"]
            row["gplain"] = graph_rows["gplain"]
    finally:
        del mgr  # 连模型、pool、graph 一起放
        gc.collect()
        torch.cuda.empty_cache()

    if only in ("decode", "graph"):
        return subprocess.call(
            [
                sys.executable,
                str(_ROOT / "multi_lora" / "examples" / "analyze_trace.py"),
                str(run_dir),
            ]
        )

    print_report([row])
    g = row["graph"]
    print(
        f"\nTPOT: DynamicCache口径 {row['tpot']:.2f} ms | StaticCache口径 "
        f"{row['tpot_static']:.2f} ms -> 换入+replay {g['tpot']:.2f} ms"
        f"  (replay-only 基准 {g['tpot_replay_only']:.2f} ms;"
        f"graph vs gplain 省 launch ≈ {row['gplain']['cpu_ms'] - g['cpu_ms']:.0f} ms/步)"
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
