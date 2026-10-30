#!/usr/bin/env python3
"""Qwen3-4B 实际工况基准:多 adapter batch 的 prefill + decode 全流程计时。

流程与在线 serving 一致:

1. prefill:N 条请求(每行绑不同 adapter,含 base)一次 forward,量 TTFT;
2. decode:逐 token 自回归,每步输入 [N, 1],量 TPOT;
3. 全部走 MultiLoRAManager.forward_with_lora,与 examples 里的推理路径一致。

backend 可选 triton / torch,分别跑一次即可对比两套 LoRA 实现的耗时。

Example::

    python multi_lora/examples/bench_qwen4b_e2e.py --num-decode 32
    python multi_lora/examples/bench_qwen4b_e2e.py --backend torch --num-decode 32
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multi_lora.hf_runner import (  # noqa: E402
    load_causal_lm,
    load_tokenizer,
    model_input_device,
    pick_device,
    pick_dtype,
    release_model,
)
from multi_lora.manager import MultiLoRAManager  # noqa: E402

DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_A = "GestaltLabs/Jeff-1"
ADAPTER_B = "/root/autodl-tmp/qwen3_4b_r64_lora/v2-20261022-131552/last-checkpoint"

# (adapter_name or None = base, prompt);三行混批,覆盖 heterogeneous batch
DEFAULT_REQUESTS: List[Tuple[Optional[str], str]] = [
    (
        "a",
        "<|im_start|>user\nList three countries and their capitals.\n"
        "<|im_start|>assistant\n",
    ),
    (
        None,
        "<|im_start|>user\nList three countries and their capitals.\n"
        "<|im_start|>assistant\n",
    ),
    (
        "b",
        "<|im_start|>user\nSummarize context-faithful generation in one sentence.\n"
        "<|im_start|>assistant\n",
    ),
]


def parse_args() -> argparse.Namespace:
    """解析命令行;模型路径、decode 步数、后端与显存选项."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--adapter-a", default=ADAPTER_A)
    p.add_argument("--adapter-b", default=ADAPTER_B)
    p.add_argument(
        "--num-decode",
        type=int,
        default=32,
        help="decode 步数(每步生成 1 个 token)",
    )
    p.add_argument(
        "--prefill-iters", type=int, default=5, help="prefill 重复次数,用于取均值"
    )
    p.add_argument("--warmup", type=int, default=3, help="正式计时前的预热步数")
    p.add_argument(
        "--backend", choices=("triton", "torch"), default="triton", help="LoRA 后端"
    )
    return p.parse_args()


def cuda_ms(fn):
    """用 CUDA event 计时一次调用,返回 (毫秒, 返回值)."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def summarize(name: str, times: List[float]) -> None:
    """打印一组耗时的均值/中位数/最小值和吞吐."""
    mean = statistics.fmean(times)
    print(
        f"{name}: mean={mean:.3f} ms  median={statistics.median(times):.3f} ms  "
        f"min={min(times):.3f} ms  max={max(times):.3f} ms"
    )


@torch.inference_mode()
def main() -> int:
    """跑 prefill + decode 全流程并打印耗时统计."""
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for Triton SGEMM")
        return 1

    device = pick_device()
    dtype = pick_dtype(device)
    tokenizer = load_tokenizer(args.base_model)
    prompts = [p for _, p in DEFAULT_REQUESTS]
    enc = tokenizer(prompts, padding=True, return_tensors="pt")

    print(
        f"base={args.base_model} device={device} dtype={dtype} backend={args.backend}"
    )
    base = load_causal_lm(
        args.base_model,
        device=device,
        dtype=dtype,
    )
    mgr = MultiLoRAManager.from_pretrained(
        base,
        {"a": args.adapter_a, "b": args.adapter_b},
        max_loras_per_batch=len(DEFAULT_REQUESTS),
        max_lora_rank=64,
        lora_backend=args.backend,
    )

    try:
        enc = {k: v.to(model_input_device(mgr.model)) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        batch, seq_len = input_ids.shape
        lora_names = mgr.registry.normalize_names(
            [n for n, _ in DEFAULT_REQUESTS])
        print(
            f"batch={batch} prompt_len={seq_len} "
            f"wrapped_modules={len(mgr.lora_wrappers)} "
            f"decode_steps={args.num_decode}"
        )

        def prefill():
            """一次整 batch prefill(带 KV cache,模拟真实请求入口)."""
            return mgr.forward_with_lora(
                input_ids=input_ids,
                attention_mask=attn,
                lora_names=lora_names,
                seq_lens=[seq_len] * batch,
                use_cache=True,
            )

        # 预热:prefill + decode 都跑一遍,排除 Triton 首次编译和显存分配
        out = prefill()
        past = out.past_key_values
        next_ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        decode_lens = [1] * batch  # CPU list,不用建 GPU 张量
        for _ in range(args.warmup):
            out = mgr.forward_with_lora(
                input_ids=next_ids,
                attention_mask=None,
                seq_lens=decode_lens,
                lora_names=lora_names,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)

        # ---- prefill 计时 ----
        prefill_times = []
        for _ in range(args.prefill_iters):
            ms, _ = cuda_ms(prefill)
            prefill_times.append(ms)
        print()
        summarize("prefill", prefill_times)
        print(f"  -> TTFT ≈ {statistics.fmean(prefill_times):.3f} ms")

        # ---- decode 计时:先重新 prefill 拿到干净的 cache ----
        out = prefill()
        past = out.past_key_values
        next_ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)

        starts = []
        ends = []
        for _ in range(args.num_decode):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = mgr.forward_with_lora(
                input_ids=next_ids,
                attention_mask=None,
                seq_lens=decode_lens,
                lora_names=lora_names,
                past_key_values=past,  # * kv cache
                use_cache=True,
            )
            end.record()
            starts.append(start)
            ends.append(end)
            past = out.past_key_values
            next_ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()

        decode_times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        print()
        summarize("decode", decode_times)
        per_step_ms = statistics.fmean(decode_times)
        print(
            f"  -> TPOT ≈ {per_step_ms:.3f} ms/token  "
            f"batch_throughput ≈ {batch * 1000 / per_step_ms:.1f} tok/s"
        )

        # 生成内容抽检,确认 decode 确实在产出 token
        ids = next_ids.flatten().tolist()
        print(f"last step token ids={ids}")
        for i, tid in enumerate(ids):
            print(f"  row[{i}] last_token={tokenizer.decode([tid])!r}")

        total = statistics.fmean(prefill_times) + args.num_decode * per_step_ms
        print(
            f"\nend-to-end: {total:.1f} ms for {args.num_decode} tokens "
            f"(batch={batch})"
        )
        print(
            f"pool_total_evictions={mgr.pool.eviction_policy.eviction_count}")
    finally:
        release_model(mgr.model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
