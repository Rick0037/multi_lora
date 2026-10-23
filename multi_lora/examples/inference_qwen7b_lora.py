#!/usr/bin/env python3
"""Qwen2-7B-Instruct + multi-LoRA text generation (one load, multiple adapters).

Uses MultiLoRAManager: registry + GPU weight pool + PEFT set_adapter for
generation. Distinct adapters in one process share the same base weights.

Example (container, ~24GB GPU or use --low-vram on ~10GB):

    cd /home/multi_lora
    python multi_lora/examples/inference_qwen7b_lora.py

    python multi_lora/examples/inference_qwen7b_lora.py --low-vram --gpu-id 0 \\
        --max-new-tokens 64

    # Custom prompts: adapter name per line (empty = base), then prompt text until blank line
    python multi_lora/examples/inference_qwen7b_lora.py --prompts-file my_prompts.txt
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from peft import PeftModel

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multi_lora.hf_runner import (  # noqa: E402
    generate_text,
    load_causal_lm,
    load_tokenizer,
    model_input_device,
    pick_device,
    pick_dtype,
    release_model,
)
from multi_lora.manager import MultiLoRAManager  # noqa: E402

# DEFAULT_BASE = "Qwen/Qwen2-7B-Instruct"
# ADAPTER_A = "FuturisticVibes/Qwen2-7B-TestInstructFinetune-LORA"
# ADAPTER_B = "Bibaolong/Context-Faithful-Qwen2-7B-Instruct"


DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"
# rank 16 只训练了qkvo
ADAPTER_A = "GestaltLabs/Jeff-1"
# rank 64 qkvo 以及 mlp 部分
ADAPTER_B = "/root/autodl-tmp/qwen3_4b_r64_lora/v2-20261022-131552/last-checkpoint"

# (adapter_name or None for base, prompt)
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
        "<|im_start|>user\nSummarize the idea of context-faithful generation in one sentence.\n"
        "<|im_start|>assistant\n",
    ),
]


def parse_args() -> argparse.Namespace:
    """解析命令行;模型/适配器路径、生成参数与 batched prefill 开关。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument(
        "--adapter-a", default=ADAPTER_A, help="PEFT adapter id/path (name: a)"
    )
    p.add_argument(
        "--adapter-b", default=ADAPTER_B, help="PEFT adapter id/path (name: b)"
    )
    p.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional file: each block is adapter line then prompt until blank line",
    )
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-loras-per-batch", type=int, default=3)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--low-vram", action="store_true")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument(
        "--batched-prefill",
        action="store_true",
        help="After generation, run one grouped prefill on all prompts (sanity check)",
    )
    return p.parse_args()


def load_requests(path: Path) -> List[Tuple[Optional[str], str]]:
    """解析 prompts 文件;首行适配器名(空/base/none 表示 base),其余为 prompt 文本。"""
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    out: List[Tuple[Optional[str], str]] = []
    for block in blocks:
        lines = block.splitlines()
        name_raw = lines[0].strip()
        if name_raw.lower() in ("", "base", "none"):
            name = None
        else:
            name = name_raw
        prompt = "\n".join(lines[1:]).strip()
        if not prompt:
            raise ValueError(f"empty prompt in block starting with {name_raw!r}")
        out.append((name, prompt))
    return out


@torch.inference_mode()
def generate_with_peft(
    model: PeftModel,
    tokenizer,
    prompt: str,
    *,
    peft_adapter: Optional[str],
    max_new_tokens: int,
    temperature: float,
) -> str:
    """用 PEFT set_adapter/disable_adapter 对单条 prompt 做自回归生成。"""
    if peft_adapter is None:
        # base 模型;临时关闭 LoRA
        ctx = model.disable_adapter()
    else:
        model.set_adapter(peft_adapter)
        ctx = nullcontext()
    with ctx:
        return generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


def run_one_request(
    mgr: MultiLoRAManager,
    tokenizer,
    adapter_name: Optional[str],
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """获取适配器引用、预热 GPU pool slot、生成文本并释放引用。"""
    lora_id = mgr.registry.acquire(adapter_name)
    try:
        # 将当前 LoRA 加载到 pool,供后续 PEFT forward 使用
        info = mgr.prepare_lora_batch([lora_id])
        peft_name = mgr.peft_name_for_row(lora_id)
        label = adapter_name if adapter_name else "base"
        print(
            f"  pool slot={info.weight_indices[0]} "
            f"rank={info.lora_ranks[info.weight_indices[0]]} "
            f"evictions={info.evictions}"
        )
        return generate_with_peft(
            mgr.model,
            tokenizer,
            prompt,
            peft_adapter=peft_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    finally:
        mgr.registry.release(lora_id)


def batched_prefill_check(
    mgr: MultiLoRAManager,
    tokenizer,
    requests: List[Tuple[Optional[str], str]],
) -> None:
    """一次 grouped forward;每行不同 LoRA,仅 prefill 并打印末 token logits 范数。"""
    texts = [p for _, p in requests]
    enc = tokenizer(texts, padding=True, return_tensors="pt")
    device = model_input_device(mgr.model)
    enc = {k: v.to(device) for k, v in enc.items()}
    lora_ids = mgr.registry.resolve_names([n for n, _ in requests])
    logits, info = mgr.forward_logits_grouped(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        lora_ids=lora_ids,
    )
    # 每条序列 prefill 最后一个有效位置
    last_idx = enc["attention_mask"].sum(dim=1) - 1
    norms = []
    for i, li in enumerate(last_idx.tolist()):
        norms.append(logits[i, li].float().norm().item())
    print("\n--- batched prefill (grouped forward) ---")
    print(f"weight_indices={info.weight_indices} last_hidden_norms={norms}")


def main() -> int:
    """加载 Qwen2-7B 与 multi-LoRA manager,逐条请求生成并可选 batched prefill 校验。"""
    args = parse_args()
    requests = (
        load_requests(args.prompts_file) if args.prompts_file else DEFAULT_REQUESTS
    )

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = pick_device()
    else:
        device = pick_device()
    dtype = pick_dtype(device)

    print("multi_lora Qwen2-7B multi-LoRA inference")
    print(
        f"base={args.base_model} device={device} dtype={dtype} low_vram={args.low_vram}"
    )
    print(f"adapters: a={args.adapter_a}")
    print(f"          b={args.adapter_b}")
    print(f"requests={len(requests)}")

    tokenizer = load_tokenizer(args.base_model)
    base = load_causal_lm(
        args.base_model,
        device=device,
        dtype=dtype,
        low_vram=args.low_vram,
        gpu_id=args.gpu_id,
    )
    mgr = MultiLoRAManager.from_pretrained(
        base,
        {"a": args.adapter_a, "b": args.adapter_b},
        max_loras_per_batch=args.max_loras_per_batch,
        max_lora_rank=args.max_lora_rank,
    )

    try:
        for i, (adapter_name, prompt) in enumerate(requests):
            label = adapter_name if adapter_name else "base"
            print(f"\n=== [{i}] adapter={label} ===")
            preview = prompt.replace("\n", "\\n")[:120]
            print(f"prompt: {preview}...")
            completion = run_one_request(
                mgr,
                tokenizer,
                adapter_name,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            print(f"--- completion ---\n{completion}\n")

        if args.batched_prefill:
            batched_prefill_check(mgr, tokenizer, requests)

        print(f"pool_total_evictions={mgr.pool.eviction_policy.eviction_count}")
    finally:
        release_model(mgr.model)

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
