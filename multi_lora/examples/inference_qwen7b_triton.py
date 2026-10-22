#!/usr/bin/env python3
"""Qwen2-7B: 整模型 multi-LoRA prefill(attention 四投影 Triton segment GEMM).

热路径为 MultiLoRAManager.forward_with_triton_lora;batch 级 prepare_lora_batch,
各层 LinearWithLoRA 内 sgemm_lora_a/b。默认仅替换 q/k/v/o_proj(MVP);
可用 --all-lora-modules 替换 weight store 中全部 LoRA 模块并与 PEFT grouped 对比。

Example::

    cd /home/multi_lora
    python multi_lora/examples/inference_qwen7b_triton.py
    python multi_lora/examples/inference_qwen7b_triton.py --low-vram --gpu-id 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multi_lora.backend.triton_backend import (
    TritonLoRABackend,
    _HAS_TRITON_KERNELS,
)  # noqa: E402
from multi_lora.hf_runner import (  # noqa: E402
    load_causal_lm,
    load_tokenizer,
    model_input_device,
    pick_device,
    pick_dtype,
    release_model,
)
from multi_lora.manager import ATTENTION_LORA_SUFFIXES, MultiLoRAManager  # noqa: E402

DEFAULT_BASE = "Qwen/Qwen2-7B-Instruct"
ADAPTER_A = "FuturisticVibes/Qwen2-7B-TestInstructFinetune-LORA"
ADAPTER_B = "Bibaolong/Context-Faithful-Qwen2-7B-Instruct"

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
    """解析命令行;模型路径、LoRA 池容量、Triton 覆盖范围与 PEFT 对比开关。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--adapter-a", default=ADAPTER_A)
    p.add_argument("--adapter-b", default=ADAPTER_B)
    p.add_argument("--prompts-file", type=Path, default=None)
    p.add_argument("--max-loras-per-batch", type=int, default=3)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--low-vram", action="store_true")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument(
        "--all-lora-modules",
        action="store_true",
        help="Replace all LoRA modules (enables full logits compare vs PEFT)",
    )
    p.add_argument(
        "--compare-peft",
        action="store_true",
        help="Compare last-token logits norm vs forward_logits_grouped (needs --all-lora-modules)",
    )
    return p.parse_args()


def load_requests(path: Path) -> List[Tuple[Optional[str], str]]:
    """从文本文件读取请求块;首行适配器名,后续为 prompt。"""
    text = path.read_text(encoding="utf-8")
    # 空行分隔多个 (adapter, prompt) 块
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    out: List[Tuple[Optional[str], str]] = []
    for block in blocks:
        lines = block.splitlines()
        name_raw = lines[0].strip()
        # base/none/空 表示不使用 LoRA
        name = None if name_raw.lower() in ("", "base", "none") else name_raw
        prompt = "\n".join(lines[1:]).strip()
        if not prompt:
            raise ValueError(f"empty prompt in block starting with {name_raw!r}")
        out.append((name, prompt))
    return out


@torch.inference_mode()
def main() -> int:
    """加载 Qwen2-7B + Triton backend,执行 batch prefill 并可选对比 PEFT。"""
    args = parse_args()
    requests = (
        load_requests(args.prompts_file) if args.prompts_file else DEFAULT_REQUESTS
    )

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for Triton SGEMM")
        return 1
    if not _HAS_TRITON_KERNELS:
        print("ERROR: Triton kernels not importable")
        return 1

    device = pick_device()
    dtype = pick_dtype(device)
    # 默认仅替换 attention 四投影;--all-lora-modules 时替换全部 LoRA 层
    suffixes = None if args.all_lora_modules else ATTENTION_LORA_SUFFIXES

    print("multi_lora Qwen2-7B full-model Triton prefill (MVP)")
    print(f"base={args.base_model} device={device} dtype={dtype}")
    print(f"lora_target_suffixes={suffixes or 'ALL'}")

    tokenizer = load_tokenizer(args.base_model)
    texts = [p for _, p in requests]
    enc = tokenizer(texts, padding=True, return_tensors="pt")

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
        lora_backend="triton",
        lora_target_suffixes=suffixes,
    )
    if not isinstance(mgr._lora_backend, TritonLoRABackend):
        print(f"ERROR: expected TritonLoRABackend, got {type(mgr._lora_backend)}")
        return 1

    print(f"wrapped_modules={len(mgr.lora_wrappers)} backend={mgr._lora_backend.name}")

    try:
        enc = {k: v.to(model_input_device(mgr.model)) for k, v in enc.items()}
        adapter_names = [n for n, _ in requests]
        lora_ids = mgr.registry.resolve_names(adapter_names)
        if not isinstance(lora_ids, list):
            lora_ids = [lora_ids]

        # 整模型 Triton prefill;每行可绑定不同 LoRA
        out = mgr.forward_with_triton_lora(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            lora_ids=lora_ids,
            use_cache=False,
        )
        logits = out.logits if hasattr(out, "logits") else out
        # 取每条序列 prefill 最后一个有效 token 的 logits
        last = enc["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[rows, last]
        for i, (name, _) in enumerate(requests):
            label = name if name else "base"
            print(
                f"  row[{i}] adapter={label} last_logit_norm={last_logits[i].float().norm().item():.4f}"
            )

        if args.compare_peft:
            if suffixes is not None:
                print(
                    "WARN: --compare-peft requires --all-lora-modules for full parity"
                )
            # PEFT grouped forward 作为数值参考
            peft_logits, _ = mgr.forward_logits_grouped(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                lora_ids=lora_ids,
            )
            diff = (last_logits - peft_logits[rows, last]).abs().max().item()
            ok = torch.allclose(
                last_logits, peft_logits[rows, last], rtol=1e-2, atol=1e-2
            )
            print(f"PEFT vs Triton last-token max_abs_diff={diff:.6g} allclose={ok}")
            if suffixes is None and not ok:
                return 1

        print(f"pool_total_evictions={mgr.pool.eviction_policy.eviction_count}")
        print("OK: forward_with_triton_lora prefill completed")
    finally:
        release_model(mgr.model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
