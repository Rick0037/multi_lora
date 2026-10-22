#!/usr/bin/env python3
"""Phase-2 smoke: one batch, different LoRA per row — compare grouped vs row-wise logits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multi_lora_phase2.batching import (  # noqa: E402
    forward_logits_grouped,
    forward_logits_rowwise_reference,
    load_peft_multi_adapter,
    logits_close,
)
from multi_lora_phase2.hf_runner import (  # noqa: E402
    load_causal_lm,
    load_tokenizer,
    model_input_device,
    pick_device,
    pick_dtype,
    release_model,
)

DEFAULT_BASE = "Qwen/Qwen2-7B-Instruct"
ADAPTER_A = "FuturisticVibes/Qwen2-7B-TestInstructFinetune-LORA"
ADAPTER_B = "Bibaolong/Context-Faithful-Qwen2-7B-Instruct"


def parse_args():
    """解析命令行;双 adapter 路径、每行 ``lora_ids``、设备与低显存选项。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--adapter-a", default=ADAPTER_A)
    p.add_argument("--adapter-b", default=ADAPTER_B)
    p.add_argument(
        "--names",
        default="a,b",
        help="PEFT adapter names (comma-separated), must match --lora-ids keys",
    )
    p.add_argument(
        "--lora-ids",
        default="a,,b,a,b,",
        help="Per-row adapter name or empty for base; comma-separated, same length as prompts",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=2e-2,
        help="Relative tolerance for last-token BF16 logits comparison",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=5e-1,
        help="Absolute tolerance for last-token BF16 logits comparison",
    )
    p.add_argument("--low-vram", action="store_true")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return p.parse_args()


def main() -> int:
    """加载双 LoRA,对比 ``forward_logits_grouped`` 与逐行对照的 logits 是否一致。"""
    args = parse_args()
    names = [x.strip() for x in args.names.split(",") if x.strip()]
    if len(names) != 2:
        raise SystemExit("expected --names a,b (two adapter names)")

    tokenizer = load_tokenizer(args.base_model)
    texts = [
        "<|im_start|>user\nSay hi in one word.\n<|im_start|>assistant\n",
        "<|im_start|>user\nSay bye in one word.\n<|im_start|>assistant\n",
        "<|im_start|>user\nName one fruit.\n<|im_start|>assistant\n",
        "<|im_start|>user\nName one animal.\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite one color.\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite one city.\n<|im_start|>assistant\n",
    ]
    enc = tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    raw_ids = [x.strip() for x in args.lora_ids.split(",")]
    if len(raw_ids) != len(texts):
        raise SystemExit(
            "--lora-ids must have one entry per prompt (use empty for base)"
        )
    lora_ids: list[str | None] = [x if x else None for x in raw_ids]

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = pick_device()
    else:
        device = pick_device()
    dtype = pick_dtype(device)

    base = load_causal_lm(
        args.base_model,
        device=device,
        dtype=dtype,
        lora_path=None,
        low_vram=args.low_vram,
        gpu_id=args.gpu_id,
    )
    adapters = {names[0]: args.adapter_a, names[1]: args.adapter_b}
    model = load_peft_multi_adapter(base, adapters, first=names[0])
    model.eval()

    input_ids = input_ids.to(model_input_device(model))
    attention_mask = attention_mask.to(model_input_device(model))

    g = forward_logits_grouped(
        model, input_ids=input_ids, attention_mask=attention_mask, lora_ids=lora_ids
    )
    r = forward_logits_rowwise_reference(
        model, input_ids=input_ids, attention_mask=attention_mask, lora_ids=lora_ids
    )

    if attention_mask is None:
        last_idx = torch.full(
            (g.size(0),), g.size(1) - 1, device=g.device, dtype=torch.long
        )
    else:
        last_idx = attention_mask.sum(dim=1).to(device=g.device, dtype=torch.long) - 1
    rows = torch.arange(g.size(0), device=g.device)
    g_last = g[rows, last_idx]
    r_last = r[rows, last_idx]

    # Full-sequence logits can differ at BF16 precision because Qwen uses different
    # batch-size dependent CUDA kernels for grouped sub-batches vs rowwise bs=1.
    # The serving-relevant prefill check is the final valid token per request.
    full_max_abs = (g - r).abs().max().item()
    last_max_abs = (g_last - r_last).abs().max().item()
    top1_equal = torch.equal(g_last.argmax(dim=-1), r_last.argmax(dim=-1))
    close = logits_close(g_last, r_last, rtol=args.rtol, atol=args.atol)
    ok = close and top1_equal
    print(
        "grouped vs rowwise "
        f"full_max_abs_diff={full_max_abs:.6g} "
        f"last_max_abs_diff={last_max_abs:.6g} "
        f"last_close={close} top1_equal={top1_equal} ok={ok}"
    )
    release_model(model)
    if not ok:
        raise SystemExit(1)
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
