#!/usr/bin/env python3
"""Phase-3 smoke: registry + GPU pool + manager forward vs rowwise reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multi_lora_phase3.batching import forward_logits_rowwise_reference, logits_close
from multi_lora_phase3.hf_runner import load_causal_lm, load_tokenizer, pick_device, pick_dtype, release_model
from multi_lora_phase3.manager import MultiLoRAManager

DEFAULT_BASE = "Qwen/Qwen2-7B-Instruct"
ADAPTER_A = "FuturisticVibes/Qwen2-7B-TestInstructFinetune-LORA"
ADAPTER_B = "Bibaolong/Context-Faithful-Qwen2-7B-Instruct"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--adapter-a", default=ADAPTER_A)
    p.add_argument("--adapter-b", default=ADAPTER_B)
    p.add_argument("--low-vram", action="store_true")
    p.add_argument("--gpu-id", type=int, default=0)
    args = p.parse_args()

    device = pick_device()
    dtype = pick_dtype(device)
    tokenizer = load_tokenizer(args.base_model)
    texts = [
        "<|im_start|>user\nSay hi.\n<|im_start|>assistant\n",
        "<|im_start|>user\nSay bye.\n<|im_start|>assistant\n",
    ]
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
        max_loras_per_batch=2,
        max_lora_rank=64,
    )
    id_a = mgr._peft_name_to_id["a"]
    lora_ids = [id_a, None]

    enc = {k: v.to(next(mgr.model.parameters()).device) for k, v in enc.items()}
    grouped, info = mgr.forward_logits_grouped(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        lora_ids=lora_ids,
    )
    peft_keys = [mgr.peft_name_for_row(x) for x in lora_ids]
    ref = forward_logits_rowwise_reference(
        mgr.model,
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        lora_ids=peft_keys,
    )
    ok = logits_close(grouped, ref, rtol=2e-3, atol=2e-3)
    print(f"weight_indices={info.weight_indices} evictions={info.evictions} close={ok}")

    id_b = mgr._peft_name_to_id["b"]
    mgr.prepare_lora_batch([id_a, id_b])
    mgr.prepare_lora_batch([id_a, None])
    print(f"pool_total_evictions={mgr.pool.eviction_policy.eviction_count}")

    release_model(mgr.model)
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
