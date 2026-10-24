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

from multi_lora.batching import forward_logits_rowwise_reference
from multi_lora.hf_runner import (
    load_causal_lm,
    load_tokenizer,
    pick_device,
    pick_dtype,
    release_model,
)
from multi_lora.manager import MultiLoRAManager

# DEFAULT_BASE = "Qwen/Qwen2-7B-Instruct"
# ADAPTER_A = "FuturisticVibes/Qwen2-7B-TestInstructFinetune-LORA"
# ADAPTER_B = "Bibaolong/Context-Faithful-Qwen2-7B-Instruct"


DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"
# rank 16 只训练了qkvo
ADAPTER_A = "GestaltLabs/Jeff-1"
# rank 64 qkvo 以及 mlp 部分
ADAPTER_B = "/root/autodl-tmp/qwen3_4b_r64_lora/v2-20261022-131552/last-checkpoint"


def main() -> int:
    """Phase-3 冒烟;manager grouped forward 与逐行参考 logits 对比,并演练 pool 驱逐。"""
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
    id_a = "a"
    # 第一行用 adapter a,第二行 base(None)
    lora_names = [id_a, None]

    enc = {k: v.to(next(mgr.model.parameters()).device) for k, v in enc.items()}
    grouped, info = mgr.forward_logits_peft(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        lora_names=lora_names,
    )
    # 内部 id 就是 adapter 名,逐行参考直接用同一份 lora_names
    peft_keys = lora_names
    ref = forward_logits_rowwise_reference(
        mgr.model,
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        lora_names=peft_keys,
    )
    ok = torch.allclose(grouped, ref, rtol=2e-3, atol=2e-3)
    print(f"weight_indices={info.weight_indices} close={ok}")

    id_b = "b"
    # 连续两次 prepare,触发 pool 槽位复用/驱逐
    mgr.prepare_lora_batch([id_a, id_b])
    mgr.prepare_lora_batch([id_a, None])
    print(f"pool_total_evictions={mgr.pool.eviction_policy.eviction_count}")

    release_model(mgr.model)
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
