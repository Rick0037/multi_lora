#!/usr/bin/env python3
"""Phase-1 smoke: Qwen2-7B-Instruct + one or two PEFT LoRA adapters (serial runs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow `python examples/smoke_single_adapter.py` from multi_lora/
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from multi_lora_phase3.hf_runner import (  # noqa: E402
    ModelSpec,
    generate_text,
    load_causal_lm,
    load_tokenizer,
    pick_device,
    pick_dtype,
    release_model,
)

DEFAULT_BASE = "Qwen/Qwen2-7B-Instruct"
# Standard PEFT LoRA (rank=32, no modules_to_save); Qwen2-7B family.
DEFAULT_LORA_1 = "FuturisticVibes/Qwen2-7B-TestInstructFinetune-LORA"
# Second adapter for serial switch; set empty via --lora-2 "" to skip.
DEFAULT_LORA_2 = "Bibaolong/Context-Faithful-Qwen2-7B-Instruct"

DEFAULT_PROMPT = (
    "<|im_start|>user\nList three countries and their capitals.\n"
    "<|im_start|>assistant\n"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--lora-1", default=DEFAULT_LORA_1, help="First LoRA (HF id or local path)")
    p.add_argument(
        "--lora-2",
        default=DEFAULT_LORA_2,
        help="Second LoRA for serial switch test (empty to skip)",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base-model-only generation (saves one full load).",
    )
    p.add_argument(
        "--low-vram",
        action="store_true",
        help="Offload part of the 7B model to CPU (for ~10GB free GPU).",
    )
    p.add_argument("--gpu-id", type=int, default=1, help="GPU index when using --low-vram")
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Force compute device (auto = cuda if available).",
    )
    return p.parse_args()


def run_case(
    spec: ModelSpec,
    *,
    base_model: str,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    low_vram: bool,
    gpu_id: int,
    device_name: str,
) -> str:
    if device_name == "cpu":
        device = torch.device("cpu")
        dtype = pick_dtype(device)
    else:
        device = pick_device() if device_name == "auto" else torch.device("cuda")
        dtype = pick_dtype(device)

    label = spec.label or (spec.lora_path or "base")

    print(f"\n=== [{label}] device={device} dtype={dtype} low_vram={low_vram} ===")
    if spec.lora_path:
        print(f"    base={base_model}")
        print(f"    lora={spec.lora_path}")
    else:
        print(f"    base={base_model} (no LoRA)")

    model = load_causal_lm(
        base_model,
        device=device if device_name != "auto" or not low_vram else device,
        dtype=dtype,
        lora_path=spec.lora_path,
        low_vram=low_vram and device_name != "cpu",
        gpu_id=gpu_id,
    )
    try:
        text = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    finally:
        release_model(model)

    print(f"--- completion ---\n{text}\n")
    return text


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.base_model)

    cases: list[ModelSpec] = []
    if not args.skip_base:
        cases.append(ModelSpec(base_model=args.base_model, label="base"))
    cases.append(
        ModelSpec(
            base_model=args.base_model,
            lora_path=args.lora_1,
            label="lora-1",
        )
    )
    if args.lora_2:
        cases.append(
            ModelSpec(
                base_model=args.base_model,
                lora_path=args.lora_2,
                label="lora-2",
            )
        )

    print("multi_lora phase-1 smoke")
    print(f"base_model={args.base_model}")
    print(f"cases={[c.label for c in cases]}")

    for spec in cases:
        run_case(
            spec,
            base_model=args.base_model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            low_vram=args.low_vram,
            gpu_id=args.gpu_id,
            device_name=args.device,
        )

    print("OK: all cases finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
