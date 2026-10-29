#!/usr/bin/env python3
"""Qwen3-4B-Instruct + multi-LoRA text generation (one load, multiple adapters).

Uses MultiLoRAManager: registry + GPU weight pool + PEFT set_adapter for
generation. Distinct adapters in one process share the same base weights.

Example (container, ~24GB GPU):

    cd /home/multi_lora
    python multi_lora/examples/inference_qwen4b_lora.py

    python multi_lora/examples/inference_qwen4b_lora.py \\
        --max-new-tokens 64

    # Custom prompts: adapter name per line (empty = base), then prompt text until blank line
    python multi_lora/examples/inference_qwen4b_lora.py --prompts-file my_prompts.txt
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
    p.add_argument(
        "--batched-prefill",
        action="store_true",
        help="After generation, run one grouped prefill on all prompts (sanity check)",
    )
    p.add_argument(
        "--triton",
        action="store_true",
        help="Also生成一遍 Triton kernel 路径,并与 PEFT 分组结果逐 token 对比",
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
            raise ValueError(
                f"empty prompt in block starting with {name_raw!r}")
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


@torch.inference_mode()
def greedy_decode_peft(
    model: PeftModel,
    tokenizer,
    prompt: str,
    *,
    peft_adapter: Optional[str],
    max_new_tokens: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """PEFT 路径逐步贪心解码;作为 Triton 对比的参考,返回 (新 token, 每步 logits)."""
    enc = tokenizer(prompt, return_tensors="pt")
    device = model_input_device(model)
    cur_ids = enc["input_ids"].to(device)
    cur_mask = enc["attention_mask"].to(device)

    # base 行要临时关掉 LoRA;adapter 行切到对应 adapter
    if peft_adapter is None:
        ctx = model.disable_adapter()
    else:
        model.set_adapter(peft_adapter)
        ctx = nullcontext()

    new_ids: List[torch.Tensor] = []
    step_logits: List[torch.Tensor] = []
    past = None
    with ctx:
        for _ in range(max_new_tokens):
            out = model(
                input_ids=cur_ids,
                attention_mask=cur_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1]
            step_logits.append(logits)
            nxt = logits.argmax(dim=-1, keepdim=True)
            new_ids.append(nxt)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
            cur_ids = nxt
            cur_mask = torch.cat(
                [cur_mask, cur_mask.new_ones((cur_mask.size(0), 1))], dim=1
            )
    return torch.cat(new_ids, dim=1)[0], step_logits


@torch.inference_mode()
def greedy_decode_triton(
    mgr: MultiLoRAManager,
    tokenizer,
    prompt: str,
    *,
    lora_name: Optional[str],
    max_new_tokens: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Triton 路径逐步贪心解码:prefill 和 decode 都走 forward_with_lora."""
    enc = tokenizer(prompt, return_tensors="pt")
    device = model_input_device(mgr.model)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    names = [lora_name]

    out = mgr.forward_with_lora(
        input_ids=ids,
        attention_mask=mask,
        lora_names=names,
        use_cache=True,
    )
    past = out.past_key_values
    logits = out.logits[:, -1]

    new_ids: List[torch.Tensor] = []
    step_logits: List[torch.Tensor] = []
    # decode 阶段每行只有 1 个 token;显式传 seq_lens 才走 decode 分支,
    # 否则 manager 会按 attention_mask 的列数当成 prefill 段长
    decode_lens = [1] * ids.size(0)  # CPU list,不用建 GPU 张量
    for _ in range(max_new_tokens):
        step_logits.append(logits)
        nxt = logits.argmax(dim=-1, keepdim=True)
        new_ids.append(nxt)
        if int(nxt.item()) == tokenizer.eos_token_id:
            break
        out = mgr.forward_with_lora(
            input_ids=nxt,
            attention_mask=None,
            seq_lens=decode_lens,
            lora_names=names,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1]
    return torch.cat(new_ids, dim=1)[0], step_logits


def compare_generations(
    label: str,
    tokenizer,
    peft_ids: torch.Tensor,
    peft_logits: List[torch.Tensor],
    tri_ids: torch.Tensor,
    tri_logits: List[torch.Tensor],
) -> bool:
    """逐 token 与逐步 logits 对比;分歧处打印 PEFT 的 top1-top2 间距."""
    n = min(len(peft_logits), len(tri_logits))
    step_diffs = [(peft_logits[i] - tri_logits[i]).abs().max().item()
                  for i in range(n)]
    max_diff = max(step_diffs) if step_diffs else 0.0
    mean_diff = sum(step_diffs) / len(step_diffs) if step_diffs else 0.0

    same = torch.equal(peft_ids, tri_ids)
    print(
        f"  [{label}] tokens_match={same} peft_len={peft_ids.numel()} "
        f"triton_len={tri_ids.numel()}"
    )
    print(
        f"  [{label}] per-step logits max_abs_diff={max_diff:.6g} "
        f"mean_abs_diff={mean_diff:.6g}"
    )
    print(f"  [{label}] step_diffs=" +
          " ".join(f"{d:.3g}" for d in step_diffs))

    if same:
        return True

    # 分歧位置:看 PEFT 参考的 top1-top2 间距,判断是不是近似平票被 bf16 噪声翻盘
    m = min(peft_ids.numel(), tri_ids.numel())
    bad = (peft_ids[:m] != tri_ids[:m]).nonzero()
    first_bad = int(bad[0, 0]) if bad.numel() else m
    if first_bad < len(peft_logits):
        top2 = peft_logits[first_bad].topk(2, dim=-1).values[0]
        gap = (top2[0] - top2[1]).item()
        p_tok = tokenizer.decode([int(peft_ids[first_bad])])
        t_tok = tokenizer.decode([int(tri_ids[first_bad])])
        print(
            f"  [{label}] first_mismatch at token {first_bad}: "
            f"peft={p_tok!r} triton={t_tok!r} peft_top1_top2_gap={gap:.4g}"
        )
    return False


def run_one_request(
    mgr: MultiLoRAManager,
    tokenizer,
    adapter_name: Optional[str],
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """校验适配器名、预热 GPU pool slot,然后生成文本。"""
    lora_id = mgr.registry.normalize_names(adapter_name)

    # 将当前 LoRA 加载到 pool,供后续 PEFT forward 使用
    info = mgr.prepare_lora_batch([lora_id])
    label = adapter_name if adapter_name else "base"
    print(
        f"  pool slot={info.weight_indices[0]} "
        f"rank={info.lora_ranks[info.weight_indices[0]]} "
        f"scalings={info.scalings[:3]}"
    )
    return generate_with_peft(
        mgr.model,
        tokenizer,
        prompt,
        peft_adapter=lora_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


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
    lora_names = mgr.registry.normalize_names([n for n, _ in requests])
    logits, info = mgr.forward_logits_peft(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        lora_names=lora_names,
    )
    # 每条序列 prefill 最后一个有效位置
    last_idx = enc["attention_mask"].sum(dim=1) - 1
    norms = []
    for i, li in enumerate(last_idx.tolist()):
        norms.append(logits[i, li].float().norm().item())
    print("\n--- batched prefill (grouped forward) ---")
    print(f"weight_indices={info.weight_indices} last_hidden_norms={norms}")


def main() -> int:
    """加载 Qwen3-4B 与 multi-LoRA manager,逐条请求生成并可选 batched prefill 校验。"""
    args = parse_args()
    requests = (
        load_requests(
            args.prompts_file) if args.prompts_file else DEFAULT_REQUESTS
    )

    device = pick_device()
    dtype = pick_dtype(device)

    print("multi_lora Qwen3-4B multi-LoRA inference")
    print(f"base={args.base_model} device={device} dtype={dtype}")
    print(f"adapters: a={args.adapter_a}")
    print(f"          b={args.adapter_b}")
    print(f"requests={len(requests)}")

    tokenizer = load_tokenizer(args.base_model)
    base = load_causal_lm(
        args.base_model,
        device=device,
        dtype=dtype,
    )
    mgr = MultiLoRAManager.from_pretrained(
        base,
        {"a": args.adapter_a, "b": args.adapter_b},
        max_loras_per_batch=args.max_loras_per_batch,
        max_lora_rank=args.max_lora_rank,
        # --triton 时用 Triton kernel 算 LoRA delta,否则走 torch 参考实现
        lora_backend="triton" if args.triton else "torch",
        # 本脚本走 PEFT 原生生成路径(set_adapter / disable_adapter),
        # 需要保留 LoraLayer,不能换成 LinearWithLoRA
        convert_lora_layers=False,
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

        # PEFT 参考必须在替换层之前跑:替换后模型里没有 LoraLayer,
        # set_adapter / disable_adapter 会直接报错
        peft_refs: List[Tuple[torch.Tensor, List[torch.Tensor]]] = []
        if args.triton:
            for adapter_name, prompt in requests:
                lora_id = mgr.registry.normalize_names(adapter_name)
                peft_refs.append(
                    greedy_decode_peft(
                        mgr.model,
                        tokenizer,
                        prompt,
                        peft_adapter=lora_id,
                        max_new_tokens=args.max_new_tokens,
                    )
                )

        if args.batched_prefill:
            batched_prefill_check(mgr, tokenizer, requests)

        if args.triton:
            mgr.convert_to_lora_layers()
            print(
                f"\n--- Triton kernel generation vs PEFT (greedy) "
                f"wrapped_modules={len(mgr.lora_wrappers)} "
                f"backend={mgr._lora_backend.name} ---"
            )
            all_match = True
            for i, (adapter_name, prompt) in enumerate(requests):
                label = adapter_name if adapter_name else "base"
                lora_id = mgr.registry.normalize_names(adapter_name)
                tri_ids, tri_step = greedy_decode_triton(
                    mgr,
                    tokenizer,
                    prompt,
                    lora_name=lora_id,
                    max_new_tokens=args.max_new_tokens,
                )
                peft_ids, peft_step = peft_refs[i]
                print(f"\n=== [{i}] adapter={label} ===")
                print(
                    "  peft  : "
                    + tokenizer.decode(peft_ids, skip_special_tokens=True)
                    .strip()
                    .replace("\n", " ")
                )
                print(
                    "  triton: "
                    + tokenizer.decode(tri_ids, skip_special_tokens=True)
                    .strip()
                    .replace("\n", " ")
                )
                match = compare_generations(
                    label, tokenizer, peft_ids, peft_step, tri_ids, tri_step
                )
                all_match = all_match and match
            print(f"\ntriton_vs_peft tokens_all_match={all_match}")

        print(
            f"pool_total_evictions={mgr.pool.eviction_policy.eviction_count}")
    finally:
        release_model(mgr.model)

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
