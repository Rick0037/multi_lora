"""Tests for multi_lora.batching (grouped vs row-wise reference)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, GPT2LMHeadModel

from multi_lora.batching import (
    forward_logits_grouped,
    forward_logits_rowwise_reference,
    group_row_indices,
    load_peft_multi_adapter,
    validate_lora_batch,
)


@pytest.fixture(scope="module")
def tiny_peft_two_adapters(tmp_path_factory):
    """准备 tiny-gpt2 上的两个 PEFT 适配器目录与 tokenizer。"""
    tmp = tmp_path_factory.mktemp("peft_adapters")
    base = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    for name, seed in (("a", 101), ("b", 202)):
        torch.manual_seed(seed)
        m = copy.deepcopy(base)
        cfg = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn"],
        )
        peft_m = get_peft_model(m, cfg)
        peft_m.save_pretrained(tmp / name)

    return {"tokenizer": tokenizer, "adapter_root": tmp}


def test_group_row_indices():
    """group_row_indices 应按 LoRA id 将 batch 行索引分组。"""
    g = group_row_indices(["x", None, "x", "y", None])
    assert list(g.keys()) == ["x", None, "y"]
    assert g["x"] == [0, 2]
    assert g[None] == [1, 4]
    assert g["y"] == [3]


def test_validate_lora_batch():
    """validate_lora_batch 应正确判断 distinct LoRA 数量是否超过槽位上限。"""
    assert validate_lora_batch(["a", "a", None], max_loras_per_batch=3)
    assert not validate_lora_batch(["a", "b", None], max_loras_per_batch=2)


def test_grouped_matches_rowwise(tiny_peft_two_adapters):
    """grouped forward 与 row-wise 参考在 tiny-gpt2 上 logits 应一致。"""
    tokenizer = tiny_peft_two_adapters["tokenizer"]
    root: Path = tiny_peft_two_adapters["adapter_root"]

    base = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    model = load_peft_multi_adapter(
        base,
        {"a": str(root / "a"), "b": str(root / "b")},  # adapter mapping
        first="a",
    )
    model.eval()

    texts = ["hello", "world test", "x", "y z"]
    enc = tokenizer(
        texts,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    lora_ids = ["a", None, "b", "a"]  # 直接用 adapter 名作为内部 id

    g = forward_logits_grouped(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        lora_ids=lora_ids,
    )
    r = forward_logits_rowwise_reference(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        lora_ids=lora_ids,
    )

    assert g.shape == r.shape
    assert torch.allclose(g, r, rtol=1e-3, atol=1e-4)
