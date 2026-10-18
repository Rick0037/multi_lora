"""Phase-3: MultiLoRAManager + pool vs phase-2 grouped forward."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, GPT2LMHeadModel

from multi_lora_phase3.batching import forward_logits_rowwise_reference, logits_close
from multi_lora_phase3.layers import lora_delta
from multi_lora_phase3.manager import MultiLoRAManager


@pytest.fixture(scope="module")
def tiny_mgr(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p3_adapters")
    base = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    for name, seed in (("a", 11), ("b", 22)):
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
        get_peft_model(m, cfg).save_pretrained(tmp / name)

    mgr = MultiLoRAManager.from_pretrained(
        GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2"),
        {"a": str(tmp / "a"), "b": str(tmp / "b")},
        max_loras_per_batch=2,
        max_lora_rank=8,
    )
    return mgr, tokenizer


def test_manager_forward_matches_rowwise(tiny_mgr):
    mgr, tokenizer = tiny_mgr
    texts = ["hi", "bye there"]
    enc = tokenizer(texts, padding=True, return_tensors="pt")
    id_a = mgr._peft_name_to_id["a"]
    lora_ids = [id_a, None]

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
    assert logits_close(grouped, ref, rtol=1e-3, atol=1e-4)
    assert len(info.weight_indices) == 2


def test_pool_gpu_matches_cpu_store(tiny_mgr):
    mgr, _ = tiny_mgr
    id_a = mgr._peft_name_to_id["a"]
    mgr.prepare_lora_batch([id_a])
    slot = mgr.pool.get_buffer_id(id_a)
    key = next(iter(mgr._weight_store[id_a].modules))
    a_gpu, b_gpu = mgr.pool.get_slot_weights(slot, key)
    rank, _ = mgr.pool._slot_meta[slot]
    a_cpu, b_cpu = mgr._weight_store[id_a].modules[key]
    assert torch.allclose(a_gpu[:rank], a_cpu.to(dtype=a_gpu.dtype), rtol=0, atol=1e-6)
    assert torch.allclose(b_gpu[:, :rank], b_cpu.to(dtype=b_gpu.dtype), rtol=0, atol=1e-6)

    x = torch.randn(2, a_cpu.shape[1])
    d_cpu = lora_delta(x, a_cpu, b_cpu, mgr._weight_store[id_a].scaling, rank=rank)
    d_gpu = lora_delta(x, a_gpu, b_gpu, mgr._weight_store[id_a].scaling, rank=rank)
    assert torch.allclose(d_cpu, d_gpu, rtol=1e-4, atol=1e-5)


def test_eviction_across_batches(tiny_mgr):
    mgr, _ = tiny_mgr
    id_a, id_b = mgr._peft_name_to_id["a"], mgr._peft_name_to_id["b"]
    mgr.prepare_lora_batch([id_a, id_b])
    mgr.prepare_lora_batch([id_a, None])
    assert mgr.pool.eviction_policy.eviction_count >= 1
    assert id_b not in mgr.pool.uid_to_buffer_id
