"""CUDA Graph 录下来的那一步 decode,重放结果应与直接跑一致.

bench 只负责测速度,录得对不对由这里保证。
"""

from __future__ import annotations

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from multi_lora.graph_decode import DecodeStep, capture_graph
from multi_lora.manager import MultiLoRAManager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA Graph 需要 GPU")

LOGIT_ATOL = 1e-2     # bf16 下 logits 的容差
CHECK_STEPS = 5       # 比几步
BATCH = 4


@pytest.fixture(scope="module")
def graph_mgr(tmp_path_factory):
    """tiny-gpt2 + a/b 两个 adapter,放在 CUDA 上走 triton 后端."""
    tmp = tmp_path_factory.mktemp("graph_adapters")
    base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").cuda()
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    cfg = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=["c_attn"])
    for name, seed in (("a", 11), ("b", 22)):
        torch.manual_seed(seed)
        get_peft_model(copy.deepcopy(base), cfg).save_pretrained(tmp / name)

    mgr = MultiLoRAManager.from_pretrained(
        base,
        {"a": str(tmp / "a"), "b": str(tmp / "b")},
        max_loras_per_batch=2,
        max_lora_rank=4,
        lora_backend="triton",
    )
    return mgr, tokenizer


def _prefill(mgr, enc, names, start, max_len):
    """建一套常驻张量,prefill 填 KV,返回 (step, 首 token)."""
    step = DecodeStep.new(mgr, BATCH, start, max_len)
    first = step.prefill(
        mgr, enc["input_ids"], enc["attention_mask"], names)
    step.input_ids.copy_(first)
    mgr._setup_forward_batch(names, seq_lens=[1] * BATCH)
    return step, first


@torch.inference_mode()
def test_replay_matches_direct_run(graph_mgr):
    """一路录成 graph、一路直接跑,同样走 CHECK_STEPS 步,logits 应一致."""
    mgr, tokenizer = graph_mgr
    enc = tokenizer(["one two three four"] * BATCH, padding=True,
                    return_tensors="pt").to("cuda")
    names = ["a", "b", "a", "b"]
    start = int(enc["input_ids"].shape[1])
    # 每步写一格:彩排 3 + 录制 1 + 比对 CHECK_STEPS
    max_len = start + 3 + 1 + CHECK_STEPS + 2

    recorded = _prefill(mgr, enc, names, start, max_len)[0]
    graph, out = capture_graph(mgr, recorded)

    direct = _prefill(mgr, enc, names, start, max_len)[0]

    for step in range(CHECK_STEPS):
        graph.replay()
        mine = out.logits[:, -1]
        theirs = direct.run(mgr).logits[:, -1]
        gap = (mine.float() - theirs.float()).abs().max().item()
        assert gap <= LOGIT_ATOL, f"第 {step} 步 max|diff|={gap:.4f}"
        nxt = mine.argmax(dim=-1, keepdim=True)
        recorded.input_ids.copy_(nxt)
        direct.input_ids.copy_(nxt)


@torch.inference_mode()
def test_capture_reuses_same_tensors(graph_mgr):
    """录制不该重新分配张量:capture 前后常驻张量的地址必须不变."""
    mgr, tokenizer = graph_mgr
    enc = tokenizer(["one two three four"] * BATCH, padding=True,
                    return_tensors="pt").to("cuda")
    names = ["a", "b", "a", "b"]
    start = int(enc["input_ids"].shape[1])
    step, _ = _prefill(mgr, enc, names, start, start + 8)

    addresses = [t.data_ptr() for t in
                 (step.kv.layers[0].keys, step.input_ids, step.position_ids,
                  step.kv_pos, step.attn_mask)]
    capture_graph(mgr, step)

    after = [t.data_ptr() for t in
             (step.kv.layers[0].keys, step.input_ids, step.position_ids,
              step.kv_pos, step.attn_mask)]
    assert addresses == after
