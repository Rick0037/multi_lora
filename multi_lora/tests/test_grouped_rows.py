"""T8 行分组合并:相邻同名行 RLE 成一个 segment.

三层验证:
    1. group_rows 纯逻辑(交错 = 不合并,相邻 = 合并,prefill 段长相加)
    2. kernel 级:同一份 x/权重,分组 metadata(GEMM 路径) vs 逐行
       metadata(GEMV 路径)的 A/B 输出一致,且与朴素 torch 参考一致
    3. manager 级:grouped 名字走 forward_with_lora,logits 与逐行
       PEFT golden 一致,且 metadata 真的压缩了(bs = 组数)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from multi_lora.batch_info import KernelInputMetaData, group_rows
from multi_lora.batching import forward_logits_rowwise_reference
from multi_lora.kernels import sgemm_lora_a_fwd, sgemm_lora_b_fwd
from multi_lora.manager import MultiLoRAManager

# ---------- 1. RLE 纯逻辑 ----------


def test_interleaved_names_are_noop():
    """交错排布没有相邻重复,输出与输入一一对应."""
    names = ["a", "b", "a", "b"]
    group_names, group_lens = group_rows(names, [])
    assert group_names == ["a", "b", "a", "b"]
    assert group_lens == [1, 1, 1, 1]


def test_consecutive_names_merge():
    """相邻同名行合并,None(基座)行也一样."""
    names = ["a", "a", "b", "b", "b", None, None]
    group_names, group_lens = group_rows(names, [])
    assert group_names == ["a", "b", None]
    assert group_lens == [2, 3, 2]


def test_prefill_lens_sum_up():
    """prefill 的行 token 数相加成段长."""
    group_names, group_lens = group_rows(["a", "a", "b"], [5, 7, 3])
    assert group_names == ["a", "b"]
    assert group_lens == [12, 3]


def test_len_mismatch_raises():
    with pytest.raises(ValueError):
        group_rows(["a", "b"], [1])


# ---------- 2. kernel 级:分组 vs 逐行 ----------


def _make_meta(lens, widx, ranks, scalings, dev):
    indptr = [0]
    for length in lens:
        indptr.append(indptr[-1] + length)
    return KernelInputMetaData(
        bs=len(lens),
        seg_lens=torch.tensor(lens, dtype=torch.int32, device=dev),
        seg_indptr=torch.tensor(indptr, dtype=torch.int32, device=dev),
        weight_indices=torch.tensor(widx, dtype=torch.int32, device=dev),
        lora_ranks=torch.tensor(ranks, dtype=torch.int32, device=dev),
        scalings=torch.tensor(scalings, dtype=torch.float32, device=dev),
        max_len=max(lens),
    )


def _naive_delta(x, a_stack, b_stack, widx_rows, ranks, scalings):
    """逐行朴素参考:delta[r] = scaling * x[r] @ A[:r]ᵀ @ B[:, :r]ᵀ."""
    deltas = []
    for row, slot in enumerate(widx_rows):
        rank = ranks[slot]
        if rank == 0:
            deltas.append(torch.zeros(b_stack.shape[1], dtype=x.dtype))
            continue
        a = a_stack[slot, :rank].to(x.dtype)
        b = b_stack[slot, :, :rank].to(x.dtype)
        deltas.append((x[row] @ a.T @ b.T) * scalings[slot])
    return torch.stack(deltas)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 跑 Triton kernel")
@pytest.mark.parametrize("rows_per_adapter", [2, 8])
def test_grouped_matches_rowwise_decode(rows_per_adapter):
    """decode 形状:分组(GEMM 路径, max_len>1) vs 逐行(GEMV 路径, max_len=1)."""
    dev, dt = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)
    # out_dim 必须是 256 的倍数:B 的 GEMM 路径 store 缺 n 方向 mask(T12),
    # 非 256 倍数会越界写,这里不测那个 bug
    in_dim, out_dim, max_rank = 96, 256, 64
    ranks = [8, 16]                       # 两个 slot 不同 rank
    scalings = [1.0, 2.0]

    a_stack = torch.randn(2, max_rank, in_dim, device=dev, dtype=dt) * 0.1
    b_stack = torch.randn(2, out_dim, max_rank, device=dev, dtype=dt) * 0.1
    rows = rows_per_adapter * 2
    x = torch.randn(rows, in_dim, device=dev, dtype=dt)
    widx_rows = [0] * rows_per_adapter + [1] * rows_per_adapter
    base = torch.randn(rows, out_dim, device=dev, dtype=dt)  # 两条路径共用

    # 分组:每 adapter 一段;逐行:每行一段
    meta_group = _make_meta(
        [rows_per_adapter] * 2, [0, 1], ranks, scalings, dev)
    meta_row = _make_meta([1] * rows, widx_rows, ranks, scalings, dev)
    assert meta_group.max_len > 1 and meta_row.max_len == 1  # GEMM vs GEMV

    outs = {}
    for tag, meta in (("group", meta_group), ("row", meta_row)):
        mid = torch.zeros(rows, max_rank, device=dev, dtype=dt)
        mid = sgemm_lora_a_fwd(x, a_stack, meta, mid)
        y = sgemm_lora_b_fwd(mid, b_stack, meta, base_output=base.clone())
        outs[tag] = (mid, y)

    # A 中间量:各行的 rank 截断区外必须都是 0
    for tag in ("group", "row"):
        mid = outs[tag][0]
        for row, slot in enumerate(widx_rows):
            assert (mid[row, ranks[slot]:] == 0).all(), (tag, row)

    # 分组 == 逐行,且都 == base + 朴素 delta
    ref = base + _naive_delta(x, a_stack, b_stack, widx_rows, ranks, scalings)
    torch.testing.assert_close(outs["group"][1], outs["row"][1], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(outs["group"][1], ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(outs["row"][1], ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 跑 Triton kernel")
def test_grouped_matches_rowwise_prefill():
    """prefill 形状:多 token 段合并前后的 A/B 输出一致."""
    dev, dt = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(1)
    in_dim, out_dim, max_rank = 96, 256, 64  # out_dim 取 256 倍数,避开 T12
    ranks, scalings = [8, 16], [1.0, 2.0]
    a_stack = torch.randn(2, max_rank, in_dim, device=dev, dtype=dt) * 0.1
    b_stack = torch.randn(2, out_dim, max_rank, device=dev, dtype=dt) * 0.1

    row_lens = [5, 5, 3, 3]              # 行 0/1 用 a,行 2/3 用 b
    rows = len(row_lens)
    total = sum(row_lens)
    x = torch.randn(total, in_dim, device=dev, dtype=dt)
    widx_rows = [0, 0, 1, 1]
    base = torch.randn(total, out_dim, device=dev, dtype=dt)

    meta_group = _make_meta([10, 6], [0, 1], ranks, scalings, dev)
    meta_row = _make_meta(row_lens, widx_rows, ranks, scalings, dev)
    assert meta_group.max_len == 10 and meta_row.max_len == 5  # 都走 GEMM

    outs = {}
    for tag, meta in (("group", meta_group), ("row", meta_row)):
        mid = torch.zeros(total, max_rank, device=dev, dtype=dt)
        mid = sgemm_lora_a_fwd(x, a_stack, meta, mid)
        y = sgemm_lora_b_fwd(mid, b_stack, meta, base_output=base.clone())
        outs[tag] = y

    torch.testing.assert_close(outs["group"], outs["row"], rtol=2e-2, atol=2e-2)


# ---------- 3. manager 级:grouped forward vs PEFT golden ----------


@pytest.fixture(scope="module")
def tiny_mgr(tmp_path_factory):
    """tiny-gpt2 + a/b 两个 adapter,torch 后端(数值路径与 triton 相同)."""
    tmp = tmp_path_factory.mktemp("t8_adapters")
    base = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    for name, seed in (("a", 11), ("b", 22)):
        torch.manual_seed(seed)
        m = copy.deepcopy(base)
        cfg = LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=["c_attn"],
        )
        get_peft_model(m, cfg).save_pretrained(tmp / name)

    mgr = MultiLoRAManager.from_pretrained(
        GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2"),
        {"a": str(tmp / "a"), "b": str(tmp / "b")},
        max_loras_per_batch=2,
        max_lora_rank=8,
        lora_backend="torch",
    )
    return mgr, tokenizer


def test_grouped_forward_matches_rowwise(tiny_mgr):
    """grouped 名字走完整 forward,logits 与逐行 PEFT golden 一致."""
    mgr, tokenizer = tiny_mgr
    texts = ["hi", "bye there", "ok", "sure thing"]
    enc = tokenizer(texts, padding=True, return_tensors="pt")
    seq = int(enc["input_ids"].shape[1])

    names = ["a", "a", "b", "b"]          # 相邻重复 → 2 段
    out = mgr.forward_with_lora(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        lora_names=names,
        seq_lens=[seq] * len(names),
    )
    # metadata 真的压缩了:bs = 组数,段长 = 2*seq
    assert mgr._resident_meta.bs == 2
    assert mgr._resident_meta.max_len == 2 * seq

    ref = forward_logits_rowwise_reference(
        mgr.model,
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        lora_names=names,
    )
    torch.testing.assert_close(out.logits, ref, rtol=1e-3, atol=1e-4)


def test_interleaved_forward_keeps_bs(tiny_mgr):
    """交错名字不合并,bs 仍是行数(行为与改动前一致)."""
    mgr, tokenizer = tiny_mgr
    texts = ["hi", "bye there", "ok", "sure thing"]
    enc = tokenizer(texts, padding=True, return_tensors="pt")
    seq = int(enc["input_ids"].shape[1])

    names = ["a", "b", "a", "b"]
    mgr.forward_with_lora(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        lora_names=names,
        seq_lens=[seq] * len(names),
    )
    assert mgr._resident_meta.bs == 4
    assert mgr._resident_meta.max_len == seq
