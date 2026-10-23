"""Tests for multi_lora_phase2.batching (grouped vs row-wise reference)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from multi_lora_phase2.batching import (
    forward_logits_grouped,
    forward_logits_rowwise_reference,
    group_row_indices,
    load_peft_multi_adapter,
    validate_lora_batch,
)

VOCAB = 128
HIDDEN = 32


def _tiny_base() -> GPT2LMHeadModel:
    """按 config 直接构造 2 层 tiny GPT-2,避免下载 ``sshleifer/tiny-gpt2``。"""
    torch.manual_seed(1234)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=VOCAB,
            n_positions=64,
            n_embd=HIDDEN,
            n_layer=2,
            n_head=4,
            n_inner=64,
        )
    )


def _save_adapter(base, path: Path, *, r: int, alpha: float, seed: int) -> None:
    """在 ``base`` 的深拷贝上注入 LoRA 并落盘。

    ``init_lora_weights=False`` 让 ``lora_B`` 也随机初始化;否则默认 ``lora_B=0``,
    ``ΔW = B@A = 0``,adapter 与基座输出完全相同,测试就验证不了切换是否生效。
    """
    torch.manual_seed(seed)
    peft_m = get_peft_model(
        copy.deepcopy(base),
        LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn"],
            init_lora_weights=False,
        ),
    )
    peft_m.save_pretrained(path)


# * 整个测试模块只执行一次
@pytest.fixture(scope="module")
def tiny_adapters(tmp_path_factory):
    """造三个 adapter:r4(α=8)、r8(α=16)、以及与 r4 同权重仅 α 不同的 r4_alpha16。"""
    tmp = tmp_path_factory.mktemp("peft_adapters")
    base = _tiny_base()
    _save_adapter(base, tmp / "r4", r=4, alpha=8, seed=101)
    _save_adapter(base, tmp / "r8", r=8, alpha=16, seed=202)
    # 同 seed + 同 r => lora_A/lora_B 完全一致,只有 alpha 不同,用于验证 scaling
    _save_adapter(base, tmp / "r4_alpha16", r=4, alpha=16, seed=101)
    return {"base": base, "root": tmp}


def _load_single(base, path) -> PeftModel:
    model = PeftModel.from_pretrained(
        copy.deepcopy(base), str(path), is_trainable=False
    )
    model.eval()
    return model


def _fixed_batch():
    """4 条长度不一的假 token(0 为 pad),用于覆盖 padding 与行号错位。"""
    input_ids = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 8],
            [9, 0, 0, 0, 0],
            [10, 11, 12, 13, 0],
        ]
    )
    return input_ids, (input_ids != 0).long()


def test_group_row_indices():
    """``group_row_indices`` 应按首次出现顺序分组且行号正确。"""
    g = group_row_indices(["x", None, "x", "y", None])
    assert list(g.keys()) == ["x", None, "y"]
    assert g["x"] == [0, 2]
    assert g[None] == [1, 4]
    assert g["y"] == [3]


def test_validate_lora_batch():
    """distinct adapter 数(含基座 ``None``)不得超过 ``max_loras_per_batch``。"""
    assert validate_lora_batch(["a", "a", None], max_loras_per_batch=3)
    assert not validate_lora_batch(["a", "b", None], max_loras_per_batch=2)


def test_grouped_matches_rowwise(tiny_adapters):
    """分组前向 logits,逐行对照(验证分组与 scatter 的位置正确性)。"""
    base = tiny_adapters["base"]
    root: Path = tiny_adapters["root"]

    model = load_peft_multi_adapter(
        copy.deepcopy(base),
        {"a": str(root / "r4"), "b": str(root / "r8")},
        first="a",
    )
    model.eval()

    input_ids, attention_mask = _fixed_batch()
    lora_ids = ["a", None, "b", "a"]

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


def test_adapter_switch_effective(tiny_adapters):
    """同一输入下 a / 基座 / b 三行输出必须互不相同。

    捕获 ``set_adapter`` 未生效(各轮都沿用上一个 adapter)以及 adapter 权重被串用。
    """
    base = tiny_adapters["base"]
    root: Path = tiny_adapters["root"]

    model = load_peft_multi_adapter(
        copy.deepcopy(base),
        {"a": str(root / "r4"), "b": str(root / "r8")},
        first="a",
    )
    model.eval()

    # 三行输入完全相同,差异只可能来自 adapter
    input_ids = torch.tensor([[1, 2, 3, 4, 5]] * 3)
    attention_mask = torch.ones_like(input_ids)
    lora_ids = ["a", None, "b"]

    g = forward_logits_grouped(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        lora_ids=lora_ids,
    )

    assert not torch.allclose(
        g[0], g[1], rtol=1e-3, atol=1e-4
    ), "adapter a 与基座应不同"
    assert not torch.allclose(
        g[2], g[1], rtol=1e-3, atol=1e-4
    ), "adapter b 与基座应不同"
    assert not torch.allclose(g[0], g[2], rtol=1e-3, atol=1e-4), "adapter a 与 b 应不同"


def test_rank_mix_matches_solo(tiny_adapters):
    """r=4 与 r=8 混装后,每个 adapter 的输出应与单独加载时完全一致。

    捕获:rank 混用导致加载错位、两个 adapter 权重互相覆盖、切换后仍用旧 adapter。
    """
    base = tiny_adapters["base"]
    root: Path = tiny_adapters["root"]

    multi = load_peft_multi_adapter(
        copy.deepcopy(base),
        {"a": str(root / "r4"), "b": str(root / "r8")},
        first="a",
    )
    multi.eval()
    solo_a = _load_single(base, root / "r4")
    solo_b = _load_single(base, root / "r8")

    input_ids, attention_mask = _fixed_batch()
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    with torch.inference_mode():
        multi.set_adapter("a")
        got_a = multi(**kwargs).logits
        multi.set_adapter("b")
        got_b = multi(**kwargs).logits
        exp_a = solo_a(**kwargs).logits
        exp_b = solo_b(**kwargs).logits

    assert not torch.allclose(
        exp_a, exp_b, rtol=1e-3, atol=1e-4
    ), "两个 adapter 应产生不同输出"
    assert torch.allclose(got_a, exp_a, rtol=1e-4, atol=1e-5), "r4 输出应等于单独加载"
    assert torch.allclose(got_b, exp_b, rtol=1e-4, atol=1e-5), "r8 输出应等于单独加载"


def test_scaling_follows_alpha_over_r(tiny_adapters):
    """验证 scaling = alpha / r：同权重下 alpha 翻倍,ΔW 应精确翻倍。"""
    base = tiny_adapters["base"]
    root: Path = tiny_adapters["root"]

    m8 = _load_single(base, root / "r4")  # alpha=8, r=4 -> scaling=2
    m16 = _load_single(base, root / "r4_alpha16")  # alpha=16, r=4 -> scaling=4

    x = torch.randn(2, 5, HIDDEN)

    def delta(model) -> torch.Tensor:
        layer = model.base_model.model.transformer.h[0].attn.c_attn
        with torch.no_grad():
            return layer(x) - layer.base_layer(x)

    d8, d16 = delta(m8), delta(m16)
    assert d8.abs().max() > 0, "LoRA 分支应有非零贡献"
    assert torch.allclose(d16, 2 * d8, rtol=1e-5, atol=1e-6), "alpha 翻倍,ΔW 应翻倍"

    # 直接按 ΔW = (alpha/r) * (x @ Aᵀ @ Bᵀ) 手算对照
    layer = m8.base_model.model.transformer.h[0].attn.c_attn
    a = layer.lora_A["default"].weight.detach()
    b = layer.lora_B["default"].weight.detach()
    with torch.no_grad():
        expected = (x @ a.T) @ b.T * (8 / 4)
    assert torch.allclose(
        d8, expected, rtol=1e-5, atol=1e-6
    ), "scaling 公式应为 alpha/r"
