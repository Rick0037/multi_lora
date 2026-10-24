"""第四步;整模型 LinearWithLoRA + 后端."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from multi_lora.layers import lora_delta
from multi_lora.manager import MultiLoRAManager


class ToyLM(nn.Module):
    """测试用最小线性模型,单层 Linear 便于验证 LoRA 包装。"""

    def __init__(self):
        """初始化 Linear(16→32) 并带 bias。"""
        super().__init__()
        self.lin = nn.Linear(16, 32, bias=True)

    def forward(self, x):
        """执行 base linear forward。"""
        return self.lin(x)


@pytest.fixture(scope="module")
def tiny_linear_mgr(tmp_path_factory):
    """构造带 a/b 两个 LoRA 适配器的 MultiLoRAManager(torch 后端)。"""
    tmp = tmp_path_factory.mktemp("p4_linear")
    cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=["lin"],
    )
    for name, seed in (("a", 11), ("b", 22)):
        torch.manual_seed(seed)
        m = ToyLM()
        get_peft_model(m, cfg).save_pretrained(tmp / name)

    mgr = MultiLoRAManager.from_pretrained(
        ToyLM(),
        {"a": str(tmp / "a"), "b": str(tmp / "b")},
        max_loras_per_batch=3,
        max_lora_rank=8,
        lora_backend="torch",
    )
    return mgr


def _ref_rowwise(mgr, x, lora_names):
    """逐行参考;base linear + 每行单独 lora_delta,用于数值对比。"""
    mgr.prepare_lora_batch(lora_names)
    linear = mgr.lora_wrappers["lin"].base_layer
    yb = linear(x)
    out = yb.clone()
    for i, name in enumerate(lora_names):
        slot = mgr.pool.get_slot_id(name)
        r, s = mgr.pool._slot_rank_scaling[slot]
        if r == 0:
            continue
        a, b = mgr.pool.get_slot_weights(slot, "lin")
        # 仅更新第 i 行
        out[i : i + 1] = yb[i : i + 1] + lora_delta(x[i : i + 1], a, b, s, rank=r)
    return out


def test_full_model_torch_matches_row_reference(tiny_linear_mgr):
    """整模型 forward_with_lora 输出应与逐行参考一致。"""
    mgr = tiny_linear_mgr
    x = torch.randn(
        3, 16, device=next(mgr.model.parameters()).device, dtype=mgr.pool.dtype
    )
    id_a, id_b = "a", "b"
    lora_names = [id_a, None, id_b]

    y = mgr.forward_with_lora(x=x, lora_names=lora_names)
    y_ref = _ref_rowwise(mgr, x, lora_names)
    assert torch.allclose(y, y_ref, rtol=1e-4, atol=1e-5)


def test_kernel_input_metadata_tensor_shapes(tiny_linear_mgr):
    """build_kernel_input_metadata 产出的张量形状应符合 segment GEMM 约定。"""
    mgr = tiny_linear_mgr
    id_a = "a"
    meta = mgr.build_kernel_input_metadata([id_a, None])
    assert meta.weight_indices.shape == (2,)
    assert meta.lora_ranks.shape[0] == mgr.max_loras_per_batch
    # 两行各一段;indptr = [0, 1, 2]
    assert meta.seg_indptr.tolist() == [0, 1, 2]


def test_decode_repeated_forward(tiny_linear_mgr):
    """模拟 decode;每步 [B, in],batch 级 prepare 可重复调用。"""
    mgr = tiny_linear_mgr
    x = torch.randn(
        2, 16, device=next(mgr.model.parameters()).device, dtype=mgr.pool.dtype
    )
    id_a, id_b = "a", "b"
    lora_names = [id_a, id_b]
    y1 = mgr.forward_with_lora(x=x, lora_names=lora_names)
    # 相同输入与 lora_names 重复 forward,结果应稳定
    y2 = mgr.forward_with_lora(x=x, lora_names=lora_names)
    assert torch.allclose(y1, y2)
