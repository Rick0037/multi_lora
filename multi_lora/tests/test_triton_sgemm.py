"""CUDA Triton 整模型 forward vs Torch backend(ToyLM)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from multi_lora.manager import MultiLoRAManager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton SGEMM test",
)


class ToyLM(nn.Module):
    """CUDA 测试用 Toy 线性模型。"""

    def __init__(self):
        """初始化 Linear(64→128)。"""
        super().__init__()
        self.lin = nn.Linear(64, 128)

    def forward(self, x):
        """执行 base linear forward。"""
        return self.lin(x)


@pytest.fixture(scope="module")
def cuda_mgr_pair(tmp_path_factory):
    """在同一 base 权重上构造 torch 与 triton 两个 manager,便于对比。"""
    tmp = tmp_path_factory.mktemp("triton_pair")
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=["lin"],
    )
    for name, seed in (("a", 1), ("b", 2)):
        torch.manual_seed(seed)
        m = ToyLM().cuda()
        get_peft_model(m, cfg).save_pretrained(tmp / name)

    # 固定 base 权重,消除随机初始化差异
    base_sd = ToyLM().cuda().state_dict()
    adapters = {"a": str(tmp / "a"), "b": str(tmp / "b")}

    base_torch = ToyLM().cuda()
    base_torch.load_state_dict(base_sd)
    torch_mgr = MultiLoRAManager.from_pretrained(
        base_torch,
        adapters,
        max_loras_per_batch=3,
        max_lora_rank=16,
        lora_backend="torch",
    )

    base_triton = ToyLM().cuda()
    base_triton.load_state_dict(base_sd)
    triton_mgr = MultiLoRAManager.from_pretrained(
        base_triton,
        adapters,
        max_loras_per_batch=3,
        max_lora_rank=16,
        lora_backend="triton",
    )
    return torch_mgr, triton_mgr


def test_triton_matches_torch_full_model(cuda_mgr_pair):
    """Triton 与 Torch 后端整模型 forward 输出应在容差内一致。"""
    torch_mgr, triton_mgr = cuda_mgr_pair
    x = torch.randn(8, 64, device="cuda", dtype=torch_mgr.pool.dtype)
    pattern = ["a", None, "b", "a", None, "b", "a", "b"]

    def _ids(mgr):
        """pattern 里的名字就是内部 id,原样返回。"""
        return [n if n else None for n in pattern]

    y_t = torch_mgr.forward_with_lora(x=x, lora_names=_ids(torch_mgr))
    y_tr = triton_mgr.forward_with_lora(x=x, lora_names=_ids(triton_mgr))
    assert torch.allclose(y_t, y_tr, rtol=1e-2, atol=1e-2)
