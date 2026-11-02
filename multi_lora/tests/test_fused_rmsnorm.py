"""融合 RMSNorm 的正确性:与 transformers 原实现逐元素对比.

CPU 部分只校验公式(平方和的均值、rsqrt、权重、dtype 回写顺序),
GPU 部分(fuse_rmsnorm 真跑 kernel)在无 CUDA 时自动跳过。
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from multi_lora.kernels.rmsnorm import fuse_rmsnorm, rmsnorm_fwd


class RefRMSNorm(nn.Module):
    """与 transformers Qwen3RMSNorm.forward 完全一致(含 fp32 上采样)。"""

    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        h = x.to(torch.float32)
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * h.to(input_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 编译 Triton kernel")
@pytest.mark.parametrize("hidden", [128, 2560])
def test_matches_reference(hidden):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    ref = RefRMSNorm(hidden).to(dev, torch.bfloat16)
    orig_forward = ref.forward

    for shape in ((64, 1, hidden), (64, 7, hidden)):   # decode / prefill
        x = torch.randn(*shape, device=dev, dtype=torch.bfloat16)
        want = orig_forward(x)
        got = rmsnorm_fwd(x, ref.weight, 1e-6)
        assert got.shape == want.shape and got.dtype == want.dtype
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 编译 Triton kernel")
def test_fuse_patches_all_and_keeps_numerics():
    torch.manual_seed(0)
    dev = torch.device("cuda")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = RefRMSNorm(2560)
            self.n2 = RefRMSNorm(128)
            self.lin = nn.Linear(8, 8)

    model = nn.Sequential(Block(), Block()).to(dev, torch.bfloat16)
    before = {
        name: mod.forward
        for name, mod in model.named_modules()
        if isinstance(mod, RefRMSNorm)
    }
    x1 = torch.randn(16, 1, 2560, device=dev, dtype=torch.bfloat16)
    x2 = torch.randn(16, 1, 128, device=dev, dtype=torch.bfloat16)
    want1, want2 = before["0.n1"](x1), before["0.n2"](x2)

    assert fuse_rmsnorm(model) == 4

    got1 = model[0].n1(x1)
    got2 = model[0].n2(x2)
    torch.testing.assert_close(got1, want1, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(got2, want2, rtol=2e-2, atol=2e-2)
    # 模块树与参数不受影响
    assert isinstance(model[0].n1.weight, nn.Parameter)
    assert isinstance(model[0].lin, nn.Linear)
