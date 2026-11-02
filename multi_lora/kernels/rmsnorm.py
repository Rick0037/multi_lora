"""融合 RMSNorm:把 transformers 的 4~5 个 kernel 合成 1 个.

Qwen3RMSNorm.forward 现在是
    to(fp32) → pow(2) → mean(-1) → rsqrt+eps → mul(x) → mul(weight) → to(bf16)
decode 下每层 4 个 norm(2 个 hidden + q_norm/k_norm),36 层 = 145 个实例,
实测 653 次 elementwise + 145 次 reduce,合计约 4.9ms/步。

本 kernel 一次完成 fp32 上采样 → 平方和 → rsqrt → 乘权重 → 回写原 dtype,
数学与原地实现一致(同样在 fp32 里算方差)。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,           # [rows, n_cols] 连续
    W,           # [n_cols] 权重(任意浮点 dtype)
    Y,           # [rows, n_cols] 输出,通常与 X 同地址(in-place 友好)
    stride,      # 行步长(= n_cols)
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    # 均值只除 n_cols,不用 mask 修正:越界项已被 other=0 置零
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y + row * stride + cols, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x: [..., H] → 归一化后的同形状张量(新分配,不改入参)."""
    hidden = x.shape[-1]
    rows = x.numel() // hidden
    out = torch.empty_like(x)

    if x.dim() > 2:
        x2 = x.reshape(rows, hidden)
        y2 = out.reshape(rows, hidden)
    else:
        x2, y2 = x, out

    block = triton.next_power_of_2(hidden)
    _rmsnorm_kernel[(rows,)](
        x2,
        weight,
        y2,
        hidden,
        hidden,
        eps,
        BLOCK=block,
        num_warps=8 if block >= 2048 else 4,
    )
    return out


def fuse_rmsnorm(model) -> int:
    """把模型里所有 *RMSNorm 实例的 forward 换成融合 kernel,返回替换个数.

    只改实例方法,不动模块树 / 参数,所以 state_dict、device、dtype 都不受影响。
    判据是类名以 RMSNorm 结尾且带 variance_epsilon(transformers 的通用约定)。
    """
    replaced = 0
    for module in model.modules():
        if not type(module).__name__.endswith("RMSNorm"):
            continue
        eps = getattr(module, "variance_epsilon", None)
        weight = getattr(module, "weight", None)
        if eps is None or weight is None:
            continue

        def _forward(hidden_states, _w=weight, _eps=float(eps)):
            return rmsnorm_fwd(hidden_states, _w, _eps)

        module.forward = _forward
        replaced += 1
    return replaced
