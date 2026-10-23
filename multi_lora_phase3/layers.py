"""基于池内 A/B 权重的朴素 LoRA delta(PyTorch matmul).

用于验证 GPU 槽位中的权重与 scaling;第四步可由 Triton segment GEMM 替代。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def lora_delta(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    *,
    rank: int,
) -> torch.Tensor:
    """计算 LoRA 对线性层的增量;delta(x) = scaling * (x @ A.T) @ B.T.

    x: [*, in]; lora_a: [max_rank, in] 仅前 rank 行有效;lora_b: [out, max_rank] 仅前 rank 列有效。
    """
    a = lora_a[:rank]
    b = lora_b[:, :rank]
    flat = x.reshape(-1, x.shape[-1])
    delta = (flat @ a.T @ b.T) * scaling
    return delta.reshape(*x.shape[:-1], b.shape[0])
