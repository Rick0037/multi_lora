"""LoRA 线性层:基座 Linear 的输出 + LoRA delta."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from multi_lora.backend.base import LoRABackend

# 进程级开关:关掉后 LinearWithLoRA 只跑基座 GEMM,用于测「不含 LoRA」的基线耗时
LORA_ENABLED = True


def set_lora_enabled(enabled: bool) -> None:
    """全局开/关 LoRA delta 计算;同一进程同一模型,只跳过 delta 那一步."""
    global LORA_ENABLED
    LORA_ENABLED = enabled


def lora_delta(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    *,
    rank: int,
) -> torch.Tensor:
    """朴素 PyTorch 版LoRA delta = scaling * x @ A^T @ B^T(单测 / 数值对照用)."""
    a = lora_a[:rank]  # [rank, in]
    b = lora_b[:, :rank]  # [out, rank]
    flat = x.reshape(-1, x.shape[-1])
    delta = (flat @ a.T @ b.T) * scaling

    out_shape = list(x.shape[:-1])
    out_shape.append(b.shape[0])
    return delta.reshape(out_shape)


class LinearWithLoRA(nn.Module):
    """替换 PEFT 的 LoraLayer;forward = 基座 GEMM + 后端算的 LoRA delta."""

    # baseline * x + x * a * b
    #! 这里的a_buffer, b_buffer 实际上是那个组大 batch 出来的 padding的大矩阵
    #! backend 计算的时候还需要根据索引去分组算
    def __init__(self, base_layer: nn.Module) -> None:
        """base_layer 是 PEFT 包裹下的原始 Linear;A/B 由 manager 每步注入."""
        super().__init__()
        self.base_layer = base_layer
        self.backend: Optional[LoRABackend] = None
        self.A_buffer: Optional[torch.Tensor] = None
        self.B_buffer: Optional[torch.Tensor] = None

    def set_lora_info(
        self,
        backend: LoRABackend,
        a_buffer: torch.Tensor,
        b_buffer: torch.Tensor,
    ) -> None:
        """每个 forward step 前由 manager 注入:后端 + 按槽位排列的权重 [K, ...]."""
        self.backend = backend
        self.A_buffer = a_buffer
        self.B_buffer = b_buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先算基座输出,再把 LoRA delta 累加进去."""
        y_base = self.base_layer(x)
        if not LORA_ENABLED:
            return y_base
        if self.backend is None or self.A_buffer is None or self.B_buffer is None:
            return y_base
        #! 这时候是有padding的
        # 输入可能是 [B, S, H] 或 [T, H];统一展平成 [tokens, hidden]
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y_base.reshape(-1, y_base.shape[-1])
        y_flat = self.backend.apply_lora(
            x_flat, self.A_buffer, self.B_buffer, y_flat)
        return y_flat.reshape(y_base.shape)
