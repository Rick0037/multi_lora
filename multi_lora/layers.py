"""LoRA 线性层包装;基座 Linear + segment GEMM delta."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from multi_lora.backend.base import LoRABackend


def lora_delta(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    *,
    rank: int,
) -> torch.Tensor:
    """朴素 PyTorch LoRA delta,用于单测 / 数值对照(非 Triton 热路径)."""
    a = lora_a[:rank]
    b = lora_b[:, :rank]
    flat = x.reshape(-1, x.shape[-1])
    # delta = scaling * x @ A^T @ B^T
    delta = (flat @ a.T @ b.T) * scaling
    return delta.reshape(*x.shape[:-1], b.shape[0])


class LinearWithLoRA(nn.Module):
    """替换 PEFT LoraLayer;在 forward 内完成 base GEMM + LoRA segment GEMM."""

    def __init__(self, base_layer: nn.Module, module_key: str) -> None:
        """base_layer 为 PEFT 包裹下的原始 Linear;module_key 与 pool / weight store 对齐."""
        super().__init__()
        self.base_layer = base_layer
        self.module_key = module_key
        self.backend: Optional[LoRABackend] = None
        self.A_buffer: Optional[torch.Tensor] = None
        self.B_buffer: Optional[torch.Tensor] = None

    def set_lora_info(
        self,
        backend: LoRABackend,
        a_buffer: torch.Tensor,
        b_buffer: torch.Tensor,
    ) -> None:
        """每个 forward step 前由 manager 注入;backend 元数据 + 堆叠权重指针."""
        self.backend = backend
        self.A_buffer = a_buffer
        self.B_buffer = b_buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先算基座输出,再在同一展平 token 维上叠加 LoRA delta."""
        y_base = self.base_layer(x)
        if self.backend is None or self.A_buffer is None or self.B_buffer is None:
            return y_base

        # HF 线性层输入可能是 [B,S,H] 或 [T,H];统一展平为 [tokens, hidden]
        in_features = x.shape[-1]
        out_features = y_base.shape[-1]
        x2 = x.reshape(-1, in_features)
        y2 = y_base.reshape(-1, out_features)
        # apply_lora 内部;sgemm_a → sgemm_b(B 可 fused add 到 y2)
        y2 = self.backend.apply_lora(x2, self.A_buffer, self.B_buffer, y2)
        return y2.reshape(y_base.shape)
