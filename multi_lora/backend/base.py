"""LoRA 计算后端抽象（Torch / Triton 等实现共享的接口）."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from multi_lora.batch_info import KernelInputMetaData


class LoRABackend(ABC):
    """多槽位 + 按 segment 选权重的 LoRA 计算后端基类."""

    name: str = "abstract"

    def __init__(self) -> None:
        """初始化时 kernel 元数据为空，需由 manager 在 forward 前注入."""
        self.kernel_input_metadata: Optional[KernelInputMetaData] = None

    def set_kernel_input_metadata(self, metadata: KernelInputMetaData) -> None:
        """绑定本 step 的 segment / slot 元数据，供 A/B SGEMM 读取."""
        self.kernel_input_metadata = metadata

    @abstractmethod
    def run_lora_a_sgemm(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """LoRA A 阶段：x [S,in] × A[slot,:r,:] → 中间激活 [S,max_rank].

        weights 形状 [K, max_rank, in]，K 为 pool 槽位数。
        """

    @abstractmethod
    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """LoRA B 阶段：中间 [S,r] × B[slot,:, :r] × scaling → delta，可 fused 加到 base_output."""

    def apply_lora(
        self,
        x: torch.Tensor,
        a_stack: torch.Tensor,
        b_stack: torch.Tensor,
        base_output: torch.Tensor,
    ) -> torch.Tensor:
        """串联 A/B 两阶段 SGEMM，默认 B 阶段 in-place 累加到基座输出."""
        # 先算 x @ A^T → [S, max_rank]
        mid = self.run_lora_a_sgemm(x, a_stack)
        # 再算 mid @ B^T * scale，并加到 y_base
        return self.run_lora_b_sgemm(mid, b_stack, base_output=base_output)
