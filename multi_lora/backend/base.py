"""LoRA 计算后端抽象(Torch / Triton 等实现共享的接口)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from multi_lora.batch_info import KernelInputMetaData


class LoRABackend(ABC):
    """多槽位 + 按 segment 选权重的 LoRA 计算后端基类."""

    name: str = "abstract"

    def __init__(self) -> None:
        """初始化时 kernel 元数据为空,需由 manager 在 forward 前注入."""
        self.kernel_input_metadata: Optional[KernelInputMetaData] = None
        # A 阶段的中间量 (rows, max_rank),常驻复用;A 写完紧接着就被 B 读走
        self._mid = torch.empty((0, 0))

    def mid_buffer(self, rows: int, max_rank: int, like: torch.Tensor) -> torch.Tensor:
        """取一块 (rows, max_rank) 的中间缓冲;装得下就复用,地址不变.

        列宽固定为 max_rank,所以只切行 self._mid[:rows] 仍是连续张量。
        """
        need_new = (
            self._mid.shape[0] < rows
            or self._mid.shape[1] != max_rank
            or self._mid.device != like.device
            or self._mid.dtype != like.dtype
        )
        if need_new:
            self._mid = torch.empty(
                (rows, max_rank), dtype=like.dtype, device=like.device)
        return self._mid[:rows]

    def set_kernel_input_metadata(self, metadata: KernelInputMetaData) -> None:
        """绑定本 step 的 segment / slot 元数据,供 A/B SGEMM 读取."""
        self.kernel_input_metadata = metadata

    @abstractmethod
    def run_lora_a_sgemm(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """LoRA A 阶段;x [S,in] * A[slot,:r,:] → 中间激活 [S,max_rank].

        weights 形状 [K, max_rank, in],K 为 pool 槽位数。
        """

    @abstractmethod
    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """LoRA B 阶段;中间 [S,r] * B[slot,:, :r] * scaling → delta,可 fused 加到 base_output."""

    def apply_lora(
        self,
        x: torch.Tensor,
        a_stack: torch.Tensor,
        b_stack: torch.Tensor,
        base_output: torch.Tensor,
    ) -> torch.Tensor:
        """串联 A/B 两阶段 SGEMM,默认 B 阶段 in-place 累加到基座输出."""
        # 先算 x @ A^T → [S, max_rank]
        mid = self.run_lora_a_sgemm(x, a_stack)
        # 再算 mid @ B^T * scale,并加到 y_base
        """
        - `x` :`[S, in]` ,S = batch 内所有 token 展平后的总数
        - `a_stack` :`[K, max_rank, in]` ,`K = max_loras_per_batch（槽位总数）` ,每一格是一个 adapter 的 A
        - `b_stack` :`[K, out, max_rank]` ,同上
        - `y_base` :`[S, out]` ,基座 Linear 已经算好的输出
        - 返回:`y_base + LoRA delta`
        内部两步

        1. A 阶段:把`x` 按 segment 切开,第 seg 段用`weight_indices[seg]` 选出槽位 slot,算`x[段] @ a_stack[slot, :r].T` →`[S, max_rank]`
        2. B 阶段:`mid[段] @ b_stack[slot, :, :r].T * scaling[slot]` ,fused 加到`y_base`
        所以“分块乘整个 x”这个说法基本对,但准确说是: 按 segment 分行挑权重 ,而不是把 x 本身切块做分块 GEMM。segment 的边界来自`seg_indptr` /`seg_lens` ,权重来自`weight_indices` 。

        它不管的事

        - 基座 Linear 的乘 —— 在 LinearWithLoRA.forward 里先算好再传进来
        - 槽位分配、权重加载 —— pool 负责
        - 模型结构 —— manager 负责
        """
        return self.run_lora_b_sgemm(mid, b_stack, base_output=base_output)
