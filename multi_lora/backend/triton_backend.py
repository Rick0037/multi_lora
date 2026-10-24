"""Triton segment-SGEMM 后端(CUDA);CPU 或无 Triton 时回退 Torch."""

from __future__ import annotations

import torch

from multi_lora.backend.base import LoRABackend
from multi_lora.backend.torch_backend import TorchLoRABackend
from multi_lora.batch_info import KernelInputMetaData

try:
    from multi_lora.kernels import sgemm_lora_a_fwd, sgemm_lora_b_fwd

    _HAS_TRITON_KERNELS = True
except ImportError:
    _HAS_TRITON_KERNELS = False


class TritonLoRABackend(LoRABackend):
    """LoRA A/B 经 Triton segment GEMM 一次 launch 覆盖全 batch 各 segment."""

    name = "triton"

    def __init__(self) -> None:
        """内嵌 Torch 后端,用于非 CUDA 或 kernel 不可用时的回退."""
        super().__init__()
        self._torch = TorchLoRABackend()

    def set_kernel_input_metadata(self, metadata: KernelInputMetaData) -> None:
        """同步元数据到 Triton 与 Torch 回退路径."""
        super().set_kernel_input_metadata(metadata)
        self._torch.set_kernel_input_metadata(metadata)

    def run_lora_a_sgemm(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """CUDA + Triton 可用时走 kernel;否则委托 Torch 逐 segment 实现."""
        if not (x.is_cuda and _HAS_TRITON_KERNELS):
            return self._torch.run_lora_a_sgemm(x, weights)
        return sgemm_lora_a_fwd(
            x.contiguous(), weights.contiguous(), self._int32_kernel_input_metadata()
        )

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """B 阶段支持 fused add;kernel 内 load 基座输出再 store 累加结果."""
        if not (x.is_cuda and _HAS_TRITON_KERNELS):
            return self._torch.run_lora_b_sgemm(x, weights, base_output=base_output)
        return sgemm_lora_b_fwd(
            x.contiguous(),
            weights.contiguous(),
            self._int32_kernel_input_metadata(),
            base_output=base_output,
        )

    def _int32_kernel_input_metadata(self) -> KernelInputMetaData:
        """Triton kernel 要求 weight_indices 等为 int32;必要时克隆一份视图."""
        meta = self.kernel_input_metadata
        if meta is None:
            raise RuntimeError("kernel_input_metadata not set")
        if meta.weight_indices.dtype == torch.int32:
            return meta
        return KernelInputMetaData(
            bs=meta.bs,
            seg_indptr=meta.seg_indptr,
            weight_indices=meta.weight_indices.to(torch.int32),
            lora_ranks=meta.lora_ranks,
            scalings=meta.scalings,
            max_len=meta.max_len,
            seg_lens=meta.seg_lens,
        )
