"""PyTorch 默认后端;按 segment / slot 分组 matmul(golden / CPU 回退)."""

from __future__ import annotations

from typing import Optional

import torch

from multi_lora.backend.base import LoRABackend


class TorchLoRABackend(LoRABackend):
    """朴素 PyTorch 实现;逐 segment 选 slot,便于单测与 Triton 数值对照."""

    name = "torch"

    def run_lora_a_sgemm(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """对每个 segment 用其 weight_indices 指向的 slot 做 x_seg @ A^T."""
        if self.kernel_input_metadata is None:
            raise RuntimeError(
                "kernel_input_metadata not set; call set_kernel_input_metadata first"
            )
        if x.dim() != 2:
            raise ValueError("TorchLoRABackend expects x [S, in]")

        bi = self.kernel_input_metadata
        s_total, _ = x.shape
        max_rank = weights.shape[1]
        y = torch.zeros(s_total, max_rank, device=x.device, dtype=x.dtype)

        for seg in range(int(bi.num_segments)):
            # seg_indptr 界定该 segment 在展平 token 维上的 [start, end)
            start = int(bi.seg_indptr[seg].item())
            end = int(bi.seg_indptr[seg + 1].item())
            if start >= end:
                continue
            # 本 segment 使用的 pool 槽位编号
            slot = int(bi.weight_indices[seg].item())
            r = int(bi.lora_ranks[slot].item())
            if r == 0:
                continue  # 基座 slot;rank=0,无 LoRA 贡献
            a = weights[slot, :r, :].to(dtype=x.dtype)
            y[start:end, :r] = x[start:end] @ a.T
        return y

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """对每个 segment 做 mid @ B^T * scaling,可选 fused 加到 base_output."""
        if self.kernel_input_metadata is None:
            raise RuntimeError(
                "kernel_input_metadata not set; call set_kernel_input_metadata first"
            )
        if x.dim() != 2:
            raise ValueError("TorchLoRABackend expects x [S, rank]")

        bi = self.kernel_input_metadata
        s_total, _ = x.shape
        out_dim = weights.shape[1]
        if base_output is None:
            y = torch.zeros(s_total, out_dim, device=x.device, dtype=x.dtype)
        else:
            y = base_output

        for seg in range(int(bi.num_segments)):
            start = int(bi.seg_indptr[seg].item())
            end = int(bi.seg_indptr[seg + 1].item())
            if start >= end:
                continue
            slot = int(bi.weight_indices[seg].item())
            r = int(bi.lora_ranks[slot].item())
            if r == 0:
                continue
            bmat = weights[slot, :, :r].to(dtype=x.dtype)
            scale = bi.scalings[slot].to(dtype=x.dtype)
            # delta = scaling * (x @ B^T)
            delta = (x[start:end, :r] @ bmat.T) * scale
            if base_output is None:
                y[start:end] = delta
            else:
                y[start:end] = y[start:end] + delta
        return y
