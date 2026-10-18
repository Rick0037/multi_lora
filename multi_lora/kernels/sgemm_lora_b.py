"""LoRA segment GEMM B 阶段 Triton kernel：mid @ B^T * scale，fused 累加到 output."""

import torch
import triton
import triton.language as tl

from multi_lora.batch_info import KernelInputMetaData


@triton.jit
def _sgemm_lora_b_kernel(
    x,
    weights,
    output,
    N,
    K,
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    scalings,
):
    """Segment batched LoRA B GEMM，结果 scaled 后 in-place 加到 output（基座输出）."""

    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    if rank == 0:
        return

    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    seg_start = tl.load(seg_indptr + batch_id)
    scaling = tl.load(scalings + w_index)
    K = tl.minimum(K, rank)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = (x + seg_start * x_stride_0) + (
        s_offset[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # scaling * delta + 已有基座输出 → fused add
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (output + seg_start * output_stride_0) + (
        s_offset[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = s_offset[:, None] < seg_len
    partial_sum += tl.load(output_ptr, mask=output_mask)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def sgemm_lora_b_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    kernel_input_metadata: KernelInputMetaData,
    base_output: torch.Tensor = None,
) -> torch.Tensor:
    """Host 入口：LoRA B segment SGEMM，可选 fused 累加到 base_output.

    x: (S, max_rank)；weights: (K, out_dim, max_rank)；output: (S, out_dim)。
    """
    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    S = x.shape[0]
    N = weights.shape[-2]
    R = weights.shape[-1]
    assert x.shape[-1] == R

    BLOCK_S = 16
    BLOCK_R = 16
    BLOCK_N = 256

    grid = (
        triton.cdiv(kernel_input_metadata.max_len, BLOCK_S) * triton.cdiv(N, BLOCK_N),
        kernel_input_metadata.bs,
    )

    if base_output is None:
        output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    _sgemm_lora_b_kernel[grid](
        x,
        weights,
        output,
        N,
        R,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        kernel_input_metadata.seg_lens,
        kernel_input_metadata.seg_indptr,
        kernel_input_metadata.weight_indices,
        kernel_input_metadata.lora_ranks,
        BLOCK_S,
        BLOCK_N,
        BLOCK_R,
        kernel_input_metadata.scalings,
    )
    return output
