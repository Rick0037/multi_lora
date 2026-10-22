"""LoRA segment GEMM A 阶段 Triton kernel;x @ A^T,按 segment 选 slot."""

import torch
import triton
import triton.language as tl

from multi_lora.batch_info import KernelInputMetaData


@triton.jit
def _sgemm_lora_a_kernel(
    x,
    weights,
    output,
    N,
    K,
    stack_num,
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
):
    """Segment batched LoRA A GEMM;output[seg] = x[seg] @ weights[slot]^T."""

    # axis=1;每个 program 对应 batch 内一个 segment(通常一条序列)
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    if rank == 0:
        return  # 基座 slot;无 LoRA,输出区域保持 0

    pid = tl.program_id(axis=0)
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)

    # 有效输出列数 = min(max_rank*stack_num, rank*stack_num)
    N = tl.minimum(N, rank * stack_num)

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

    # K 维分块累加;partial_sum += x_tile @ w_tile
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (output + seg_start * output_stride_0) + (
        s_offset[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def sgemm_lora_a_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    kernel_input_metadata: KernelInputMetaData,
    stack_num: int = 1,
) -> torch.Tensor:
    """Host 入口;LoRA A segment SGEMM.

    x: (S, in_dim);weights: (K, max_rank, in_dim);output: (S, max_rank)。
    grid 第二维 = bs(segment 数);第一维覆盖 max_len * rank 分块。
    """
    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    S = x.shape[0]
    R = weights.shape[-2]
    K = weights.shape[-1]
    assert x.shape[-1] == K

    BLOCK_S = 16
    BLOCK_K = 256
    BLOCK_R = 16

    grid = (
        triton.cdiv(kernel_input_metadata.max_len, BLOCK_S) * triton.cdiv(R, BLOCK_R),
        kernel_input_metadata.bs,
    )

    output = torch.empty((S, R), device=x.device, dtype=x.dtype)
    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        R,
        K,
        stack_num,
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
        BLOCK_R,
        BLOCK_K,
    )
    return output
