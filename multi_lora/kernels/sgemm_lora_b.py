"""LoRA segment GEMM B 阶段 Triton kernel;mid @ B^T * scale,fused 累加到 output."""

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
    """Segment batched LoRA B GEMM,结果 scaled 后 in-place 加到 output(基座输出)."""

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
    # weights [K_slots, out_dim, max_rank], 按正常顺序读取
    w_ptrs = (weights + w_index * w_stride_0) + (
        n_offset[:, None] * w_stride_1 + k_offset[None, :] * w_stride_2
    )

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        # 读出来的 w_tile 是 [out_dim, rank], 需要转置成 [rank, out_dim]
        w_tile_raw = tl.load(
            w_ptrs,
            mask=(n_offset[:, None] < N) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.trans(w_tile_raw)  # [out_dim, rank] -> [rank, out_dim]
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


@triton.jit
def _gemv_lora_b_kernel(
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
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """decode 专用 GEMV 路径:mid 行向量 @ B^T * scaling,fused 累加到 output.

    K = rank ≤ 64,一轮归约;out 维按 BLOCK_N 分块。
    """

    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return

    pid_n = tl.program_id(axis=0)
    scaling = tl.load(scalings + w_index)
    K = tl.minimum(K, rank)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offset = tl.arange(0, BLOCK_K)
    seg_start = tl.load(seg_indptr + batch_id)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + k_offset
        x_vec = tl.load(  # mid 的唯一一行 [BLOCK_K] = [rank]
            x + seg_start * x_stride_0 + kk * x_stride_1,
            mask=k_offset < K - k * BLOCK_K,
            other=0.0,
        )
        w_tile = tl.load(  # [BLOCK_N, BLOCK_K] = (out_dim, rank) 行主序
            weights + w_index * w_stride_0
            + n_offset[:, None] * w_stride_1 + kk[None, :] * w_stride_2,
            mask=(n_offset[:, None] < N) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        acc += tl.sum(w_tile * x_vec[None, :], axis=1)

    # scaling * delta + 已有基座输出 → fused add
    acc *= scaling
    acc = acc.to(x.dtype.element_ty)
    out_ptr = output + seg_start * output_stride_0 + n_offset * output_stride_1
    acc += tl.load(out_ptr, mask=n_offset < N)
    tl.store(out_ptr, acc, mask=n_offset < N)


def sgemm_lora_b_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    kernel_input_metadata: KernelInputMetaData,
    base_output: torch.Tensor = None,
) -> torch.Tensor:
    """Host 入口;LoRA B segment SGEMM,可选 fused 累加到 base_output.

    x: (S, max_rank);weights: (K, out_dim, max_rank);output: (S, out_dim)。
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

    # T13/T14 dispatch:纯 decode(每段 1 token)走 GEMV,其余走 GEMM。
    # 与 A 阶段同判据(max_len 是 CPU 侧 int),graph capture 时选定。
    if kernel_input_metadata.max_len == 1:
        _gemv_lora_b_kernel[(triton.cdiv(N, 256), kernel_input_metadata.bs)](
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
            kernel_input_metadata.seg_indptr,
            kernel_input_metadata.weight_indices,
            kernel_input_metadata.lora_ranks,
            kernel_input_metadata.scalings,
            BLOCK_N=256,
            BLOCK_K=64,
        )
        return output

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
