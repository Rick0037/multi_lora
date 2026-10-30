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
    seg_lens,  # * lens 实际是三个数字【16，16， 22】
    seg_indptr,  # * index的start 以及end
    weight_indices,  # * slot id
    lora_ranks,  # * lora_ranks
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Segment batched LoRA A GEMM;output[seg] = x[seg] @ weights[slot]^T."""

    # axis=1;每个 program 对应 batch 内一个 segment(通常一条序列)
    batch_id = tl.program_id(axis=1)  # * blockidx.y
    w_index = tl.load(weight_indices + batch_id)
    #! 这里在基座模型none的情况下，
    # * 只是lork_ranks 的list 取w_index 的位置，实际rank == 0
    rank = tl.load(lora_ranks + w_index)

    if rank == 0:
        return  # 基座 slot;无 LoRA,输出区域保持 0

    pid = tl.program_id(axis=0)
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)

    # 有效输出列数 = min(max_rank*stack_num, rank*stack_num)
    # * 但实际工程实现会加上 N 维切分来提升小 batch 场景的并行度。这不违反论文思想，只是工程优化。
    N = tl.minimum(N, rank * stack_num)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    # * 到底这个pid在s上是什么样子的，在n上是什么样子的
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    """
    - `s_offset`：当前线程块负责的**行号**（相对 segment 内部），向量长度 `BLOCK_S=16`
    - `n_offset`：当前线程块负责的**输出列号**rank 维，向量长度 `BLOCK_N=16`
    - `k_offset`, K 维in_dim的**列号**，向量长度 `BLOCK_K=256`，用于循环累加
    它们是后面计算内存地址的基础坐标。
    """
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    # * (x + seg_start * x_stride_0) x上的segment 起点
    # * s_offset[:, None] * x_stride_0 segment中的行
    # * k_offset[None, :] * x_stride_1 segment中的列

    # * (s,k) * (k, n)
    # ! 对于x 来说 虽然是二维但是实际上 seq_len 维度是拼接起来的，
    # * 要真正的到位到自己需要的 【seq_len】小段, IC, 针对【seq_small, IC】 进行 行方向的切分
    # * 行方向的切分是x_stride_0， 之后定位自己在哪个列x_stride_1
    x_ptrs = (x + seg_start * x_stride_0) + (
        s_offset[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )

    #! weights [K_slots, max_rank, in_dim], 按正常顺序读取
    # * w_stride_0 是因为他们没有压缩成batch，在除了batch之外是max_rank, in_dim的，
    # * 下面相乘的时候需要去 进行转置
    # * 所以也是先batch， 在行， 在列， 跟上面的顺序一样， 但是 w_stride_1 的写法+1
    w_ptrs = (weights + w_index * w_stride_0) + (
        n_offset[:, None] * w_stride_1 + k_offset[None, :] * w_stride_2
    )

    # K 维分块累加;partial_sum += x_tile @ w_tile
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    # * split k 进行累加， reduce， 最终把相关的按照 担忧segment的
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(  # * tl loar 通过mask去， 取数据 并限制mask的范围
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (
                k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        # 读出来的 w_tile 是 [rank, in_dim], 需要转置成 [in_dim, rank]
        w_tile_raw = tl.load(
            w_ptrs,
            mask=(n_offset[:, None] < N) & (
                k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.trans(w_tile_raw)  # [rank, in_dim] -> [in_dim, rank]
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1  # * k在这里对应的是stride_1
        w_ptrs += BLOCK_K * w_stride_2  # * k在这里对应的是stride_2

    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (output + seg_start * output_stride_0) + (
        s_offset[:, None] * output_stride_0 +
        n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def sgemm_lora_a_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,  # * -> [OC, IC] 因为是linear出来的
    kernel_input_metadata: KernelInputMetaData,
    output: torch.Tensor,
    stack_num: int = 1,
) -> torch.Tensor:
    """Host 入口;LoRA A segment SGEMM.

    x: (S, in_dim);weights: (K, max_rank, in_dim);output: (S, max_rank)。
    output 由调用方给(常驻缓冲),这里只负责写,不再每次分配。
    grid 第二维 = bs(segment 数);第一维覆盖 max_len * rank 分块。
    """
    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    # x [M, K] * w [k, N], 但是本文中的weight转置了
    M = x.shape[0]
    N = weights.shape[-2]
    K = weights.shape[-1]
    assert x.shape[-1] == K

    BLOCK_M = 16
    BLOCK_K = 256
    BLOCK_N = 16

    # ? 为什么输入都是拼接在一起的， grid的第二维还是batch size
    # ! 阅读punica的论文里面的kernel 实现
    grid = (
        triton.cdiv(kernel_input_metadata.max_len, BLOCK_M) *
        triton.cdiv(N, BLOCK_N),
        kernel_input_metadata.bs,
    )

    output = output[:M, :N]  # 常驻缓冲可能比本次大, 取本次的行/列
    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        N,
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
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    return output
