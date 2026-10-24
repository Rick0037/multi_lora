"""KernelInputMetaData;送入 Triton segment GEMM kernel 的 batch 张量元数据.

seg_indptr / weight_indices / lora_ranks / scalings 等字段告诉 kernel;
- 每个 segment(通常对应一条请求)在 x 里的 token 范围;
- 该 segment 应读取 pool 的哪个 slot 权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from multi_lora.pool import BatchLoRAInfo


@dataclass
class KernelInputMetaData:
    """Triton / 自定义 kernel 调用所需的 segment batch 元数据(GPU 张量视图)."""

    bs: int  # batch 内序列条数(与 weight_indices 长度一致)
    num_segments: int  # segment 个数,
    seg_indptr: torch.Tensor  # 前缀和;seg i 的 token 在 x 中起始于 seg_indptr[i]
    weight_indices: torch.Tensor  # 每个 segment 使用的 pool slot 编号
    lora_ranks: torch.Tensor  # 按 slot 索引的有效 rank(基座 slot 为 0)
    scalings: torch.Tensor  # 按 slot 索引的 LoRA scaling
    max_len: Optional[int]  # 本 batch 最长 segment,用于 Triton grid  sizing
    seg_lens: Optional[torch.Tensor]  # 每个 segment 的 token 数


def kernel_input_metadata_from_pool_lists(
    *,
    weight_indices: List[int],
    lora_ranks: List[int],
    scalings: List[float],
    device: torch.device,
    dtype: torch.dtype,
    seq_lens: Optional[torch.Tensor] = None,
) -> KernelInputMetaData:
    """把 pool 输出的 list 元数据转成 GPU 张量,供 kernel 直接读取."""
    bs = len(weight_indices)
    # * weight_indices-> 对应 gpu的buffer是哪个 [1, 0, 2]
    # kernel 侧 weight_indices / ranks 用 int32;scalings 与激活 dtype 对齐
    # * 创建tensor, list -> tensor
    w = torch.tensor(weight_indices, device=device, dtype=torch.int32)
    ranks = torch.tensor(lora_ranks, device=device, dtype=torch.int32)
    scal = torch.tensor(scalings, device=device, dtype=dtype)

    if seq_lens is None:
        # * decode 默认;每行 1 个 token
        # * 每行 1 个 segment,
        # * seg_indptr = [0,1,2,...,B]
        seg_lens = torch.ones(bs, device=device, dtype=torch.int32)
        seg_indptr = torch.arange(bs + 1, device=device, dtype=torch.int32)
        num_segments = bs
        max_len = 1
    else:
        # * prefill;每条序列可有多个 token,seg_indptr 由 seq_lens 前缀和构造
        seg_lens = seq_lens.to(device=device, dtype=torch.int32)
        num_segments = int(seg_lens.numel())
        seg_indptr = torch.zeros(num_segments + 1, device=device, dtype=torch.int32)
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
        max_len = int(seg_lens.max().item())
    """
        ! num_segments = 3, 3 条请求即使都用同一个 adapter, 也是 3 个 segment
        ! adapter 只通过`weight_indices` 映射到槽位。
        (Pdb) seg_lens
        tensor([13, 13, 22], device='cuda:0', dtype=torch.int32)
        (Pdb) seg_indptr
        tensor([ 0, 13, 26, 48], device='cuda:0', dtype=torch.int32)
    """

    return KernelInputMetaData(
        bs=bs,
        num_segments=num_segments,
        seg_indptr=seg_indptr,
        weight_indices=w,
        lora_ranks=ranks,
        scalings=scal,
        max_len=max_len,
        seg_lens=seg_lens,
    )


def kernel_input_metadata_from_pool_batch(
    pool_batch: "BatchLoRAInfo",
    *,
    device: torch.device,
    dtype: torch.dtype,
    seq_lens: Optional[torch.Tensor] = None,
) -> KernelInputMetaData:
    """从 pool.build_batch_info 的 CPU list 结构一键构造 kernel 元数据."""
    return kernel_input_metadata_from_pool_lists(
        weight_indices=pool_batch.weight_indices,  # {1, 0, 2}
        lora_ranks=pool_batch.lora_ranks,  # {16, 0, 64}
        scalings=pool_batch.scalings,  # {scale_0, scale_1}
        device=device,
        dtype=dtype,
        seq_lens=seq_lens,  # * 每个的长度 {len_1, len_2, len_3}
    )
