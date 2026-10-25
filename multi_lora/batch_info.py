"""batch 元数据的两份表示:CPU list 版和 GPU 张量版.

pool.build_batch_info -> BatchLoRAInfo(CPU list)
to_kernel_input_metadata -> KernelInputMetaData(GPU 张量,kernel 直接读)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class BatchLoRAInfo:
    """一次 batch 的槽位信息(CPU list)."""

    weight_indices: List[int]  # 长度 = batch 行数;第 i 行用哪个槽位
    lora_ranks: List[int]  # 长度 = 槽位总数 K;每个槽位的 rank
    scalings: List[float]  # 长度 = 槽位总数 K;每个槽位的 scaling


@dataclass
class KernelInputMetaData:
    """kernel 调用所需的 metadata(GPU 张量).

    一条序列就是一个 segment(即使多条序列用同一个 adapter 也是多个 segment):
        seg_indptr / seg_lens   这个 segment 在展平 token 维上的位置
        weight_indices[seg]     这个 segment 用哪个槽位
        lora_ranks[slot]        该槽位的 rank(基座为 0)
        scalings[slot]          该槽位的 scaling
    """

    bs: int  # segment 数(= batch 内序列数)
    seg_indptr: torch.Tensor  # 前缀和;seg i 的 token 从 seg_indptr[i] 开始
    weight_indices: torch.Tensor  # 每个 segment 用哪个槽位
    lora_ranks: torch.Tensor  # 每个槽位的 rank
    scalings: torch.Tensor  # 每个槽位的 scaling
    max_len: int  # 最长 segment 的 token 数,用于算 grid
    seg_lens: torch.Tensor  # 每个 segment 的 token 数


def to_kernel_input_metadata(
    pool_batch: BatchLoRAInfo,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seq_lens: Optional[torch.Tensor] = None,
) -> KernelInputMetaData:
    """把 CPU list 转成 GPU 张量。

    seq_lens 只在 prefill 传:表示展平后 x 里每段有几个 token。
    因此 x 必须是紧密排列的,不能有 padding。
    """
    bs = len(pool_batch.weight_indices)

    weight_indices = torch.tensor(
        pool_batch.weight_indices, device=device, dtype=torch.int32
    )
    lora_ranks = torch.tensor(pool_batch.lora_ranks, device=device, dtype=torch.int32)
    scalings = torch.tensor(pool_batch.scalings, device=device, dtype=dtype)

    #! 如果 `lora_names = ['a', 'a', 'b']` 会得到`weight_indices = [0, 0, 1]` 。
    #! 因为 decoder阶段 可能不需要mask
    if seq_lens is None:
        # decode:每行 1 个 token
        seg_lens = torch.ones(bs, device=device, dtype=torch.int32)
        seg_indptr = torch.arange(bs + 1, device=device, dtype=torch.int32)
        max_len = 1
    else:
        # prefill:每行可以有多个 token
        # * seg_lens-> tensor([13, 13, 22], device='cuda:0')
        # * seg_indptr-> tensor([ 0, 13, 26, 48], device='cuda:0', dtype=torch.int32)
        seg_lens = seq_lens.to(device=device, dtype=torch.int32)
        seg_indptr = torch.zeros(bs + 1, device=device, dtype=torch.int32)
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
        max_len = int(seg_lens.max().item())

    return KernelInputMetaData(
        bs=bs,
        seg_indptr=seg_indptr,  # ! 每个人要管多少行
        weight_indices=weight_indices,  # ! 在buffer中是如何划分的 buffer shape 【max_batch, x, y】
        lora_ranks=lora_ranks,
        scalings=scalings,
        max_len=max_len,
        seg_lens=seg_lens,
    )
