"""batch 元数据的两份表示:CPU list 版和 GPU 张量版.

pool.build_batch_info -> BatchLoRAInfo(CPU list)
to_kernel_input_metadata -> KernelInputMetaData(GPU 张量,kernel 直接读)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

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
    seq_lens: Optional[Sequence[int]] = None,
) -> KernelInputMetaData:
    """把 CPU list 转成 GPU 张量。

    seq_lens 表示 x 展平后每段占多少行:
      decode 不传 -> 每段 1 个 token
      prefill 传  -> CPU 上的序列,如 [13, 13, 22](传 tensor 也支持,但会多一次同步)
    seg_lens 必须和 x 的实际布局一致(padded 输入就是 padded 长度),
    kernel 按 seg_indptr 取行,不对会读错位置。
    """
    bs = len(pool_batch.weight_indices)
    if isinstance(seq_lens, torch.Tensor):
        seq_lens = seq_lens.tolist()
    # * lens -> [13, 13, 22]:每个 segment 多少 token
    lens = [1] * bs if seq_lens is None else [int(v) for v in seq_lens]
    assert len(lens) == bs, f"seq_lens has {len(lens)} entries, expected {bs}"

    # * indptr -> [0, 13, 26, 48]:前缀和,seg i 的 token 从 indptr[i] 开始
    indptr = [0] * (bs + 1)
    for i, length in enumerate(lens):
        indptr[i + 1] = indptr[i] + length

    #! 如果 `lora_names = ['a', 'a', 'b']` 会得到 `weight_indices = [0, 0, 1]`,
    #! 多条序列复用同一个槽位; lora_ranks / scalings 的长度是槽位总数 K,不是行数。
    return KernelInputMetaData(
        bs=bs,
        # ! weight_indices 是在 buffer 中如何划分的;pool 的 buffer shape [max_batch, x, y]
        weight_indices=torch.tensor(
            pool_batch.weight_indices, device=device, dtype=torch.int32),
        lora_ranks=torch.tensor(
            pool_batch.lora_ranks, device=device, dtype=torch.int32),
        seg_lens=torch.tensor(lens, device=device, dtype=torch.int32),
        # ! 每个人要管多少行:kernel 靠 seg_indptr 到展平的 x 里取自己那几行
        seg_indptr=torch.tensor(indptr, device=device, dtype=torch.int32),
        scalings=torch.tensor(pool_batch.scalings, device=device, dtype=dtype),
        max_len=max(lens, default=0),  # 最长一段,用来算 grid
    )
