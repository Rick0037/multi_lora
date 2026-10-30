"""batch 元数据的两份表示:CPU list 版和 GPU 张量版.

pool.build_batch_info -> BatchLoRAInfo(CPU list)
empty_kernel_input_metadata -> 0 容量的初值(第一次写装不下,会自动分配)
write_kernel_input_metadata -> KernelInputMetaData(GPU 张量,kernel 直接读)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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
    seg_lens: torch.Tensor  # 每个 segment 的 token 数
    seg_indptr: torch.Tensor  # 前缀和;seg i 的 token 从 seg_indptr[i] 开始
    weight_indices: torch.Tensor  # 每个 segment 用哪个槽位
    lora_ranks: torch.Tensor  # 每个槽位的 rank
    scalings: torch.Tensor  # 每个槽位的 scaling
    max_len: int  # 最长 segment 的 token 数,用于算 grid


def segment_lens(bs: int, seq_lens: List[int]) -> Tuple[List[int], List[int]]:
    """每段多少 token,以及它们的前缀和。

    decode  传 []          -> lens = [1, 1, 1]      indptr = [0, 1, 2, 3]
    prefill 传 [13, 13, 22] -> lens = [13, 13, 22]  indptr = [0, 13, 26, 48]

    indptr 比 lens 多一格:indptr[i] 是第 i 段在展平的 x 里从第几行开始。
    """
    if not seq_lens:
        # decode:每行 1 个 token
        lens = [1] * bs
    else:
        # prefill:每行可以有多个 token
        # * seg_lens-> tensor([13, 13, 22], device='cuda:0')
        # * seg_indptr-> tensor([ 0, 13, 26, 48], device='cuda:0', dtype=torch.int32)
        assert len(
            seq_lens) == bs, f"seq_lens 要每段一个:{len(seq_lens)} 个 vs {bs} 段"
        lens = [int(v) for v in seq_lens]

    indptr = [0]
    for length in lens:
        indptr.append(indptr[-1] + length)
    return lens, indptr


def fill_column(
    column: torch.Tensor, values: list, pad: float
) -> None:
    """整列覆盖:前 len(values) 格写 values,剩下的格补 pad。

    写完之后 column 还是原来那个张量、原来那个长度,所以显存地址不变。
    """
    size = column.shape[0]
    assert len(values) <= size, f"{len(values)} 个数字装不进 {size} 格"
    new_values = list(values) + [pad] * (size - len(values))
    column.copy_(torch.tensor(new_values, dtype=column.dtype))


def empty_kernel_input_metadata(
    *, device: torch.device, dtype: torch.dtype
) -> KernelInputMetaData:
    """0 容量的一版:第一次写入时装不下,会自动换成真的尺寸。"""
    return KernelInputMetaData(
        bs=0,
        seg_lens=torch.empty(0, dtype=torch.int32, device=device),
        seg_indptr=torch.empty(0, dtype=torch.int32, device=device),
        weight_indices=torch.empty(0, dtype=torch.int32, device=device),
        lora_ranks=torch.empty(0, dtype=torch.int32, device=device),
        scalings=torch.empty(0, dtype=dtype, device=device),
        max_len=0,
    )


def write_kernel_input_metadata(
    meta: KernelInputMetaData,
    pool_batch: BatchLoRAInfo,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> KernelInputMetaData:
    """把这次 batch 的 metadata 写进去,返回能直接喂给 kernel 的对象。

    meta 传上一次返回的对象(第一次传 empty_kernel_input_metadata 那版)。
    装得下就原地改写:返回的还是同一个 meta、显存地址不变,CUDA Graph 才 replay 得了。
    装不下(bs 变多)才按这次的大小重新分配,device / dtype 就是给这时候用的。
    lora_ranks / scalings 的长度恒为 pool 的槽位数,分配一次后不再变。

    seq_lens 传 [] 就是 decode(每段 1 个 token);prefill 传每段多少 token。
    seg_lens 必须和 x 的实际布局一致(padded 输入就是 padded 长度),
    kernel 按 seg_indptr 取行,不对会读错位置。
    """
    #! 如果 `lora_names = ['a', 'a', 'b']` 会得到`weight_indices = [0, 0, 1]` 。
    #! 因为 decoder阶段 可能不需要mask
    rows = len(pool_batch.weight_indices)  # 这次有几段
    lens, indptr = segment_lens(rows, seq_lens)

    # 装不下:按这次的大小分配一批新的
    if rows > meta.weight_indices.shape[0]:
        return KernelInputMetaData(
            bs=rows,
            seg_lens=torch.tensor(lens, device=device, dtype=torch.int32),
            seg_indptr=torch.tensor(
                indptr, device=device, dtype=torch.int32),  # ! 每个人要管多少行
            # ! 在buffer中是如何划分的 buffer shape 【max_batch, x, y】
            weight_indices=torch.tensor(
                pool_batch.weight_indices, device=device, dtype=torch.int32),
            lora_ranks=torch.tensor(
                pool_batch.lora_ranks, device=device, dtype=torch.int32),
            scalings=torch.tensor(
                pool_batch.scalings, device=device, dtype=dtype),
            max_len=max(lens),  # 最长一段,用来算 grid
        )

    # 装得下:整列覆盖,张量和地址都不变。多出来的格一律填该列的空值,
    # 这样整列单独拿出来看也是自洽的,不靠 kernel 只读前 bs 格兜底。
    fill_column(meta.weight_indices, pool_batch.weight_indices, 0)  # ! 每段用哪个槽位
    fill_column(meta.lora_ranks, pool_batch.lora_ranks, 0)  # rank 0 = 基座
    fill_column(meta.seg_lens, lens, 0)  # 空段 0 个 token
    # 空段的起点 = 总行数,于是 seg_indptr 整列仍然单调不减
    fill_column(meta.seg_indptr, indptr, indptr[-1])  # ! 每个人要管多少行
    fill_column(meta.scalings, pool_batch.scalings, 0.0)
    meta.bs = rows
    meta.max_len = max(lens)
    return meta
