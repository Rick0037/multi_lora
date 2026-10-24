"""KernelInputMetaData;送入 Triton segment GEMM kernel 的 batch 张量元数据.

seg_indptr / weight_indices / lora_ranks / scalings 等字段告诉 kernel;
- 每个 segment(通常对应一条请求)在 x 里的 token 范围;
- 该 segment 应读取 pool 的哪个 slot 权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from multi_lora.pool import BatchLoRAInfo


@dataclass
class KernelInputMetaData:
    """Triton / 自定义 kernel 调用所需的 segment batch 元数据(GPU 张量视图)."""

    bs: int  # batch 内序列条数(与 weight_indices 长度一致),也是一条序列一个 segment
    seg_indptr: torch.Tensor  # 前缀和;seg i 的 token 在 x 中起始于 seg_indptr[i]
    weight_indices: torch.Tensor  # 每个 segment 使用的 pool slot 编号
    lora_ranks: torch.Tensor  # 按 slot 索引的有效 rank(基座 slot 为 0)
    scalings: torch.Tensor  # 按 slot 索引的 LoRA scaling
    max_len: Optional[int]  # 本 batch 最长 segment,用于 Triton grid  sizing
    seg_lens: Optional[torch.Tensor]  # 每个 segment 的 token 数


def kernel_input_metadata_from_pool_batch(
    pool_batch: "BatchLoRAInfo",
    *,
    device: torch.device,
    dtype: torch.dtype,
    seq_lens: Optional[torch.Tensor] = None,
) -> KernelInputMetaData:
    """把 pool.build_batch_info 的 CPU list 转成 kernel 直接读取的 GPU 张量."""
    bs = len(pool_batch.weight_indices)
    # kernel 侧 weight_indices / ranks 用 int32;scalings 与激活 dtype 对齐
    w = torch.tensor(pool_batch.weight_indices, device=device, dtype=torch.int32)
    ranks = torch.tensor(pool_batch.lora_ranks, device=device, dtype=torch.int32)
    scal = torch.tensor(pool_batch.scalings, device=device, dtype=dtype)

    if seq_lens is None:
        # * decode 默认;每行 1 个 token
        seg_lens = torch.ones(bs, device=device, dtype=torch.int32)
        seg_indptr = torch.arange(bs + 1, device=device, dtype=torch.int32)
        max_len = 1
    else:
        # * prefill;每条序列可有多个 token,seg_indptr 由 seq_lens 前缀和构造
        seg_lens = seq_lens.to(device=device, dtype=torch.int32)
        seg_indptr = torch.zeros(bs + 1, device=device, dtype=torch.int32)
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
        max_len = int(seg_lens.max().item())
    """
        ! bs = 3, 3 条请求即使都用同一个 adapter, 也是 3 个 segment
        ! adapter 只通过`weight_indices` 映射到槽位。
        (Pdb) seg_lens
        tensor([13, 13, 22], device='cuda:0', dtype=torch.int32)
        (Pdb) seg_indptr
        tensor([ 0, 13, 26, 48], device='cuda:0', dtype=torch.int32)
    """

    return KernelInputMetaData(
        bs=bs,
        seg_indptr=seg_indptr,
        weight_indices=w,
        lora_ranks=ranks,
        scalings=scal,
        max_len=max_len,
        seg_lens=seg_lens,
    )
