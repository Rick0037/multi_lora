"""T19:一个 chunk 的散布合成一个 kernel.

v11 里每个 (key, chunk) 用两条高级索引赋值,一步 8 chunk * 144 key * 2
= 2304 个 index_put_ kernel,由 1152 轮 Python 循环发起(~23ms CPU)。
本 kernel 用偏移表一次做完:A 段按槽位连续平铺,B 段按 (out, rank) 跨步写。

池子的 A/B 是 144+144 个独立张量,kernel 没法收这么多参数;这里取任一池
张量当 base,其余张量的位置表示成相对 base 的元素偏移(int64,可为负),
kernel 里只剩标准的指针加法。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_stage_kernel(
    stage_ptr,  # [chunk, total] 连续暂存,行 i = 第 i 个 adapter 的整块
    base_ptr,  # 池内任一张量,其余池张量按 a_delta/b_delta 相对它定位
    slots_ptr,  # [h] int64 槽位号
    a_delta_tbl,  # [n] int64 A[key] 相对 base 的元素偏移
    a_off_tbl,  # [n] int32 bank 布局里 A 段起点
    a_numel_tbl,  # [n] int32 A 段长度 = rank*in
    a_sstride_tbl,  # [n] int32 A[key] 槽位步长 = max_rank*in
    b_delta_tbl,  # [n] int64 B[key] 相对 base 的元素偏移
    b_off_tbl,  # [n] int32 bank 布局里 B 段起点
    b_numel_tbl,  # [n] int32 B 段长度 = out*rank
    b_sstride_tbl,  # [n] int32 B[key] 槽位步长 = out*max_rank
    b_rstride_tbl,  # [n] int32 B[key] 行步长 = max_rank
    total,  # stage 行宽(元素数)
    h,  # 本 chunk 的 adapter 数
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_k = tl.program_id(0)  # 第几个 key
    pid_c = tl.program_id(1)  # 段内第几块
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)

    # ---- A 段:池内每槽位连续,直接平铺 ----
    # bank 的 A 段是 [rank, in] 行优先,恰好等于 A[key][slot] 的前 rank 行
    a_numel = tl.load(a_numel_tbl + pid_k)
    a_mask = offs < a_numel
    a_off = tl.load(a_off_tbl + pid_k)
    a_sstride = tl.load(a_sstride_tbl + pid_k)
    a_dst = base_ptr + tl.load(a_delta_tbl + pid_k)
    for i in range(h):
        slot = tl.load(slots_ptr + i)
        v = tl.load(stage_ptr + i * total + a_off + offs, mask=a_mask)
        tl.store(a_dst + slot * a_sstride + offs, v, mask=a_mask)

    # ---- B 段:池内是 (out, max_rank),rank 在最后一维,跨步写 ----
    # bank 的 B 段是 [out, rank] 行优先;池内 (o, r) 的地址跨了 max_rank
    b_numel = tl.load(b_numel_tbl + pid_k)
    b_mask = offs < b_numel
    b_off = tl.load(b_off_tbl + pid_k)
    b_sstride = tl.load(b_sstride_tbl + pid_k)
    b_rstride = tl.load(b_rstride_tbl + pid_k)
    b_dst = base_ptr + tl.load(b_delta_tbl + pid_k)
    row = offs // RANK
    col = offs % RANK
    for i in range(h):
        slot = tl.load(slots_ptr + i)
        v = tl.load(stage_ptr + i * total + b_off + offs, mask=b_mask)
        tl.store(b_dst + slot * b_sstride + row * b_rstride + col, v, mask=b_mask)


def scatter_stage(stage, slots, tables, total, h, rank, BLOCK=8192):
    """一个 (rank 组, chunk) 的散布。tables 来自 _RankPlan.build_tables。

    与 python 路径(高级索引赋值)语义一致:只写前 rank 行/列,
    padding 区的旧数据不动。
    """
    grid = (tables["n"], triton.cdiv(tables["max_numel"], BLOCK))
    _scatter_stage_kernel[grid](
        stage,
        tables["base"],
        slots,
        tables["a_delta"],
        tables["a_off"],
        tables["a_numel"],
        tables["a_sstride"],
        tables["b_delta"],
        tables["b_off"],
        tables["b_numel"],
        tables["b_sstride"],
        tables["b_rstride"],
        total,
        h,
        RANK=rank,
        BLOCK=BLOCK,
        num_warps=8,
    )
