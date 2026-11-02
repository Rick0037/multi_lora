"""scheduling.TierRotation 的纯逻辑:批次组成、换入压力、两种排布."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from multi_lora.scheduling import build_rotation

# 每档 rank 各 8 个名字,模拟 128 变体的 rank 分布
RANK_OF = {f"q{rank:02d}_{i}": rank for rank in (8, 16, 32, 64) for i in range(8)}


def test_interleaved_cycles_picks():
    """交错排布:i % K 轮转,K 个不同 adapter 均匀铺满."""
    rot = build_rotation(8, 4, rank_of=RANK_OF, batch_size=16)
    names = rot.next_names()
    assert len(names) == 16
    assert len(set(names)) == 8
    assert names[0] == names[8]  # 周期 = K


def test_grouped_emits_blocks():
    """分组排布:同名行连续,多重集与交错版完全一致."""
    inter = build_rotation(8, 4, rank_of=RANK_OF, batch_size=16)
    grouped = build_rotation(8, 4, rank_of=RANK_OF, batch_size=16, grouped=True)
    a, b = inter.next_names(), grouped.next_names()
    assert sorted(a) == sorted(b)          # 同一组 picks
    assert b[:2] == [b[0]] * 2             # 每 adapter 连续 2 行
    assert len(set(b)) == 8


def test_k1_whole_batch_one_adapter():
    """K=1:全 batch 一个 adapter(来自 r32 组)."""
    rot = build_rotation(1, 1, rank_of=RANK_OF, batch_size=16, grouped=True)
    names = rot.next_names()
    assert len(set(names)) == 1
    assert names[0] in RANK_OF and RANK_OF[names[0]] == 32


def test_k64_two_layouts_identical():
    """K=64:picks 已是 64 个,分组与交错输出相同(k64 档不受开关影响)."""
    a = build_rotation(64, 0, rank_of=RANK_OF, batch_size=64)
    b = build_rotation(64, 0, rank_of=RANK_OF, batch_size=64, grouped=True)
    assert a.next_names() == b.next_names()


def test_rank_mix_and_used_names():
    """rank 混合恒 1/4;used = 每组 per+move 个 × 4 组."""
    rot = build_rotation(8, 4, rank_of=RANK_OF, batch_size=64)
    assert rot.rank_mix() == "r8:2 r16:2 r32:2 r64:2"
    assert len(rot.used_names()) == (8 // 4 + 4 // 4) * 4  # 3 × 4 = 12


def test_swap_advances_picks():
    """每步前进 H/4:k8swap4 一步后 8 个 pick 换掉 4 个."""
    rot = build_rotation(8, 4, rank_of=RANK_OF, batch_size=16)
    a = set(rot.next_names())
    b = set(rot.next_names())
    assert len(a & b) == 4


def test_swap0_picks_frozen():
    """H=0:名字每步不变(k64swap0 对照档的依据)."""
    rot = build_rotation(8, 0, rank_of=RANK_OF, batch_size=16)
    a = rot.next_names()
    b = rot.next_names()
    assert a == b
