"""调度策略层:batch 组装、换入压力生成、容量规划.

从 fix_batch_test_each_cost.py 抽出的 TierRotation:每个 rank 组一个滑动
窗口,决定"这一步哪些 adapter 进 batch、每行用哪个"。纯策略,不含测量
(计时 / profiler 都留在 bench 侧),也不碰请求状态(KV / mask 归
graph_decode.DecodeStep)。

约定(与 bench 五档对齐,改这里 = 改 workload 定义):
    rank 分组固定为 8/16/32/64 四组,K=1 的特例从 r32 组取;
    每步从每组取 per=K/4 个、前进 move=H/4 个 → batch 里四种 rank
    恒各占 1/4(平均 rank 不变),且每步恰好换入 H 个。
"""

from __future__ import annotations

from typing import Dict, List, Mapping

RANK_GROUPS = (8, 16, 32, 64)
K1_RANK = 32  # K=1 特例用的 rank 组


class TierRotation:
    """adapter 轮换器:每步产出一组名字,制造稳定的换入压力."""

    def __init__(
        self,
        k: int,
        swap: int,
        groups: Dict[int, List[str]],
        *,
        batch_size: int,
        grouped: bool = False,
    ) -> None:
        """k=batch 内不同 adapter 数,swap=每步换入=换出个数(H=0 不换入).

        groups 是 {rank: [adapter 名]} 的池子;batch_size 是每步的行数;
        grouped=True 时行按 adapter 连续排布(T8 分组,配合 manager 的
        group_rows 把相邻同名行合成一个 segment),False 时交错轮转。
        """
        self.k = k
        self.swap = swap
        self.groups = groups
        self.batch_size = batch_size
        self.grouped = grouped
        self.per = k // 4  # 每个 rank 组驻留几个
        self.move = swap // 4  # 每个 rank 组每步前进几个
        self.step = 0

    def next_names(self) -> List[str]:
        """本步 batch_size 个名字(每行一个),不同 adapter 恰好 K 个.

        两种排布的 adapter 多重集、rank 混合、每步换入压力完全相同,
        只有行的顺序不同;grouped 让 lora kernel 的 M 从 1 提到
        batch_size/K。
        """
        start = self.step
        self.step += 1
        if self.k == 1:
            pool = self.groups[K1_RANK]
            picks = [pool[start % len(pool)]]
        else:
            picks = []
            for rank in RANK_GROUPS:
                pool = self.groups[rank]
                base = start * self.move
                picks += [pool[(base + j) % len(pool)] for j in range(self.per)]
        if self.grouped and len(picks) < self.batch_size:
            per = self.batch_size // len(picks)
            return [picks[j // per] for j in range(self.batch_size)]
        return [picks[i % len(picks)] for i in range(self.batch_size)]

    def rank_mix(self) -> str:
        """batch 的 rank 组成(报告用)."""
        if self.k == 1:
            return f"r{K1_RANK}:1"
        return " ".join(f"r{r}:{self.per}" for r in RANK_GROUPS)

    def used_names(self) -> List[str]:
        """能轮换到的全部名字;注册即常驻显存,调用方只把这些装进来."""
        if self.k == 1:
            return list(self.groups[K1_RANK])
        out = []
        for rank in RANK_GROUPS:
            out += self.groups[rank]
        return out


def build_rotation(
    k: int,
    swap: int,
    *,
    rank_of: Mapping[str, int],
    batch_size: int,
    grouped: bool = False,
) -> TierRotation:
    """按 rank 把 adapter 分组,每组只留窗口会用到的 K/4+H/4 个.

    rank_of 是 {adapter 名: rank};K=1 在 r32 组取 1+H 个。
    """
    need = 1 + swap if k == 1 else k // 4 + swap // 4
    groups = {
        r: [n for n, v in rank_of.items() if v == r][:need]
        for r in RANK_GROUPS
    }
    return TierRotation(
        k, swap, groups, batch_size=batch_size, grouped=grouped
    )
