"""槽位满时的换出策略:LRU(最久没用)或 FIFO(最早装入)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Set

# order 中的顺序就是"越靠前越先被换出"的顺序
_KINDS = ("lru", "fifo")


class EvictionPolicy:
    """记录 adapter 的使用顺序,并据此挑出要被换出的那一个.

    - lru:  每次用到就挪到队尾,队首是最久没用的
    - fifo: 只在第一次出现时入队,之后顺序不变
    基座(None)永远不入队:它没有可重载的权重,只能作为最后手段被换出。
    """

    def __init__(self, kind: str = "lru") -> None:
        if kind not in _KINDS:
            raise ValueError(f"unknown eviction policy: {kind}; try lru | fifo")
        self.kind = kind
        self.order: "OrderedDict[str, None]" = OrderedDict()
        self.eviction_count = 0

    def mark_used(self, name: Optional[str]) -> None:
        """本 batch 用到该 adapter 时更新顺序."""
        if name is None:
            return
        if self.kind == "lru":
            self.order.pop(name, None)  # LRU:先删再插 = 挪到队尾
        self.order[name] = None

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        """从候选里挑出最该换出的 adapter."""
        for name in self.order:
            if name in candidates:
                self.eviction_count += 1
                return name
        if None in candidates:  # 只剩基座槽可换
            self.eviction_count += 1
            return None
        raise RuntimeError(f"no evictable adapter in {candidates}")

    def remove(self, name: Optional[str]) -> None:
        """槽位回收后,把该 adapter 从顺序表中删掉."""
        if name is not None:
            self.order.pop(name, None)


def get_eviction_policy(name: str) -> EvictionPolicy:
    """按名字创建策略:lru | fifo."""
    return EvictionPolicy(name.lower().strip())
