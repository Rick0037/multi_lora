"""LoRA GPU 槽位驱逐策略(LRU / FIFO).

在 pool.prepare_lora_batch 中;当槽位满且需加载新 uid 时,从可驱逐候选中选 victim。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Set


class EvictionPolicy(ABC):
    """槽位满时的 adapter 驱逐策略抽象."""

    def __init__(self) -> None:
        self.eviction_count = 0

    @abstractmethod
    def mark_used(self, uid: Optional[str]) -> None:
        """本 batch 用到 uid 时更新策略状态(LRU 挪到末尾 / FIFO 记录首次出现)."""

    @abstractmethod
    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        """从 candidates 中选出将被驱逐的 uid(可为 None=基座槽)."""

    @abstractmethod
    def remove(self, uid: Optional[str]) -> None:
        """victim 槽回收后,从策略结构中移除该 uid."""


class LRUEvictionPolicy(EvictionPolicy):
    """最近最少使用;淘汰最久未 mark_used 的 adapter."""

    def __init__(self) -> None:
        super().__init__()
        # key 顺序 = 从旧到新;mark_used 时 pop 再 append
        self.access_order: OrderedDict[str, None] = OrderedDict()

    def mark_used(self, uid: Optional[str]) -> None:
        # None(基座)不进入 LRU 链;基座槽没有可重载的 LoRA 权重,不参与"谁值得留在显存"的排序
        if uid is not None:
            self.access_order.pop(uid, None)
            self.access_order[uid] = None

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        # 从最旧端扫描,第一个落在 candidates 里的即 victim
        for uid in list(self.access_order.keys()):
            if uid in candidates:
                self.eviction_count += 1
                return uid
        # LRU 链里扫不到候选时,只可能是仅剩基座槽可驱逐(None 不入链),此时兜底返回 None
        if None in candidates:
            self.eviction_count += 1
            return None
        raise RuntimeError(f"no LRU victim in {candidates}")

    def remove(self, uid: Optional[str]) -> None:
        if uid is not None:
            self.access_order.pop(uid, None)


class FIFOEvictionPolicy(EvictionPolicy):
    """先进先出;淘汰最早装入 pool 的 adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.insertion_order: OrderedDict[str, None] = OrderedDict()

    def mark_used(self, uid: Optional[str]) -> None:
        # 新 uid 首次出现时加入队尾,之后不再移动
        if uid is not None and uid not in self.insertion_order:
            self.insertion_order[uid] = None

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        for uid in list(self.insertion_order.keys()):
            if uid in candidates:
                self.eviction_count += 1
                return uid
        if None in candidates:
            self.eviction_count += 1
            return None
        raise RuntimeError(f"no FIFO victim in {candidates}")

    def remove(self, uid: Optional[str]) -> None:
        if uid is not None:
            self.insertion_order.pop(uid, None)


def get_eviction_policy(name: str) -> EvictionPolicy:
    """按名称实例化驱逐策略(lru | fifo)."""
    policies = {"lru": LRUEvictionPolicy, "fifo": FIFOEvictionPolicy}
    if name not in policies:
        raise ValueError(f"unknown eviction policy: {name}")
    return policies[name]()
