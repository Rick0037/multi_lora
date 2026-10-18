"""LoRA GPU 槽位驱逐策略（LRU / FIFO）.

在 pool.prepare_lora_batch 中：当 max_loras_per_batch 个槽都被占满、且当前 batch
还需要加载新 uid 时，从当前不在本 batch 且非 pinned 的 uid 里选受害者，释放其槽位。

- LRU：mark_used 把刚用到的 uid 挪到 OrderedDict 末尾；select_victim 从头部挑最久未用的。
- FIFO：按首次 mark_used 插入顺序淘汰最早装入 pool 的 adapter。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Set


class EvictionPolicy(ABC):
    @abstractmethod
    def mark_used(self, uid: Optional[str]) -> None: ...

    @abstractmethod
    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]: ...

    @abstractmethod
    def remove(self, uid: Optional[str]) -> None: ...


class LRUEvictionPolicy(EvictionPolicy):
    def __init__(self) -> None:
        # key 顺序 = 从旧到新；mark_used 时 pop 再 append 等价于最近使用的在末尾。
        self.access_order: OrderedDict[str, None] = OrderedDict()
        self.eviction_count = 0

    def mark_used(self, uid: Optional[str]) -> None:
        # None（基座）不进入 LRU 链
        if uid is not None:
            self.access_order.pop(uid, None)
            self.access_order[uid] = None

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        # 从 LRU 链最旧端扫描，第一个落在 candidates 里的即淘汰目标。
        for uid in list(self.access_order.keys()):
            if uid in candidates:
                self.eviction_count += 1
                return uid
        # 若仅剩基座槽可驱逐（例如 batch 全为 LoRA），允许 victim=None。
        if None in candidates:
            self.eviction_count += 1
            return None
        raise RuntimeError(f"no LRU victim in {candidates}")

    def remove(self, uid: Optional[str]) -> None:
        if uid is not None:
            self.access_order.pop(uid, None)


class FIFOEvictionPolicy(EvictionPolicy):
    def __init__(self) -> None:
        # 仅记录首次出现的顺序；mark_used 对已存在 key 不调整顺序。
        self.insertion_order: OrderedDict[str, None] = OrderedDict()
        self.eviction_count = 0

    def mark_used(self, uid: Optional[str]) -> None:
        # 新 uid 第一次出现时加入队尾，之后不再移动（FIFO）。
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
    policies = {"lru": LRUEvictionPolicy, "fifo": FIFOEvictionPolicy}
    if name not in policies:
        raise ValueError(f"unknown eviction policy: {name}")
    return policies[name]()
