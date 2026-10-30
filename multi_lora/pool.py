"""GPU 槽位池:固定 K 个槽位,负责把 CPU 权重装进去、槽满时换出.

PEFT 权重 --抽取--> CPU AdapterWeights --装载--> 本模块的 GPU 槽位
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import torch

from multi_lora.batch_info import BatchLoRAInfo
from multi_lora.eviction import EvictionPolicy
from multi_lora.registry import LoraName
from multi_lora.weights import AdapterWeights


class _GpuBuffers:
    """显存里的 LoRA 权重,包含所有槽位的总和(不是某一个 adapter)。

    显存里有 max_loras_per_batch 个给 adapter 预留的格子,编号 0 ~ K-1
    (K = max_loras_per_batch,这个编号就是"槽位号")。
    每个格子放一个 adapter 的 A/B,所以:
        A[key] 形状 [K, max_lora_rank, in]:A[key][槽位号] 是那个格子的 A
        B[key] 形状 [K, out, max_lora_rank]:B[key][槽位号] 是那个格子的 B
    rank 不足 max_lora_rank 的部分补 0,kernel 只看前 rank 行/列。
    """

    def __init__(
        self,
        max_loras_per_batch: int,
        max_lora_rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_loras_per_batch = max_loras_per_batch
        self.max_lora_rank = max_lora_rank
        self.device = device
        self.dtype = dtype
        self.A: Dict[str, torch.Tensor] = {}
        self.B: Dict[str, torch.Tensor] = {}

    def ensure(self, weights: AdapterWeights) -> None:
        """第一次见到某个 module 时,按它的形状建 buffer;已存在的跳过."""

        #! 外面是遍历使用的, 这里只是针对一个adapter做的操作
        #! 创建新的A[key] -> 一个tensor 但是zeros
        for key in weights.modules:
            if key in self.A:
                continue
            a_cpu, b_cpu = weights.modules[key]
            in_dim = int(a_cpu.shape[1])
            out_dim = int(b_cpu.shape[0])
            self.A[key] = torch.zeros(
                (self.max_loras_per_batch, self.max_lora_rank, in_dim),
                device=self.device,
                dtype=self.dtype,
            )
            self.B[key] = torch.zeros(
                (self.max_loras_per_batch, out_dim, self.max_lora_rank),
                device=self.device,
                dtype=self.dtype,
            )

    def write(self, slot: int, weights: AdapterWeights) -> None:
        """把该 adapter 的权重写进槽位;先清空整个槽位,避免残留旧权重."""
        for key in self.A:
            self.A[key][slot].zero_()
            self.B[key][slot].zero_()

        for key in weights.modules:
            a_cpu, b_cpu = weights.modules[key]
            #! 每个人只写自己slot位置的
            self.A[key][slot, : weights.rank].copy_(a_cpu.to(dtype=self.dtype))
            self.B[key][slot, :, : weights.rank].copy_(
                b_cpu.to(dtype=self.dtype))

    def get(self, module_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """取出某个 module 的全部槽位权重."""
        if module_key not in self.A:
            raise KeyError(f"module {module_key!r} not in pool")
        return self.A[module_key], self.B[module_key]


class LoRAMemoryPool:
    """K 个 GPU 格子(槽位),编号 0 ~ K-1。每格是:空 / 基座 / 某个 adapter。

    槽位号就是"显存里给 adapter 预留的格子编号"。
    kernel 用不了字符串名字,只能按编号取权重,所以这里维护名字和编号的对应:
        name_to_slot: adapter 名 -> 槽位号
        slot_names:        槽位号 -> adapter 名(None 表示基座);不在表里 = 空格子
        _slot_rank_scaling: 槽位号 -> (rank, scaling)
        buffers:           格子里的 A/B 权重(见 _GpuBuffers)
    """

    def __init__(
        self,
        max_loras_per_batch: int,
        max_lora_rank: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        eviction_policy: str = "lru",
    ) -> None:
        """只建映射,不分配显存;buffer 由 ensure_buffers 按 module 形状创建."""
        self.max_loras_per_batch = max_loras_per_batch
        self.max_lora_rank = max_lora_rank
        self.device = device
        self.dtype = dtype
        self.eviction_policy = EvictionPolicy(eviction_policy)

        self.name_to_slot: Dict[LoraName, int] = {}
        self.slot_names: Dict[int, LoraName] = {}
        self._slot_rank_scaling: Dict[int, Tuple[int, float]] = {}
        self.buffers = _GpuBuffers(
            max_loras_per_batch=max_loras_per_batch,
            max_lora_rank=max_lora_rank,
            device=device,
            dtype=dtype,
        )

    def get_slot_id(self, name: LoraName) -> int:
        """查 adapter 当前所在的槽位号;不在池里则报错."""
        if name not in self.name_to_slot:
            raise KeyError(f"lora {name!r} not resident in pool")
        return self.name_to_slot[name]

    def get_module_buffers(self, module_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回某个 module 的全部槽位权重,可直接喂给 segment GEMM."""
        return self.buffers.get(module_key)

    def get_slot_weights(
        self, slot: int, module_key: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """取单个槽位的 A/B 视图(测试与参考实现用)."""
        a_stack, b_stack = self.buffers.get(module_key)
        return a_stack[slot], b_stack[slot]

    def ensure_buffers(self, weights: AdapterWeights) -> None:
        """建该 adapter 用到的 module 的 GPU buffer;已存在的跳过(幂等)."""
        self.buffers.ensure(weights)

    def prepare_lora_batch(
        self,
        lora_names: Set[LoraName],
        *,
        weight_store: Dict[LoraName, AdapterWeights],
    ) -> None:
        """保证 lora_names 都在 GPU 上:缺的逐个装槽,槽满就换出别的 adapter."""
        for name in lora_names:
            self.eviction_policy.mark_used(name)

        for name in lora_names:
            if name in self.name_to_slot:
                continue
            slot = self._alloc_slot(lora_names)
            self._load_name_to_slot(name, slot, weight_store)

    def build_batch_info(self, lora_names: List[LoraName]) -> BatchLoRAInfo:
        """按行给出槽位号,按槽位给出 rank / scaling(kernel 用槽位号去查表)."""
        #!
        weight_indices: List[int] = []
        for name in lora_names:
            weight_indices.append(self.get_slot_id(name))
        # weight_indices 实际上是本次batch的数量
        # rank / scalings 的长度是槽位总数 K,不是 batch 行数
        lora_ranks = [0] * self.max_loras_per_batch
        scalings = [0.0] * self.max_loras_per_batch
        for slot in self._slot_rank_scaling:
            rank, scaling = self._slot_rank_scaling[slot]
            lora_ranks[slot] = rank
            scalings[slot] = scaling

        return BatchLoRAInfo(
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
        )

    def _alloc_slot(self, keep_names: Set[LoraName]) -> int:
        """找一个可写入的槽位:优先空槽,否则换出不在本 batch 里的 adapter."""
        for slot in range(self.max_loras_per_batch):
            if slot not in self.slot_names:
                return slot

        # 候选 = 不在本 batch 里的槽;基座(None)单独记,只有没别的可选时才换它
        candidates: Set[LoraName] = set()
        base_is_evictable = False
        for slot in self.slot_names:
            name = self.slot_names[slot]
            if name in keep_names:
                continue
            if name is None:
                base_is_evictable = True
            else:
                candidates.add(name)

        if not candidates:
            if not base_is_evictable:
                raise ValueError(
                    "no evictable slot: batch needs more adapters than max_loras_per_batch"
                )
            candidates.add(None)

        victim = self.eviction_policy.select_victim(candidates)
        slot = self.name_to_slot.pop(victim)
        self.eviction_policy.remove(victim)
        self.slot_names.pop(slot)
        self._slot_rank_scaling.pop(slot, None)
        return slot

    def _load_name_to_slot(
        self,
        name: LoraName,
        slot: int,
        weight_store: Dict[LoraName, AdapterWeights],
    ) -> None:
        """把 CPU 权重拷进指定槽位;基座(None)只登记 rank=0,不写权重."""
        self.name_to_slot[name] = slot
        self.slot_names[slot] = name

        if name is None:
            self._slot_rank_scaling[slot] = (0, 0.0)
            return

        weights = weight_store[name]
        if weights.rank > self.max_lora_rank:
            raise ValueError(
                f"adapter {name!r} rank {weights.rank} > max_lora_rank {self.max_lora_rank}"
            )

        # 池被单独使用(没经过 manager 注册)时,在这里补建
        self.buffers.ensure(weights)
        self.buffers.write(slot, weights)
        self._slot_rank_scaling[slot] = (weights.rank, weights.scaling)
