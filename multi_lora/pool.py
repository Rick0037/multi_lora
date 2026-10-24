"""GPU 槽位池:固定 K 个槽位,负责把 CPU 权重装进去、槽满时换出.

PEFT 权重 --抽取--> CPU AdapterWeights --装载--> 本模块的 GPU 槽位
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import torch

from multi_lora.batch_info import BatchLoRAInfo
from multi_lora.eviction import EvictionPolicy, get_eviction_policy
from multi_lora.registry import LoraName
from multi_lora.weights import AdapterWeights


class _GpuBuffers:
    """每个 module 一份堆叠权重:A [K, max_rank, in] / B [K, out, max_rank].

    只负责显存里的权重:建 buffer、清槽位、写权重、按 module 取出来。
    槽位怎么分配、谁被换出,是 LoRAMemoryPool 的事。
    """

    def __init__(
        self,
        num_slots: int,
        max_rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_slots = num_slots
        self.max_rank = max_rank
        self.device = device
        self.dtype = dtype
        self.A: Dict[str, torch.Tensor] = {}
        self.B: Dict[str, torch.Tensor] = {}

    def ensure(self, weights: AdapterWeights) -> None:
        """第一次见到某个 module 时,按它的形状建 buffer;已存在的跳过."""
        for key in weights.modules:
            if key in self.A:
                continue
            a_cpu, b_cpu = weights.modules[key]
            in_dim = int(a_cpu.shape[1])
            out_dim = int(b_cpu.shape[0])
            self.A[key] = torch.zeros(
                (self.num_slots, self.max_rank, in_dim),
                device=self.device,
                dtype=self.dtype,
            )
            self.B[key] = torch.zeros(
                (self.num_slots, out_dim, self.max_rank),
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
            self.A[key][slot, : weights.rank].copy_(a_cpu.to(dtype=self.dtype))
            self.B[key][slot, :, : weights.rank].copy_(b_cpu.to(dtype=self.dtype))

    def get(self, module_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """取出某个 module 的全部槽位权重."""
        if module_key not in self.A:
            raise KeyError(f"module {module_key!r} not in pool")
        return self.A[module_key], self.B[module_key]


class LoRAMemoryPool:
    """K 个 GPU 槽位,每格是三种状态之一:空 / 基座 / 某个 adapter。

    name_to_buffer_id: adapter 名 -> 槽位号
    slot_names:        槽位号 -> adapter 名(None 表示基座);不在表里就是空槽
    _slot_rank_scaling: 槽位号 -> (rank, scaling)
    buffers:           显存里的 A/B 权重(见 _GpuBuffers)
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
        self.eviction_policy: EvictionPolicy = get_eviction_policy(eviction_policy)

        self.name_to_buffer_id: Dict[LoraName, int] = {}
        self.slot_names: Dict[int, LoraName] = {}
        self._slot_rank_scaling: Dict[int, Tuple[int, float]] = {}
        self.buffers = _GpuBuffers(
            num_slots=max_loras_per_batch,
            max_rank=max_lora_rank,
            device=device,
            dtype=dtype,
        )

    def get_buffer_id(self, name: LoraName) -> int:
        """查 adapter 当前所在的槽位号;不在池里则报错."""
        if name not in self.name_to_buffer_id:
            raise KeyError(f"lora {name!r} not resident in pool")
        return self.name_to_buffer_id[name]

    def get_module_buffers(self, module_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回某个 module 的全部槽位权重,可直接喂给 segment GEMM."""
        return self.buffers.get(module_key)

    def get_slot_weights(
        self, slot_id: int, module_key: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """取单个槽位的 A/B 视图(测试与参考实现用)."""
        a_stack, b_stack = self.buffers.get(module_key)
        return a_stack[slot_id], b_stack[slot_id]

    def ensure_buffers(self, weights: AdapterWeights) -> None:
        """建该 adapter 用到的 module 的 GPU buffer;已存在的跳过(幂等)."""
        self.buffers.ensure(weights)

    def prepare_lora_batch(
        self,
        cur_names: Set[LoraName],
        *,
        weight_store: Dict[LoraName, AdapterWeights],
    ) -> None:
        """保证 cur_names 都在 GPU 上:缺的逐个装槽,槽满就换出别的 adapter."""
        for name in cur_names:
            self.eviction_policy.mark_used(name)

        for name in cur_names:
            if name in self.name_to_buffer_id:
                continue
            slot = self._alloc_slot(cur_names)
            self._load_name_to_slot(name, slot, weight_store)

    def build_batch_info(self, lora_ids: List[LoraName]) -> BatchLoRAInfo:
        """按行给出槽位号,按槽位给出 rank / scaling(kernel 用槽位号去查表)."""
        weight_indices: List[int] = []
        for name in lora_ids:
            weight_indices.append(self.get_buffer_id(name))

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

    def _alloc_slot(self, keep: Set[LoraName]) -> int:
        """找一个可写入的槽位:优先空槽,否则换出不在本 batch 里的 adapter."""
        for slot in range(self.max_loras_per_batch):
            if slot not in self.slot_names:
                return slot

        # 候选 = 不在本 batch 里的槽;基座(None)单独记,只有没别的可选时才换它
        candidates: Set[LoraName] = set()
        base_is_evictable = False
        for slot in self.slot_names:
            name = self.slot_names[slot]
            if name in keep:
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
        slot = self.name_to_buffer_id.pop(victim)
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
        self.name_to_buffer_id[name] = slot
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
