"""GPU 槽位池;管理 LoRA 权重的 GPU 驻留与驱逐."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import torch

from multi_lora.eviction import EvictionPolicy, get_eviction_policy
from multi_lora.weights import AdapterWeights

LoraName = Optional[str]


class _EmptySlot:
    """槽位未占用时的占位标记(与 None = 基座区分)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "|EMPTY|"


EMPTY_SLOT = _EmptySlot()


@dataclass
class BatchLoRAInfo:
    """单次 batch 的槽位索引与 rank/scaling 元数据;用于 kernel_input_metadata 构造."""

    weight_indices: List[int]
    lora_ranks: List[int]
    scalings: List[float]
    evictions: int = 0


class LoRAMemoryPool:
    """管理固定数量的 GPU 槽位;每个槽驻留一个 adapter 的权重,或空闲/基座占位。"""

    def __init__(
        self,
        max_loras_per_batch: int,
        max_lora_rank: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        eviction_policy: str = "lru",
    ) -> None:
        """只建映射,不分配显存;buffer 由 finalize_buffers 懒创建。"""
        self.max_loras_per_batch = max_loras_per_batch
        self.max_lora_rank = max_lora_rank
        self.device = device
        self.dtype = dtype
        self.eviction_policy: EvictionPolicy = get_eviction_policy(eviction_policy)

        self.name_to_buffer_id: Dict[LoraName, int] = {}
        self.buffer_id_to_name: List[Union[LoraName, _EmptySlot]] = [
            EMPTY_SLOT
        ] * max_loras_per_batch
        self._slot_meta: Dict[int, Tuple[int, float]] = {}
        self._last_evictions = 0

        self.A_buffers: Dict[str, torch.Tensor] = {}
        self.B_buffers: Dict[str, torch.Tensor] = {}

    def get_buffer_id(self, name: LoraName) -> int:
        """查询 adapter 名当前驻留的 GPU 槽位编号."""
        if name not in self.name_to_buffer_id:
            raise KeyError(f"lora {name!r} not resident in pool")
        return self.name_to_buffer_id[name]

    def get_module_buffers(self, module_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回可直接喂给 segment GEMM 的堆叠权重 [K, ...]."""
        if module_key not in self.A_buffers:
            raise KeyError(f"module {module_key!r} not in pool")
        return self.A_buffers[module_key], self.B_buffers[module_key]

    def get_slot_weights(
        self, slot_id: int, module_key: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单槽视图(测试 / 参考实现用)."""
        a_stack, b_stack = self.get_module_buffers(module_key)
        return a_stack[slot_id], b_stack[slot_id]

    def finalize_buffers(self, module_shapes: Dict[str, Tuple[int, int]]) -> None:
        """按各 module 的 (in_dim, out_dim) 预分配堆叠 GPU buffer;只在首次创建。"""
        for key, (in_dim, out_dim) in module_shapes.items():
            if key in self.A_buffers:
                continue
            # A [K, max_rank, in],B [K, out, max_rank]
            self.A_buffers[key] = torch.zeros(
                (self.max_loras_per_batch, self.max_lora_rank, in_dim),
                device=self.device,
                dtype=self.dtype,
            )
            self.B_buffers[key] = torch.zeros(
                (self.max_loras_per_batch, out_dim, self.max_lora_rank),
                device=self.device,
                dtype=self.dtype,
            )

    def prepare_lora_batch(
        self,
        cur_names: Set[LoraName],
        *,
        weight_store: Dict[LoraName, AdapterWeights],
    ) -> None:
        """保证本 batch 的 adapter 都在 GPU;槽不够就驱逐不在本 batch 的,已驻留的跳过。"""
        for name in cur_names:
            self.eviction_policy.mark_used(name)

        evictions_before = self.eviction_policy.eviction_count

        for name in cur_names:
            if name in self.name_to_buffer_id:
                continue
            buffer_id = self._alloc_slot(cur_names)
            self._load_name_to_slot(name, buffer_id, weight_store)
            self.name_to_buffer_id[name] = buffer_id
            self.buffer_id_to_name[buffer_id] = name

        self._last_evictions = self.eviction_policy.eviction_count - evictions_before

    def build_batch_info(self, lora_ids: List[LoraName]) -> BatchLoRAInfo:
        """adapter 名 → per-row slot 与 per-slot rank/scaling;weight_indices 长度 = 行数。"""
        weight_indices: List[int] = []
        ranks = [0] * self.max_loras_per_batch
        scalings = [0.0] * self.max_loras_per_batch
        for name in lora_ids:
            slot = self.get_buffer_id(name)
            weight_indices.append(slot)
            r, s = self._slot_meta[slot]
            ranks[slot] = r
            scalings[slot] = s
        return BatchLoRAInfo(
            weight_indices=weight_indices,
            lora_ranks=ranks,
            scalings=scalings,
            evictions=self._last_evictions,
        )

    def _alloc_slot(self, cur_names: Set[LoraName]) -> int:
        """找可复用 slot:先空槽,再驱逐一个不在本 batch 的 adapter。"""
        for i in range(self.max_loras_per_batch):
            if self.buffer_id_to_name[i] == EMPTY_SLOT:
                return i

        candidates: Set[LoraName] = set()
        for i in range(self.max_loras_per_batch):
            name = self.buffer_id_to_name[i]
            if name == EMPTY_SLOT or name in cur_names:
                continue
            candidates.add(name)  # type: ignore[arg-type]

        if not candidates:
            raise ValueError(
                "no evictable pool slot (batch exceeds max_loras_per_batch)"
            )
        non_none = candidates - {None}
        victim = self.eviction_policy.select_victim(non_none or candidates)
        victim_slot = self.name_to_buffer_id.pop(victim)  # type: ignore[arg-type]
        self.eviction_policy.remove(victim)  # type: ignore[arg-type]
        self.buffer_id_to_name[victim_slot] = EMPTY_SLOT
        self._slot_meta.pop(victim_slot, None)
        return victim_slot

    def _load_name_to_slot(
        self,
        name: LoraName,
        buffer_id: int,
        weight_store: Dict[LoraName, AdapterWeights],
    ) -> None:
        """从 weight_store 取 CPU 权重,拷贝到 GPU 槽 buffer_id;None(基座) 只写 meta。"""
        if name is None:
            self._slot_meta[buffer_id] = (0, 0.0)
            return

        aw = weight_store[name]
        if aw.rank > self.max_lora_rank:
            raise ValueError(
                f"adapter {name!r} rank {aw.rank} > max_lora_rank {self.max_lora_rank}"
            )
        # TODO: 槽位复用时只清了本 adapter 拥有的模块,其余模块保留上一个 adapter 的残留权重,
        # 缺模块的 adapter(如只有 qkvo)会读到旧 MLP 权重。应在装载前按 slot 清空全部 A/B buffer。
        if any(key not in self.A_buffers for key in aw.modules):
            raise KeyError(
                f"adapter {name!r} modules not in pool; call finalize_buffers first"
            )
        for key, (a_cpu, b_cpu) in aw.modules.items():
            a_gpu = self.A_buffers[key][buffer_id]
            b_gpu = self.B_buffers[key][buffer_id]
            a_gpu.zero_()
            b_gpu.zero_()
            a_gpu[: aw.rank].copy_(a_cpu.to(dtype=self.dtype))
            b_gpu[:, : aw.rank].copy_(b_cpu.to(dtype=self.dtype))

        self._slot_meta[buffer_id] = (aw.rank, aw.scaling)
