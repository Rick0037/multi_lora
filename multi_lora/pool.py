"""GPU LoRA 权重槽位池：在有限槽位数内为不同 adapter 驻留 GPU 权重.

槽位维堆叠的 A_buffers / B_buffers 由 segment GEMM 直接读取，避免每步 stack copy。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import torch

from multi_lora.eviction import EvictionPolicy, get_eviction_policy
from multi_lora.registry import LoRARef
from multi_lora.weights import AdapterWeights


class _EmptySlot:
    """槽位未占用时的占位标记（与 None uid 区分）."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "|EMPTY|"


EMPTY_SLOT = _EmptySlot()
Uid = Optional[str]  # None 表示基座模型（无 LoRA 权重）


@dataclass
class BatchLoRAInfo:
    """单次 batch 的 LoRA 槽位索引与 rank/scaling 元数据（CPU list 形式）."""

    weight_indices: List[int]
    lora_ranks: List[int]
    scalings: List[float]
    evictions: int = 0


class LoRAMemoryPool:
    """max_loras_per_batch 个 GPU 槽位；uid（含 None=基座）映射到 slot."""

    def __init__(
        self,
        *,
        max_loras_per_batch: int,
        max_lora_rank: int,
        device: torch.device,
        dtype: torch.dtype,
        eviction_policy: str = "lru",
    ) -> None:
        """预分配槽位映射表；各 module 的 A/B buffer 在 finalize_buffers 时按需创建."""
        self.max_loras_per_batch = max_loras_per_batch
        self.max_lora_rank = max_lora_rank
        self.device = device
        self.dtype = dtype
        self.eviction_policy: EvictionPolicy = get_eviction_policy(eviction_policy)

        # uid ↔ slot 双向映射
        self.uid_to_buffer_id: Dict[Uid, int] = {}
        self.buffer_id_to_uid: List[Union[Uid, _EmptySlot]] = [
            EMPTY_SLOT
        ] * max_loras_per_batch
        # slot -> (rank, scaling)，供 build_batch_info 填充 per-slot 数组
        self._slot_meta: Dict[int, Tuple[int, float]] = {}

        # module_key -> [K, max_rank, in] / [K, out, max_rank]
        self.A_buffers: Dict[str, torch.Tensor] = {}
        self.B_buffers: Dict[str, torch.Tensor] = {}

    def get_buffer_id(self, uid: Uid) -> int:
        """查询 uid 当前驻留的 GPU 槽位编号."""
        if uid not in self.uid_to_buffer_id:
            raise KeyError(f"uid {uid!r} not resident in pool")
        return self.uid_to_buffer_id[uid]

    def get_module_buffers(self, module_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回可直接喂给 segment GEMM 的堆叠权重 [K, ...]."""
        if module_key not in self.A_buffers:
            raise KeyError(f"module_key {module_key!r} not in pool buffers")
        return self.A_buffers[module_key], self.B_buffers[module_key]

    def get_slot_weights(self, slot_id: int, module_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """单槽视图（测试 / 参考实现用）."""
        a_stack, b_stack = self.get_module_buffers(module_key)
        return a_stack[slot_id], b_stack[slot_id]

    def finalize_buffers(self, weight_store: Dict[Uid, AdapterWeights]) -> None:
        """根据 CPU weight store 预分配各 module 的堆叠 GPU buffer（仅首次创建）."""
        module_shapes: Dict[str, Tuple[int, int]] = {}
        for aw in weight_store.values():
            if not aw.modules:
                continue
            for key, (a_cpu, b_cpu) in aw.modules.items():
                in_dim = int(a_cpu.shape[1])
                out_dim = int(b_cpu.shape[0])
                if key in module_shapes:
                    old_in, old_out = module_shapes[key]
                    if (old_in, old_out) != (in_dim, out_dim):
                        raise ValueError(
                            f"shape mismatch for module {key!r}: "
                            f"{module_shapes[key]} vs {(in_dim, out_dim)}"
                        )
                else:
                    module_shapes[key] = (in_dim, out_dim)

        k = self.max_loras_per_batch
        for key, (in_dim, out_dim) in module_shapes.items():
            if key not in self.A_buffers:
                # 按 max_rank 上界 padding，实际有效 rank 由 lora_ranks 控制
                self.A_buffers[key] = torch.zeros(
                    k,
                    self.max_lora_rank,
                    in_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.B_buffers[key] = torch.zeros(
                    k,
                    out_dim,
                    self.max_lora_rank,
                    device=self.device,
                    dtype=self.dtype,
                )

    def prepare_lora_batch(
        self,
        cur_uids: Set[Uid],
        *,
        weight_store: Dict[Uid, AdapterWeights],
        lora_refs: Dict[str, LoRARef],
    ) -> None:
        """确保 cur_uids 中每个 uid 都有 GPU 槽位；必要时驱逐旧 adapter."""
        self.finalize_buffers(weight_store)

        # 本 batch 用到的 uid 标记为最近使用，供 LRU 驱逐
        for uid in cur_uids:
            self.eviction_policy.mark_used(uid)

        evictions_before = self.eviction_policy.eviction_count

        for uid in cur_uids:
            if uid in self.uid_to_buffer_id:
                continue  # 已驻留，跳过 H2D
            buffer_id = self._alloc_slot(cur_uids, lora_refs)
            self._load_uid_to_slot(uid, buffer_id, weight_store)
            self.uid_to_buffer_id[uid] = buffer_id
            self.buffer_id_to_uid[buffer_id] = uid

        self._last_evictions = self.eviction_policy.eviction_count - evictions_before

    def build_batch_info(self, lora_ids: List[Uid]) -> BatchLoRAInfo:
        """根据每条序列的 uid 构造 per-row slot 索引与 per-slot rank/scaling.

        weight_indices[i] = 第 i 条序列所用 uid 的 slot；
        lora_ranks[slot] / scalings[slot] 按下标索引槽位（非按行）。
        """
        weight_indices: List[int] = []
        ranks = [0] * self.max_loras_per_batch
        scalings = [0.0] * self.max_loras_per_batch
        for uid in lora_ids:
            slot = self.get_buffer_id(uid)
            weight_indices.append(slot)
            r, s = self._slot_meta[slot]
            ranks[slot] = r
            scalings[slot] = s
        return BatchLoRAInfo(
            weight_indices=weight_indices,
            lora_ranks=ranks,
            scalings=scalings,
            evictions=getattr(self, "_last_evictions", 0),
        )

    def _alloc_slot(self, cur_uids: Set[Uid], lora_refs: Dict[str, LoRARef]) -> int:
        """分配空闲槽位，或驱逐一个不在本 batch 且非 pinned 的 uid 后复用其槽."""
        # 1) 优先用物理空槽
        for i in range(self.max_loras_per_batch):
            if self.buffer_id_to_uid[i] == EMPTY_SLOT:
                return i

        # 2) 无空槽：候选 = 已占用 ∧ 不在 cur_uids ∧ 非 pinned
        candidates: Set[Uid] = set()
        for i in range(self.max_loras_per_batch):
            uid = self.buffer_id_to_uid[i]
            if uid == EMPTY_SLOT or uid in cur_uids:
                continue
            if uid is not None:
                ref = lora_refs.get(uid)
                if ref and ref.pinned:
                    continue
            candidates.add(uid)  # type: ignore[arg-type]

        if not candidates:
            raise ValueError(
                "no evictable pool slot (pinned adapters or batch exceeds max_loras_per_batch)"
            )
        # 优先驱逐 LoRA 槽；仅在无其它候选时驱逐基座槽（None）
        non_none = candidates - {None}
        victim = self.eviction_policy.select_victim(non_none or candidates)
        # 回收 victim 的映射，返回其 slot 号复用
        victim_slot = self.uid_to_buffer_id.pop(victim)  # type: ignore[arg-type]
        self.eviction_policy.remove(victim)  # type: ignore[arg-type]
        self.buffer_id_to_uid[victim_slot] = EMPTY_SLOT
        self._slot_meta.pop(victim_slot, None)
        return victim_slot

    def _load_uid_to_slot(
        self,
        uid: Uid,
        buffer_id: int,
        weight_store: Dict[Uid, AdapterWeights],
    ) -> None:
        """把 weight_store 中 uid 的 CPU 权重 copy 到 GPU 槽位 buffer_id."""
        if uid is None:
            # 基座 slot：无真实 LoRA 张量，rank=0 使 kernel 成为 no-op
            self._slot_meta[buffer_id] = (0, 0.0)
            return

        if uid not in weight_store:
            raise KeyError(f"adapter weights not in store: {uid}")
        aw = weight_store[uid]
        if aw.rank > self.max_lora_rank:
            raise ValueError(f"rank {aw.rank} > max_lora_rank {self.max_lora_rank}")

        for key, (a_cpu, b_cpu) in aw.modules.items():
            if key not in self.A_buffers:
                self.finalize_buffers(weight_store)
            a_gpu = self.A_buffers[key][buffer_id]
            b_gpu = self.B_buffers[key][buffer_id]
            a_gpu.zero_()
            b_gpu.zero_()
            # 只拷贝有效 rank 区域，其余 padding 保持 0
            a_gpu[: aw.rank].copy_(a_cpu.to(dtype=self.dtype))
            b_gpu[:, : aw.rank].copy_(b_cpu.to(dtype=self.dtype))

        self._slot_meta[buffer_id] = (aw.rank, aw.scaling)
