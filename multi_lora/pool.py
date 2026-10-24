"""GPU 槽位池:把 adapter 的权重放进固定的 K 个 GPU 槽位,并负责装载与换出.

权重一共三层,本模块是最底下一层:
    PEFT 里的权重  --抽取-->  CPU 上的 AdapterWeights  --装载-->  本模块的 GPU 槽位
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch

from multi_lora.batch_info import BatchLoRAInfo
from multi_lora.eviction import EvictionPolicy, get_eviction_policy
from multi_lora.weights import AdapterWeights

# adapter 名;None 是一个合法的 key,表示"基座"(不使用任何 LoRA)
LoraName = Optional[str]

# 空槽位的占位标记。不能直接用 None,因为 None 已经被"基座"占用了
_EMPTY = object()


class LoRAMemoryPool:
    """K 个 GPU 槽位;每格放一个 adapter 的 A/B 权重,或空槽,或基座。

    - name_to_buffer_id: adapter 名 -> 槽位号
    - buffer_id_to_name: 槽位号 -> adapter 名 / _EMPTY
    - _slot_meta:        槽位号 -> (rank, scaling)
    """

    def __init__(
        self,
        max_loras_per_batch: int,
        max_lora_rank: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        eviction_policy: str = "lru",
    ) -> None:
        """只建立映射,不分配显存;buffer 由 finalize_buffers 按需创建."""
        self.max_loras_per_batch = max_loras_per_batch
        self.max_lora_rank = max_lora_rank
        self.device = device
        self.dtype = dtype
        self.eviction_policy: EvictionPolicy = get_eviction_policy(eviction_policy)

        self.name_to_buffer_id: Dict[LoraName, int] = {}
        self.buffer_id_to_name: List[object] = [_EMPTY] * max_loras_per_batch
        self._slot_meta: Dict[int, Tuple[int, float]] = {}

        # module_key -> 堆叠后的权重,形状 [K, max_rank, in] / [K, out, max_rank]
        self.A_buffers: Dict[str, torch.Tensor] = {}
        self.B_buffers: Dict[str, torch.Tensor] = {}

    # ---------- 查询 ----------

    def get_buffer_id(self, name: LoraName) -> int:
        """查 adapter 当前所在的槽位号;不在池里则报错."""
        if name not in self.name_to_buffer_id:
            raise KeyError(f"lora {name!r} not resident in pool")
        return self.name_to_buffer_id[name]

    def get_module_buffers(self, module_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回某个 module 的全部槽位权重,可直接喂给 segment GEMM."""
        if module_key not in self.A_buffers:
            raise KeyError(f"module {module_key!r} not in pool")
        return self.A_buffers[module_key], self.B_buffers[module_key]

    def get_slot_weights(
        self, slot_id: int, module_key: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """取单个槽位的 A/B 视图(测试与参考实现用)."""
        a_stack, b_stack = self.get_module_buffers(module_key)
        return a_stack[slot_id], b_stack[slot_id]

    # ---------- 建 buffer ----------

    def finalize_buffers(self, module_shapes: Dict[str, Tuple[int, int]]) -> None:
        """按各 module 的 (in_dim, out_dim) 分配 GPU buffer;已存在的直接跳过."""
        for key, (in_dim, out_dim) in module_shapes.items():
            if key in self.A_buffers:
                continue
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

    # ---------- 每个 forward step 的入口 ----------

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
        for slot in self._slot_meta:
            rank, scaling = self._slot_meta[slot]
            lora_ranks[slot] = rank
            scalings[slot] = scaling

        return BatchLoRAInfo(
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
        )

    # ---------- 内部:找槽位 / 写权重 ----------

    def _alloc_slot(self, keep: Set[LoraName]) -> int:
        """返回一个可写入的槽位:优先空槽,否则换出不在本 batch 里的 adapter."""
        for slot in range(self.max_loras_per_batch):
            if self.buffer_id_to_name[slot] is _EMPTY:
                return slot

        # 本 batch 正在用的槽不能换;基座(None)只能作为最后手段
        evictable: Set[LoraName] = set()
        for name in self.buffer_id_to_name:
            if name is _EMPTY:
                continue
            if name in keep:
                continue
            evictable.add(name)

        if not evictable:
            raise ValueError(
                "no evictable slot: batch needs more adapters than max_loras_per_batch"
            )

        # 优先换出真正的 adapter;只剩基座槽时才换基座
        candidates = set(evictable)
        candidates.discard(None)
        if not candidates:
            candidates = evictable

        victim = self.eviction_policy.select_victim(candidates)
        slot = self.name_to_buffer_id.pop(victim)
        self.eviction_policy.remove(victim)
        self.buffer_id_to_name[slot] = _EMPTY
        self._slot_meta.pop(slot, None)
        return slot

    def _load_name_to_slot(
        self,
        name: LoraName,
        slot: int,
        weight_store: Dict[LoraName, AdapterWeights],
    ) -> None:
        """把 CPU 权重拷进指定槽位;基座(None)只登记 rank=0,不写权重."""
        self.name_to_buffer_id[name] = slot
        self.buffer_id_to_name[slot] = name

        if name is None:
            self._slot_meta[slot] = (0, 0.0)
            return

        weights = weight_store[name]
        if weights.rank > self.max_lora_rank:
            raise ValueError(
                f"adapter {name!r} rank {weights.rank} > max_lora_rank {self.max_lora_rank}"
            )

        # 外部没调 finalize_buffers 时,按当前 adapter 的形状补建(幂等)
        shapes: Dict[str, Tuple[int, int]] = {}
        for key in weights.modules:
            a, b = weights.modules[key]
            shapes[key] = (int(a.shape[1]), int(b.shape[0]))
        self.finalize_buffers(shapes)

        # 先清空整个槽位,避免复用时残留上一个 adapter 的权重
        for key in self.A_buffers:
            self.A_buffers[key][slot].zero_()
            self.B_buffers[key][slot].zero_()

        for key in weights.modules:
            a_cpu, b_cpu = weights.modules[key]
            self.A_buffers[key][slot, : weights.rank].copy_(a_cpu.to(dtype=self.dtype))
            self.B_buffers[key][slot, :, : weights.rank].copy_(
                b_cpu.to(dtype=self.dtype)
            )

        self._slot_meta[slot] = (weights.rank, weights.scaling)
