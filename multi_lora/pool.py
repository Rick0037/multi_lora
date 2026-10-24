"""GPU LoRA 权重槽位池;在有限槽位数内为不同 adapter 驻留 GPU 权重.

槽位维堆叠的 A_buffers / B_buffers 由 segment GEMM 直接读取,避免每步 stack copy。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import torch

from multi_lora.eviction import EvictionPolicy, get_eviction_policy
from multi_lora.weights import AdapterWeights


class _EmptySlot:
    """槽位未占用时的占位标记(与 None = 基座区分)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "|EMPTY|"


EMPTY_SLOT = _EmptySlot()
LoraName = Optional[str]  # None 表示基座模型(无 LoRA 权重)


@dataclass
class BatchLoRAInfo:
    """一次 batch 的槽位元数据;weight_indices 按行,ranks/scalings 按 slot(K 长)."""

    weight_indices: List[int]
    lora_ranks: List[int]
    scalings: List[float]
    evictions: int = 0


class LoRAMemoryPool:
    """K = max_loras_per_batch 个 GPU 槽位;槽内是 adapter 名、None(基座) 或 EMPTY_SLOT."""

    def __init__(
        self,
        *,
        max_loras_per_batch: int,
        max_lora_rank: int,
        device: torch.device,
        dtype: torch.dtype,
        eviction_policy: str = "lru",
    ) -> None:
        """只建映射,不分配显存;buffer 由 finalize_buffers 懒创建。"""
        # K: 同时驻留 GPU 的不同 adapter（含基座槽位）的个数上界。
        self.max_loras_per_batch = max_loras_per_batch
        self.max_lora_rank = max_lora_rank
        self.device = device
        self.dtype = dtype
        self.eviction_policy: EvictionPolicy = get_eviction_policy(eviction_policy)

        # * name to buffer index, None（基座）也可占用一个槽位。
        self.name_to_buffer_id: Dict[LoraName, int] = {}
        # 列表下标就是 slot_id;Union 表示槽内绑定的是 adapter 名或 EMPTY_SLOT 空闲标记。
        self.buffer_id_to_name: List[Union[LoraName, _EmptySlot]] = [
            EMPTY_SLOT
        ] * max_loras_per_batch
        # slot -> (rank, scaling),供 build_batch_info 填充 per-slot 数组
        self._slot_meta: Dict[int, Tuple[int, float]] = {}
        # 上一轮 prepare 触发的驱逐次数,供 build_batch_info 读取
        self._last_evictions = 0

        # module_key -> [K, max_rank, in] / [K, out, max_rank]
        # K = max_loras_per_batch(槽位总数),第一维就是槽位号
        # key 例如 'layers.0.self_attn.q_proj';同一 key 的 A/B 按槽位维堆叠。
        self.A_buffers: Dict[str, torch.Tensor] = {}
        self.B_buffers: Dict[str, torch.Tensor] = {}

    def get_buffer_id(self, name: LoraName) -> int:
        """adapter 名 → 槽位号;不在池中抛 KeyError。"""
        if name not in self.name_to_buffer_id:
            raise KeyError(f"lora {name!r} not resident in pool")
        return self.name_to_buffer_id[name]

    def get_module_buffers(self, module_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """该 module 的堆叠权重:A [K, max_rank, in],B [K, out, max_rank]。"""
        if module_key not in self.A_buffers:
            raise KeyError(f"module_key {module_key!r} not in pool buffers")
        return self.A_buffers[module_key], self.B_buffers[module_key]

    def get_slot_weights(
        self, slot_id: int, module_key: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单槽视图(测试用):A [max_rank, in],B [out, max_rank]。"""
        a_stack, b_stack = self.get_module_buffers(module_key)
        return a_stack[slot_id], b_stack[slot_id]

    def finalize_buffers(self, weight_store: Dict[LoraName, AdapterWeights]) -> None:
        """为出现过的 module 预分配 GPU buffer;已存在则跳过,in/out 不一致报错。"""
        module_shapes: Dict[str, Tuple[int, int]] = {}
        # import pdb

        # pdb.set_trace()
        for aw in weight_store.values():
            if not aw.modules:
                continue

            for key, (a_cpu, b_cpu) in aw.modules.items():
                # * 'layers.0.self_attn.q_proj'
                """
                (Pdb) in_dim
                2560
                (Pdb) out_dim
                4096
                """
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
                # 按 max_rank 上界 padding,实际有效 rank 由 lora_ranks 控制
                #! 第一个维度是K, 也就是把整个buffer内的权重都保存下来了
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
        cur_names: Set[LoraName],  # 当前批次需要的所有 adapter 名，包含 None（基座）
        *,
        weight_store: Dict[LoraName, AdapterWeights],  # 已加载的 CPU 权重集合
    ) -> None:
        """保证本 batch 的 adapter 都在 GPU;槽不够就驱逐不在本 batch 的,已驻留的跳过。"""
        # * 没有用
        # self.finalize_buffers(weight_store)

        # 本 batch 用到的 adapter 标记为最近使用,供 LRU 驱逐
        # * (1) 标记使用, 实际上是加队列里面, 起的名字不好
        for name in cur_names:
            self.eviction_policy.mark_used(name)

        evictions_before = self.eviction_policy.eviction_count

        for name in cur_names:
            if name in self.name_to_buffer_id:
                continue  # 已驻留,跳过 H2D
            # * 驱逐找到一个可以的buffer_id
            buffer_id = self._alloc_slot(cur_names)
            # * 把权重挂上去 weight store
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
        # 1) 优先用物理空槽
        for i in range(self.max_loras_per_batch):
            if self.buffer_id_to_name[i] == EMPTY_SLOT:
                return i

        # 2) 无空槽;候选 = 已占用 ∧ 不在 cur_names
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
        # 优先驱逐 LoRA 槽;仅在无其它候选时驱逐基座槽(None)
        non_none = candidates - {None}
        victim = self.eviction_policy.select_victim(non_none or candidates)
        # 一次只回收一个 victim;还有未驻留 adapter 时由外层循环再次调用
        # 回收 victim 的映射,返回其 slot 号复用
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
        """CPU → GPU 拷贝;None 只写 rank=0,rank 超上限报错。"""
        if name is None:
            # 基座 slot;无真实 LoRA 张量,rank=0 使 kernel 成为 no-op
            self._slot_meta[buffer_id] = (0, 0.0)
            return

        if name not in weight_store:
            raise KeyError(f"adapter weights not in store: {name}")
        aw = weight_store[name]
        if aw.rank > self.max_lora_rank:
            raise ValueError(f"rank {aw.rank} > max_lora_rank {self.max_lora_rank}")

        # TODO: 槽位复用时只清了本 adapter 拥有的模块,其余模块保留上一个 adapter 的残留权重,
        # 缺模块的 adapter(如只有 qkvo)会读到旧 MLP 权重。应在装载前按 slot 清空全部 A/B buffer。
        if any(key not in self.A_buffers for key in aw.modules):
            self.finalize_buffers(weight_store)
        for key, (a_cpu, b_cpu) in aw.modules.items():
            a_gpu = self.A_buffers[key][buffer_id]
            b_gpu = self.B_buffers[key][buffer_id]
            a_gpu.zero_()
            b_gpu.zero_()
            # 槽内 A 为 [max_rank, in],B 为 [out, max_rank]
            # 只拷贝有效 rank 区域,其余 padding 保持 0
            a_gpu[: aw.rank].copy_(a_cpu.to(dtype=self.dtype))
            b_gpu[:, : aw.rank].copy_(b_cpu.to(dtype=self.dtype))

        self._slot_meta[buffer_id] = (aw.rank, aw.scaling)
