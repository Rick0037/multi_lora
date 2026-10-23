"""GPU LoRA 权重槽位池;在有限槽位数内为不同 adapter 驻留 GPU 权重。

核心机制;
- 固定 max_loras_per_batch 个槽位 slot_id(0 .. K-1);每个槽可绑定一个 uid
  (lora_id 或 None 表示基座)。
- prepare_lora_batch(cur_uids);保证当前 batch 出现的每个 uid 都在某个 slot 里;
  若槽满,从不在 cur_uids、且非 pinned 的已占槽中选 victim,驱逐后复用该 slot。
- build_batch_info(lora_ids);为 batch 内每一行给出 weight_indices[i](第i行对应的uid lora所在slot),
  以及各槽的 lora_ranks / scalings,供 segment GEMM 按行选权重。

实现上按模块路径 key 在各槽内存储 LoRA A/B;由 uid 映射到可计算的 GPU 张量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import torch

from multi_lora_phase3.eviction import EvictionPolicy, get_eviction_policy
from multi_lora_phase3.registry import LoRARef
from multi_lora_phase3.weights import AdapterWeights


class _EmptySlot:
    __slots__ = ()

    def __repr__(self) -> str:
        return "|EMPTY|"


EMPTY_SLOT = _EmptySlot()
Uid = Optional[str]
# 单个槽内的数据;{模块 key : (A_gpu, B_gpu)};A 为 [max_rank, in],B 为 [out, max_rank],有效 rank 用 meta 记录。
SlotTensors = Dict[str, Tuple[torch.Tensor, torch.Tensor]]
# * str = "layer_0.self_attn.q_proj"
# * Tuple[torch.Tensor, torch.Tensor] -> a_weight, b_weight


@dataclass
class BatchLoRAInfo:
    """单次 batch 的 LoRA 槽位索引与 rank/scaling 元数据."""

    weight_indices: List[int]
    lora_ranks: List[int]
    scalings: List[float]
    evictions: int = 0


class LoRAMemoryPool:
    """总共 max_loras_per_batch 个 GPU 槽位;uid(含 None=基座)映射到 slot.
    有多的就分配,没有就evict
    """

    def __init__(
        self,
        *,
        max_loras_per_batch: int,
        max_lora_rank: int,
        device: torch.device,
        dtype: torch.dtype,
        eviction_policy: str = "lru",
    ) -> None:
        # K;单步 forward 最多同时驻留 GPU 的不同 adapter(+基座)个数上界。
        self.max_loras_per_batch = max_loras_per_batch
        # 所有槽上 A/B 张量第一维按此上界分配;实际 adapter rank 可更小,有效部分 [:rank]。
        # A 为 [max_rank, in],B 为 [out, max_rank]
        self.max_lora_rank = max_lora_rank
        self.device = device
        self.dtype = dtype
        self.eviction_policy: EvictionPolicy = get_eviction_policy(eviction_policy)

        # uid -> 槽位下标;None(基座)也可占用一个槽位。
        self.uid_to_buffer_id: Dict[Uid, int] = {}
        # 列表为当前绑定的 uid,即槽位index为i的槽位绑定的uid,如果i槽位没有任何lora,则为EMPTY_SLOT 表示该槽空闲可分配。
        # * Union[X, Y] 表示类型是`X` 或`Y` 要不是empty 要不是uid
        self.buffer_id_to_uid: List[Union[Uid, _EmptySlot]] = [
            EMPTY_SLOT
        ] * max_loras_per_batch
        #! buffer 都指的是这个
        # slot_id -> module_key -> (A_gpu, B_gpu)
        self._slot_tensors: Dict[int, SlotTensors] = {}
        # rank, scaling
        self._slot_meta: Dict[int, Tuple[int, float]] = {}

    def get_buffer_id(self, uid: Uid) -> int:
        if uid not in self.uid_to_buffer_id:
            raise KeyError(f"uid {uid!r} not resident in pool")
        return self.uid_to_buffer_id[uid]

    def get_slot_weights(
        self, slot_id: int, module_key: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._slot_tensors[slot_id][module_key]

    def prepare_lora_batch(
        self,
        cur_uids: Set[Uid],  # * 当前批次的所有uid
        *,
        weight_store: Dict[Uid, AdapterWeights],  # * 之前所有的weight
        lora_refs: Dict[str, LoRARef],  # * 之前的info
    ) -> None:
        """确保 cur_uids 中每个 uid 都在 GPU 槽位中,如不在则分配slot;必要时驱逐并 H2D 拷贝.

        调用顺序;先对所有 cur_uids mark_used(更新 LRU/FIFO),再逐个确保 resident;
        这样本 batch 仍要用的 uid 不会在选 victim 时进入 candidates。
        """
        # * 实际上是在lru 里面去更新一个标志位
        for uid in cur_uids:
            self.eviction_policy.mark_used(uid)

        # * 就是一个数据统计
        evictions_before = self.eviction_policy.eviction_count
        # * 这里是直接传出来的包括none {None, '3b27455aebac4567bbd9b15b3341896e'}
        for uid in cur_uids:
            # * 已在 GPU,无需再次 H2D, 外层
            if uid in self.uid_to_buffer_id:
                continue
            # 为当前请求需要的所有lora id分配buffer
            buffer_id = self._alloc_slot(cur_uids, lora_refs)
            # 为当前请求需要的所有lora id对应的lora weight加载到上述gpu buffer
            self._load_uid_to_slot(uid, buffer_id, weight_store)
            self.uid_to_buffer_id[uid] = buffer_id
            self.buffer_id_to_uid[buffer_id] = uid

        # 供上层打印/单测观测;本轮 prepare 触发了几次 select_victim。
        self._last_evictions = self.eviction_policy.eviction_count - evictions_before

    def build_batch_info(self, lora_ids: List[Uid]) -> BatchLoRAInfo:
        """输入该batch所有seq对应的lora id(uid:str),据此为每条seq拿到 lora weight与 per-slot rank/scaling.

        weight_indices 长度 = batch,第 i 行为第 i 条序列所用 uid 的 slot。
        lora_ranks / scalings 长度 = max_loras_per_batch,按下标索引对应槽位。
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
        # 1) 优先用物理空槽。
        for i in range(self.max_loras_per_batch):
            if self.buffer_id_to_uid[i] == EMPTY_SLOT:
                return i

        # 2) 无空槽;构造可驱逐候选 ;已占用slot但却不在本 batch、且非 pinned的lora id buffer。
        candidates: Set[Uid] = set()
        # * 找出不用的gpu, NONE 也包括
        for i in range(self.max_loras_per_batch):
            uid = self.buffer_id_to_uid[i]
            # * 该lora id被当前某个请求选中
            if uid == EMPTY_SLOT or uid in cur_uids:
                continue
            # 该lora id未被当前某个请求选中,则写入驱逐列表
            if uid is not None:
                ref = lora_refs.get(uid)
                if ref and ref.pinned:
                    continue
            candidates.add(uid)  # type: ignore[arg-type]

        if not candidates:
            raise ValueError(
                "no evictable pool slot (pinned adapters or batch exceeds max_loras_per_batch)"
            )
        # 驱逐逻辑,见eviction.py
        # 优先驱逐 LoRA 槽;仅在无其它候选时驱逐基座槽(None)。
        non_none = candidates - {None}
        if non_none:
            victim = self.eviction_policy.select_victim(non_none)
        else:
            victim = self.eviction_policy.select_victim(candidates)

        # 回收 被驱逐slot: victim 的映射与 GPU 张量;返回该slot 号复用。
        # * 只选择出了一个victim_slot
        victim_slot = self.uid_to_buffer_id.pop(victim)  # type: ignore[arg-type]
        self.eviction_policy.remove(victim)  # type: ignore[arg-type]
        self.buffer_id_to_uid[victim_slot] = EMPTY_SLOT
        self._slot_tensors.pop(victim_slot, None)
        self._slot_meta.pop(victim_slot, None)
        return victim_slot

    def _load_uid_to_slot(
        self,
        uid: Uid,
        buffer_id: int,
        weight_store: Dict[Uid, AdapterWeights],
    ) -> None:
        """
        将预先在weight_store这个cpu里的lora weight加载到gpu上对应的buffer Id即slot里面
        """
        if uid is None:
            # 基座;不占真实 LoRA 张量;rank/scaling 置 0,仅占用 slot 映射。
            self._slot_tensors[buffer_id] = {}
            self._slot_meta[buffer_id] = (0, 0.0)
            return

        if uid not in weight_store:
            raise KeyError(f"adapter weights not in store: {uid}")
        # * 取出当前的weight
        aw = weight_store[uid]
        if aw.rank > self.max_lora_rank:
            raise ValueError(f"rank {aw.rank} > max_lora_rank {self.max_lora_rank}")

        slot: SlotTensors = {}
        for key, (a_cpu, b_cpu) in aw.modules.items():
            # * 预分配最大 rank 大小的 GPU buffer,下面只填前 aw.rank 行/列,其余为 0。
            a_gpu = torch.zeros(
                self.max_lora_rank,
                a_cpu.shape[1],  # torch.Size([16, 2560])
                device=self.device,  # GPU
                dtype=self.dtype,
            )
            b_gpu = torch.zeros(
                b_cpu.shape[0],  # torch.Size([4096, 16])
                self.max_lora_rank,
                device=self.device,  # GPU
                dtype=self.dtype,
            )

            # CPU -> GPU;将权重写入槽位张量。
            a_gpu[: aw.rank].copy_(a_cpu.to(dtype=self.dtype))
            b_gpu[:, : aw.rank].copy_(b_cpu.to(dtype=self.dtype))
            slot[key] = (a_gpu, b_gpu)  # ! 'layers.0.self_attn.q_proj'
        # *dict_keys(['layers.0.self_attn.q_proj', 'layers.0.self_attn.k_proj' ......
        self._slot_tensors[buffer_id] = slot
        self._slot_meta[buffer_id] = (aw.rank, aw.scaling)
