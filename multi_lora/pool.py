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
        # T17 异步装载:专用 copy stream + 完成事件(仅 cuda;CPU 测试走同步)
        self._copy_stream = None
        self._load_event = None
        # T17b 分组暂存:(key, rank) -> (pinned_a, pinned_b, 容量);缺 module 的零源缓存
        self._stage: Dict[tuple, tuple] = {}
        self._zeros: Dict[tuple, tuple] = {}

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

    def write_many(self, pairs) -> None:
        """把一批 adapter 装进各自槽位(T17b:按 (module,rank) 分组 + pinned 暂存)。

        三个版本的教训:
          v6 逐条 pageable:9216 次同步仪式 ≈ 350ms CPU
          v8 装箱补零:次数少了但字节翻倍(h2d GPU 118→275ms)
          v9 pinned 逐条:字节紧凑、GPU 只忙 153ms,但 9216 次 Python 级
             copy_ 的调度(~70µs/次)≈ 580ms CPU,成了新瓶颈(TPOT 732ms)
        本版把表达方式改成每 (module, rank) 一组 6 条语句:
          cat 聚合进 pinned 暂存(紧凑,不补零) → 1 次异步 H2D → 1 次散布
        torch 调用 ~3.5k,聚合的 CPU memcpy(~1.4GB)与 GPU 搬运流水。
        pairs = [(槽位号, AdapterWeights), ...],k64swap32 时长度 32。
        """
        if not pairs:
            return
        if self.device.type == "cuda":
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream()
                self._load_event = torch.cuda.Event()
            with torch.cuda.stream(self._copy_stream):
                self._load_grouped(pairs, non_blocking=True)
            self._load_event.record(self._copy_stream)
        else:
            self._load_grouped(pairs, non_blocking=False)

    def _load_grouped(self, pairs, non_blocking: bool) -> None:
        """按 rank 分组:同组内各 adapter 的 (key, A/B) 形状一致,才能 cat 成块."""
        by_rank: Dict[int, list] = {}
        for slot, w in pairs:
            by_rank.setdefault(w.rank, []).append((slot, w))
        for rank, group in by_rank.items():
            h = len(group)
            slots = torch.tensor([s for s, _ in group], device=self.device)
            for key in self.A:
                in_dim = self.A[key].shape[2]
                out_dim = self.B[key].shape[1]
                pa, pb = self._ensure_stage(
                    key, rank, h, in_dim, out_dim, pinned=non_blocking
                )
                a_list, b_list = [], []
                for _, w in group:
                    if key in w.modules:
                        a_src, b_src = w.modules[key]
                    else:
                        # 该 adapter 没此 module:聚合时填零(kernel 只读前 rank 行)
                        a_src, b_src = self._zero_src(key, rank, in_dim, out_dim)
                    a_list.append(
                        a_src if a_src.dtype == self.dtype else a_src.to(self.dtype)
                    )
                    b_list.append(
                        b_src if b_src.dtype == self.dtype else b_src.to(self.dtype)
                    )
                # stack 把 h 个 [rank, in] 叠成 [h, rank, in](cat 是沿已有
                # 维度粘接,会给成 [h*rank, in],形状对不上 out 会触发 resize)
                torch.stack(a_list, 0, out=pa[:h])
                torch.stack(b_list, 0, out=pb[:h])
                ga = pa[:h].to(self.device, non_blocking=non_blocking)
                gb = pb[:h].to(self.device, non_blocking=non_blocking)
                # 槽位行宽是 max_lora_rank,只散前 rank 行/列;尾部的旧行
                # kernel 按 rank 截断读不到,不清
                self.A[key][slots, :rank] = ga
                self.B[key][slots, :, :rank] = gb

    def _ensure_stage(self, key, rank, h, in_dim, out_dim, pinned):
        """(key, rank) 的 pinned 暂存,容量不够就翻倍重分配(总 pinned ≈ 紧凑字节量)."""
        ent = self._stage.get((key, rank))
        if ent is None or ent[2] < h:
            cap = h if ent is None else max(h, ent[2] * 2)
            ent = (
                torch.empty(cap, rank, in_dim, dtype=self.dtype, pin_memory=pinned),
                torch.empty(cap, out_dim, rank, dtype=self.dtype, pin_memory=pinned),
                cap,
            )
            self._stage[(key, rank)] = ent
        return ent[0], ent[1]

    def _zero_src(self, key, rank, in_dim, out_dim):
        """缺 module 的零源(按 (key, rank) 缓存),cat 时占位成全零权重."""
        z = self._zeros.get((key, rank))
        if z is None:
            z = (
                torch.zeros(rank, in_dim, dtype=self.dtype),
                torch.zeros(out_dim, rank, dtype=self.dtype),
            )
            self._zeros[(key, rank)] = z
        return z

    def wait_load(self) -> None:
        """计算流等待最近一次异步装载完成(prepare 末尾调,forward 前生效)."""
        if self._copy_stream is not None:
            torch.cuda.current_stream().wait_event(self._load_event)

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
        """保证 lora_names 都在 GPU 上:缺的逐个分配槽位,槽满就换出别的 adapter.

        装槽动作按批聚合(T16):先收集本步全部 (slot, weights),最后一次
        write_many —— H2D 次数与 batch 内 adapter 数解耦,恒为 2*module 数。
        """
        for name in lora_names:
            self.eviction_policy.mark_used(name)

        pairs = []
        for name in lora_names:
            if name in self.name_to_slot:
                continue
            slot = self._alloc_slot(lora_names)
            self.name_to_slot[name] = slot
            self.slot_names[slot] = name
            if name is None:
                # 基座只登记 rank=0,不占权重
                self._slot_rank_scaling[slot] = (0, 0.0)
                continue
            weights = weight_store[name]
            if weights.rank > self.max_lora_rank:
                raise ValueError(
                    f"adapter {name!r} rank {weights.rank} > max_lora_rank {self.max_lora_rank}"
                )
            self.buffers.ensure(weights)
            self._slot_rank_scaling[slot] = (weights.rank, weights.scaling)
            pairs.append((slot, weights))
        self.buffers.write_many(pairs)
        # 计算流排在异步搬运之后:本函数返回后,当前流上的 forward/replay
        # 一定读得到刚写入的槽位
        self.buffers.wait_load()

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
