"""GPU 槽位池:固定 K 个槽位,负责把 CPU 权重装进去、槽满时换出.

PEFT 权重 --抽取--> CPU AdapterWeights --装载--> 本模块的 GPU 槽位
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import torch

from multi_lora.batch_info import BatchLoRAInfo
from multi_lora.eviction import EvictionPolicy
from multi_lora.registry import LoraName
from multi_lora.weights import AdapterWeights

# T18:一次往 GPU 暂存里聚合几个 adapter。暂存显存 = chunk * 单 adapter 扁平字节,
# 4 个 rank64 adapter 约 344MB;再大就该分块,再小则 H2D 发起次数回升。
_BANK_CHUNK = 4


class _RankPlan:
    """某个 rank 下 adapter 权重的扁平布局:先排所有 A 块,再排所有 B 块。

    同 rank 的所有 adapter 共用这一套偏移,所以每个 adapter 的整块内容可以
    一次连续 H2D;池里按 key 切视图散回去。
    """

    """
    AdapterWeights.modules  (144 个散张量: q.A, q.B, v.A, v.B, ...)
        │  按 _RankPlan 的偏移,逐段 copy_ 进一块连续 pinned buffer
        ▼
    bank = [q.A | k.A | v.A | o.A | ... | q.B | k.B | ...]   ← 一维,长 total
        存进 self._banks[id(weights)],带强引用防 id 复用
    """

    def __init__(self, rank: int, keys: List[str], dims: Dict[str, Tuple[int, int]]):
        self.rank = rank
        self.keys = keys
        self.a_off: Dict[str, int] = {}
        self.b_off: Dict[str, int] = {}
        self.a_shape: Dict[str, Tuple[int, int]] = {}
        self.b_shape: Dict[str, Tuple[int, int]] = {}
        off = 0
        for key in keys:
            in_dim, _ = dims[key]
            self.a_off[key] = off
            self.a_shape[key] = (rank, in_dim)
            off += rank * in_dim
        for key in keys:
            _, out_dim = dims[key]
            self.b_off[key] = off
            self.b_shape[key] = (out_dim, rank)
            off += out_dim * rank
        self.total = off
        #! T19:GPU 上的散布描述表(懒建,见 build_tables), 如何使用GPU去进行切片于解包
        self.tables = None

    def build_tables(self, buffers, device) -> dict:
        """把 144 个 key 的池地址/步长/段偏移打包成 GPU 张量,给散布 kernel 用。

        池张量按 key 各自独立分配,无法当 kernel 参数逐个传;这里取第一个
        A 当 base,其余位置表示成相对它的元素偏移(int64,可为负)。
        A/B 建好后地址不再变,_plan_for 在 key 集合变化时整体作废重建。
        """
        keys = self.keys
        a, b = buffers.A, buffers.B
        base = a[keys[0]]
        el = base.element_size()

        def i32(xs):
            return torch.tensor(xs, dtype=torch.int32, device=device)

        def i64(xs):
            return torch.tensor(xs, dtype=torch.int64, device=device)

        a_numel = [self.rank * self.a_shape[k][1] for k in keys]
        b_numel = [self.b_shape[k][0] * self.rank for k in keys]
        self.tables = {
            "base": base,
            "a_delta": i64([(a[k].data_ptr() - base.data_ptr()) // el for k in keys]),
            "a_off": i32([self.a_off[k] for k in keys]),
            "a_numel": i32(a_numel),
            "a_sstride": i32([a[k].stride(0) for k in keys]),
            "b_delta": i64([(b[k].data_ptr() - base.data_ptr()) // el for k in keys]),
            "b_off": i32([self.b_off[k] for k in keys]),
            "b_numel": i32(b_numel),
            "b_sstride": i32([b[k].stride(0) for k in keys]),
            "b_rstride": i32([b[k].stride(1) for k in keys]),
            "n": len(keys),
            "max_numel": max(max(a_numel), max(b_numel)),
        }
        return self.tables


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
        # T19 散布实现:kernel(默认)/python;LORA_SCATTER_IMPL=python 切回参考实现
        self.scatter_impl = os.environ.get("LORA_SCATTER_IMPL", "kernel")
        # T17 异步装载:专用 copy stream + 完成事件(仅 cuda;CPU 测试走同步)
        self._copy_stream = None
        self._load_event = None
        # T18:每个 adapter 一份"整块 pinned bank"(布局见 _RankPlan),换入时整块 H2D
        self._banks: Dict[int, tuple] = {}
        self._plans: Dict[int, "_RankPlan"] = {}
        self._plan_keys: Optional[tuple] = None
        # T18:bank 建好后释放 weight_store 的散张量(权重只留一份,省约 4.2GB 主机内存)
        self.release_modules = True
        # T18:GPU 侧聚合暂存 [chunk, total],跨 rank 复用,不随 rank 数叠加
        self._stage: Optional[torch.Tensor] = None

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
        """把一批 adapter 装进各自槽位(T18:整块 pinned bank 一次 H2D)。

        版本账(k64swap32,每步 32 个 adapter):
          v6 逐条 pageable   9216 次同步仪式 ≈ 350ms CPU
          v8 装箱补零        次数少但字节翻倍(h2d GPU 118→275ms)
          v9 pinned 逐条     GPU 只忙 153ms,但 9216 次调度 ≈ 580ms CPU
          v10 按(module,rank)聚合  TPOT 732→428ms,但多一层 CPU→CPU 拷贝
                             (~1.4GB/步)仍占大头
        本版首次换入时把整个 adapter 的 A/B 按固定布局拼成一块 pinned bank,
        之后每次重装就是"32 次整块 H2D + 288*2 次 GPU 内散布":CPU 不再搬运
        权重字节,只发起拷贝;H2D 全是连续大块。
        pairs = [(槽位号, AdapterWeights), ...],k64swap32 时长度 32。
        """
        if not pairs:
            return
        if self.device.type == "cuda":
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream()
                self._load_event = torch.cuda.Event()
            # 抢在计算流之前:上一轮 forward/replay 还在读这些槽位时不能覆写
            self._copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._copy_stream):
                self._load_flat(pairs, non_blocking=True)
            #! 流切换, 切换到加载过程, 但是外部其实还在同步等加载流结束
            self._load_event.record(self._copy_stream)
        else:
            self._load_flat(pairs, non_blocking=False)

    def _load_flat(self, pairs, non_blocking: bool) -> None:
        by_rank: Dict[int, list] = {}
        # * <rank, [slot_id, weight]> 同样rank的分一组
        for slot, w in pairs:
            by_rank.setdefault(w.rank, []).append((slot, w))
        """
        第1轮: (5, 甲)   甲.rank=8  → by_rank[8] 追加 (5, 甲)
        第2轮: (17, 乙)  乙.rank=8  → by_rank[8] 追加 (17, 乙)
        第3轮: (3, 丙)   丙.rank=16 → by_rank[16] 追加 (3, 丙)
        """
        # T19:GPU 且配置为 kernel 时走单 kernel 散布;否则 python 高级索引(参考实现)
        use_kernel = self.device.type == "cuda" and self.scatter_impl == "kernel"
        if use_kernel:
            from multi_lora.kernels.scatter_bank import scatter_stage

        for rank, group in by_rank.items():  # 32个按rank分4桶(8/16/32/64各8个)
            plan = self._plan_for(rank)  # 该rank的偏移表
            stage = self._ensure_stage(plan.total)  # GPU暂存,形如如下, 装不下了继续
            """
            stage(rank=8 时)
            行0 [ ................................ ]  ← 宽 total,正好装一个 adapter 的全部
            行1 [ ................................ ]
            行2 [ ................................ ]
            行3 [ ................................ ]
            """
            if non_blocking:
                # 暂存跨 rank 复用、可能在默认流分配却在本流使用,登记一下
                # 免得重分配后被分配器提前复用
                stage.record_stream(torch.cuda.current_stream())
            # T19:144 个 key 的池地址/步长/段偏移,一次打包给 kernel
            tables = plan.tables or plan.build_tables(self, self.device)
            # * 分组循环, 每次处理_BANK_CHUNK个
            for start in range(0, len(group), _BANK_CHUNK):
                # * range(0, 8, 4)` 就是`[0, 4]` ——从 0 开始，每次加 4。所以这个循环跑两轮。
                chunk = group[start : start + _BANK_CHUNK]
                h = len(chunk)
                for i, (_, w) in enumerate(chunk):
                    # 整块 pinned 源 → 暂存第 i 行,一次连续异步 H2D
                    stage[i].copy_(self._bank_for(w, plan), non_blocking=non_blocking)
                    # * rank = 4 中的某一个权重直接压缩到stage的一行中了
                slots = torch.tensor([s for s, _ in chunk], device=self.device)
                #! 某一个chunk 4个adapter对应的 `slots = [5, 17, 3, 40]`
                # *------------权重分发----------------
                # T19:一个 kernel 做完本 chunk 全部 144 key 的 A+B 散布
                # (v11 是 144×2 条 index_put_,1152 轮 Python 循环 ≈23ms CPU)
                if use_kernel:
                    scatter_stage(stage, slots, tables, plan.total, h, rank)
                    continue
                for key in plan.keys:  # 144 个 module 挨个来, 先看 key="q_proj"
                    a_off = plan.a_off[key]  #! 在play中寻找 自己矩阵a 对应的offset
                    a_numel = plan.rank * plan.a_shape[key][1]  # * 实际需要的数组
                    self.A[key][slots, :rank] = stage[:h, a_off : a_off + a_numel].view(
                        h, *plan.a_shape[key]
                    )
                    b_off = plan.b_off[key]
                    b_numel = plan.b_shape[key][0] * rank
                    self.B[key][slots, :, :rank] = stage[
                        :h, b_off : b_off + b_numel
                    ].view(h, *plan.b_shape[key])

    def _plan_for(self, rank: int) -> "_RankPlan":
        """取该 rank 的扁平布局;module 集合变了就作废旧布局和旧 bank."""
        keys = tuple(self.A)
        if keys != self._plan_keys:
            self._plans.clear()
            self._banks.clear()
            self._plan_keys = keys
        plan = self._plans.get(rank)
        if plan is None:
            dims = {k: (self.A[k].shape[2], self.B[k].shape[1]) for k in keys}
            plan = _RankPlan(rank, list(keys), dims)
            self._plans[rank] = plan
        return plan

    def _bank_for(self, weights: AdapterWeights, plan: "_RankPlan") -> torch.Tensor:
        """该 adapter 的整块 pinned 源;首次用到时拼一次,之后按 id 复用。

        缺 module 的位置填零(kernel 只按 rank 截断读,与旧零源语义一致)。
        拼好后按 release_modules 释放 weight_store 里的散张量:同一份权重只留
        bank 一份,省一份主机内存(96 个 adapter 约 4.2GB)。
        """
        ent = self._banks.get(id(weights))
        if ent is not None and ent[0] is weights and ent[1] == plan.rank:
            return ent[2]
        if weights.flattened:
            # 释放过的权重拼不回新布局:宁可报错,不能静默写错数据
            raise RuntimeError(
                f"adapter(rank{plan.rank}) 的权重已扁平化并释放,"
                f"module 布局从 {len(plan.keys)} 个变过之后无法重建 bank"
            )
        flat = torch.empty(
            plan.total,
            dtype=self.dtype,
            pin_memory=self.device.type == "cuda",
        )
        # plan.keys -> ['weight_keys1', 'weight_keys2']
        for weight_key in plan.keys:
            mod = weights.modules.get(weight_key)
            a_off, (a_r, a_in) = plan.a_off[weight_key], plan.a_shape[weight_key]
            a_numel = a_r * a_in
            b_off, (b_out, b_r) = plan.b_off[weight_key], plan.b_shape[weight_key]
            b_numel = b_out * b_r
            if mod is None:
                flat[a_off : a_off + a_numel].zero_()
                flat[b_off : b_off + b_numel].zero_()
                continue
            flat[a_off : a_off + a_numel].copy_(mod[0].reshape(-1).to(self.dtype))
            flat[b_off : b_off + b_numel].copy_(mod[1].reshape(-1).to(self.dtype))
        # 同时持有 weights 的强引用,防止 id 被复用后指到别的对象
        self._banks[id(weights)] = (weights, plan.rank, flat)
        if self.release_modules:
            weights.modules = {}
            weights.flattened = True
        return flat

    def _ensure_stage(self, total: int) -> torch.Tensor:
        """GPU 聚合暂存,返回 [chunk, total] 的连续视图;跨 rank 复用一块扁平buffer.

        扁平分配是为了让不同 rank 的行宽不同也各自连续:H2D 目标连续才走
        cudaMemcpyAsync,非连续会退化成搬运 kernel。
        """
        need = _BANK_CHUNK * total
        st = self._stage
        if st is None or st.numel() < need:
            self._stage = torch.empty(need, device=self.device, dtype=self.dtype)
        return self._stage[:need].view(_BANK_CHUNK, total)

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
        lora_names -> 本次的lora权重合集
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
        # pairs -> [slot, weights<weight_key, struct<rank, scale, A, B>>]
        self.buffers.write_many(pairs)
        # 计算流排在异步搬运之后:
        #! 必须等待权重全部都读完
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
