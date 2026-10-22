"""Multi-LoRA 编排;Registry + CPU 权重库 + GPU Pool +(可选)PEFT 分组前向.

第三步整体流水线(读代码时建议按此顺序跟);
1) from_pretrained / register_adapter;HF 基座 + 多 PEFT adapter;同时
   extract_adapter_weights 把每套 LoRA 的 A/B 存到 _weight_store[lora_id](CPU)。
2) registry;name <-> lora_id;acquire 用于请求生命周期(本类 forward 路径可不调用)。
3) prepare_lora_batch;对当前 batch 的 distinct lora_id(含 None)调用 pool.prepare_lora_batch,
   把需要的权重 H2D 装进有限个 GPU 槽;返回 BatchLoRAInfo 供后续 kernel 按行选 slot。
4) forward_logits_grouped;整模型 logits 仍走 PEFT(与第二步一致)。
当前为单进程同步 API;未实现调度器侧异步 H2D overlap 与 CUDA Graph。
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Set, Union

import torch
from peft import PeftModel

from multi_lora_phase3.batching import (
    LoraKey,
    forward_logits_grouped,
    forward_logits_rowwise_reference,
    load_peft_multi_adapter,
    validate_lora_batch,
)
from multi_lora_phase3.pool import BatchLoRAInfo, LoRAMemoryPool, Uid
from multi_lora_phase3.registry import LoRARef, LoRARegistry
from multi_lora_phase3.weights import AdapterWeights, extract_adapter_weights


class MultiLoRAManager:
    """Multi-LoRA 编排;注册表、CPU 权重库、GPU 池与 PEFT 分组前向.

    成员分工;
    - self.model;PeftModel,负责正确 logits 的参考前向。
    - self.registry;lora模型的元数据与 name/id。
    - self.pool;GPU 槽位与驱逐;与 _weight_store 配合完成谁该在 GPU 上。
    - _id_to_peft_name;把 stable lora_id 映射回 PEFT 的 adapter_name,供第二步 batching 使用。
    """

    def __init__(
        self,
        model: PeftModel,
        *,
        max_loras_per_batch: int = 4,
        max_lora_rank: int = 64,
        eviction_policy: str = "lru",
        registry: Optional[LoRARegistry] = None,
    ) -> None:
        self.model = model
        self.max_loras_per_batch = max_loras_per_batch
        self.max_lora_rank = max_lora_rank
        self.registry = registry or LoRARegistry()

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.pool = LoRAMemoryPool(
            max_loras_per_batch=max_loras_per_batch,
            max_lora_rank=max_lora_rank,
            device=device,
            dtype=dtype,
            eviction_policy=eviction_policy,
        )

        # lora_id -> CPU 上的 AdapterWeights(第三步 pool 的 H2D 源)。
        self._weight_store: Dict[Optional[str], AdapterWeights] = {}
        # lora_id <-> PEFT adapter 名字;batching 层仍用 str adapter 名调用 set_adapter。
        self._id_to_peft_name: Dict[str, str] = {}
        self._peft_name_to_id: Dict[str, str] = {}

    @classmethod
    def from_pretrained(
        cls,
        base_model: torch.nn.Module,
        adapters: Mapping[str, str],
        *,
        max_loras_per_batch: int = 4,
        max_lora_rank: int = 64,
        eviction_policy: str = "lru",
        first: Optional[str] = None,
    ) -> "MultiLoRAManager":
        """加载基座 + 多个 PEFT adapter,并注册到 manager."""
        names = list(adapters.keys())
        primary = first or names[0]
        peft_model = load_peft_multi_adapter(base_model, adapters, first=primary)
        mgr = cls(
            peft_model,
            max_loras_per_batch=max_loras_per_batch,
            max_lora_rank=max_lora_rank,
            eviction_policy=eviction_policy,
        )
        for peft_name, path in adapters.items():
            ref = LoRARef(
                lora_name=peft_name,
                lora_path=path,
            )
            mgr.register_adapter(ref, peft_adapter_name=peft_name)
        return mgr

    def register_adapter(self, ref: LoRARef, *, peft_adapter_name: str) -> None:
        """登记一个适配器;registry 元数据 + CPU 权重快照 + id 与 PEFT 名互查。"""
        self.registry.register(ref)
        weights = extract_adapter_weights(self.model, peft_adapter_name)
        self._weight_store[ref.lora_id] = weights
        self._id_to_peft_name[ref.lora_id] = peft_adapter_name
        self._peft_name_to_id[peft_adapter_name] = ref.lora_id
        # 运行池分配的 max_lora_rank 至少覆盖已注册 adapter 的最大 rank。
        self.max_lora_rank = max(self.max_lora_rank, weights.rank)

    def validate_lora_batch(
        self,
        lora_ids: Sequence[LoraKey],
        *,
        num_pinned_in_pool: int = 0,
    ) -> bool:
        """校验本 batch 中 distinct adapter 数量不超过槽位上限,并考虑 pinned 占用的不可驱逐槽。"""
        # distinct表示不同的lora adapter,pinned可以理解为钉子户,这一部分永不驱逐
        if not validate_lora_batch(
            lora_ids, max_loras_per_batch=self.max_loras_per_batch
        ):
            return False
        pinned_in_batch = 0
        for uid in set(lora_ids):
            if uid is None:
                continue
            ref = self.registry.get_by_id(uid)
            pinned_in_batch += int(ref.pinned)
        # 非 pinned 槽位数量须能容纳本 batch 中的 distinct adapter
        required = len(set(lora_ids)) - pinned_in_batch
        vacancy = self.max_loras_per_batch - num_pinned_in_pool
        return required <= vacancy

    def prepare_lora_batch(self, lora_ids: Sequence[LoraKey]) -> BatchLoRAInfo:
        """把本 batch 涉及的 uid 装入 GPU pool,返回 per-row slot 元数据.

        会为基座注入 weight_store[None] 占位,使 None uid 与 LoRA 共用统一的槽位映射。
        """
        if not self.validate_lora_batch(lora_ids):
            raise ValueError(
                f"invalid lora batch: distinct={len(set(lora_ids))} "
                f"max={self.max_loras_per_batch}"
            )

        cur_uids: Set[Uid] = set(lora_ids)
        # 基座 slot;无真实 LoRA 张量;pool 仍可为 None 分配 buffer_id,便于 per-row weight_indices 统一语义。
        if None not in self._weight_store:
            self._weight_store[None] = AdapterWeights(rank=0, scaling=0.0, modules={})

        # lora_id -> LoRARef;供 pool 判断 pinned,不参与权重拷贝。
        refs = {ref.lora_id: ref for ref in self.registry.all_refs()}
        self.pool.prepare_lora_batch(
            cur_uids,
            weight_store=self._weight_store,
            lora_refs=refs,
        )
        return self.pool.build_batch_info(list(lora_ids))

    def peft_name_for_row(self, lora_id: LoraKey) -> LoraKey:
        """pool/请求用 lora_id;PEFT forward 仍用 adapter 名."""
        if lora_id is None:
            return None
        return self._id_to_peft_name[lora_id]

    def resolve_request_lora_ids(
        self, names: Union[str, List[Optional[str]], None]
    ) -> Union[Optional[str], List[Optional[str]]]:
        return self.registry.acquire(names)

    def forward_logits_grouped(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        lora_ids: Sequence[LoraKey],
    ) -> tuple[torch.Tensor, BatchLoRAInfo]:
        """先 prepare_lora_batch(GPU 槽位 + BatchLoRAInfo),再走第二步分组 PEFT forward.

        注意;logits 仍由 HF PEFT 计算;batch_info 反映 pool 槽位与驱逐情况。
        """
        batch_info = self.prepare_lora_batch(lora_ids)
        # 将每行的内部 lora_id 转回 PEFT adapter 名;None 保持为基座路径。
        peft_keys = [self.peft_name_for_row(uid) for uid in lora_ids]
        logits = forward_logits_grouped(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            lora_ids=peft_keys,
        )
        return logits, batch_info
