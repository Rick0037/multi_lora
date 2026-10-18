"""Multi-LoRA 编排：Registry + CPU 权重库 + GPU Pool + Triton 整模型 forward."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Mapping, Optional, Sequence, Set, Union

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer

from multi_lora.batching import (
    LoraKey,
    forward_logits_grouped,
    forward_logits_rowwise_reference,
    load_peft_multi_adapter,
    validate_lora_batch,
)
from multi_lora.backend.base import LoRABackend
from multi_lora.backend.registry import get_lora_backend
from multi_lora.batch_info import KernelInputMetaData, kernel_input_metadata_from_pool_batch
from multi_lora.layers import LinearWithLoRA
from multi_lora.model_utils import (
    ATTENTION_LORA_SUFFIXES,
    get_inference_model,
    module_key_matches,
    replace_submodule,
)
from multi_lora.pool import BatchLoRAInfo, LoRAMemoryPool, Uid
from multi_lora.registry import LoRARef, LoRARegistry
from multi_lora.weights import AdapterWeights, extract_adapter_weights


class MultiLoRAManager:
    """Multi-LoRA 编排：注册表、CPU 权重库、GPU 池、整模型 Triton LoRA 层."""

    def __init__(
        self,
        model: PeftModel,
        *,
        max_loras_per_batch: int = 4,
        max_lora_rank: int = 64,
        eviction_policy: str = "lru",
        registry: Optional[LoRARegistry] = None,
        lora_backend: str = "torch",
        lora_target_suffixes: Optional[Sequence[str]] = None,
        convert_lora_layers: bool = True,
    ) -> None:
        """构造 manager：初始化 pool、backend，可选将 LoraLayer 替换为 LinearWithLoRA."""
        self.model = model
        self.max_loras_per_batch = max_loras_per_batch
        self.max_lora_rank = max_lora_rank
        self.lora_target_suffixes = lora_target_suffixes
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

        # lora_id -> CPU 权重；None key 为基座占位
        self._weight_store: Dict[Optional[str], AdapterWeights] = {}
        self._id_to_peft_name: Dict[str, str] = {}
        self._peft_name_to_id: Dict[str, str] = {}

        self._lora_backend: LoRABackend = get_lora_backend(lora_backend)
        self.inference_model = get_inference_model(model)
        self.lora_wrappers: Dict[str, LinearWithLoRA] = {}

        if convert_lora_layers:
            self.convert_to_lora_layers()

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
        lora_backend: str = "torch",
        lora_target_suffixes: Optional[Sequence[str]] = None,
        convert_lora_layers: bool = True,
    ) -> "MultiLoRAManager":
        """加载多 adapter PEFT 模型并注册到 manager."""
        names = list(adapters.keys())
        primary = first or names[0]
        peft_model = load_peft_multi_adapter(base_model, adapters, first=primary)
        mgr = cls(
            peft_model,
            max_loras_per_batch=max_loras_per_batch,
            max_lora_rank=max_lora_rank,
            eviction_policy=eviction_policy,
            lora_backend=lora_backend,
            lora_target_suffixes=lora_target_suffixes,
            convert_lora_layers=False,
        )
        for peft_name, path in adapters.items():
            ref = LoRARef(
                lora_name=peft_name,
                lora_path=path,
            )
            mgr.register_adapter(ref, peft_adapter_name=peft_name)
        if convert_lora_layers:
            mgr.convert_to_lora_layers()
        return mgr

    def register_adapter(self, ref: LoRARef, *, peft_adapter_name: str) -> None:
        """注册 adapter 元数据，并从 PEFT 抽取 CPU 权重写入 weight_store."""
        self.registry.register(ref)
        weights = extract_adapter_weights(self.model, peft_adapter_name)
        self._weight_store[ref.lora_id] = weights
        self._id_to_peft_name[ref.lora_id] = peft_adapter_name
        self._peft_name_to_id[peft_adapter_name] = ref.lora_id
        self.max_lora_rank = max(self.max_lora_rank, weights.rank)
        self.pool.finalize_buffers(self._weight_store)

    def convert_to_lora_layers(self) -> None:
        """将目标 LoraLayer 替换为 LinearWithLoRA（Triton 热路径）."""
        known_keys: Set[str] = set()
        for aw in self._weight_store.values():
            known_keys.update(aw.modules.keys())

        self.lora_wrappers.clear()
        for full_name, mod in list(self.inference_model.named_modules()):
            if not isinstance(mod, LoraLayer):
                continue
            module_key = full_name
            if module_key not in known_keys:
                continue
            if not module_key_matches(module_key, target_suffixes=self.lora_target_suffixes):
                continue
            base = mod.get_base_layer() if hasattr(mod, "get_base_layer") else mod.base_layer
            wrapper = LinearWithLoRA(base, module_key)
            replace_submodule(self.inference_model, full_name, wrapper)
            self.lora_wrappers[module_key] = wrapper

    def validate_lora_batch(
        self,
        lora_ids: Sequence[LoraKey],
        *,
        num_pinned_in_pool: int = 0,
    ) -> bool:
        """校验 distinct adapter 数与 pinned 槽位是否允许本 batch 准入."""
        if not validate_lora_batch(lora_ids, max_loras_per_batch=self.max_loras_per_batch):
            return False
        pinned_in_batch = 0
        for uid in set(lora_ids):
            if uid is None:
                continue
            ref = self.registry.get_by_id(uid)
            pinned_in_batch += int(ref.pinned)
        required = len(set(lora_ids)) - pinned_in_batch
        vacancy = self.max_loras_per_batch - num_pinned_in_pool
        return required <= vacancy

    def prepare_lora_batch(self, lora_ids: Sequence[LoraKey]) -> BatchLoRAInfo:
        """把本 batch 涉及的 uid 装入 GPU pool，返回 per-row slot 元数据."""
        if not self.validate_lora_batch(lora_ids):
            raise ValueError(
                f"invalid lora batch: distinct={len(set(lora_ids))} "
                f"max={self.max_loras_per_batch}"
            )

        cur_uids: Set[Uid] = set(lora_ids)
        # 基座 slot 占位，使 None uid 与 LoRA 共用统一 slot 语义
        if None not in self._weight_store:
            self._weight_store[None] = AdapterWeights(rank=0, scaling=0.0, modules={})

        refs = {ref.lora_id: ref for ref in self.registry.all_refs()}
        self.pool.prepare_lora_batch(
            cur_uids,
            weight_store=self._weight_store,
            lora_refs=refs,
        )
        return self.pool.build_batch_info(list(lora_ids))

    def build_kernel_input_metadata(
        self, lora_ids: Sequence[LoraKey], *, seq_lens: Optional[torch.Tensor] = None
    ) -> tuple[KernelInputMetaData, BatchLoRAInfo]:
        """prepare_lora_batch + 将 CPU list 元数据转为 GPU KernelInputMetaData."""
        pool_info = self.prepare_lora_batch(lora_ids)
        meta = kernel_input_metadata_from_pool_batch(
            pool_info,
            batch_size=len(lora_ids),
            device=self.pool.device,
            dtype=self.pool.dtype,
            seq_lens=seq_lens,
        )
        return meta, pool_info

    def _setup_forward_batch(
        self,
        lora_ids: Sequence[LoraKey],
        *,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> tuple[KernelInputMetaData, BatchLoRAInfo]:
        """每个 forward step 前：注入 backend 元数据 + 各 LinearWithLoRA 的 A/B buffer."""
        meta, pool_info = self.build_kernel_input_metadata(lora_ids, seq_lens=seq_lens)
        self._lora_backend.set_kernel_input_metadata(meta)
        for module_key, wrapper in self.lora_wrappers.items():
            a_buf, b_buf = self.pool.get_module_buffers(module_key)
            wrapper.set_lora_info(self._lora_backend, a_buf, b_buf)
        return meta, pool_info

    @torch.inference_mode()
    def forward_with_triton_lora(
        self,
        *,
        lora_ids: Sequence[LoraKey],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """整模型 forward：batch 级 prepare + 各 LinearWithLoRA 内 segment GEMM."""
        if seq_lens is None and attention_mask is not None:
            # prefill：每条序列有效 token 数 = mask 求和
            seq_lens = attention_mask.sum(dim=1)

        self._setup_forward_batch(lora_ids, seq_lens=seq_lens)

        # 未 convert 的 PEFT LoraLayer 需 disable，避免与 LinearWithLoRA 双算
        disable_ctx = (
            self.model.disable_adapter()
            if self._has_peft_lora_layers()
            else nullcontext()
        )
        with disable_ctx:
            if x is not None:
                return self.inference_model(x, **kwargs)
            if input_ids is None:
                raise ValueError("forward_with_triton_lora requires input_ids or x")
            return self.inference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

    def _has_peft_lora_layers(self) -> bool:
        """检查模型中是否仍存在未替换的 PEFT LoraLayer."""
        for _, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                return True
        return False

    def peft_name_for_row(self, lora_id: LoraKey) -> LoraKey:
        """将内部 lora_id 映射回 PEFT adapter 名称（供 grouped forward 使用）."""
        if lora_id is None:
            return None
        return self._id_to_peft_name[lora_id]

    def resolve_request_lora_ids(
        self, names: Union[str, List[Optional[str]], None]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """解析请求中的 adapter 名并 acquire 引用计数，返回 lora_id."""
        return self.registry.acquire(names)

    def forward_logits_grouped(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        lora_ids: Sequence[LoraKey],
    ) -> tuple[torch.Tensor, BatchLoRAInfo]:
        """PEFT 分组 forward（golden 参考路径，不走 Triton kernel）."""
        batch_info = self.prepare_lora_batch(lora_ids)
        peft_keys = [self.peft_name_for_row(uid) for uid in lora_ids]
        logits = forward_logits_grouped(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            lora_ids=peft_keys,
        )
        return logits, batch_info


__all__ = ["MultiLoRAManager", "ATTENTION_LORA_SUFFIXES"]
