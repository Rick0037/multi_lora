"""Multi-LoRA 编排;Registry + CPU 权重库 + GPU Pool + Triton 整模型 forward."""

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
from multi_lora.batch_info import (
    KernelInputMetaData,
    kernel_input_metadata_from_pool_batch,
)
from multi_lora.layers import LinearWithLoRA
from multi_lora.model_utils import (
    ATTENTION_LORA_SUFFIXES,
    get_inference_model,
    module_key_matches,
    replace_submodule,
    normalize_module_key,
)
from multi_lora.pool import BatchLoRAInfo, LoRAMemoryPool, Uid
from multi_lora.registry import LoRARef, LoRARegistry
from multi_lora.weights import AdapterWeights, extract_adapter_weights


class MultiLoRAManager:
    """Multi-LoRA 编排;注册表、CPU 权重库、GPU 池、整模型 Triton LoRA 层."""

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
    ) -> None:
        """构造 manager;初始化 pool、backend,可选将 LoraLayer 替换为 LinearWithLoRA."""
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

        # lora_id -> CPU 权重;None key 为基座占位;这是 pool H2D 拷贝的源
        self._weight_store: Dict[Optional[str], AdapterWeights] = {}
        # lora_id <-> PEFT adapter 名;batching 层仍用 str adapter 名调用 set_adapter
        self._id_to_peft_name: Dict[str, str] = {}
        self._peft_name_to_id: Dict[str, str] = {}

        self._lora_backend: LoRABackend = get_lora_backend(lora_backend)
        # * 纯洁的model, 没有peft的model 封装,
        self.inference_model = get_inference_model(model)
        self.lora_wrappers: Dict[str, LinearWithLoRA] = {}

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
            lora_target_suffixes=lora_target_suffixes,  # *("q_proj", "k_proj", "v_proj", "o_proj")
        )
        for peft_name, path in adapters.items():
            ref = LoRARef(
                lora_name=peft_name,
                lora_path=path,
            )
            mgr.register_adapter(ref, peft_adapter_name=peft_name)
        if convert_lora_layers:
            mgr.convert_to_lora_layers()  # * peft.lora -> linear with lora
        return mgr

    def register_adapter(self, ref: LoRARef, *, peft_adapter_name: str) -> None:
        """注册 adapter 元数据,并从 PEFT 抽取 CPU 权重写入 weight_store."""
        self.registry.register(ref)
        # * 抽取的是单个adapter的权重
        weights = extract_adapter_weights(self.model, peft_adapter_name)
        self._weight_store[ref.lora_id] = weights
        self._id_to_peft_name[ref.lora_id] = peft_adapter_name
        self._peft_name_to_id[peft_adapter_name] = ref.lora_id
        #! max_lora_rank如果更新但是 pool的max 没有更新啊
        self.max_lora_rank = max(self.max_lora_rank, weights.rank)
        self.pool.finalize_buffers(self._weight_store)

    def convert_to_lora_layers(self) -> None:
        """将目标 LoraLayer 替换为 LinearWithLoRA(Triton 热路径)."""
        known_keys: Set[str] = set()
        for aw in self._weight_store.values():
            known_keys.update(aw.modules.keys())
        # * 'layers.31.mlp.up_proj', 'layers.20.mlp.gate_proj',
        # *'layers.19.self_attn.k_proj', 'layers.18.mlp.up_proj',
        self.lora_wrappers.clear()
        for full_name, mod in list(self.inference_model.named_modules()):
            # * full_name = 'model.layers.0.self_attn.q_proj'
            if not isinstance(mod, LoraLayer):
                continue
            module_key = normalize_module_key(full_name)
            # * module_key 'layers.0.self_attn.q_proj'

            if module_key not in known_keys:
                continue
            # * 只要对应的lora_target_suffixes
            if not module_key_matches(
                module_key, target_suffixes=self.lora_target_suffixes
            ):
                continue
            #! 由于前缀问题 只有qkv o 没有别的
            base = (
                mod.get_base_layer()
                if hasattr(mod, "get_base_layer")
                else mod.base_layer
            )
            # import pdb

            # pdb.set_trace()
            # * base = Linear(in_features=2560, out_features=4096, bias=False)
            # * inference_model 'layers.0.self_attn.q_proj'
            wrapper = LinearWithLoRA(base, module_key)
            #! 使用新的linear 替换老linear
            replace_submodule(self.inference_model, full_name, wrapper)
            # * wrapper 和lora wappers中共享一个内存
            self.lora_wrappers[module_key] = wrapper

    def validate_lora_batch(self, lora_ids: Sequence[LoraKey]) -> bool:
        """校验本 batch 的 distinct adapter 数不超过槽位上限."""
        # distinct 表示不同的 lora adapter
        return validate_lora_batch(
            lora_ids, max_loras_per_batch=self.max_loras_per_batch
        )

    def prepare_lora_batch(self, lora_ids: Sequence[LoraKey]) -> BatchLoRAInfo:
        """把本 batch 涉及的 uid 装入 GPU pool,返回 per-row slot 元数据."""
        if not self.validate_lora_batch(lora_ids):
            raise ValueError(
                f"invalid lora batch: distinct={len(set(lora_ids))} "
                f"max={self.max_loras_per_batch}"
            )

        cur_uids: Set[Uid] = set(lora_ids)

        # 把本 batch 需要的权重装进 GPU 槽位;槽满时由 pool 驱逐旧 adapter
        self.pool.prepare_lora_batch(
            cur_uids,  # * 当前请求的 uid 集合,含 None(基座)
            weight_store=self._weight_store,  # * CPU 上的全部 adapter 权重
        )
        # 为 batch 内每一行给出 weight_indices,以及各槽的 lora_ranks / scalings
        return self.pool.build_batch_info(list(lora_ids))

    def build_kernel_input_metadata(
        self, lora_ids: Sequence[LoraKey], *, seq_lens: Optional[torch.Tensor] = None
    ) -> tuple[KernelInputMetaData, BatchLoRAInfo]:
        """prepare_lora_batch + 将 CPU list 元数据转为 GPU KernelInputMetaData."""
        # * pool_info = BatchLoRAInfo(weight_indices=[2, 0, 1], lora_ranks=[0, 64, 16],
        # * scalings=[0.0, 2.0, 2.0], evictions=0)
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
        """每个 forward step 前;
        注入 backend 元数据 + 各 LinearWithLoRA 的 A/B buffer."""
        meta, pool_info = self.build_kernel_input_metadata(lora_ids, seq_lens=seq_lens)
        self._lora_backend.set_kernel_input_metadata(meta)
        for module_key, wrapper in self.lora_wrappers.items():
            # a_buf [K, max_rank, in]、b_buf [K, out, max_rank];K = max_loras_per_batch(槽位总数)
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
        """整模型 forward;batch 级 prepare + 各 LinearWithLoRA 内 segment GEMM."""
        if seq_lens is None and attention_mask is not None:
            # prefill;每条序列有效 token 数 = mask 求和
            seq_lens = attention_mask.sum(dim=1)
        """
        (Pdb) attention_mask
        tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        device='cuda:0')

        (Pdb) seq_lens
        tensor([13, 13, 22], device='cuda:0')
        """
        self._setup_forward_batch(lora_ids, seq_lens=seq_lens)

        # 未 convert 的 PEFT LoraLayer 需 disable,避免与 LinearWithLoRA 双算
        disable_ctx = (
            self.model.disable_adapter()
            if self._has_peft_lora_layers()
            else nullcontext()
        )
        with disable_ctx:
            # * 区分一下到底是处理一个x, 还是处理整个input_ids
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
        """将内部 lora_id 映射回 PEFT adapter 名称(供 grouped forward 使用)."""
        if lora_id is None:
            return None
        return self._id_to_peft_name[lora_id]

    def forward_logits_grouped(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        lora_ids: Sequence[LoraKey],
    ) -> tuple[torch.Tensor, BatchLoRAInfo]:
        """PEFT 分组 forward(golden 参考路径,不走 Triton kernel)."""
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
