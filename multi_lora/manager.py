"""Multi-LoRA 编排;Registry + CPU 权重缓存 + GPU 槽位池 + Triton/Torch backend."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Mapping, Optional, Sequence, Set

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer

from multi_lora.backend.base import LoRABackend
from multi_lora.backend.registry import get_lora_backend
from multi_lora.batch_info import (
    KernelInputMetaData,
    kernel_input_metadata_from_pool_batch,
)
from multi_lora.batching import (
    LoraKey,
    forward_logits_grouped,
    load_peft_multi_adapter,
    validate_lora_batch,
)
from multi_lora.layers import LinearWithLoRA
from multi_lora.model_utils import (
    ATTENTION_LORA_SUFFIXES,
    get_inference_model,
    module_key_matches,
    replace_submodule,
)
from multi_lora.pool import BatchLoRAInfo, LoRAMemoryPool, LoraName
from multi_lora.registry import LoRARef, LoRARegistry
from multi_lora.weights import (
    AdapterWeights,
    collect_module_shapes,
    extract_adapter_weights,
    normalize_module_key,
)


class MultiLoRAManager:
    """Multi-LoRA 调度器;registry(元数据) + _weight_store(CPU) + pool(GPU) + backend(kernel)."""

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

        self._weight_store: Dict[str, AdapterWeights] = {}
        self._lora_backend: LoRABackend = get_lora_backend(lora_backend)
        self.inference_model = get_inference_model(model)
        self.lora_wrappers: Dict[str, LinearWithLoRA] = {}
        self._has_peft_layers = True

    @classmethod
    def from_pretrained(
        cls,
        base_model: torch.nn.Module,
        adapters: Mapping[str, str],
        *,
        max_loras_per_batch: int = 4,
        max_lora_rank: int = 64,
        eviction_policy: str = "lru",
        lora_backend: str = "torch",
        lora_target_suffixes: Optional[Sequence[str]] = None,
        convert_lora_layers: bool = True,
    ) -> "MultiLoRAManager":
        """加载多 adapter PEFT 模型并注册到 manager."""
        peft_model = load_peft_multi_adapter(base_model, adapters)
        mgr = cls(
            peft_model,
            max_loras_per_batch=max_loras_per_batch,
            max_lora_rank=max_lora_rank,
            eviction_policy=eviction_policy,
            lora_backend=lora_backend,
            lora_target_suffixes=lora_target_suffixes,  # *("q_proj", "k_proj", "v_proj", "o_proj")
        )
        for name, path in adapters.items():
            mgr.register_adapter(LoRARef(lora_name=name, lora_path=path))
        if convert_lora_layers:
            mgr.convert_to_lora_layers()  # * peft.lora -> linear with lora
        return mgr

    def register_adapter(self, ref: LoRARef) -> None:
        """注册 adapter 元数据,并从 PEFT 抽取 CPU 权重写入 weight_store."""
        self.registry.register(ref)
        weights = extract_adapter_weights(self.model, ref.lora_name)
        self._weight_store[ref.lora_name] = weights
        shapes = collect_module_shapes(self._weight_store)
        self.pool.finalize_buffers(shapes)

    def convert_to_lora_layers(self) -> None:
        """将目标 LoraLayer 替换为 LinearWithLoRA(Triton 热路径)."""
        known_keys: Set[str] = set()
        for aw in self._weight_store.values():
            known_keys.update(aw.modules.keys())

        self.lora_wrappers.clear()
        for full_name, mod in list(self.inference_model.named_modules()):
            if not isinstance(mod, LoraLayer):
                continue
            module_key = normalize_module_key(full_name)

            if module_key not in known_keys:
                continue
            if not module_key_matches(
                module_key, target_suffixes=self.lora_target_suffixes
            ):
                continue

            base = (
                mod.get_base_layer()
                if hasattr(mod, "get_base_layer")
                else mod.base_layer
            )
            wrapper = LinearWithLoRA(base)
            replace_submodule(self.inference_model, full_name, wrapper)
            self.lora_wrappers[module_key] = wrapper

        self._has_peft_layers = any(
            isinstance(m, LoraLayer) for m in self.inference_model.modules()
        )

    def prepare_lora_batch(self, lora_ids: Sequence[LoraKey]) -> BatchLoRAInfo:
        """把本 batch 涉及的 adapter 装入 GPU pool,返回 per-row slot 元数据."""
        if not validate_lora_batch(
            lora_ids, max_loras_per_batch=self.max_loras_per_batch
        ):
            raise ValueError(
                f"invalid lora batch: distinct={len(set(lora_ids))} "
                f"max={self.max_loras_per_batch}"
            )

        cur_names: Set[LoraName] = set(lora_ids)

        # 把本 batch 需要的权重装进 GPU 槽位;槽满时由 pool 驱逐旧 adapter
        self.pool.prepare_lora_batch(
            cur_names,  # * 当前请求的 adapter 名集合,含 None(基座)
            weight_store=self._weight_store,  # * CPU 上的全部 adapter 权重
        )
        # 为 batch 内每一行给出 weight_indices,以及各槽的 lora_ranks / scalings
        return self.pool.build_batch_info(list(lora_ids))

    def build_kernel_input_metadata(
        self, lora_ids: Sequence[LoraKey], *, seq_lens: Optional[torch.Tensor] = None
    ) -> KernelInputMetaData:
        """prepare_lora_batch 并转成 GPU 版 KernelInputMetaData."""
        # * pool_info = BatchLoRAInfo(weight_indices=[2, 0, 1], lora_ranks=[0, 64, 16],
        # * scalings=[0.0, 2.0, 2.0], evictions=0)
        pool_info = self.prepare_lora_batch(lora_ids)
        return kernel_input_metadata_from_pool_batch(
            pool_info,
            device=self.pool.device,
            dtype=self.pool.dtype,
            seq_lens=seq_lens,
        )

    def _setup_forward_batch(
        self,
        lora_ids: Sequence[LoraKey],
        *,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """每个 forward step 前;
        注入 backend 元数据 + 各 LinearWithLoRA 的 A/B buffer."""
        meta = self.build_kernel_input_metadata(lora_ids, seq_lens=seq_lens)
        self._lora_backend.set_kernel_input_metadata(meta)
        for module_key, wrapper in self.lora_wrappers.items():
            # a_buf [K, max_rank, in]、b_buf [K, out, max_rank];K = max_loras_per_batch(槽位总数)
            a_buf, b_buf = self.pool.get_module_buffers(module_key)
            wrapper.set_lora_info(self._lora_backend, a_buf, b_buf)

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
            self.model.disable_adapter() if self._has_peft_layers else nullcontext()
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

    def forward_logits_grouped(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        lora_ids: Sequence[LoraKey],
    ) -> tuple[torch.Tensor, BatchLoRAInfo]:
        """PEFT 分组 forward(golden 参考路径,不走 Triton kernel)."""
        batch_info = self.prepare_lora_batch(lora_ids)
        logits = forward_logits_grouped(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            lora_ids=list(lora_ids),
        )
        return logits, batch_info


__all__ = ["MultiLoRAManager", "ATTENTION_LORA_SUFFIXES"]
