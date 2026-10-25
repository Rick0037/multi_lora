"""多 LoRA 调度器:把 registry / CPU 权重 / GPU 槽位池 / kernel 后端串起来."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Mapping, Optional, Sequence

import torch
from peft import PeftModel

from multi_lora.backend.base import LoRABackend
from multi_lora.backend.registry import get_lora_backend
from multi_lora.batch_info import (
    BatchLoRAInfo,
    KernelInputMetaData,
    to_kernel_input_metadata,
)
from multi_lora.batching import (
    forward_logits_grouped,
    load_peft_multi_adapter,
    validate_lora_batch,
)
from multi_lora.layers import LinearWithLoRA
from multi_lora.model_utils import (
    ATTENTION_LORA_SUFFIXES,
    convert_lora_layers,
    get_inference_model,
    has_lora_layers,
)
from multi_lora.pool import LoRAMemoryPool
from multi_lora.registry import LoRARef, LoRARegistry, LoraName
from multi_lora.weights import AdapterWeights, extract_adapter_weights


class MultiLoRAManager:
    """多 LoRA 调度器,四块分工:

    - registry       元数据(name -> path)
    - _weight_store  CPU 上的 adapter 权重
    - pool           GPU 槽位,只放本 batch 用得到的 adapter
    - _lora_backend  A/B 两段 segment GEMM(torch 或 triton)

    一次 forward 的流程:
        1. prepare_lora_batch:把用到的 adapter 装进 GPU 槽位(槽满就换出);
        2. build_kernel_input_metadata:把槽位信息转成 kernel 要的张量;
        3. 模型 forward:每个 LinearWithLoRA 调 backend 算 基座输出 + LoRA delta。
    """

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
        """建 pool / backend,并把模型剥到真正执行 forward 的那一层."""
        self.model = model
        self.max_loras_per_batch = max_loras_per_batch
        self.lora_target_suffixes = lora_target_suffixes
        self.registry = registry or LoRARegistry()

        device = next(model.parameters()).device
        self.pool = LoRAMemoryPool(
            max_loras_per_batch=max_loras_per_batch,
            max_lora_rank=max_lora_rank,
            device=device,
            dtype=next(model.parameters()).dtype,
            eviction_policy=eviction_policy,
        )

        self._weight_store: Dict[str, AdapterWeights] = {}
        self._lora_backend: LoRABackend = get_lora_backend(lora_backend)
        self.inference_model = get_inference_model(model)
        # * 针对base model 可能会被lora替换的记录下来的wrappers, 是linear的引用
        self.lora_wrappers: Dict[str, LinearWithLoRA] = {}
        # 是否还有没被替换掉的 PEFT LoraLayer(需要 disable,否则会重复算 LoRA)
        self._has_peft_layers = has_lora_layers(self.model)

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
        """加载多 adapter 的 PEFT 模型,注册进 manager,并按需替换 LoraLayer."""
        # * create peft model from base model and add adapter model
        peft_model = load_peft_multi_adapter(base_model, adapters)
        mgr = cls(
            peft_model,
            max_loras_per_batch=max_loras_per_batch,
            max_lora_rank=max_lora_rank,
            eviction_policy=eviction_policy,
            lora_backend=lora_backend,
            lora_target_suffixes=lora_target_suffixes,
        )  #! 创建一大堆类内的成员变量
        #! a,b 先登记
        for name, path in adapters.items():
            mgr.register_adapter(LoRARef(lora_name=name, lora_path=path))
        #! 一起进行替换, 只做一次, 替换一个并集
        if convert_lora_layers:
            mgr.convert_to_lora_layers()
        return mgr

    def register_adapter(self, ref: LoRARef) -> None:
        """登记元数据、抽取 CPU 权重,并为新 module 分配 GPU buffer."""
        #! registry only lora model
        self.registry.register(ref)
        #! load adapter weight to cpu
        weights = extract_adapter_weights(self.model, ref.lora_name)
        #! 此时有两份, 一份在peft model上面, 一个在weight_store中
        self._weight_store[ref.lora_name] = weights
        #! 针对多个adapter的 每一个lora weight 都创建了一块空白的pool, max batch的pool
        self.pool.ensure_buffers(weights)

    def convert_to_lora_layers(self) -> None:
        """把目标 LoraLayer 换成 LinearWithLoRA(走 backend 的快路径)."""
        module_keys: Set[str] = set()

        #! _weight_store 中全部替换也没事, 数值上是等价的, 后续证明
        for weights in self._weight_store.values():
            # print(weights)
            for key in weights.modules:
                module_keys.add(key)

        # 必须和 extract_adapter_weights 用同一个模型根,否则模块名前缀对不上
        self.lora_wrappers = convert_lora_layers(
            self.model,
            module_keys,
            lora_target_suffixes=self.lora_target_suffixes,
        )
        self._has_peft_layers = has_lora_layers(self.inference_model)
        #! 可能在模型包括了mlp 但是后缀是qkv的情况, 出现了true
        print("_has_peft_layers", self._has_peft_layers)

    # ---------- 每个 forward step 的准备 ----------

    def prepare_lora_batch(self, lora_names: Sequence[LoraName]) -> BatchLoRAInfo:
        """把本 batch 用到的 adapter 装进 GPU pool,返回槽位信息."""
        if not validate_lora_batch(
            lora_names, max_loras_per_batch=self.max_loras_per_batch
        ):
            raise ValueError(
                f"invalid lora batch: distinct={len(set(lora_names))} "
                f"max={self.max_loras_per_batch}"
            )

        # 自动加set了
        self.pool.prepare_lora_batch(
            set(lora_names),  # 本 batch 需要的 adapter(含 None = 基座)
            weight_store=self._weight_store,
        )
        return self.pool.build_batch_info(list(lora_names))

    def build_kernel_input_metadata(
        self, lora_names: Sequence[LoraName], *, seq_lens: Optional[torch.Tensor] = None
    ) -> KernelInputMetaData:
        """prepare_lora_batch + 转成 kernel 读取的 GPU 张量."""
        batch_info = self.prepare_lora_batch(lora_names)
        return to_kernel_input_metadata(
            batch_info,
            device=self.pool.device,
            dtype=self.pool.dtype,
            seq_lens=seq_lens,
        )

    def _setup_forward_batch(
        self,
        lora_names: Sequence[LoraName],
        *,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """forward 前:给 backend 设置 metadata,给每个 LinearWithLoRA 注入 buffer."""
        metadata = self.build_kernel_input_metadata(lora_names, seq_lens=seq_lens)
        self._lora_backend.set_kernel_input_metadata(metadata)

        #! 所有的
        for module_key, wrapper in self.lora_wrappers.items():
            # 'model.layers.0.self_attn.q_proj
            # LinearWithLoRA(
            # (base_layer): Linear(in_features=2560, out_features=4096, bias=False)
            # )
            #! [key][max_pre_batch, x, x], 这里在基座模型的索引下面是藏数据
            a_buf, b_buf = self.pool.get_module_buffers(module_key)
            #! b_buf.shape -> [max_pre_batch, x, x]
            wrapper.set_lora_info(self._lora_backend, a_buf, b_buf)

    # ---------- 前向 ----------

    @torch.inference_mode()
    def forward_with_lora(
        self,
        *,
        lora_names: Sequence[LoraName],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """整模型 forward:先准备 batch,再让各 LinearWithLoRA 里算 LoRA delta.

        torch 和 triton 两个后端都走这个入口,具体由 lora_backend 决定。
        """
        # * (1) 看还有多少的序列
        if seq_lens is None and attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1)

        # * (2) 准备 batch:装槽位 + 生成 kernel 元数据 + 把 buffer 注入每个 wrapper(linear)
        # lora_names 本次进来的 names ['a', None, 'b']
        self._setup_forward_batch(lora_names, seq_lens=seq_lens)

        # 没被替换的 PEFT LoraLayer 要关掉,避免和 LinearWithLoRA 重复计算
        # * (3) 关掉没被替换的 PEFT 层(避免算两遍)
        #! 如果微调了包括MLP，启动suffix 需要是全部的linear，不然就有错误
        if self._has_peft_layers:
            context = self.model.disable_adapter()
        else:
            context = nullcontext()

        # * (4) forward
        with context:
            if x is not None:
                return self.inference_model(x, **kwargs)
            if input_ids is None:
                raise ValueError("forward_with_lora requires input_ids or x")
            return self.inference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

    def forward_logits_peft(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        lora_names: Sequence[LoraName],
    ) -> tuple[torch.Tensor, BatchLoRAInfo]:
        """PEFT 分组 forward(golden 参考路径,不走 kernel)."""
        batch_info = self.prepare_lora_batch(lora_names)
        logits = forward_logits_grouped(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            lora_names=list(lora_names),
        )
        return logits, batch_info


__all__ = ["MultiLoRAManager", "ATTENTION_LORA_SUFFIXES"]
