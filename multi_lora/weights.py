"""从 PEFT 模型里把 LoRA 权重抽出来,存成 CPU 上的 AdapterWeights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer

# module_key -> (A [rank, in], B [out, rank])
ModuleWeights = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class AdapterWeights:
    """单个 adapter 的全部 LoRA 权重(在 CPU 上)."""

    rank: int
    scaling: float
    modules: ModuleWeights


def normalize_module_key(name: str) -> str:
    """去掉 PEFT 加的前缀,让模块名和 pool 的 buffer key 对得上."""
    for prefix in ("base_model.model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def extract_adapter_weights(model: PeftModel, adapter_name: str) -> AdapterWeights:
    """遍历模型里的 LoraLayer,取出指定 adapter 的 A/B(会切换当前 adapter)."""
    model.set_adapter(adapter_name)

    modules: ModuleWeights = {}
    rank: Optional[int] = None
    scaling: Optional[float] = None

    for full_name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if adapter_name not in getattr(module, "lora_A", {}):
            continue

        # PEFT 里 A 是 [rank, in],B 是 [out, rank]
        a = module.lora_A[adapter_name].weight.detach().cpu().clone()
        b = module.lora_B[adapter_name].weight.detach().cpu().clone()
        r = int(module.r[adapter_name])

        if rank is None:
            rank = r
            scaling = float(module.lora_alpha[adapter_name]) / r
        elif rank != r:
            raise ValueError(f"rank mismatch in adapter {adapter_name}: {rank} vs {r}")

        modules[normalize_module_key(full_name)] = (a, b)

    if not modules:
        raise ValueError(f"no LoRA weights found for adapter {adapter_name}")
    assert rank is not None and scaling is not None
    return AdapterWeights(rank=rank, scaling=scaling, modules=modules)
