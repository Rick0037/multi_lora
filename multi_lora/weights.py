"""从 PEFT 模型抽取 LoRA 权重;供 CPU 缓存与 GPU pool 装载."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer

ModuleWeights = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class AdapterWeights:
    """单个 adapter 的全部 LoRA 权重(CPU);A [r, in],B [out, r]."""

    rank: int
    scaling: float
    modules: ModuleWeights


def normalize_module_key(name: str) -> str:
    """归一化模块名;剥掉 PEFT 前缀,与 pool buffer key 对齐."""
    for prefix in ("base_model.model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def extract_adapter_weights(model: PeftModel, adapter_name: str) -> AdapterWeights:
    """遍历 LoraLayer,抽取指定 adapter 的 lora_A / lora_B;必须先 set_adapter."""
    model.set_adapter(adapter_name)
    modules: ModuleWeights = {}
    rank: Optional[int] = None
    scaling: Optional[float] = None

    for full_name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if adapter_name not in getattr(module, "lora_A", {}):
            continue
        lora_a_layer = module.lora_A[adapter_name]
        lora_b_layer = module.lora_B[adapter_name]
        a = lora_a_layer.weight.detach().cpu().clone()
        b = lora_b_layer.weight.detach().cpu().clone()
        raw_r = (
            module.r[adapter_name]
            if isinstance(getattr(module, "r", None), dict)
            else module.r
        )
        fan_in = int(lora_a_layer.in_features)
        fan_out = int(lora_b_layer.out_features)

        if a.shape == (fan_in, raw_r):
            a = a.T
        if a.shape != (raw_r, fan_in):
            raise ValueError(
                f"lora_A shape {tuple(a.shape)} != ({raw_r}, {fan_in}) for {full_name}"
            )
        if b.shape == (raw_r, fan_out):
            b = b.T
        if b.shape != (fan_out, raw_r):
            raise ValueError(
                f"lora_B shape {tuple(b.shape)} != ({fan_out}, {raw_r}) for {full_name}"
            )
        key = normalize_module_key(full_name)
        modules[key] = (a, b)
        r = int(raw_r)
        if rank is None:
            rank = r
            alpha = getattr(module, "lora_alpha", r)
            if isinstance(alpha, dict):
                alpha = alpha.get(adapter_name, r)
            scaling = float(alpha) / float(r)
        elif rank != r:
            raise ValueError(f"rank mismatch in adapter {adapter_name}: {rank} vs {r}")

    if not modules:
        raise ValueError(f"no LoRA weights found for adapter {adapter_name}")
    assert rank is not None and scaling is not None
    return AdapterWeights(rank=rank, scaling=scaling, modules=modules)


def collect_module_shapes(
    weight_store: Dict[Optional[str], AdapterWeights],
) -> Dict[str, Tuple[int, int]]:
    """遍历 weight_store,收集全部 module 的 (in_dim, out_dim);用于 pool.finalize_buffers."""
    shapes: Dict[str, Tuple[int, int]] = {}
    for aw in weight_store.values():
        for key, (a_cpu, b_cpu) in aw.modules.items():
            in_dim, out_dim = int(a_cpu.shape[1]), int(b_cpu.shape[0])
            if key in shapes:
                old_in, old_out = shapes[key]
                if (old_in, old_out) != (in_dim, out_dim):
                    raise ValueError(
                        f"shape mismatch for {key!r}: {shapes[key]} vs {(in_dim, out_dim)}"
                    )
            else:
                shapes[key] = (in_dim, out_dim)
    return shapes
