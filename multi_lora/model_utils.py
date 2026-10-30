"""模型结构操作:替换子模块、把 PEFT 的 LoraLayer 换成本项目的 LinearWithLoRA."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch.nn as nn
from peft.tuners.lora import LoraLayer

from multi_lora.layers import LinearWithLoRA
from multi_lora.weights import normalize_module_key

# 默认只给 attention 的 q/k/v/o 挂 LoRA
ATTENTION_LORA_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def get_inference_model(peft_model: nn.Module) -> nn.Module:
    """剥掉 PeftModel / 外层包装,返回真正执行 forward 的 HF 模型."""
    model = peft_model
    if hasattr(model, "base_model"):
        model = model.base_model
    if hasattr(model, "model"):
        model = model.model
    return model


def replace_submodule(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """按点分路径(a.b.c)原地替换子模块."""
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def module_key_matches(
    module_key: str,
    *,
    lora_target_suffixes: Optional[List[str]],
) -> bool:
    """该模块名是否命中目标后缀;不传后缀表示全部命中."""
    if lora_target_suffixes is None:
        return True

    for suffix in lora_target_suffixes:
        if module_key.endswith(suffix):
            return True
    return False


def has_lora_layers(model: nn.Module) -> bool:
    """模型里是否还留着 PEFT 的 LoraLayer(留着就必须 disable,否则会重复算)."""
    for module in model.modules():
        if isinstance(module, LoraLayer):
            return True
    return False


def convert_lora_layers(
    model: nn.Module,
    module_keys: Set[str],
    *,
    lora_target_suffixes: Optional[List[str]] = None,
) -> Dict[str, LinearWithLoRA]:
    """把命中目标的 LoraLayer 原地换成 LinearWithLoRA,返回 module_key -> wrapper."""
    wrappers: Dict[str, LinearWithLoRA] = {}

    for full_name, module in list(model.named_modules()):
        if not isinstance(module, LoraLayer):
            continue
        # * 需要同时符合key map中的数据以及前缀中的数据, lora_target_suffixes
        module_key = normalize_module_key(full_name)
        if module_key not in module_keys:
            continue
        if not module_key_matches(
            module_key, lora_target_suffixes=lora_target_suffixes
        ):
            continue

        wrapper = LinearWithLoRA(module.get_base_layer())
        replace_submodule(model, full_name, wrapper)
        wrappers[module_key] = wrapper

    return wrappers
