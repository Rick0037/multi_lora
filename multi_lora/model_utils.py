"""模型模块替换与 LoRA 目标模块匹配工具."""

from __future__ import annotations

from typing import Optional, Sequence

import torch.nn as nn

from multi_lora.weights import _module_key

# MVP 默认只替换 attention 四投影；None 表示 weight store 里出现的全部模块都替换
ATTENTION_LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")


def replace_submodule(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """按点分路径（如 layers.0.self_attn.q_proj）原地替换子模块."""
    parts = module_name.split(".")
    parent = root
    # 逐级下钻到父模块
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def module_key_matches(
    module_key: str,
    *,
    target_suffixes: Optional[Sequence[str]],
) -> bool:
    """判断该 module_key 是否在本次 convert 的目标集合内."""
    if target_suffixes is None:
        return True
    # 支持 ...q_proj 全路径后缀或最后一层模块名匹配
    return any(
        module_key.endswith(suffix) or module_key.split(".")[-1] == suffix
        for suffix in target_suffixes
    )


def get_inference_model(peft_model: nn.Module) -> nn.Module:
    """剥离 PeftModel 外壳，得到实际执行 forward 的内部 HF 模型."""
    model = peft_model
    if hasattr(model, "base_model"):
        model = model.base_model
    if hasattr(model, "model"):
        model = model.model
    return model


def normalize_module_key(full_name: str) -> str:
    """把 named_modules 全名归一化为与 weight store 一致的 key."""
    return _module_key(full_name)
