"""从 PEFT 模型抽取 LoRA A/B 权重,供 CPU 缓存与 GPU pool 装载.

第三步数据流;
1. MultiLoRAManager.register_adapter 调用 extract_adapter_weights,把每个 PEFT adapter
   名对应的全部 LoraLayer 的 A/B 张量拉到 CPU,存进 _weight_store[lora_id]。
2. LoRAMemoryPool._load_uid_to_slot 从 weight_store 取 CPU 张量,按 max_lora_rank 做
   padding 后 copy_ 到 GPU 槽位,供第四步 fused kernel 或当前 layers.lora_delta 使用。

注意;此处仅抽取各模块 A/B 并按统一 rank 上界 padding,不做 qkv 等模块的权重合并。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer

# 本文件的作用是提取peft模型中的lora权重,并将其存储在AdapterWeights中

# * key -> name, value -> [weight_a, weight_b]
ModuleWeights = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class AdapterWeights:
    """单个 adapter 的全部 LoRA 权重(CPU).

    modules;归一化后的模块路径 -> (lora_A, lora_B)。
    A 形状 [r, in],B 形状 [out, r];与常见 LoRA 公式 delta = (x @ A.T) @ B.T * scaling 一致。
    """

    rank: int
    scaling: float
    modules: ModuleWeights


def _module_key(name: str) -> str:
    """归一化模块名,便于 pool 按 key 索引."""
    for prefix in ("base_model.model.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def extract_adapter_weights(model: PeftModel, adapter_name: str) -> AdapterWeights:
    """遍历 LoraLayer,抽取指定 adapter 的 lora_A / lora_B.

    必须先 set_adapter,否则 lora_A 字典里可能没有该 adapter 的键。
    """
    model.set_adapter(adapter_name)  # 是PeftModel类下的一个函数
    # * adapter_name:"a"
    modules: ModuleWeights = {}
    rank: Optional[int] = None
    scaling: Optional[float] = None  # * alpha / rank

    for full_name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if adapter_name not in getattr(module, "lora_A", {}):
            continue
        lora_a_layer = module.lora_A[adapter_name]
        lora_b_layer = module.lora_B[adapter_name]
        """
        (Pdb) lora_a_layer
        Linear(in_features=2560, out_features=16, bias=False)
        (Pdb) lora_b_layer
        Linear(in_features=16, out_features=4096, bias=False)
        lora_a_layer.weight 原来一直在GPU
        """
        a = lora_a_layer.weight.detach().cpu().clone()
        b = lora_b_layer.weight.detach().cpu().clone()
        # * rank

        raw_r = (
            module.r[adapter_name]
            if isinstance(getattr(module, "r", None), dict)
            else module.r
        )
        fan_in = int(lora_a_layer.in_features)  # * 2560 * 16,
        fan_out = int(lora_b_layer.out_features)  # * 16 * 4096
        # nn.Linear 的 weight 为 [out_features, in_features];标准 LoRA A: [r, fan_in],B: [fan_out, r]。
        if a.shape == (fan_in, raw_r):
            a = a.T
        if a.shape != (raw_r, fan_in):
            raise ValueError(
                f"lora_A shape {tuple(a.shape)} != ({raw_r}, {fan_in}) after normalize for {full_name}"
            )
        if b.shape == (raw_r, fan_out):
            b = b.T
        if b.shape != (fan_out, raw_r):
            raise ValueError(
                f"lora_B shape {tuple(b.shape)} != ({fan_out}, {raw_r}) after normalize for {full_name}"
            )
        key = _module_key(full_name)
        # * 'layers.0.self_attn.q_proj'
        modules[key] = (a, b)
        r = int(raw_r)
        if rank is None:
            rank = r
            # scaling = alpha / r,与 HF PEFT 默认 LoRA 缩放一致。
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
