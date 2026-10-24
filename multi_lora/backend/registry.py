"""LoRA 计算后端工厂;按名称实例化 Torch 或 Triton 实现."""

from __future__ import annotations

from multi_lora.backend.base import LoRABackend
from multi_lora.backend.torch_backend import TorchLoRABackend
from multi_lora.backend.triton_backend import TritonLoRABackend


def get_lora_backend(name: str) -> LoRABackend:
    """根据 "torch" / "triton" 返回对应 LoRA 后端实例."""
    name = name.lower().strip()
    if name == "torch":
        return TorchLoRABackend()
    if name == "triton":
        return TritonLoRABackend()
    raise ValueError(f"unknown lora backend: {name}; try torch | triton")
