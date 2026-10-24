"""LoRA 元数据注册与引用计数."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass(frozen=True)
class LoRARef:
    """LoRA adapter 元数据;lora_name 同时作为 GPU pool 的 key."""

    lora_name: str
    lora_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.lora_name:
            raise ValueError("lora_name cannot be empty")


class LoRARegistry:
    """adapter 元数据注册表;name ↔ path 映射 + 引用计数(acquire/release 供示例模拟生命周期)."""

    def __init__(self) -> None:
        self._registry: Dict[str, LoRARef] = {}
        self._counters: Dict[str, int] = {}

    def register(self, lora_ref: LoRARef) -> None:
        """注册新 adapter;lora_name 不可重复."""
        if lora_ref.lora_name in self._registry:
            raise ValueError(f"LoRA name already registered: {lora_ref.lora_name}")
        self._registry[lora_ref.lora_name] = lora_ref
        self._counters.setdefault(lora_ref.lora_name, 0)

    def get(self, lora_name: str) -> LoRARef:
        """按名称查 LoRARef;不存在抛错."""
        ref = self._registry.get(lora_name)
        if ref is None:
            raise ValueError(
                f"unknown LoRA name: {lora_name}; loaded: {list(self._registry)}"
            )
        return ref

    def resolve_names(
        self, lora_names: Union[str, List[Optional[str]], None]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """校验 adapter 名,并把空串/None 归一为 None;保持输入形状."""
        if lora_names is None:
            return None
        if isinstance(lora_names, str):
            if lora_names == "":
                return None
            return self.get(lora_names).lora_name
        out: List[Optional[str]] = []
        for name in lora_names:
            if name is None or name == "":
                out.append(None)
            else:
                out.append(self.get(name).lora_name)
        return out

    def acquire(
        self, lora_names: Union[str, List[Optional[str]], None]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """校验名称,增加引用计数."""
        ids = self.resolve_names(lora_names)
        if isinstance(ids, list):
            for lid in ids:
                if lid is not None:
                    self._counters[lid] += 1
        elif ids is not None:
            self._counters[ids] += 1
        return ids

    def release(self, lora_name: Optional[str]) -> None:
        """减少引用计数;计数为 0 时再释放抛错."""
        if lora_name is None:
            return
        count = self._counters.get(lora_name, 0)
        if count <= 0:
            raise ValueError(f"release without acquire for {lora_name}")
        self._counters[lora_name] = count - 1
