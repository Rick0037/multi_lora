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
        """注册 adapter 元数据;重名报错."""
        if lora_ref.lora_name in self._registry:
            raise ValueError(f"LoRA name already registered: {lora_ref.lora_name}")
        self._registry[lora_ref.lora_name] = lora_ref
        self._counters[lora_ref.lora_name] = 0

    def get_by_name(self, lora_name: str) -> LoRARef:
        """查元数据;不存在抛错."""
        ref = self._registry.get(lora_name)
        if ref is None:
            raise ValueError(
                f"unknown LoRA name: {lora_name}; loaded: {list(self._registry)}"
            )
        return ref

    def resolve_names(
        self, lora_names: Union[str, List[Optional[str]], None]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """校验 adapter 名存在,空串/None 归一为 None,保持输入形状(单值/列表)."""
        if lora_names is None:
            return None
        if isinstance(lora_names, str):
            return None if not lora_names else self.get_by_name(lora_names).lora_name
        return [
            None if not name else self.get_by_name(name).lora_name
            for name in lora_names
        ]

    def acquire(
        self, lora_names: Union[str, List[Optional[str]], None]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """resolve + 引用计数 +1(仅非 None)."""
        result = self.resolve_names(lora_names)
        if isinstance(result, list):
            for name in result:
                if name is not None:
                    self._counters[name] += 1
        elif result is not None:
            self._counters[result] += 1
        return result

    def release(self, lora_name: Optional[str]) -> None:
        """引用计数 -1."""
        if lora_name is not None:
            self._counters[lora_name] -= 1
