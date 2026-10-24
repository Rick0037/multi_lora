"""adapter 元数据注册表:name -> path 映射."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# adapter 名;None 是合法的值,表示"基座"(不使用任何 LoRA)
LoraName = Optional[str]


@dataclass(frozen=True)
class LoRARef:
    """一个 adapter 的元数据;lora_name 同时是 GPU 槽位池里的 key."""

    lora_name: str
    lora_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.lora_name:
            raise ValueError("lora_name cannot be empty")


class LoRARegistry:
    """元数据注册表;这里只存 name/path,真正的权重由 manager 抽取."""

    def __init__(self) -> None:
        self._registry: Dict[str, LoRARef] = {}

    def register(self, lora_ref: LoRARef) -> None:
        """登记元数据;同名重复注册直接报错."""
        if lora_ref.lora_name in self._registry:
            raise ValueError(f"LoRA name already registered: {lora_ref.lora_name}")
        self._registry[lora_ref.lora_name] = lora_ref

    def get_by_name(self, lora_name: str) -> LoRARef:
        """按名字查元数据;不存在则报错."""
        ref = self._registry.get(lora_name)
        if ref is None:
            raise ValueError(
                f"unknown LoRA name: {lora_name}; loaded: {list(self._registry)}"
            )
        return ref

    def resolve_names(
        self, lora_names: Union[str, List[Optional[str]], None]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """校验名字存在;空串 / None 统一成 None(基座),保持输入是单值还是列表."""
        if lora_names is None:
            return None

        # 单个名字:直接查,空串当成基座
        if isinstance(lora_names, str):
            if not lora_names:
                return None
            return self.get_by_name(lora_names).lora_name

        # 列表:逐个查,空串 / None 都转成 None
        result: List[Optional[str]] = []
        for name in lora_names:
            if not name:
                result.append(None)
            else:
                result.append(self.get_by_name(name).lora_name)
        return result
