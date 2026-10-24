"""LoRA 适配器注册表;元数据、稳定 ID 与引用计数。

职责(与 pool.py / manager.py 的配合);
- 对外暴露人类可读的 lora_name;GPU 槽位也用同一个名字作 key。
- acquire / release 做引用计数;模拟 tokenizer 侧请求开始/结束,防止卸载仍在用的 adapter。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Union


# * 元数据 lora_name, lora_path, 这里的register是不包括none基座模型的
@dataclass(frozen=True)
class LoRARef:
    """元数据集合体;内部 id 直接复用 lora_name."""

    lora_name: str
    lora_path: Optional[str] = None

    def __post_init__(self) -> None:
        """每次构造 LoRARef 后执行,校验 lora_name 非空."""
        if not self.lora_name:
            raise ValueError("lora_name cannot be empty")

    def __str__(self) -> str:
        """可读字符串,便于日志打印."""
        parts = [
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if getattr(self, f.name) is not None
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


# * manager得一部分
class LoRARegistry:
    """维护 name -> LoRARef,供调用方查询元数据,并通过引用计数跟踪进行中的请求.
    lora的实体路径
    数据流;
    1. 请求传入可读的 lora_name。
    2. resolve_names/acquire 校验名称并归一(空串/None → None)。
    3. manager.prepare_lora_batch 用这些名字作为 pool 的 key。
    """

    def __init__(self, initial: Optional[List[LoRARef]] = None) -> None:
        """可选传入初始 LoRARef 列表并完成 register."""
        # lora_name -> LoRARef
        self._registry: Dict[str, LoRARef] = {}
        # lora_name -> 进行中的请求数(acquire 加一,release 减一)。
        self._counters: Dict[str, int] = {}
        if initial:
            for ref in initial:
                self.register(ref)

    def register(self, lora_ref: LoRARef) -> None:
        """注册新 adapter;lora_name 不可重复."""
        if lora_ref.lora_name in self._registry:
            raise ValueError(f"LoRA name already registered: {lora_ref.lora_name}")
        self._registry[lora_ref.lora_name] = lora_ref
        # setdefault 仅在 key 不存在时插入 0,已有计数不会被清零。
        self._counters.setdefault(lora_ref.lora_name, 0)

    def get_by_name(self, lora_name: str) -> LoRARef:
        """按可读名称查 LoRARef."""
        ref = self._registry.get(lora_name)
        if ref is None:
            raise ValueError(
                f"unknown LoRA name: {lora_name}; loaded: {list(self._registry)}"
            )
        return ref

    # *------------------ 仅测试使用---------------------
    # * 如果要进行卸载在register里面的时候最好cpu的weight都给卸载了, 仅仅这里这只是 meta info
    def unregister(self, lora_name: str) -> str:
        """注销 adapter;若有进行中的 acquire 则拒绝."""
        ref = self._registry.get(lora_name)
        if ref is None:
            raise ValueError(f"unknown LoRA name: {lora_name}")
        if self._counters.get(ref.lora_name, 0) > 0:
            raise ValueError(
                f"cannot unregister {lora_name}: {self._counters[ref.lora_name]} in-flight requests"
            )
        del self._registry[lora_name]
        self._counters.pop(ref.lora_name, None)
        return ref.lora_name

    def resolve_names(
        self, lora_names: Union[str, List[Optional[str]], None]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """校验 adapter 名存在,并把 None/空串归一为 None(保持输入形状).

        单值进单值出,列表进列表出(逐位保留 None)。
        """
        if lora_names is None:
            return None
        if isinstance(lora_names, str):
            if lora_names == "":
                return None
            return self.get_by_name(lora_names).lora_name
        out: List[Optional[str]] = []
        for name in lora_names:
            if name is None or name == "":
                out.append(None)
            else:
                out.append(self.get_by_name(name).lora_name)
        return out

    def acquire(
        self, lora_names: Union[str, List[Optional[str]], None]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """校验并归一名称,同时增加非 None 的引用计数(表示该 adapter 正被请求占用)。"""
        ids = self.resolve_names(lora_names)
        if isinstance(ids, list):
            for lid in ids:
                if lid is not None:
                    self._counters[lid] = self._counters.get(lid, 0) + 1
        elif ids is not None:
            self._counters[ids] = self._counters.get(ids, 0) + 1
        return ids

    def release(self, lora_name: Optional[str]) -> None:
        """结束一次占用;计数为 0 时再释放会报错,避免下溢."""
        if lora_name is None:
            return
        count = self._counters.get(lora_name, 0)
        if count <= 0:
            raise ValueError(f"release without acquire for {lora_name}")
        self._counters[lora_name] = count - 1
