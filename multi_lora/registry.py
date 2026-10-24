"""LoRA 适配器注册表;元数据、稳定 ID 与引用计数。

职责(与 pool.py / manager.py 的配合);
- 对外暴露人类可读的 lora_name;对内用稳定 lora_id(uuid)与 GPU 槽位 uid 对齐。
- acquire / release 做引用计数;模拟 tokenizer 侧请求开始/结束,防止卸载仍在用的 adapter。
- touch 供 LRU 驱逐侧按最近使用排序(本仓库 pool 的 LRU 按 uid;registry 的 LRU 可按 name 扩展)。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Union
from uuid import uuid4


@dataclass(frozen=True)
class LoRARef:
    """适配器元数据;稳定 lora_id、可读 lora_name / lora_path.

    默认 lora_id 是随机生成的 UUID 字符串,不再直接等同于 lora_name。
    """

    # 未显式传入 lora_id 时,每次构造都会调用 default_factory 生成 UUID。
    lora_id: str = field(default_factory=lambda: uuid4().hex)
    lora_name: Optional[str] = None
    lora_path: Optional[str] = None

    def __post_init__(self) -> None:
        """每次构造 LoRARef 后执行,校验 lora_id 非空."""
        if not self.lora_id:
            raise ValueError("lora_id cannot be empty")

    def __str__(self) -> str:
        """可读字符串,便于日志打印."""
        parts = [
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if getattr(self, f.name) is not None
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class LoRARegistry:
    """维护 name -> LoRARef,供调用方查询元数据,并通过引用计数跟踪进行中的请求.
    lora的实体路径
    数据流;
    1. 请求传入可读的 lora_name。
    2. resolve_names/acquire 将名称解析为 lora_id 列表。
    3. manager.prepare_lora_batch 用这些 uid 作为 pool 的 key。
    """

    def __init__(self, initial: Optional[List[LoRARef]] = None) -> None:
        """可选传入初始 LoRARef 列表并完成 register."""
        # lora_name -> LoRARef;OrderedDict 便于 LRU 把最近访问的 name 移到末尾。
        # value 保存原来的 LoRARef 对象;OrderedDict 默认按插入顺序迭代。
        self._registry: OrderedDict[str, LoRARef] = OrderedDict()
        # lora_id -> 进行中的请求数(acquire 加一,release 减一)。
        self._counters: Dict[str, int] = {}
        if initial:
            for ref in initial:
                self.register(ref)

    def register(self, lora_ref: LoRARef) -> None:
        """注册新 adapter;lora_name 不可重复."""
        if lora_ref.lora_name is None:
            raise ValueError("lora_name is required for register")
        if lora_ref.lora_name in self._registry:
            raise ValueError(f"LoRA name already registered: {lora_ref.lora_name}")
        self._registry[lora_ref.lora_name] = lora_ref
        # setdefault 仅在 key 不存在时插入 0,已有计数不会被清零。
        self._counters.setdefault(lora_ref.lora_id, 0)

    # 仅测试使用
    def unregister(self, lora_name: str) -> str:
        """注销 adapter;若有进行中的 acquire 则拒绝."""
        ref = self._registry.get(lora_name)
        if ref is None:
            raise ValueError(f"unknown LoRA name: {lora_name}")
        if self._counters.get(ref.lora_id, 0) > 0:
            raise ValueError(
                f"cannot unregister {lora_name}: {self._counters[ref.lora_id]} in-flight requests"
            )
        del self._registry[lora_name]
        self._counters.pop(ref.lora_id, None)
        return ref.lora_id

    def get_by_name(self, lora_name: str) -> LoRARef:
        """按可读名称查 LoRARef."""
        ref = self._registry.get(lora_name)
        if ref is None:
            raise ValueError(
                f"unknown LoRA name: {lora_name}; loaded: {list(self._registry)}"
            )
        return ref

    def get_by_id(self, lora_id: str) -> LoRARef:
        """按稳定 lora_id 查 LoRARef."""
        for ref in self._registry.values():
            if ref.lora_id == lora_id:
                return ref
        raise ValueError(f"unknown LoRA id: {lora_id}")

    def resolve_names(
        self, lora_names: Union[str, List[Optional[str]], None]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """将请求里的 name(或 None=基座)解析为 lora_id(不修改引用计数).

        输入输出形状保持一致:单值进单值出,列表进列表出(逐位保留 None)。
        """
        if lora_names is None:
            return None
        if isinstance(lora_names, str):
            if lora_names == "":
                return None
            return self.get_by_name(lora_names).lora_id
        out: List[Optional[str]] = []
        for name in lora_names:
            if name is None or name == "":
                out.append(None)
            else:
                out.append(self.get_by_name(name).lora_id)
        return out

    def acquire(
        self, lora_names: Union[str, List[Optional[str]], None]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """解析 name 并增加非 None 的 lora_id 计数(表示该 adapter 正被某请求占用)。"""
        ids = self.resolve_names(lora_names)
        if isinstance(ids, list):
            for lid in ids:
                if lid is not None:
                    self._counters[lid] = self._counters.get(lid, 0) + 1
        elif ids is not None:
            self._counters[ids] = self._counters.get(ids, 0) + 1
        return ids

    def release(self, lora_id: Optional[str]) -> None:
        if lora_id is None:
            return
        count = self._counters.get(lora_id, 0)
        if count <= 0:
            raise ValueError(f"release without acquire for {lora_id}")
        self._counters[lora_id] = count - 1

    def touch(self, lora_name: str) -> LoRARef:
        """LRU;访问时把 name 移到 OrderedDict 末尾."""
        ref = self.get_by_name(lora_name)
        self._registry.move_to_end(lora_name)
        return ref

    def all_refs(self) -> List[LoRARef]:
        """返回当前已注册的全部 LoRARef."""
        return list(self._registry.values())

    @property
    def names(self) -> List[str]:
        """已注册的 adapter 可读名称列表."""
        return list(self._registry.keys())
