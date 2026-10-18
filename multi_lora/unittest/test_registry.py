"""Tests for multi_lora.registry."""

import pytest

from multi_lora.registry import LoRARef, LoRARegistry


def test_register_acquire_release():
    """注册、acquire、release、unregister 的基本生命周期应正常。"""
    reg = LoRARegistry()
    ref = LoRARef(lora_name="a", lora_path="/path/a")
    reg.register(ref)
    lid = reg.acquire("a")
    assert lid == ref.lora_id
    reg.release(lid)
    reg.unregister("a")


def test_acquire_list_and_base():
    """acquire 列表时 None/空字符串应映射为 base（None）。"""
    reg = LoRARegistry()
    reg.register(LoRARef(lora_name="x", lora_path="p"))
    ids = reg.acquire(["x", None, ""])
    assert ids[0] == reg.get_by_name("x").lora_id
    assert ids[1] is None
    assert ids[2] is None


def test_unregister_with_inflight():
    """仍有 inflight 引用时 unregister 应抛出 ValueError。"""
    reg = LoRARegistry()
    reg.register(LoRARef(lora_name="b", lora_path="p"))
    lid = reg.acquire("b")
    with pytest.raises(ValueError):
        reg.unregister("b")
    reg.release(lid)
    reg.unregister("b")
