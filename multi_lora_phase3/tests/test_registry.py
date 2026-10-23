"""Tests for multi_lora_phase3.registry."""

import pytest

from multi_lora_phase3.registry import LoRARef, LoRARegistry


def test_register_acquire_release():
    reg = LoRARegistry()
    ref = LoRARef(lora_name="a", lora_path="/path/a")
    reg.register(ref)
    lid = reg.acquire("a")
    assert lid == ref.lora_id
    reg.release(lid)
    reg.unregister("a")


def test_acquire_list_and_base():
    reg = LoRARegistry()
    reg.register(LoRARef(lora_name="x", lora_path="p"))
    ids = reg.acquire(["x", None, ""])
    assert ids[0] == reg.get_by_name("x").lora_id
    assert ids[1] is None
    assert ids[2] is None


def test_unregister_with_inflight():
    reg = LoRARegistry()
    reg.register(LoRARef(lora_name="b", lora_path="p"))
    lid = reg.acquire("b")
    with pytest.raises(ValueError):
        reg.unregister("b")
    reg.release(lid)
    reg.unregister("b")
