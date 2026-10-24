"""Tests for multi_lora.registry."""

from multi_lora.registry import LoRARef, LoRARegistry


def test_register_and_normalize_names():
    """注册后能按名字查到;normalize_names 把空串 / None 归成基座。"""
    reg = LoRARegistry()
    ref = LoRARef(lora_name="a", lora_path="/path/a")
    reg.register(ref)

    assert reg.get_by_name("a") is ref
    assert reg.normalize_names("a") == ref.lora_name
    assert reg.normalize_names("") is None
    assert reg.normalize_names(None) is None

    ids = reg.normalize_names(["a", None, ""])
    assert ids[0] == ref.lora_name
    assert ids[1] is None
    assert ids[2] is None
