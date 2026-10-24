"""Tests for multi_lora.pool eviction and slot mapping."""

import torch

from multi_lora.pool import LoRAMemoryPool
from multi_lora.weights import AdapterWeights


def _fake_weights(rank: int = 4) -> AdapterWeights:
    """构造随机 LoRA 权重占位,用于 pool 单元测试。"""
    a = torch.randn(rank, 8)
    b = torch.randn(16, rank)
    return AdapterWeights(rank=rank, scaling=0.5, modules={"layer": (a, b)})


def test_pool_eviction_lru():
    """LRU 策略下槽位不足时应驱逐旧适配器并正确 build_batch_info。"""
    pool = LoRAMemoryPool(
        max_loras_per_batch=2,
        max_lora_rank=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        eviction_policy="lru",
    )
    id_a, id_b, id_c = "ida", "idb", "idc"
    store = {
        id_a: _fake_weights(),
        id_b: _fake_weights(),
        id_c: _fake_weights(),
        None: AdapterWeights(rank=0, scaling=0.0, modules={}),
    }
    # 第一批;占满 2 个槽位
    pool.prepare_lora_batch({id_a, id_b}, weight_store=store)
    assert id_a in pool.name_to_buffer_id and id_b in pool.name_to_buffer_id

    # 第二批;需要 c 与 None,应驱逐 a、b
    pool.prepare_lora_batch({id_c, None}, weight_store=store)
    assert pool.eviction_policy.eviction_count >= 1
    assert id_c in pool.name_to_buffer_id
    assert None in pool.name_to_buffer_id
    # 仅 2 槽;加载 c 与 None 后,先前的 a、b 都应被驱逐
    assert id_a not in pool.name_to_buffer_id
    assert id_b not in pool.name_to_buffer_id

    info = pool.build_batch_info([id_c, None])
    assert (
        info.weight_indices
        == [  # weight_indices 长度 = batch,第 i 行为第 i 条序列所用 adapter 的 slot。
            pool.get_buffer_id(id_c),
            pool.get_buffer_id(None),
        ]
    )
