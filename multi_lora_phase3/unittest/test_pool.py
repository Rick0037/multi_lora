"""Tests for multi_lora_phase3.pool eviction and slot mapping."""

import torch

from multi_lora_phase3.pool import LoRAMemoryPool
from multi_lora_phase3.registry import LoRARef
from multi_lora_phase3.weights import AdapterWeights


def _fake_weights(rank: int = 4) -> AdapterWeights:
    a = torch.randn(rank, 8)
    b = torch.randn(16, rank)
    return AdapterWeights(rank=rank, scaling=0.5, modules={"layer": (a, b)})


def test_pool_eviction_lru():
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
    refs = {
        id_a: LoRARef(lora_id=id_a, lora_name="a"),
        id_b: LoRARef(lora_id=id_b, lora_name="b"),
        id_c: LoRARef(lora_id=id_c, lora_name="c"),
    }

    pool.prepare_lora_batch({id_a, id_b}, weight_store=store, lora_refs=refs)
    assert id_a in pool.uid_to_buffer_id and id_b in pool.uid_to_buffer_id

    pool.prepare_lora_batch({id_c, None}, weight_store=store, lora_refs=refs)
    assert pool.eviction_policy.eviction_count >= 1
    assert id_c in pool.uid_to_buffer_id
    assert None in pool.uid_to_buffer_id
    # 仅 2 槽：加载 c 与 None 后，先前的 a、b 都应被驱逐
    assert id_a not in pool.uid_to_buffer_id
    assert id_b not in pool.uid_to_buffer_id

    info = pool.build_batch_info([id_c, None])
    assert info.weight_indices == [
        pool.get_buffer_id(id_c),
        pool.get_buffer_id(None),
    ]
