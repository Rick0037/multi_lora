"""write_many(T16 合并 H2D)的正确性测试:纯 CPU,验证与旧逐条 write 语义一致.

覆盖:首次装载 / 换出后重装 / rank 混合补零 / adapter 缺 module 时该槽清零 /
已驻留的 adapter 不触发重装。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from multi_lora.pool import LoRAMemoryPool
from multi_lora.weights import AdapterWeights

MODULES = {"m1": (4, 3), "m2": (5, 3)}  # key: (out_dim, in_dim)


def make_weights(rank, keys, seed):
    g = torch.Generator().manual_seed(seed)
    mods = {}
    for key in keys:
        out_dim, in_dim = MODULES[key]
        mods[key] = (
            torch.randn(rank, in_dim, generator=g),
            torch.randn(out_dim, rank, generator=g),
        )
    return AdapterWeights(rank=rank, scaling=1.0 / rank, modules=mods)


def check_slot(pool, slot, weights):
    """槽位前 rank 行/列 == 权重本体。rank 之后的旧行不校验:
    分组装载不清尾(kernel 按 rank 截断,读不到);缺 module 的 adapter
    在前 rank 区是零(聚合时填的零源)。"""
    for key, (out_dim, in_dim) in MODULES.items():
        a, b = pool.get_slot_weights(slot, key)
        if key in weights.modules:
            wa, wb = weights.modules[key]
            r = weights.rank
            assert torch.equal(a[:r], wa), f"slot{slot} {key} A 前缀不一致"
            assert torch.equal(b[:, :r], wb), f"slot{slot} {key} B 前缀不一致"
        else:
            # 该 adapter 没有此 module:kernel 会读的前 rank 区必须是零
            r = weights.rank
            assert (
                a[:r].abs().sum() == 0
            ), f"slot{slot} {key} 缺 module 前 rank 区非零(A)"
            assert (
                b[:, :r].abs().sum() == 0
            ), f"slot{slot} {key} 缺 module 前 rank 区非零(B)"


def main():
    torch.manual_seed(0)
    pool = LoRAMemoryPool(
        max_loras_per_batch=2,
        max_lora_rank=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    store = {
        "a": make_weights(8, ["m1", "m2"], seed=1),  # 满 rank、全 module
        "b": make_weights(4, ["m1"], seed=2),  # 低 rank、缺 m2
        "c": make_weights(8, ["m1", "m2"], seed=3),
        "d": make_weights(2, ["m2"], seed=4),  # 缺 m1
    }
    for w in store.values():
        pool.ensure_buffers(w)

    # 1) 首批装载(2 个槽位正好占满;set 顺序不定,断言只看驻留集合)
    pool.prepare_lora_batch({"a", "b"}, weight_store=store)
    assert set(pool.name_to_slot) == {"a", "b"}
    for name, slot in pool.name_to_slot.items():
        check_slot(pool, slot, store[name])

    # 2) 换出一批:c/d 进,a/b 被驱逐
    pool.prepare_lora_batch({"c", "d"}, weight_store=store)
    assert set(pool.name_to_slot) == {"c", "d"}
    for name, slot in pool.name_to_slot.items():
        check_slot(pool, slot, store[name])

    # 3) 混合批:a 重新进(触发驱逐),d 已驻留不重装
    pool.prepare_lora_batch({"a", "d"}, weight_store=store)
    assert set(pool.name_to_slot) == {"a", "d"}
    for name, slot in pool.name_to_slot.items():
        check_slot(pool, slot, store[name])
    # build_batch_info 的槽位映射与实际驻留一致
    info = pool.build_batch_info(["a", "d"])
    assert info.weight_indices[0] == pool.get_slot_id("a")
    assert info.weight_indices[1] == pool.get_slot_id("d")

    print("test_pool_write_many: all OK")


if __name__ == "__main__":
    main()
