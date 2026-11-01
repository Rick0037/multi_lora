"""write_many(T17b 分组装载)的 GPU 正确性测试。

多轮换入换出(含驱逐复用槽位)后,逐槽位对比池内前 rank 区与源权重;
覆盖 h>1(同 rank 多个)、h==1、混合 rank、缺 module 的 adapter。
异步 copy stream 路径的正确性靠 wait_load + synchronize 后再比对。

    .venv/bin/python multi_lora/tests/test_pool_write_many_gpu.py
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


def check_resident(pool, store, step):
    """当前驻留的每个槽:前 rank 区逐位等于源权重;缺 module 的前 rank 区为零."""
    ok = 0
    for name, slot in pool.name_to_slot.items():
        if name is None:
            continue
        w = store[name]
        for key in MODULES:
            a, b = pool.get_slot_weights(slot, key)
            r = w.rank
            if key in w.modules:
                wa, wb = w.modules[key]
                assert torch.equal(a[:r], wa.to(device=a.device, dtype=a.dtype)), \
                    f"step{step} {name}/{key} A 不一致"
                assert torch.equal(b[:, :r], wb.to(device=b.device, dtype=b.dtype)), \
                    f"step{step} {name}/{key} B 不一致"
            else:
                assert a[:r].abs().sum() == 0, \
                    f"step{step} {name}/{key} 缺 module 前 rank 区非零(A)"
                assert b[:, :r].abs().sum() == 0, \
                    f"step{step} {name}/{key} 缺 module 前 rank 区非零(B)"
        ok += 1
    return ok


def main():
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA")
        return 0
    dev = torch.device("cuda")
    torch.manual_seed(0)

    pool = LoRAMemoryPool(
        max_loras_per_batch=4, max_lora_rank=8, device=dev, dtype=torch.bfloat16
    )
    store = {
        "a": make_weights(8, ["m1", "m2"], 1),
        "b": make_weights(8, ["m1"], 2),       # 同 rank、缺 m2 → h=2 组 + 零源
        "c": make_weights(4, ["m1", "m2"], 3),  # 低 rank 组
        "d": make_weights(4, ["m2"], 4),
        "e": make_weights(8, ["m1", "m2"], 5),
    }
    for w in store.values():
        pool.ensure_buffers(w)
    pin = {n: AdapterWeights(w.rank, w.scaling,
                             {k: (a.pin_memory(), b.pin_memory())
                              for k, (a, b) in w.modules.items()})
           for n, w in store.items()}

    rounds = [
        {"a", "b", "c", "d"},   # 满批:rank8 h=2, rank4 h=2
        {"e"},                  # h==1(单 rank 组)
        {"a", "c", "e"},        # 混合 + 驱逐复用槽位
        {"b", "d"},             # 再来一轮复用
    ]
    for i, names in enumerate(rounds):
        pool.prepare_lora_batch(names, weight_store=pin)
        torch.cuda.synchronize()   # 等 copy stream 的异步搬运真正落池
        n = check_resident(pool, pin, i)
        print(f"round{i} {sorted(names)}: {n} 个槽位核对一致")

    print("test_pool_write_many_gpu: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
