"""write_many(T18 bank 装载 + T19 单 kernel 散布)的 GPU 正确性测试。

多轮换入换出(含驱逐复用槽位)后,逐槽位对比池内前 rank 区与源权重;
覆盖 h>1(同 rank 多个)、h==1、混合 rank、缺 module、跨 _BANK_CHUNK 分批。
散布走 _scatter_stage_kernel(指针/偏移表),CPU 侧参考实现在
test_pool_write_many.py;异步 copy stream 路径靠 wait_load + synchronize 后再比对。

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


def check_resident(pool, ref, step):
    """当前驻留的每个槽:前 rank 区逐位等于源权重;缺 module 的前 rank 区为零.

    ref 是源权重的独立副本 —— T18 会在装槽后释放 weight_store 里的散张量,
    池内容只能用这份副本校验。
    """
    ok = 0
    for name, slot in pool.name_to_slot.items():
        if name is None:
            continue
        w = ref[name]
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


def run_impl(impl, store, ref):
    """按指定散布实现(kernel/python)跑全部轮次,返回 bank 计数列表."""
    pool = LoRAMemoryPool(
        max_loras_per_batch=8, max_lora_rank=8,
        device=torch.device("cuda"), dtype=torch.bfloat16,
    )
    pool.buffers.scatter_impl = impl
    for w in store.values():
        pool.ensure_buffers(w)

    rounds = [
        {"a", "b", "c", "d"},          # 满批:rank8 h=2, rank4 h=2
        {"e"},                         # h==1(单 rank 组)
        {"a", "c", "e"},               # 混合 + 驱逐复用槽位(bank 命中)
        {"b", "d"},                    # 再来一轮复用
        {"a", "b", "e", "f", "g"},     # rank8 h=5 > _BANK_CHUNK,走分批聚合
    ]
    bank_counts = []
    for i, names in enumerate(rounds):
        pool.prepare_lora_batch(names, weight_store=store)
        torch.cuda.synchronize()   # 等 copy stream 的异步搬运真正落池
        n = check_resident(pool, ref, i)
        bank_counts.append(len(pool.buffers._banks))
        print(f"[{impl}] round{i} {sorted(names)}: {n} 个槽位核对一致, bank={bank_counts[-1]}")
    # bank 按 adapter 复用:7 个 adapter 最多 7 块,不会随轮次增长
    assert bank_counts[-1] <= len(store), f"bank 泄漏: {bank_counts}"
    # 装过的 adapter 散张量都已释放(只留 bank 一份)
    for name in ref:
        assert store[name].flattened, f"{name} 的散张量没有释放"
    return bank_counts


def main():
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA")
        return 0
    torch.manual_seed(0)

    # (rank, modules, seed):每次 run 都重建,因为 T18 会释放 store 里的散张量
    specs = {
        "a": (8, ["m1", "m2"], 1),
        "b": (8, ["m1"], 2),        # 同 rank、缺 m2 → h>1 组 + 零填充
        "c": (4, ["m1", "m2"], 3),  # 低 rank 组
        "d": (4, ["m2"], 4),
        "e": (8, ["m1", "m2"], 5),
        "f": (8, ["m1", "m2"], 6),  # 凑够 5 个同 rank,跨 chunk 边界
        "g": (8, ["m2"], 7),
    }

    # 两种散布实现都跑:kernel 是默认路径,python 是参考实现,结果必须一致
    for impl in ("kernel", "python"):
        store = {n: make_weights(r, k, s) for n, (r, k, s) in specs.items()}
        ref = {n: make_weights(r, k, s) for n, (r, k, s) in specs.items()}
        run_impl(impl, store, ref)

    print("test_pool_write_many_gpu: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
