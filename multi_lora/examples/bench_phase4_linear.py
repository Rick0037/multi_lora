#!/usr/bin/env python3
"""Micro-benchmark: 整模型 ToyLM forward_with_triton_lora vs 逐行参考."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from multi_lora.layers import lora_delta
from multi_lora.manager import MultiLoRAManager


class ToyLM(nn.Module):
    """用于基准测试的最小线性语言模型占位结构。"""

    def __init__(self):
        """初始化单层 Linear(256→512)。"""
        super().__init__()
        self.lin = nn.Linear(256, 512)

    def forward(self, x):
        """对输入执行单层线性变换。"""
        return self.lin(x)


def main():
    """对比整模型 forward 与逐行 LoRA 参考实现的耗时与数值一致性。"""
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    # 在临时目录保存两个不同 seed 的 LoRA 适配器
    tmp = Path(tempfile.mkdtemp())
    cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=["lin"],
    )
    for seed in (1, 2):
        torch.manual_seed(seed)
        m = ToyLM()
        get_peft_model(m, cfg).save_pretrained(tmp / f"a{seed}")

    # 加载双适配器 manager,后端先用 torch 便于与逐行参考对齐
    mgr = MultiLoRAManager.from_pretrained(
        ToyLM(),
        {"a": str(tmp / "a1"), "b": str(tmp / "a2")},
        max_loras_per_batch=4,
        max_lora_rank=64,
        lora_backend="torch",
    )
    id_a, id_b = mgr._peft_name_to_id["a"], mgr._peft_name_to_id["b"]
    # 交替分配 a/b 适配器,模拟 heterogeneous batch
    lora_ids = [id_a if i % 2 == 0 else id_b for i in range(args.batch)]
    device = mgr.pool.device
    dtype = mgr.pool.dtype
    x = torch.randn(args.batch, 256, device=device, dtype=dtype)

    def run_full():
        """整模型路径;manager 内部 batch prepare + 包装层 forward。"""
        return mgr.forward_with_triton_lora(x=x, lora_ids=lora_ids)

    def run_row_loop():
        """逐行参考;base linear + 每行单独 lora_delta。"""
        mgr.prepare_lora_batch(lora_ids)
        linear = mgr.lora_wrappers["lin"].base_layer
        yb = linear(x)
        acc = yb.clone()
        for i, uid in enumerate(lora_ids):
            slot = mgr.pool.get_buffer_id(uid)
            r, s = mgr.pool._slot_meta[slot]
            if r == 0:
                continue
            a, b = mgr.pool.get_slot_weights(slot, "lin")
            # 仅更新第 i 行的 LoRA 增量
            acc[i : i + 1] = yb[i : i + 1] + lora_delta(x[i : i + 1], a, b, s, rank=r)
        return acc

    # 预热,避免首次 kernel/编译开销影响计时
    for _ in range(10):
        run_full()
        run_row_loop()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        run_full()
    t1 = time.perf_counter()
    for _ in range(args.iters):
        run_row_loop()
    t2 = time.perf_counter()

    b_ms = (t1 - t0) / args.iters * 1000
    r_ms = (t2 - t1) / args.iters * 1000
    print(
        f"full_model {b_ms:.3f} ms/iter  row_loop {r_ms:.3f} ms/iter  speedup {r_ms/b_ms:.2f}x"
    )
    # 数值校验;两种路径输出应接近
    assert torch.allclose(run_full(), run_row_loop(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    main()
