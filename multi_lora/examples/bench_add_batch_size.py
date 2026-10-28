#!/usr/bin/env python3
"""异构 rank adapter 下,扫描 batch size 的 prefill / decode 性能.

adapter 来自 make_rank_variants.py 生成的 rank 8/16/32 三个 checkpoint,
batch 内按 i % 3 循环分配,保证 distinct adapter 数不超过池容量,
于是一次跑同时覆盖「异构 rank」和「batch  scaling」两件事。

计时方式与 bench_qwen4b_e2e.py 一致:prefill 重复多次取均值(TTFT),
decode 用 CUDA event 逐步记录、多轮取均值(TPOT)。

Example::

    python multi_lora/examples/bench_add_batch_size.py
    python multi_lora/examples/bench_add_batch_size.py --backend torch
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multi_lora.batching import group_row_indices  # noqa: E402
from multi_lora.hf_runner import (  # noqa: E402
    load_causal_lm,
    load_tokenizer,
    pick_device,
    pick_dtype,
    release_model,
)
from multi_lora.layers import set_lora_enabled  # noqa: E402
from multi_lora.manager import MultiLoRAManager  # noqa: E402

BASE = "Qwen/Qwen3-4B-Instruct-2507"
VARIANTS = Path("/root/autodl-tmp/qwen3_4b_rank_variants")

# 原来的两个 adapter
ADAPTER_A = "GestaltLabs/Jeff-1"  # rank 16,只训练了 qkvo
ADAPTER_B = (
    "/root/autodl-tmp/qwen3_4b_r64_lora/v2-20261022-131552/last-checkpoint"  # rank 64
)

# 名字 -> checkpoint;原有两个 + 降秩造出的三个,rank 覆盖 16/64/8/16/32
ADAPTERS: Dict[str, str] = {
    "a": ADAPTER_A,
    "b": ADAPTER_B,
    "r8": str(VARIANTS / "r8"),
    "r16": str(VARIANTS / "r16"),
    "r32": str(VARIANTS / "r32"),
}

# 名字 -> rank,只用于打印;实际 rank 由 pool 从权重里读
ADAPTER_RANKS: Dict[str, int] = {
    "a": 16, "b": 64, "r8": 8, "r16": 16, "r32": 32}

PROMPT = "user\nList three countries and their capitals.\n" "assistant\n"

# 固定实验配置:不再做成命令行参数,避免误调导致两次结果不可比
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]  # batch size 扫描点
# BATCHES = [128]  # batch size 扫描点
NUM_DECODE = 16  # decode 步数
PREFILL_ITERS = 5  # prefill 重复次数
DECODE_REPEAT = 3  # decode 重复轮数,每轮重新 prefill 拿干净 cache
WARMUP = 3  # 正式计时前的 decode 预热步数
SEED = 1234  # 固定随机种子
ATOL = 1e-3  # 绝对判据 allclose(atol)
RTOL = 5e-3  # 相对判据 mean_rel < rtol


def parse_args() -> argparse.Namespace:
    """解析命令行;只保留后端切换,其余实验配置均为模块常量."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend", choices=("triton", "torch"), default="triton", help="LoRA 后端"
    )
    return p.parse_args()


def assign_names(count: int, keys: List[str]) -> List[str]:
    """按 i % len(keys) 循环分配,distinct adapter 数不超过池容量."""
    return [keys[i % len(keys)] for i in range(count)]


def cuda_ms(fn):
    """用 CUDA event 计时一次调用,返回 (毫秒, 返回值)."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def rank_distribution(names: List[str]) -> Dict[int, int]:
    """统计这批请求里各 rank 有多少行."""
    counts: Dict[int, int] = {}
    for name in names:
        rank = ADAPTER_RANKS[name]
        counts[rank] = counts.get(rank, 0) + 1
    return counts


@torch.inference_mode()
def mixed_prefill(mgr, input_ids, attention_mask, names):
    """混合批次(Punica SGMV 式):一次 forward 覆盖所有 segment,带 KV cache."""
    out = mgr.forward_with_lora(
        input_ids=input_ids,
        attention_mask=attention_mask,
        lora_names=names,
        use_cache=True,
    )
    return out.logits, out.past_key_values


@torch.inference_mode()
def grouped_prefill(mgr, input_ids, attention_mask, names):
    """分组逐个相乘(S-LoRA 式):每组单独 prefill,各自持有一份 KV cache.

    返回 (拼回原行号的 logits, [(group_rows, past, next_ids), ...])。
    每组的 cache 只覆盖本组行,所以 decode 也必须按组进行。
    """
    groups = group_row_indices(names)
    out_logits = None
    states = []
    for adapter, rows in groups.items():
        idx = torch.tensor(rows, device=input_ids.device, dtype=torch.long)
        out = mgr.forward_with_lora(
            input_ids=input_ids.index_select(0, idx),
            attention_mask=attention_mask.index_select(0, idx),
            lora_names=[adapter] * len(rows),
            use_cache=True,
        )
        if out_logits is None:
            out_logits = torch.empty(
                input_ids.shape[0],
                *out.logits.shape[1:],
                device=out.logits.device,
                dtype=out.logits.dtype,
            )
        out_logits.index_copy_(0, idx, out.logits)
        next_ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        states.append((idx, out.past_key_values, next_ids))
    return out_logits, states


@torch.inference_mode()
def mixed_decode(mgr, next_ids, past, names, decode_lens):
    """混合批次 decode 一步."""
    out = mgr.forward_with_lora(
        input_ids=next_ids,
        attention_mask=None,
        seq_lens=decode_lens,
        lora_names=names,
        past_key_values=past,
        use_cache=True,
    )
    return out.logits[:, -1], out.past_key_values


@torch.inference_mode()
def grouped_decode(mgr, states, names):
    """分组 decode 一步:每组用自己的 cache,结果写回原行号."""
    groups = group_row_indices(names)
    out_logits = None
    new_states = []
    for (idx, past, next_ids), adapter in zip(states, groups.keys()):
        out = mgr.forward_with_lora(
            input_ids=next_ids,
            attention_mask=None,
            seq_lens=torch.ones(
                idx.numel(), device=idx.device, dtype=torch.int32),
            lora_names=[adapter] * idx.numel(),
            past_key_values=past,
            use_cache=True,
        )
        logits = out.logits[:, -1]
        if out_logits is None:
            out_logits = torch.empty(
                sum(s[0].numel() for s in states),
                logits.shape[-1],
                device=logits.device,
                dtype=logits.dtype,
            )
        out_logits.index_copy_(0, idx, logits)
        new_states.append(
            (idx, out.past_key_values, logits.argmax(dim=-1, keepdim=True))
        )
    return out_logits, new_states


@contextmanager
def lora_disabled():
    """期间全局关掉 LoRA delta,退出时必然恢复,异常也不会漏状态."""
    set_lora_enabled(False)
    try:
        yield
    finally:
        set_lora_enabled(True)


def measure_both(mgr, input_ids, attention_mask, names, decode_lens, fn):
    """同一 batch 开/关 LoRA 各跑一次 fn,返回 (带 LoRA, 基座, 净开销).

    预热与计时口径完全一致,三项都是 fn 的同构返回值(标量或元组均可)。
    """
    with_lora = fn(mgr, input_ids, attention_mask, names, decode_lens)
    with lora_disabled():
        base = fn(mgr, input_ids, attention_mask, names, decode_lens)
    if isinstance(with_lora, tuple):
        net = tuple(a - b for a, b in zip(with_lora, base))
    else:
        net = with_lora - base
    return with_lora, base, net


@torch.inference_mode()
def measure_ttft_tpot(mgr, input_ids, attention_mask, names, decode_lens):
    """跑 prefill + DECODE_REPEAT 轮 decode,返回 (ttft_ms, tpot_ms) 均值.

    供 measure_both 在开/关 LoRA 两种状态下各调一次。
    """

    def prefill():
        """一次整 batch prefill,带 KV cache."""
        return mgr.forward_with_lora(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lora_names=names,
            use_cache=True,
        )

    # 预热:排除 Triton 首次编译与显存分配
    out = prefill()
    past = out.past_key_values
    next_ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    for _ in range(WARMUP):
        out = mgr.forward_with_lora(
            input_ids=next_ids,
            attention_mask=None,
            seq_lens=decode_lens,
            lora_names=names,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)

    prefill_times = []
    for _ in range(PREFILL_ITERS):
        ms, _ = cuda_ms(prefill)
        prefill_times.append(ms)

    # decode:每轮重新 prefill 拿干净 cache(否则 past 会一直变长),逐步记 event
    step_times = []
    for _ in range(DECODE_REPEAT):
        out = prefill()
        past = out.past_key_values
        next_ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        for _ in range(NUM_DECODE):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = mgr.forward_with_lora(
                input_ids=next_ids,
                attention_mask=None,
                seq_lens=decode_lens,
                lora_names=names,
                past_key_values=past,
                use_cache=True,
            )
            end.record()
            step_times.append((start, end))
            past = out.past_key_values
            next_ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    return (
        statistics.fmean(prefill_times),
        statistics.fmean(s.elapsed_time(e) for s, e in step_times),
    )


def check_close(label, a, b):
    """计算绝对判据 allclose(ATOL) 与相对判据 mean_rel<RTOL,返回结果 dict."""
    abs_diff = (a - b).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    scale = a.abs().max().item()
    mean_rel = mean_diff / scale if scale > 0 else 0.0
    return {
        "stage": label,
        "max_diff": max_diff,
        "mean_rel": mean_rel,
        "abs_ok": torch.allclose(a, b, atol=ATOL),
        "rel_ok": mean_rel < RTOL,
    }


def verify_paths(mgr, input_ids, attention_mask, names):
    """只做数值对比:混合 vs 分组,覆盖 prefill 与一步 decode.

    返回两条记录(prefill / decode),每条是 check_close 的 dict。
    """
    bs = input_ids.shape[0]
    decode_lens = torch.ones(bs, device=input_ids.device, dtype=torch.int32)

    mp, mpast = mixed_prefill(mgr, input_ids, attention_mask, names)
    gl, gstates = grouped_prefill(mgr, input_ids, attention_mask, names)
    rows = [check_close("prefill", mp, gl)]

    mnext = mp[:, -1].argmax(dim=-1, keepdim=True)
    md_logits, _ = mixed_decode(mgr, mnext, mpast, names, decode_lens)
    gd_logits, _ = grouped_decode(mgr, gstates, names)
    rows.append(check_close("decode", md_logits, gd_logits))
    return rows


def bench_paths(mgr, input_ids, attention_mask, names, decode_lens=None):
    """只做耗时对比:混合 vs 分组,prefill 与 decode 各四项.

    返回 (mixed_prefill_ms, grouped_prefill_ms, mixed_decode_ms, grouped_decode_ms)。
    分组路径每组 shape 不同,首次会触发 Triton 重编译,必须先预热。
    decode_lens 供 measure_both 统一五参调用,不传则内部自己算。
    """
    bs = input_ids.shape[0]
    if decode_lens is None:
        decode_lens = torch.ones(
            bs, device=input_ids.device, dtype=torch.int32)
    iters = PREFILL_ITERS

    # 预热:两条路径都各跑一遍 prefill + decode,排除编译与显存分配
    mp, mpast = mixed_prefill(mgr, input_ids, attention_mask, names)
    gl, gstates = grouped_prefill(mgr, input_ids, attention_mask, names)
    mnext = mp[:, -1].argmax(dim=-1, keepdim=True)
    mixed_decode(mgr, mnext, mpast, names, decode_lens)
    grouped_decode(mgr, gstates, names)

    def time_mixed_prefill():
        mgr.forward_with_lora(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lora_names=names,
            use_cache=False,
        )

    def time_grouped_prefill():
        for adapter, rows in group_row_indices(names).items():
            idx = torch.tensor(rows, device=input_ids.device, dtype=torch.long)
            mgr.forward_with_lora(
                input_ids=input_ids.index_select(0, idx),
                attention_mask=attention_mask.index_select(0, idx),
                lora_names=[adapter] * len(rows),
                use_cache=False,
            )

    mixed_prefill_times = [cuda_ms(time_mixed_prefill)[0]
                           for _ in range(iters)]
    grouped_prefill_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        time_grouped_prefill()
        torch.cuda.synchronize()
        grouped_prefill_times.append((time.perf_counter() - t0) * 1000)

    # decode 计时:每次都重建 cache,否则 past 会一直变长
    mixed_decode_times = []
    for _ in range(iters):
        _, past = mixed_prefill(mgr, input_ids, attention_mask, names)
        nxt = torch.zeros(bs, 1, dtype=torch.long, device=input_ids.device)
        ms, _ = cuda_ms(lambda: mixed_decode(
            mgr, nxt, past, names, decode_lens))
        mixed_decode_times.append(ms)

    grouped_decode_times = []
    for _ in range(iters):
        _, states = grouped_prefill(mgr, input_ids, attention_mask, names)
        t0 = time.perf_counter()
        grouped_decode(mgr, states, names)
        torch.cuda.synchronize()
        grouped_decode_times.append((time.perf_counter() - t0) * 1000)

    return (
        statistics.fmean(mixed_prefill_times),
        statistics.fmean(grouped_prefill_times),
        statistics.fmean(mixed_decode_times),
        statistics.fmean(grouped_decode_times),
    )


@torch.inference_mode()
def main() -> int:
    """对每个 batch size 各跑一次 prefill + decode,汇总成表."""
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required")
        return 1

    # 固定种子:输入 token 与采样路径都不随机,保证两次运行结果可比
    torch.manual_seed(SEED)

    device = pick_device()
    dtype = pick_dtype(device)
    tokenizer = load_tokenizer(BASE)
    # 只 tokenize 一次最大 batch,各 batch size 复用前 bs 行
    max_bs = max(BATCHES)
    enc_full = tokenizer([PROMPT] * max_bs, padding=True, return_tensors="pt")
    enc_full = {k: v.to(device) for k, v in enc_full.items()}

    base = load_causal_lm(BASE, device=device, dtype=dtype)
    mgr = MultiLoRAManager.from_pretrained(
        base,
        ADAPTERS,
        max_loras_per_batch=len(ADAPTERS),
        max_lora_rank=64,  # 必须容纳 b 的原生 rank 64
        lora_backend=args.backend,
    )
    keys = list(ADAPTERS)

    try:
        print(
            f"base={BASE} device={device} dtype={dtype} backend={args.backend}\n"
            f"adapters={keys} ranks={sorted(ADAPTER_RANKS.values())} "
            f"wrapped_modules={len(mgr.lora_wrappers)} seed={SEED}"
        )
        print()

        header = (
            f"{'bs':>4} {'rank_mix':<16} {'TTFT(ms)':>10} "
            f"{'TPOT(ms)':>10} {'tok/s':>9} {'total(ms)':>10} "
            f"{'base_ttft':>10} {'base_tpot':>10} {'add_ttft':>9} {'add_tpot':>9}"
        )

        # ================= 一起跑:四组数据全部算完再打印 =================
        # 阶段 1:单批次 prefill/decode 延迟,以及关掉 LoRA 后的基线
        scan_rows = []
        for bs in sorted(BATCHES):
            names = assign_names(bs, keys)
            input_ids = enc_full["input_ids"][:bs]
            attn = enc_full["attention_mask"][:bs]
            # decode 每行只有 1 个 token,必须显式传 seq_lens 走 decode 语义
            decode_lens = torch.ones(bs, device=device, dtype=torch.int32)

            (ttft, tpot), (base_ttft, base_tpot), (add_ttft, add_tpot) = measure_both(
                mgr, input_ids, attn, names, decode_lens, measure_ttft_tpot
            )
            scan_rows.append(
                {
                    "bs": bs,
                    "mix": rank_distribution(names),
                    "ttft": ttft,
                    "tpot": tpot,
                    "tok_s": bs * 1000 / tpot if tpot > 0 else 0.0,
                    "total": ttft + NUM_DECODE * tpot,
                    "base_ttft": base_ttft,
                    "base_tpot": base_tpot,
                    "add_ttft": add_ttft,
                    "add_tpot": add_tpot,
                }
            )

        # 阶段 2:混合 vs 分组的数值一致性
        num_rows = []
        for bs in sorted(BATCHES):
            names = assign_names(bs, keys)
            rows = verify_paths(
                mgr,
                enc_full["input_ids"][:bs],
                enc_full["attention_mask"][:bs],
                names,
            )
            num_rows.append(
                {"bs": bs, "groups": len(
                    group_row_indices(names)), "stages": rows}
            )

        # 阶段 3:混合 vs 分组的耗时,开/关 LoRA 各一趟得到净耗时
        speed_rows, speed_net_rows = [], []
        for bs in sorted(BATCHES):
            names = assign_names(bs, keys)
            input_ids = enc_full["input_ids"][:bs]
            attn = enc_full["attention_mask"][:bs]
            decode_lens = torch.ones(bs, device=device, dtype=torch.int32)
            (pf, gf, pd, gd), _, (npf, ngf, npd, ngd) = measure_both(
                mgr, input_ids, attn, names, decode_lens, bench_paths
            )
            groups = len(group_row_indices(names))
            speed_rows.append(
                {"bs": bs, "groups": groups, "pf": pf, "gf": gf, "pd": pd, "gd": gd}
            )
            speed_net_rows.append(
                {"bs": bs, "groups": groups, "pf": npf,
                    "gf": ngf, "pd": npd, "gd": ngd}
            )

        evictions = mgr.pool.eviction_policy.eviction_count

        # ================= 分开打印 =================
        print(f"表1 混合批次 batch 扫描")
        print(header)
        print("-" * len(header))
        for r in scan_rows:
            mix_str = " ".join(f"r{k}:{v}" for k,
                               v in sorted(r["mix"].items()))
            print(
                f"{r['bs']:>4} {mix_str:<16} {r['ttft']:>10.2f} "
                f"{r['tpot']:>10.2f} {r['tok_s']:>9.1f} {r['total']:>10.1f} "
                f"{r['base_ttft']:>10.2f} {r['base_tpot']:>10.2f} "
                f"{r['add_ttft']:>9.2f} {r['add_tpot']:>9.2f}"
            )
        print(f"pool_total_evictions={evictions}")

        print()
        print(
            f"表2 数值一致性  混合批次 vs 分组逐个  "
            f"(abs: allclose atol={ATOL:g}   rel: mean_rel < {RTOL:g})"
        )
        num_header = (
            f"{'bs':>4} {'groups':>7} {'stage':>8} "
            f"{'max_diff':>10} {'mean_rel':>10} {'abs':>6} {'rel':>6}"
        )
        print(num_header)
        print("-" * len(num_header))
        abs_all_ok, rel_all_ok = True, True
        for r in num_rows:
            for row in r["stages"]:
                abs_all_ok = abs_all_ok and row["abs_ok"]
                rel_all_ok = rel_all_ok and row["rel_ok"]
                print(
                    f"{r['bs']:>4} {r['groups']:>7} {row['stage']:>8} "
                    f"{row['max_diff']:>10.4g} {row['mean_rel']:>10.3g} "
                    f"{'OK' if row['abs_ok'] else 'FAIL':>6} "
                    f"{'OK' if row['rel_ok'] else 'FAIL':>6}"
                )
        print(f"\n绝对精度全部通过 = {abs_all_ok}")
        print(f"相对精度全部通过 = {rel_all_ok}")

        print()
        print("表3 耗时  混合批次 vs 分组逐个 (ms)")
        sp_header = (
            f"{'bs':>4} {'groups':>7} "
            f"{'pf_mixed':>9} {'pf_group':>9} {'pf_x':>7} "
            f"{'dc_mixed':>9} {'dc_group':>9} {'dc_x':>7}"
        )
        print(sp_header)
        print("-" * len(sp_header))
        for r in speed_rows:
            print(
                f"{r['bs']:>4} {r['groups']:>7} "
                f"{r['pf']:>9.2f} {r['gf']:>9.2f} {r['gf'] / r['pf']:>6.2f}x "
                f"{r['pd']:>9.2f} {r['gd']:>9.2f} {r['gd'] / r['pd']:>6.2f}x"
            )

        print()
        print("表4 LoRA 净耗时  带 LoRA 时间 - 只跑基座的时间 (ms)")
        sp4_header = (
            f"{'bs':>4} {'groups':>7} "
            f"{'pf_lora':>9} {'pf_grp':>9} {'pf_x':>7} "
            f"{'dc_lora':>9} {'dc_grp':>9} {'dc_x':>7}"
        )
        print(sp4_header)
        print("-" * len(sp4_header))
        for r in speed_net_rows:
            print(
                f"{r['bs']:>4} {r['groups']:>7} "
                f"{r['pf']:>9.2f} {r['gf']:>9.2f} {r['gf'] / r['pf']:>6.2f}x "
                f"{r['pd']:>9.2f} {r['gd']:>9.2f} {r['gd'] / r['pd']:>6.2f}x"
            )
    finally:
        release_model(mgr.model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
