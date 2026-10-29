#!/usr/bin/env python3
"""
固定模型、输入长度与 batch,用 torch.profiler(CPU+CUDA、record_shapes、export_chrome_trace)
分别跑 adapter 数量 = 1 / 8 / 32 / 64 四档，每档 warmup 10 次、计时 50 次，
输出 throughput(tokens/s)、TTFT、单次 forward 的 CPU/GPU 时间占比，
以及 top-10 kernel 耗时表和 kernel launch 次数。

Example::

    python multi_lora/examples/fix_batch_test_each_cost.py
"""

import pdb
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import DeviceType, ProfilerActivity, profile, schedule

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from multi_lora.hf_runner import (  # noqa: E402
    load_causal_lm, load_tokenizer, pick_device, pick_dtype, release_model,
)
from multi_lora.manager import MultiLoRAManager  # noqa: E402

# base mode
BASE = "Qwen/Qwen3-4B-Instruct-2507"
QKVO_VARIANTS = Path("/root/autodl-tmp/qwen3_4b_qkvo_variants")
TRACE_DIR = _ROOT / "traces" / "02_fix_batch_each_cost"

# adapter model: make_qkvo_variants.py 生成的 64 个变体, 只含 qkvo, rank 按 8/16/32/64 循环
ADAPTERS = {f"q{i:02d}": str(QKVO_VARIANTS / f"q{i:02d}") for i in range(64)}

# {'q61': '/root/autodl-tmp/qwen3_4b_qkvo_variants/q61',
# 'q62': '/root/autodl-tmp/qwen3_4b_qkvo_variants/q62',
# 'q63': '/root/autodl-tmp/qwen3_4b_qkvo_variants/q63'}


# rank 直接从各 checkpoint 的 config 读,重新生成变体后不用改这里
ADAPTER_RANKS = {
    name: int(json.loads(
        (Path(path) / "adapter_config.json").read_text())["r"])
    for name, path in ADAPTERS.items()
}

# prompt
PROMPT = "user\nList three countries and their capitals.\nassistant\n"

# 测试不同得adapter
ADAPTER_COUNTS = [1, 8, 32, 64]
# n=1 档只用 1 个 adapter, 默认会拿到 keys[0]=q00(r8);
# 而另三档都是 8/16/32/64 各占 1/4(平均 rank 30), rank 差一截就没法比.
# 所以这一档改用 r32 的 q02, 让四档的总 rank 都对齐.
SOLO_ADAPTER = "q02"
BATCH_SIZE = 64
WARMUP_ITERS = 10
TIMED_ITERS = 50
SEED = 1234


"""
keys 顺序 = q00(r8), q01(r16), q02(r32), q03(r64), q04(r8), ...

k=1 → 特例走 SOLO_ADAPTER → ["q02","q02",...]   → 整 batch 1 个 adapter(rank 32)
k=2 → i%2 = 0,1,0,1… → ["q00","q01","q00",...] → 2 个,各 32 行
k=3 → i%3 = 0,1,2,0,1,2… → ["q00","q01","q02","q00",...] → 3 个,各 22/21/21 行
k=4 → i%4 循环 → ["q00","q01","q02","q03",...] → 4 个,rank 8/16/32/64 各 16 行
k=8/32/64 → rank 8/16/32/64 始终各占 1/4, 平均 rank 都是 30
"""


def assign_names(k, keys):
    """循环取前 k 个名字填满 batch,使 batch 内不同 adapter 数恰好是 k."""
    if k == 1:
        return [SOLO_ADAPTER] * BATCH_SIZE
    return [keys[i % k] for i in range(BATCH_SIZE)]


def cuda_ms(fn):
    """CUDA event 测一次 fn() 的 GPU 毫秒,返回 (ms, fn 的返回值)."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def top_kernel_table(prof, n=10):
    """top-n GPU 耗时表;剔除 ProfilerStep 容器行.

    key_averages().table() 没有过滤参数,而容器行的 self_device_time = 整窗口 GPU 跳度
    (含空档),会顶到第一名且百分比超 100%. 所以多取一行再删掉它,保证真 top-n 不变短.
    其余行的百分比是 kineto 按去重后的 busy 算的(559.637/975.292=57.38%), 无需修正.
    """
    raw = prof.key_averages().table(sort_by="self_device_time_total", row_limit=n + 1)
    return "\n".join(line for line in raw.splitlines()
                     if "ProfilerStep" not in line)


def profile_window(fn, tag, outdir):
    """profile TIMED_ITERS 次 fn(),返回单次 forward 的 CPU/GPU 构成.

    输出口径(都是 ÷TIMED_ITERS 的单次值):
      cpu_ms   = CPU 上真正在跑算子的时间(各 CPU 算子 self 相加,不含容器空档),
                 所以它小于墙钟: 实测 60.86 + 空档 40.94 = 页脚墙钟 101.80
      gpu_ms   = GPU 上真正在跑 kernel 的时间(逐条 kernel 时长,无容器重复)
      gpu_pct  = gpu_ms / (gpu_ms + cpu_ms),两边都在忙时 GPU 占的份额;
                 想看"GPU 在墙钟里占多少"要看 analyze_trace 的 gap%
      launches = 单步 kernel 数(Memcpy/Memset 不计入)

    tag: 输出文件名前缀,格式为 "{阶段}_n{档位数}"。
         阶段 = prefill 或 decode; n 后的数字 = 该 batch 内不同 adapter 的个数
         (ADAPTER_COUNTS 里的 1/8/32/64),不是 rank。
         例: "prefill_n8" -> 一次 prefill、batch 内混 8 个 adapter 的窗口,
         产出 prefill_n8.txt(表格) + prefill_n8.json.gz(gzip 压缩后的 chrome trace,
         Perfetto 直接打开这个 .gz 就行),各档各阶段互不覆盖。
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=WARMUP_ITERS,
                          active=TIMED_ITERS, repeat=1),
        record_shapes=True,
    ) as prof:
        for _ in range(WARMUP_ITERS + TIMED_ITERS):
            fn()
            torch.cuda.synchronize()
            prof.step()

    gpu_us = cpu_us = 0.0
    launches = 0
    for ev in prof.events() or []:
        # schedule 打的 ProfilerStep 容器标记要排除:它自带的 GPU 时间 = 整窗口跨度(含空档),
        # 计进来会把真 busy 重复加一遍(实测 50 步多算 4511ms)
        if ev.name.startswith("ProfilerStep"):
            continue
        if ev.device_type == DeviceType.CUDA:
            gpu_us += ev.self_device_time_total
            if "Memcpy" not in ev.name and "Memset" not in ev.name:
                launches += ev.count
        else:
            cpu_us += ev.self_cpu_time_total

    top10 = top_kernel_table(prof)
    (outdir / f"{tag}.txt").write_text(top10)
    prof.export_chrome_trace(str(outdir / f"{tag}.json.gz"))

    return {
        "cpu_ms": cpu_us / 1000 / TIMED_ITERS,
        "gpu_ms": gpu_us / 1000 / TIMED_ITERS,
        "gpu_pct": gpu_us / (gpu_us + cpu_us) * 100,
        "launches": launches / TIMED_ITERS,
        "top10": top10,
    }


@torch.inference_mode()
def run_one_tier(mgr, enc, names, device, outdir, num_adapters,
                 stages=("prefill", "decode")):
    """跑一档:先 event 测延迟,再按 stages 开 profiler 窗口.

    stages: 要 profile 哪几个阶段,默认两个都开(四档全量跑就是这个):
            只关心 decode 时传 ("decode",),就少开一个窗口、少写一个 json.gz,
            延迟测量(TTFT/TPOT)不受影响。
    """
    decode_lens = torch.ones(BATCH_SIZE, device=device, dtype=torch.int32)

    def prefill():
        return mgr.forward_with_lora(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            lora_names=names, use_cache=True)

    def decode(ids, past):
        return mgr.forward_with_lora(
            input_ids=ids, attention_mask=None, seq_lens=decode_lens,
            lora_names=names, past_key_values=past, use_cache=True)

    def next_token(out):
        return out.logits[:, -1].argmax(dim=-1, keepdim=True)

    # 阶段 A: 测 TTFT / TPOT
    for _ in range(WARMUP_ITERS):
        out = prefill()
        decode(next_token(out), out.past_key_values)

    ttfts, tpots = [], []
    for _ in range(TIMED_ITERS):
        ms, out = cuda_ms(prefill)
        ttfts.append(ms)
        ids = next_token(out)
        ms, _ = cuda_ms(lambda: decode(ids, out.past_key_values))
        tpots.append(ms)

    # 阶段 B: 分别对 prefill 和 decode 各开一个 profiler 窗口。
    # 文件名里的 n 就是本档的 adapter 数量(num_adapters),例如 n=8 表示这个 batch 混了 8 个不同 adapter。
    # decode 窗口复用窗口外预先建好的同一份 cache, 并且每步 use_cache=False 不往里追加,
    # 这样每步 decode 的输入形状都相同, 也不会把 prefill 的耗时混进来
    res = {}
    if "prefill" in stages:
        res["prefill"] = profile_window(
            prefill, f"prefill_n{num_adapters}", outdir)
    # 真的 prefill 一次
    out = prefill()
    # 它输出的下一个 token,当作 decode 的输入
    ids = next_token(out)
    if "decode" in stages:
        res["decode"] = profile_window(
            lambda: mgr.forward_with_lora(
                input_ids=ids, attention_mask=None, seq_lens=decode_lens,
                lora_names=names, past_key_values=out.past_key_values,
                use_cache=False),
            f"decode_n{num_adapters}", outdir)

    mix = " ".join(f"r{r}:{c}" for r, c in
                   sorted(Counter(ADAPTER_RANKS[n] for n in names).items()))
    tpot = statistics.fmean(tpots)
    return {
        "n": num_adapters, "mix": mix,
        "ttft": statistics.fmean(ttfts), "tpot": tpot,
        "tok_s": BATCH_SIZE * 1000 / tpot,
        **res,
    }


def print_table(title, header, lines):
    print(f"\n{title}\n{header}\n" + "-" * len(header))
    for line in lines:
        print(line)


def print_report(rows):
    """打印三张表: 延迟吞吐 / CPU-GPU 占比 / top-10 kernel."""
    print_table(
        "表1 延迟与吞吐",
        f"{'adapters':>8} {'rank_mix':<16} {'TTFT(ms)':>10} "
        f"{'TPOT(ms)':>10} {'tok/s':>9}",
        [f"{r['n']:>8} {r['mix']:<16} {r['ttft']:>10.2f} "
         f"{r['tpot']:>10.2f} {r['tok_s']:>9.1f}" for r in rows])

    stages = []
    for r in rows:
        for stage in ("prefill", "decode"):
            if not r.get(stage):
                continue
            s = r[stage]
            stages.append(
                f"{r['n']:>8} {stage:>8} {s['cpu_ms']:>9.3f} "
                f"{s['gpu_ms']:>9.3f} {s['gpu_pct']:>6.1f}% "
                f"{s['launches']:>12.1f}")
    print_table(
        "表2 单次 forward 的 CPU/GPU 占比",
        f"{'adapters':>8} {'stage':>8} {'CPU(ms)':>9} {'GPU(ms)':>9} "
        f"{'GPU%':>7} {'launch/step':>12}",
        stages)

    for r in rows:
        for stage in ("prefill", "decode"):
            if not r.get(stage):
                continue
            print(f"\n表3 adapters={r['n']} {stage} top-10 kernel")
            print(r[stage]["top10"])


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required")
        return 1

    torch.manual_seed(SEED)
    outdir = TRACE_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    dtype = pick_dtype(device)
    enc = load_tokenizer(BASE)(
        [PROMPT] * BATCH_SIZE, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    mgr = MultiLoRAManager.from_pretrained(
        load_causal_lm(BASE, device=device, dtype=dtype),
        ADAPTERS,
        # 池子槽位数 = 最大档位的 adapter 数;rank 上限 = 变体的实际 rank
        max_loras_per_batch=max(ADAPTER_COUNTS),
        max_lora_rank=max(ADAPTER_RANKS.values()),
        lora_backend="triton",
    )

    keys = list(ADAPTERS)

    try:
        rows = [run_one_tier(mgr, enc, assign_names(n, keys), device, outdir, n)
                for n in ADAPTER_COUNTS]
    finally:
        release_model(mgr.model)

    print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
