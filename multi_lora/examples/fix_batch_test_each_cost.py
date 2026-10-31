#!/usr/bin/env python3
"""
固定模型、输入长度与 batch,用 torch.profiler(CPU+CUDA、record_shapes、export_chrome_trace)
分别跑 adapter 数量 = 1 / 8 / 32 / 64 四档，每档 warmup 10 次、计时 50 次，
输出 throughput(tokens/s)、TTFT、单次 forward 的 CPU/GPU 时间占比，
以及 top-10 kernel 耗时表和 kernel launch 次数。
    python multi_lora/examples/fix_batch_test_each_cost.py
"""

import ctypes
import gc
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.profiler import DeviceType, ProfilerActivity, profile, schedule

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from multi_lora.graph_decode import DecodeStep, capture_graph  # noqa: E402
from multi_lora.hf_runner import (  # noqa: E402
    load_causal_lm,
    load_tokenizer,
    pick_device,
    pick_dtype,
)
from multi_lora.manager import MultiLoRAManager  # noqa: E402

# base model 与 make_qkvo_variants.py 生成的 128 个变体(只含 qkvo,rank 按 8/16/32/64 循环)
BASE = "Qwen/Qwen3-4B-Instruct-2507"
QKVO_VARIANTS = Path("/root/autodl-tmp/qwen3_4b_qkvo_variants")
NUM_VARIANTS = 128
TRACE_DIR = _ROOT / "traces" / "02_fix_batch_each_cost"
ADAPTERS = {f"q{i:02d}": str(QKVO_VARIANTS / f"q{i:02d}") for i in range(NUM_VARIANTS)}


def _load_adapter_ranks() -> dict:
    """从各变体的 adapter_config.json 读 rank,重新生成变体后无需改代码.

    缺变体直接退出并提示先跑 make_qkvo_variants.py。
    """
    ranks = {}
    for name, path in ADAPTERS.items():
        cfg = Path(path) / "adapter_config.json"
        if not cfg.is_file():
            raise SystemExit(
                f"adapter 变体不全({cfg} 缺失),先跑 "
                f"python multi_lora/examples/make_qkvo_variants.py "
                f"生成 {NUM_VARIANTS} 个"
            )
        ranks[name] = int(json.loads(cfg.read_text())["r"])
    return ranks


ADAPTER_RANKS = _load_adapter_ranks()
PROMPT = "user\nList three countries and their capitals.\nassistant\n"

# (K, H, steps): K=batch 内 adapter 数,H=每步换入=换出个数(H=0 是不换入对照组)。
# steps=15:kineto 事件 ~1 GiB/步(k64swap32 每步 9216 次 H2D),调大会顶爆 62 GiB;
# 表1 的 TTFT/TPOT 走 CUDA event 50 次,与 steps 无关。
TIERS = [(1, 1, 15), (8, 4, 15), (32, 16, 15), (64, 32, 15), (64, 0, 15)]
BATCH_SIZE = 64
WARMUP_ITERS = 10
TIMED_ITERS = 50
SEED = 1234
GRAPH_TPOT_ITERS = 50  # graph 端到端 TPOT 的计时步数,跟 TIMED_ITERS 对齐
REPLAY_BASE_ITERS = 25  # replay-only 基准 TPOT 步数(不换入,Graph 的纯下限)


class TierRotation:
    """adapter 轮换器:每个 rank 组一个滑动窗口,每步前进 H/4 个,
    batch 里四种 rank 恒各占 1/4(平均 rank 不变),且每步恰好换入 H 个。"""

    def __init__(self, k: int, swap: int, groups: dict):
        self.k = k
        self.swap = swap
        self.groups = groups
        self.per = k // 4  # 每个 rank 组驻留几个
        self.move = swap // 4  # 每个 rank 组每步前进几个
        self.step = 0

    def next_names(self):
        """本步 BATCH_SIZE 个名字(每行一个),不同 adapter 恰好 K 个."""
        start = self.step
        self.step += 1
        if self.k == 1:
            pool = self.groups[32]
            picks = [pool[start % len(pool)]]
        else:
            picks = []
            for rank in (8, 16, 32, 64):
                pool = self.groups[rank]
                base = start * self.move
                picks += [pool[(base + j) % len(pool)] for j in range(self.per)]
        return [picks[i % len(picks)] for i in range(BATCH_SIZE)]

    def rank_mix(self) -> str:
        """batch 的 rank 组成(表1 的 rank_mix 列)."""
        if self.k == 1:
            return "r32:1"
        return " ".join(f"r{r}:{self.per}" for r in (8, 16, 32, 64))

    def used_names(self) -> list:
        """能轮换到的全部名字;PEFT 注册即常驻显存,只把这些注册进 manager."""
        if self.k == 1:
            return list(self.groups[32])
        out = []
        for rank in (8, 16, 32, 64):
            out += self.groups[rank]
        return out


def build_rotation(k: int, swap: int) -> TierRotation:
    """每组只留窗口会用到的 K/4+H/4 个(K=1 在 r32 组取 1+H 个)."""
    need = 1 + swap if k == 1 else k // 4 + swap // 4
    groups = {
        r: [n for n, v in ADAPTER_RANKS.items() if v == r][:need]
        for r in (8, 16, 32, 64)
    }
    return TierRotation(k, swap, groups)


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
    """top-n GPU 耗时表。ProfilerStep 容器行的 GPU 时间是整窗口跨度(含空档),
    table() 又没有过滤参数,所以多取一行再删掉它。"""
    raw = prof.key_averages().table(sort_by="self_device_time_total", row_limit=n + 1)
    return "\n".join(line for line in raw.splitlines() if "ProfilerStep" not in line)


def profile_window(fn, tag, outdir, steps=TIMED_ITERS):
    """profile steps 次 fn(),返回单次 forward 的构成(均 ÷steps):
    cpu_ms / wait_ms / gpu_ms / launches。wait = CPU 在 cuda*Sync 里干等 GPU
    的时间,不计入 cpu_ms。产出 {tag}.txt + {tag}.json.gz(Perfetto 可直接开)。"""
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=WARMUP_ITERS, active=steps, repeat=1),
        record_shapes=True,
    ) as prof:
        for _ in range(WARMUP_ITERS + steps):
            fn()
            torch.cuda.synchronize()
            prof.step()

    gpu_us = cpu_us = wait_us = 0.0
    launches = 0
    waiting = ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize")
    for ev in prof.events() or []:
        # ProfilerStep 容器的 GPU 时间是整窗口跨度,计进来会重复加 busy
        if ev.name.startswith("ProfilerStep"):
            continue
        if ev.device_type == DeviceType.CUDA:
            gpu_us += ev.self_device_time_total
            if "Memcpy" not in ev.name and "Memset" not in ev.name:
                launches += ev.count
        elif ev.name.startswith(waiting):
            wait_us += ev.self_cpu_time_total
        else:
            cpu_us += ev.self_cpu_time_total

    top10 = top_kernel_table(prof)
    (outdir / f"{tag}.txt").write_text(top10)
    prof.export_chrome_trace(str(outdir / f"{tag}.json.gz"))

    # kineto 事件表离开 with 块不还内存,不主动 gc+trim 的话窗口间会累积几十 GiB
    del prof
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)

    return {
        "cpu_ms": cpu_us / 1000 / steps,
        "wait_ms": wait_us / 1000 / steps,
        "gpu_ms": gpu_us / 1000 / steps,
        "gpu_pct": gpu_us / (gpu_us + cpu_us) * 100,
        "launches": launches / steps,
        "top10": top10,
    }


@torch.inference_mode()
def run_one_tier(mgr, enc, rotation, outdir, label, steps=TIMED_ITERS, with_plain=True):
    """阶段A 用 CUDA event 测 TTFT/TPOT,再开 prefill(/plain) profiler 窗口。
    每步 forward 都从 rotation 取新的一组,warmup 也在真换入。
    with_plain=False 跳过 plain 窗口(graph 对比场景下 gplain 才是 graph 的对照)。"""
    decode_lens = [1] * BATCH_SIZE  # decode 每段 1 token

    def prefill():
        return mgr.forward_with_lora(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            lora_names=rotation.next_names(),
            seq_lens=[int(enc["input_ids"].shape[1])] * BATCH_SIZE,
            use_cache=True,
        )

    def decode(ids, past):
        return mgr.forward_with_lora(
            input_ids=ids,
            attention_mask=None,
            seq_lens=decode_lens,
            lora_names=rotation.next_names(),
            past_key_values=past,
            use_cache=True,
        )

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

    # B-1: prefill 窗口
    res = {"prefill": profile_window(prefill, f"prefill_{label}", outdir, steps)}

    # B-2: plain 窗口(StaticCache,每步真写 KV)。
    # PEFT 释放后的显存缓存先还给驱动再建 KV;KV 不能扩,按 WARMUP+steps 开
    if with_plain:
        torch.cuda.empty_cache()
        seq = int(enc["input_ids"].shape[1])
        step = DecodeStep.new(mgr, BATCH_SIZE, seq, seq + WARMUP_ITERS + steps + 4)
        step.input_ids.copy_(
            step.prefill(
                mgr, enc["input_ids"], enc["attention_mask"], rotation.next_names()
            )
        )

        def plain_decode():
            mgr._setup_forward_batch(rotation.next_names(), seq_lens=decode_lens)
            return step.run(mgr)

        res["plain"] = profile_window(plain_decode, f"plain_{label}", outdir, steps)

    tpot = statistics.fmean(tpots)
    return {
        "n": rotation.k,
        "mix": rotation.rank_mix(),
        "label": label,
        "ttft": statistics.fmean(ttfts),
        "tpot": tpot,
        "tok_s": BATCH_SIZE * 1000 / tpot,
        **res,
    }


@torch.inference_mode()
def run_graph_tier(mgr, enc, rotation, outdir, label, steps=TIMED_ITERS):
    """CUDA Graph 一档:prefill → 录制 → 换入版 TPOT → graph / gplain 两窗口。

    每步 graph 外 _setup_forward_batch(新的一组) 再 replay:常驻 metadata 地址
    不变、原地改写,graph 不用重录,换入的 H2D 全在 graph 外。gplain 是同样的
    每步换入 + 写 KV 但不录 graph 的对照。swap=0 时退化为固定名字的旧行为。
    高 swap 档 steps 必须 ~15,理由同 TIERS 注释。"""
    start = int(enc["input_ids"].shape[1])  # prefill 长度,也是 KV 起点
    # KV 不能扩:录制1+基准25+TPOT(10+50) + 两个窗口(10+steps) + 彩排/余量
    max_len = (start + 4 + REPLAY_BASE_ITERS + GRAPH_TPOT_ITERS
               + 2 * (WARMUP_ITERS + steps) + 8)
    decode_lens = [1] * BATCH_SIZE

    # 1) prefill 填 KV,首 token 当输入;metadata 先写一次(段长恒 1)
    step = DecodeStep.new(mgr, BATCH_SIZE, start, max_len)
    first_token = step.prefill(
        mgr, enc["input_ids"], enc["attention_mask"], rotation.next_names()
    )
    step.input_ids.copy_(first_token)
    mgr._setup_forward_batch(rotation.next_names(), seq_lens=decode_lens)

    # 2) 录制(内部含彩排);out.logits 每次 replay 都会被刷
    graph, out = capture_graph(mgr, step)
    torch.cuda.empty_cache()  # 彩排的 KV 没人引用了,还显存再建对照

    # 3a) replay-only 基准 TPOT(不换入):Graph 能给的纯下限
    for _ in range(WARMUP_ITERS):
        graph.replay()
    base_tpots = [cuda_ms(graph.replay)[0] for _ in range(REPLAY_BASE_ITERS)]

    # 3b) 端到端 TPOT = 换入 + replay + 采样
    def swap_replay_and_sample():
        mgr._setup_forward_batch(rotation.next_names(), seq_lens=decode_lens)
        graph.replay()
        step.input_ids.copy_(out.logits[:, -1].argmax(dim=-1, keepdim=True))

    for _ in range(WARMUP_ITERS):
        swap_replay_and_sample()
    tpots = [cuda_ms(swap_replay_and_sample)[0] for _ in range(GRAPH_TPOT_ITERS)]

    # 4) 窗口 = 换入(H2D 在 graph 外)+ replay,一起计时
    def swap_and_replay():
        mgr._setup_forward_batch(rotation.next_names(), seq_lens=decode_lens)
        graph.replay()

    graph_res = profile_window(swap_and_replay, f"graph_{label}", outdir, steps)
    graph_res["tpot"] = statistics.fmean(tpots)
    graph_res["tpot_replay_only"] = statistics.fmean(base_tpots)

    # 5) gplain 对照:与 graph 同 KV 同换入,只差不录 graph
    plain = DecodeStep.new(mgr, BATCH_SIZE, start, max_len)
    plain.prefill(mgr, enc["input_ids"], enc["attention_mask"], rotation.next_names())

    def plain_decode():
        mgr._setup_forward_batch(rotation.next_names(), seq_lens=decode_lens)
        return plain.run(mgr)

    plain_res = profile_window(plain_decode, f"gplain_{label}", outdir, steps)

    return {"graph": graph_res, "gplain": plain_res}


def print_table(title, header, lines):
    print(f"\n{title}\n{header}\n" + "-" * len(header))
    for line in lines:
        print(line)


def print_report(rows):
    """三张表:延迟吞吐 / 单次 forward CPU-GPU 占比 / top-10 kernel."""
    print_table(
        "表1 延迟与吞吐",
        f"{'tier':>11} {'rank_mix':<16} {'TTFT(ms)':>10} "
        f"{'TPOT(ms)':>10} {'tok/s':>9}",
        [
            f"{r['label']:>11} {r['mix']:<16} {r['ttft']:>10.2f} "
            f"{r['tpot']:>10.2f} {r['tok_s']:>9.1f}"
            for r in rows
        ],
    )

    stages = []
    for r in rows:
        for stage in ("prefill", "plain", "gplain", "graph"):
            if not r.get(stage):
                continue
            s = r[stage]
            stages.append(
                f"{r['label']:>11} {stage:>8} {s['cpu_ms']:>9.3f} {s['wait_ms']:>9.3f} "
                f"{s['gpu_ms']:>9.3f} {s['gpu_pct']:>6.1f}% "
                f"{s['launches']:>12.1f}"
            )
    print_table(
        "表2 单次 forward 的 CPU/GPU 占比(wait = CPU 干等 GPU 的时间,不计入 CPU;"
        "plain 是真实 decode,graph 是同一份 decode 录成 graph)",
        f"{'tier':>11} {'stage':>8} {'CPU(ms)':>9} {'wait(ms)':>9} "
        f"{'GPU(ms)':>9} {'GPU%':>7} {'launch/step':>12}",
        stages,
    )

    for r in rows:
        for stage in ("prefill", "plain", "gplain", "graph"):
            if not r.get(stage):
                continue
            print(f"\n表3 tier={r['label']} {stage} top-10 kernel")
            print(r[stage]["top10"])


def release_tier(mgr) -> None:
    """档末回收:清 weight store、还显存缓存,malloc_trim 把 glibc 空闲页还给 OS."""
    mgr._weight_store.clear()
    gc.collect()
    torch.cuda.empty_cache()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def load_enc(device, dtype):
    enc = load_tokenizer(BASE)([PROMPT] * BATCH_SIZE, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def run_tier(k: int, swap: int, steps: int) -> dict:
    """跑一档:加载 → run_one_tier 两窗口 → 回收。只注册 rotation 会用到的 adapter。"""
    torch.manual_seed(SEED)
    outdir = TRACE_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    dtype = pick_dtype(device)
    enc = load_enc(device, dtype)

    label = f"k{k}swap{swap}"
    rotation = build_rotation(k, swap)
    used = {n: ADAPTERS[n] for n in rotation.used_names()}
    base_model = load_causal_lm(BASE, device=device, dtype=dtype)
    print(f"  {label} 模型加载完")
    mgr = MultiLoRAManager.from_pretrained(
        base_model,
        used,
        # 槽位数 = 本档 adapter 数,否则有空槽就不驱逐,测不到换出
        max_loras_per_batch=k,
        max_lora_rank=max(ADAPTER_RANKS.values()),
        lora_backend="triton",
    )
    print(f"  {label} pool 建完,注册 {len(used)} 个 adapter")
    try:
        return run_one_tier(mgr, enc, rotation, outdir, label, steps)
    finally:
        release_tier(mgr)


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required")
        return 1

    # 单进程连跑五档:窗口末 gc+trim + 档末 release_tier 保证不累积(实测峰值 29.7 GiB)
    rows = []
    for k, swap, steps in TIERS:
        print(f"=== k{k}swap{swap} ===")
        rows.append(run_tier(k, swap, steps))
        print(f"done k{k}swap{swap}")

    print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
