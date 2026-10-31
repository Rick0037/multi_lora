#!/usr/bin/env python3
"""
固定模型、输入长度与 batch,用 torch.profiler(CPU+CUDA、record_shapes、export_chrome_trace)
分别跑 adapter 数量 = 1 / 8 / 32 / 64 四档，每档 warmup 10 次、计时 50 次，
输出 throughput(tokens/s)、TTFT、单次 forward 的 CPU/GPU 时间占比，
以及 top-10 kernel 耗时表和 kernel launch 次数。

Example::

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
    load_causal_lm, load_tokenizer, pick_device, pick_dtype,
)
from multi_lora.manager import MultiLoRAManager  # noqa: E402

# base mode
BASE = "Qwen/Qwen3-4B-Instruct-2507"
QKVO_VARIANTS = Path("/root/autodl-tmp/qwen3_4b_qkvo_variants")
NUM_VARIANTS = 128
TRACE_DIR = _ROOT / "traces" / "02_fix_batch_each_cost"

# adapter model: make_qkvo_variants.py 生成的 128 个变体, 只含 qkvo, rank 按 8/16/32/64 循环
ADAPTERS = {f"q{i:02d}": str(QKVO_VARIANTS / f"q{i:02d}")
            for i in range(NUM_VARIANTS)}

# {'q61': '/root/autodl-tmp/qwen3_4b_qkvo_variants/q61',
# 'q62': '/root/autodl-tmp/qwen3_4b_qkvo_variants/q62',
# 'q63': '/root/autodl-tmp/qwen3_4b_qkvo_variants/q63'}


def _load_adapter_ranks() -> dict:
    """rank 直接从各 checkpoint 的 config 读,重新生成变体后不用改这里.

    变体是 make_qkvo_variants.py 的产物,缺任何一个都直接退出并告诉补法,
    不带着不全的变体往模型加载走。
    """
    ranks = {}
    for name, path in ADAPTERS.items():
        cfg = Path(path) / "adapter_config.json"
        if not cfg.is_file():
            raise SystemExit(
                f"adapter 变体不全({cfg} 缺失),先跑 "
                f"python multi_lora/examples/make_qkvo_variants.py "
                f"生成 {NUM_VARIANTS} 个")
        ranks[name] = int(json.loads(cfg.read_text())["r"])
    return ranks


ADAPTER_RANKS = _load_adapter_ranks()

# prompt
PROMPT = "user\nList three countries and their capitals.\nassistant\n"

# 每档是 (batch 内 adapter 数 K, 每步换入=换出的个数 H, profiler 记录步数)。
# H=0 是不换入的对照组。记录步数取 15:实测 profiler 约 1 GiB/步
# (每次 pageable H2D 泄 5 条事件,k64swap32 每步 9216 次),25 步单窗口就顶 26 GiB。
# 表1 的 TTFT/TPOT 走 CUDA event 计时,跟这个数无关,仍是 50 次。
TIERS = [(1, 1, 15), (8, 4, 15), (32, 16, 15), (64, 32, 15), (64, 0, 15)]
BATCH_SIZE = 64
WARMUP_ITERS = 10
TIMED_ITERS = 50
SEED = 1234
GRAPH_TPOT_ITERS = 50      # graph 端到端 TPOT 的计时步数,跟 TIMED_ITERS 对齐


class TierRotation:
    """一档的 adapter 轮换器:每步给一组 K 个,平均 rank 与不换入时完全一致.

    关键是每个 rank 各自一个滑动窗口:组内每次前进 H/4 个,所以
    batch 里 8/16/32/64 始终各占 1/4(平均 rank 不变),同时每步相对上一步
    正好换掉 H 个,保证真的发生换入换出。
    K=1 时只从 r32 组里取 1 个,平均 rank 仍是 32。
    """

    def __init__(self, k: int, swap: int, groups: dict):
        self.k = k
        self.swap = swap
        self.groups = groups
        self.per = k // 4        # 每个 rank 组驻留几个
        self.move = swap // 4    # 每个 rank 组每步前进几个
        self.step = 0

    def next_names(self):
        """本步一组共 BATCH_SIZE 个名字(每行一个),其中不同 adapter 恰好 K 个.

        先按 rank 组取窗口里的 K 个,再循环铺到 64 行,顺序固定为 rank 8/16/32/64。
        """
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
                picks += [pool[(base + j) % len(pool)]
                          for j in range(self.per)]
        return [picks[i % len(picks)] for i in range(BATCH_SIZE)]

    def rank_mix(self) -> str:
        """本档 batch 里的 rank 组成,实量上与不换入时一致."""
        if self.k == 1:
            return "r32:1"
        return " ".join(f"r{r}:{self.per}" for r in (8, 16, 32, 64))

    def used_names(self) -> list:
        """本档轮换器能取到的全部名字 —— 也只把这些注册进 manager.

        PEFT 会把每个注册的 adapter 常驻显存,多注册一个就多占一份。
        """
        if self.k == 1:
            return list(self.groups[32])
        out = []
        for rank in (8, 16, 32, 64):
            out += self.groups[rank]
        return out


def build_rotation(k: int, swap: int) -> TierRotation:
    """按 rank 分好组的轮换器,每组只留本档会用到的那些.

    驻留 K/4 个 + 每步前进 H/4 个,窗口在这几个里循环,所以每组只需 K/4+H/4 个;
    K=1 只在 r32 组里逐个前进,取 1+H 个。
    """
    need = 1 + swap if k == 1 else k // 4 + swap // 4
    groups = {r: [n for n, v in ADAPTER_RANKS.items() if v == r][:need]
              for r in (8, 16, 32, 64)}
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
    """top-n GPU 耗时表;剔除 ProfilerStep 容器行.

    key_averages().table() 没有过滤参数,而容器行的 self_device_time = 整窗口 GPU 跳度
    (含空档),会顶到第一名且百分比超 100%. 所以多取一行再删掉它,保证真 top-n 不变短.
    其余行的百分比是 kineto 按去重后的 busy 算的(559.637/975.292=57.38%), 无需修正.
    """
    raw = prof.key_averages().table(sort_by="self_device_time_total", row_limit=n + 1)
    return "\n".join(line for line in raw.splitlines()
                     if "ProfilerStep" not in line)


def profile_window(fn, tag, outdir, steps=TIMED_ITERS):
    """profile steps 次 fn(),返回单次 forward 的 CPU/GPU 构成.

    输出口径(都是 ÷steps 的单次值):
      cpu_ms   = CPU 上真正在跑算子的时间(各 CPU 算子 self 相加,不含容器空档),
                 所以它小于墙钟: 实测 60.86 + 空档 40.94 = 页脚墙钟 101.80
      wait_ms  = CPU 在同步调用里干等 GPU 的时间(cudaDeviceSynchronize 等),不计入 cpu_ms
      gpu_ms   = GPU 上真正在跑 kernel 的时间(逐条 kernel 时长,无容器重复)
      gpu_pct  = gpu_ms / (gpu_ms + cpu_ms),两边都在忙时 GPU 占的份额;
                 想看"GPU 在墙钟里占多少"要看 analyze_trace 的 gap%
      launches = 单步 kernel 数(Memcpy/Memset 不计入)

    tag: 输出文件名前缀,格式为 "{阶段}_{档}"。
         阶段 = prefill 或 plain; 档形如 k8swap4 = batch 内 8 个 adapter、每步换入 4 个。
         例: "prefill_k8swap4" -> 一次 prefill 窗口,
         产出 prefill_k8swap4.txt(表格) + prefill_k8swap4.json.gz(gzip 压缩后的 chrome trace,
         Perfetto 直接打开这个 .gz 就行),各档各阶段互不覆盖。
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=WARMUP_ITERS,
                          active=steps, repeat=1),
        record_shapes=True,
    ) as prof:
        for _ in range(WARMUP_ITERS + steps):
            fn()
            torch.cuda.synchronize()
            prof.step()

    gpu_us = cpu_us = wait_us = 0.0
    launches = 0
    # 同步等待不算 CPU 干活,单独记一列
    waiting = ("cudaDeviceSynchronize", "cudaStreamSynchronize",
               "cudaEventSynchronize")
    for ev in prof.events() or []:
        # schedule 打的 ProfilerStep 容器标记要排除:它自带的 GPU 时间 = 整窗口跨度(含空档),
        # 计进来会把真 busy 重复加一遍(实测 50 步多算 4511ms)
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

    # kineto 的事件表是 C++ 小对象,离开 with 块不会自己还给 OS(实测退出后仍占 2.7 GiB,
    # gc+trim 后回到 0.8)。不主动清的话,一档里两个窗口会累积到几十 GiB。
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
def run_one_tier(mgr, enc, rotation, outdir, label, steps=TIMED_ITERS):
    """跑一档:先 event 测 TTFT/TPOT,再给 prefill 与 plain 各开一个 profiler 窗口.

    每次 forward 前都从 rotation 取新的一组 adapter,所以 warmup 与计时阶段
    都在真换入换出。plain 是真实 decode(每步写 KV),不录 graph。
    """
    # decode 每段 1 个 token;CPU 上的 list 即可,不用建 GPU 张量
    decode_lens = [1] * BATCH_SIZE

    def prefill():
        return mgr.forward_with_lora(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            lora_names=rotation.next_names(),
            seq_lens=[int(enc["input_ids"].shape[1])] * BATCH_SIZE,
            use_cache=True)

    def decode(ids, past):
        return mgr.forward_with_lora(
            input_ids=ids, attention_mask=None, seq_lens=decode_lens,
            lora_names=rotation.next_names(), past_key_values=past,
            use_cache=True)

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

    # 阶段 B-1: prefill 窗口。文件名里的 label 形如 k8swap4 = 本档 8 个 adapter、每步换入 4 个
    res = {"prefill": profile_window(
        prefill, f"prefill_{label}", outdir, steps)}

    # 阶段 B-2: plain 窗口。用 StaticCache 让每步形状一致,但每步真写 KV 且换入新 adapter
    # from_pretrained 中途释掉的 PEFT 权重还被 torch 缓存着(实测 reserved 比 allocated
    # 多 8 GiB),先还给驱动,免得建 KV 时贴边
    torch.cuda.empty_cache()
    seq = int(enc["input_ids"].shape[1])
    # 每步多写一格 KV 且不能扩:窗口只跑 WARMUP_ITERS + steps 步
    step = DecodeStep.new(mgr, BATCH_SIZE, seq,
                          seq + WARMUP_ITERS + steps + 4)
    step.input_ids.copy_(step.prefill(
        mgr, enc["input_ids"], enc["attention_mask"], rotation.next_names()))

    def plain_decode():
        mgr._setup_forward_batch(rotation.next_names(), seq_lens=decode_lens)
        return step.run(mgr)

    res["plain"] = profile_window(
        plain_decode, f"plain_{label}", outdir, steps)

    tpot = statistics.fmean(tpots)
    return {
        "n": rotation.k, "mix": rotation.rank_mix(),
        "label": label,
        "ttft": statistics.fmean(ttfts), "tpot": tpot,
        "tok_s": BATCH_SIZE * 1000 / tpot,
        **res,
    }


@torch.inference_mode()
def run_graph_tier(mgr, enc, names, outdir, num_adapters):
    """真实 decode 的 CUDA Graph 一档:prefill → 录制 → 测 TPOT → profile.

    plain 与 graph 两个窗口跑的完全是一样的 decode(每步真写 KV),
    唯一区别是有没有录成 graph。
    """
    start = int(enc["input_ids"].shape[1])   # prefill 长度,也是 KV 起点
    # 每步都往 KV 写一格而 KV 不能扩:录制 1 + TPOT(10+50)
    # + graph 窗口(10+50) + 不录 graph 的对照窗口(10+50) + 彩排/余量
    max_len = (start + 4 + GRAPH_TPOT_ITERS
               + 2 * (WARMUP_ITERS + TIMED_ITERS) + 8)

    # 1) 建常驻张量,prefill 填 KV,首 token 当第一个输入
    step = DecodeStep.new(mgr, BATCH_SIZE, start, max_len)
    first_token = step.prefill(
        mgr, enc["input_ids"], enc["attention_mask"], names)
    step.input_ids.copy_(first_token)
    # metadata 在 graph 外只写一次:decode 每步的段长与槽位映射都不变
    mgr._setup_forward_batch(names, seq_lens=[1] * BATCH_SIZE)

    # 2) 录制(内部含彩排);out 里的 logits 每次 replay 都会被刷
    graph, out = capture_graph(mgr, step)
    # 彩排用的那套 KV 已经没人引用了,把显存交回驱动再建下面的 plain
    torch.cuda.empty_cache()

    # 3) 端到端 TPOT:replay + 采样,跟表1 同口径(profiler 窗口里没有采样)
    def replay_and_sample():
        graph.replay()
        logits = out.logits
        step.input_ids.copy_(logits[:, -1].argmax(dim=-1, keepdim=True))

    for _ in range(WARMUP_ITERS):
        replay_and_sample()
    tpots = [cuda_ms(replay_and_sample)[0] for _ in range(GRAPH_TPOT_ITERS)]

    # 4) profiler 窗口测 replay 本身
    graph_res = profile_window(graph.replay, f"graph_n{num_adapters}", outdir)
    graph_res["tpot"] = statistics.fmean(tpots)

    # 5) 对照窗口:同样每步真写 KV,只是不录 graph
    plain = DecodeStep.new(mgr, BATCH_SIZE, start, max_len)
    plain.prefill(mgr, enc["input_ids"], enc["attention_mask"], names)
    mgr._setup_forward_batch(names, seq_lens=[1] * BATCH_SIZE)
    plain_res = profile_window(
        lambda: plain.run(mgr), f"plain_n{num_adapters}", outdir)

    return {"graph": graph_res, "plain": plain_res}


def print_table(title, header, lines):
    print(f"\n{title}\n{header}\n" + "-" * len(header))
    for line in lines:
        print(line)


def print_report(rows):
    """打印三张表: 延迟吞吐 / CPU-GPU 占比 / top-10 kernel.

    第一列 k32swap16 = 本档 batch 内 32 个 adapter、每步换入换出 16 个。
    """
    print_table(
        "表1 延迟与吞吐",
        f"{'tier':>11} {'rank_mix':<16} {'TTFT(ms)':>10} "
        f"{'TPOT(ms)':>10} {'tok/s':>9}",
        [f"{r['label']:>11} {r['mix']:<16} {r['ttft']:>10.2f} "
         f"{r['tpot']:>10.2f} {r['tok_s']:>9.1f}" for r in rows])

    stages = []
    for r in rows:
        for stage in ("prefill", "plain", "graph"):
            if not r.get(stage):
                continue
            s = r[stage]
            stages.append(
                f"{r['label']:>11} {stage:>8} {s['cpu_ms']:>9.3f} {s['wait_ms']:>9.3f} "
                f"{s['gpu_ms']:>9.3f} {s['gpu_pct']:>6.1f}% "
                f"{s['launches']:>12.1f}")
    print_table(
        "表2 单次 forward 的 CPU/GPU 占比(wait = CPU 干等 GPU 的时间,不计入 CPU;"
        "plain 是真实 decode,graph 是同一份 decode 录成 graph)",
        f"{'tier':>11} {'stage':>8} {'CPU(ms)':>9} {'wait(ms)':>9} "
        f"{'GPU(ms)':>9} {'GPU%':>7} {'launch/step':>12}",
        stages)

    for r in rows:
        for stage in ("prefill", "plain", "graph"):
            if not r.get(stage):
                continue
            print(f"\n表3 tier={r['label']} {stage} top-10 kernel")
            print(r[stage]["top10"])


def release_tier(mgr) -> None:
    """一档跑完后的回收.

    先清掉 weight store 里那 128 个 CPU 权重(每档都要重新加载一份,10.6 GiB),
    再用 malloc_trim 把 glibc 抓着的空闲页还给 OS,否则 RSS 会逐档抬高。
    """
    mgr._weight_store.clear()
    gc.collect()
    torch.cuda.empty_cache()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def load_enc(device, dtype):
    enc = load_tokenizer(BASE)(
        [PROMPT] * BATCH_SIZE, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def run_tier(k: int, swap: int, steps: int) -> dict:
    """跑一档并返回该档的三张表数据:加载 → 表1 event 计时 → prefill/plain 两窗口 → 回收.

    只注册本档会轮换到的 adapter(used_names),槽位数 = k 保证每步真换出。
    """
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

    # 五档在同一个进程里顺序跑:窗口末尾的 gc+malloc_trim(kineto 事件内存,
    # 实测能要回 91%以上,plain 峰值 29.7 vs prefill 29.4 GiB 不叠加)和
    # 档末的 release_tier(weight store + 显存缓存)负责档间不累积
    rows = []
    for k, swap, steps in TIERS:
        print(f"=== k{k}swap{swap} ===")
        rows.append(run_tier(k, swap, steps))
        print(f"done k{k}swap{swap}")

    print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
