#!/usr/bin/env python3
"""
单层 decoder 的最小复现: 用真实 Qwen3 配置但只留 num_hidden_layers=1,
随机初始化(from_config, 不下载权重), 造两个不同 rank 的 LoRA, 跑一次 prefill
+ 若干次 decode, 用 torch.profiler 导出 chrome trace 后直接结束。

trace 一步只含 1 层 -> kernel 数量少、时间轴干净, 适合看懂 launch 结构。

用法::

    python multi_lora/examples/bench_one_decoder_layer.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multi_lora.hf_runner import pick_device, pick_dtype  # noqa: E402
from multi_lora.manager import MultiLoRAManager  # noqa: E402

BASE = "Qwen/Qwen3-4B-Instruct-2507"
TRACE_DIR = _ROOT / "traces" / "03_one_decoder_layer"

# 实验配置
BATCH_SIZE = 64
PREFILL_LEN = 32
WARMUP = 3
ACTIVE = 5
MAX_RANK = 64
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_one_layer_base(cfg) -> torch.nn.Module:
    """按 cfg 随机初始化一个模型(不落盘权重);device/dtype 由调用方搬."""
    return AutoModelForCausalLM.from_config(cfg)


def make_adapter_dir(cfg, out_dir: Path, rank: int, alpha: int, seed: int) -> str:
    """建一个只有 1 层的小模型, 挂 LoRA 后存到 out_dir, 返回路径."""
    torch.manual_seed(seed)
    m = build_one_layer_base(cfg)
    lcfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0,
        bias="none", task_type="FEATURE_EXTRACTION", target_modules=TARGETS,
    )
    get_peft_model(m, lcfg).save_pretrained(out_dir)
    del m
    return str(out_dir)


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required")
        return 1
    device = pick_device()
    dtype = pick_dtype(device)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 真实 Qwen3 配置, 但只留 1 层 + 缩小 vocab, 让随机初始化与显存都很小
    cfg = AutoConfig.from_pretrained(BASE)
    cfg.num_hidden_layers = 1
    cfg.vocab_size = 4096  # 随机 token 用不到真实词表, 缩小省显存/省时间
    if hasattr(cfg, "tie_word_embeddings"):
        cfg.tie_word_embeddings = True

    # 2) 临时造两个不同 rank 的 adapter(16 / 64), 存到临时目录
    tmp = Path(tempfile.mkdtemp(prefix="onelayer_"))
    adapters = {
        "r16": make_adapter_dir(cfg, tmp / "r16", rank=16, alpha=32, seed=1),
        "r64": make_adapter_dir(cfg, tmp / "r64", rank=64, alpha=64, seed=2),
    }

    # 3) 单个随机初始化的 base, 交给 manager 挂多 adapter
    base = build_one_layer_base(cfg).to(device=device, dtype=dtype).eval()
    mgr = MultiLoRAManager.from_pretrained(
        base,
        adapters,
        max_loras_per_batch=len(adapters),
        max_lora_rank=MAX_RANK,
        lora_backend="triton",
    )
    print(f"num_layers=1 device={device} dtype={dtype} "
          f"wrapped_modules={len(mgr.lora_wrappers)} adapters={list(adapters)}")

    keys = list(adapters)
    names = [keys[i % len(keys)] for i in range(BATCH_SIZE)]
    input_ids = torch.randint(
        0, cfg.vocab_size, (BATCH_SIZE, PREFILL_LEN), device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids)
    decode_lens = [1] * BATCH_SIZE  # CPU list,不用建 GPU 张量

    def prefill(past=None):
        return mgr.forward_with_lora(
            input_ids=input_ids, attention_mask=attn,
            lora_names=names, seq_lens=[PREFILL_LEN] * BATCH_SIZE,
            past_key_values=past, use_cache=True)

    def decode_step(nxt, past):
        return mgr.forward_with_lora(
            input_ids=nxt, attention_mask=None, seq_lens=decode_lens,
            lora_names=names, past_key_values=past, use_cache=True)

    # 4) 预热
    out = prefill()
    nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    past = out.past_key_values
    for _ in range(WARMUP):
        out = decode_step(nxt, past)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    # 5) profile: 每步 = 一次干净 prefill(cache 不跨步增长) + 一次 decode
    sched = torch.profiler.schedule(
        wait=0, warmup=WARMUP, active=ACTIVE, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=sched, record_shapes=True,
    ) as prof:
        for _ in range(WARMUP + ACTIVE):
            pf = prefill()
            dnext = pf.logits[:, -1].argmax(dim=-1, keepdim=True)
            decode_step(dnext, pf.past_key_values)
            torch.cuda.synchronize()
            prof.step()

    tag = "one_layer_prefill_decode"
    prof.export_chrome_trace(str(TRACE_DIR / f"{tag}.json"))
    print("\ntop-10 kernel (self device time):")
    print(prof.key_averages().table(
        sort_by="self_device_time_total", row_limit=10))
    print(f"\ntrace -> {TRACE_DIR / (tag + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
