#!/usr/bin/env python3
"""
读取 bench_multi_lora / fix_batch_test_each_cost 导出的 chrome trace, 把 GPU 侧耗时拆成五块:
  base_gemm / lora_ab_gemm / adapter_index / adapter_h2d_copy / other_gpu

产出三样东西:
  1. stdout 上一张汇总表: 每个 trace 一行, 五块各自的 ms/forward + GPU 空档占比
  2. bucket_summary.csv 与 bucket_detail.csv(后者给每个桶内部的 kernel 名和 aten 算子名排名)
  3. <原名>.bucketed.json.gz: 原 trace 的派生副本, 只保留前 --max-steps 次 forward,
     每条 GPU 事件的 cat 改写成桶名(Perfetto 按 cat 上色, 于是五块五种颜色),
     另加两条合成泳道: 每桶一条区间并集 lane + 每桶一条 per-forward counter 折线

归桶规则(先按 kernel 名, 命不中再按发起它的 aten 算子名):
  lora_ab_gemm      _sgemm_lora_a_kernel / _sgemm_lora_b_kernel
  base_gemm         *gemm* / cutlass / cublas / nvjet / flash / fmha, 或 aten::mm/addmm/linear
  adapter_index     gather / index_select / scatter / DeviceScan / arange 一类索引与段前缀和
  adapter_h2d_copy  Memcpy HtoD/DtoH + Memset(显存槽位清零)
  other_gpu         剩下全部(逐元素乘加、cat/contiguous、norm、act、reduce ...)

Example::

    python multi_lora/examples/analyze_trace.py traces/02_fix_batch_each_cost
    python multi_lora/examples/analyze_trace.py traces/02_fix_batch_each_cost --max-steps 3
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 分篮子 到不同的bucket
BUCKETS = ["base_gemm", "lora_ab_gemm",
           "adapter_index", "adapter_h2d_copy", "other_gpu"]

# GPU kernel 名规则(顺序即优先级)
RE_LORA = re.compile(r"_sgemm_lora_[ab]_kernel")
RE_GEMM = re.compile(r"gemm|cutlass|cublas|nvjet|flash|fmha|attention", re.I)
RE_INDEX = re.compile(
    r"gather|index_select|indexSelect|scatter|index_elementwise"
    r"|DeviceScan|elementwise_kernel_with_index|searchsorted|sort",
    re.I,
)

# 发起该 kernel 的 aten 算子名规则(kernel 名命不中时兜底)
GEMM_OPS = {"aten::mm", "aten::addmm", "aten::bmm", "aten::linear",
            "aten::_scaled_dot_product_flash_attention",
            "aten::_scaled_dot_product_flash_attention_for_cpu"}
INDEX_OPS = {"aten::index_select", "aten::gather", "aten::index_put_",
             "aten::cumsum", "aten::arange", "aten::searchsorted",
             "aten::sort", "aten::argsort", "aten::scatter"}

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def load_trace(path: Path) -> dict:
    """读 .json 或 .json.gz 的 chrome trace."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def dump_trace(obj: dict, path: Path) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def build_op_index(events: List[dict]) -> Tuple[Dict[int, int], Dict[int, str]]:
    """建两张映射表:correlation -> 发起它的 aten 算子名;correlation -> runtime 事件下标."""
    corr_to_eid: Dict[int, int] = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            cid = e.get("args", {}).get("correlation")
            eid = e.get("args", {}).get("External id")
            if cid is not None:
                corr_to_eid[cid] = eid if eid is not None else -1

    eid_to_op: Dict[int, str] = {}
    eid_to_dur: Dict[int, float] = {}
    for e in events:
        if e.get("cat") != "cpu_op":
            continue
        eid = e.get("args", {}).get("External id")
        if eid is None:
            continue
        dur = e.get("dur", 0.0)
        # 同一个 External id 可能对应嵌套算子, 取最短的那层(最贴近 kernel 的算子)
        if eid not in eid_to_dur or dur < eid_to_dur[eid]:
            eid_to_dur[eid] = dur
            eid_to_op[eid] = e["name"]
    return corr_to_eid, eid_to_op


def classify(name: str, cat: str, op: Optional[str]) -> str:
    """把一条 GPU 事件归到五个桶之一."""
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "adapter_h2d_copy"
    if RE_LORA.search(name):
        return "lora_ab_gemm"
    if RE_GEMM.search(name) or (op in GEMM_OPS if op else False):
        return "base_gemm"
    if RE_INDEX.search(name) or (op in INDEX_OPS if op else False):
        return "adapter_index"
    return "other_gpu"


# 五桶太粗(基座 GEMM 一桶就吃掉 70%), 再加一组互斥的细粒度分组, 按顺序取第一个命中.
# 注: lm_head 从名字上认不出来, 归在 base_gemm 里.
GROUP_RULES = [
    ("lora_a", re.compile(r"_sgemm_lora_a_kernel")),
    ("lora_b", re.compile(r"_sgemm_lora_b_kernel")),
    ("attention", re.compile(r"flash|fmha|attention", re.I)),
    ("cat_kv", re.compile(r"CatArrayBatchedCopy|concat", re.I)),
    ("base_gemm", re.compile(r"gemm|cutlass|cublas|nvjet|gemv", re.I)),
    ("elementwise", re.compile(r"elementwise", re.I)),
]


def group_of(name: str, cat: str) -> str:
    """GPU 事件 -> 分组名(copy/memset 靠 cat 认, 其余靠 kernel 名)."""
    if cat in ("gpu_memcpy", "gpu_memset") or re.search(r"Memcpy|Memset", name):
        return "copy_memset"
    for label, rx in GROUP_RULES:
        if rx.search(name):
            return label
    return "other"


def find_steps(events: List[dict]) -> List[Tuple[str, float, float]]:
    """profiler schedule 打的 ProfilerStep#N 标记 = 一次 forward 的 CPU 区间."""
    out = [(e["name"], e["ts"], e["ts"] + e.get("dur", 0.0))
           for e in events
           if e.get("cat") == "user_annotation" and e["name"].startswith("ProfilerStep")]
    return sorted(out, key=lambda x: x[1])


def merge(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """区间并集:相邻或重叠的合一段,用于算 GPU busy 时间和画泳道."""
    merged: List[List[float]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(m[0], m[1]) for m in merged]


def analyze(path: Path, max_steps: int) -> Tuple[Dict, dict]:
    """返回该 trace 的汇总数字, 以及改写好的派生 trace."""
    data = load_trace(path)
    events = data["traceEvents"]
    corr_to_eid, eid_to_op = build_op_index(events)
    steps = find_steps(events)
    starts = [s[1] for s in steps]

    def step_of(ts: float) -> Optional[int]:
        i = bisect_right(starts, ts) - 1
        if i < 0 or ts >= steps[i][2]:
            return None
        return i

    n_steps = len(steps)
    bucket_ms = defaultdict(float)          # 桶 -> 全部 step 累计 us
    bucket_cnt = defaultdict(int)
    detail = defaultdict(lambda: [0.0, 0])  # (桶, kernel 名) -> [us, 次数]
    op_detail = defaultdict(lambda: [0.0, 0])
    group_acc = defaultdict(lambda: [0.0, 0])   # 分组 -> [us, 次数]
    per_step = defaultdict(lambda: defaultdict(float))   # step -> 桶 -> us
    step_gpu_iv = defaultdict(list)
    kept_iv: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    limit = min(n_steps, max_steps) if max_steps > 0 else n_steps
    for e in events:
        if e.get("cat") not in GPU_CATS:
            continue
        si = step_of(e["ts"])
        if si is None:
            continue
        dur = e.get("dur", 0.0)
        cid = e.get("args", {}).get("correlation")
        op = eid_to_op.get(corr_to_eid.get(cid, -1))
        b = classify(e["name"], e["cat"], op)
        e["cat"] = b                       # 派生 trace 里直接改 cat, Perfetto 按它上色
        bucket_ms[b] += dur
        bucket_cnt[b] += 1
        per_step[si][b] += dur
        step_gpu_iv[si].append((e["ts"], e["ts"] + dur))
        short = re.sub(r"<.*", "", e["name"])[:60]
        detail[(b, short)][0] += dur
        detail[(b, short)][1] += 1
        op_detail[(b, op or "?")][0] += dur
        op_detail[(b, op or "?")][1] += 1
        g = group_of(e["name"], e["cat"])
        group_acc[g][0] += dur
        group_acc[g][1] += 1
        if si < limit:
            kept_iv[b].append((e["ts"], e["ts"] + dur))

    wall_ms = [steps[i][2] - steps[i][1] for i in range(n_steps)]
    busy_ms = [sum(b - a for a, b in merge(step_gpu_iv[i]))
               for i in range(n_steps)]

    summary = {
        "trace": path.name, "steps": n_steps,
        "wall_ms": sum(wall_ms) / n_steps / 1000 if n_steps else 0.0,
        "gpu_busy_ms": sum(busy_ms) / n_steps / 1000 if n_steps else 0.0,
    }
    for b in BUCKETS:
        summary[f"{b}_ms"] = bucket_ms[b] / 1000 / n_steps if n_steps else 0.0
        summary[f"{b}_launch"] = bucket_cnt[b] / n_steps if n_steps else 0.0
    summary["gap_ms"] = summary["wall_ms"] - summary["gpu_busy_ms"]
    summary["gap_pct"] = (summary["gap_ms"] / summary["wall_ms"] * 100
                          if summary["wall_ms"] else 0.0)
    # 每个桶内部的排名,交给 main 写 csv 用(不进汇总表)
    summary["_detail"] = detail
    summary["_op_detail"] = op_detail
    summary["_groups"] = {k: [v[0] / 1000 / n_steps, v[1] / n_steps]
                          for k, v in group_acc.items()} if n_steps else {}

    derived = build_derived(data, kept_iv, per_step, steps[:limit], limit)
    return summary, derived


def build_derived(data: dict, kept_iv, per_step, steps, limit) -> dict:
    """派生 trace:原事件按桶改写 cat + 两条合成泳道, 只留前 limit 次 forward."""
    lo = steps[0][1] if steps else 0.0
    hi = steps[-1][2] if steps else float("inf")
    out = [e for e in data["traceEvents"]
           if e.get("cat") not in GPU_CATS and "ph" in e
           and lo <= e.get("ts", lo) <= hi]

    P_BUCKET, P_COUNTER, P_MARK = 900001, 900002, 900003
    meta = []
    for pid, name in ((P_BUCKET, "GPU busy by bucket"),
                      (P_COUNTER, "per-forward bucket ms"),
                      (P_MARK, "forward boundaries")):
        meta.append({"ph": "M", "pid": pid, "tid": 1, "name": "process_name",
                     "args": {"name": name}})
    for tid, b in enumerate(BUCKETS):
        meta.append({"ph": "M", "pid": P_BUCKET, "tid": tid, "name": "thread_name",
                     "args": {"name": b}})
        meta.append({"ph": "M", "pid": P_COUNTER, "tid": tid, "name": "thread_name",
                     "args": {"name": b}})

    slices = []
    for tid, b in enumerate(BUCKETS):
        for s, e in merge(kept_iv[b]):
            slices.append({"ph": "X", "cat": b, "name": b, "pid": P_BUCKET,
                           "tid": tid, "ts": s, "dur": e - s})
    counters = []
    for si in range(limit):
        for tid, b in enumerate(BUCKETS):
            counters.append({"ph": "C", "pid": P_COUNTER, "tid": tid,
                             "name": b, "ts": steps[si][1],
                             "args": {b: round(per_step[si].get(b, 0.0) / 1000, 3)}})
    marks = [{"ph": "i", "s": "g", "pid": P_MARK, "tid": 1, "cat": "forward",
              "name": f"iter#{si} {steps[si][0]}", "ts": steps[si][1]}
             for si in range(limit)]

    new = dict(data)
    new["traceEvents"] = meta + out + slices + counters + marks
    return new


def print_table(summaries: List[Dict]) -> None:
    hdr = (f"{'trace':<22} {'steps':>5} {'wall':>7} " +
           " ".join(f"{b[:9]:>9}" for b in BUCKETS) +
           f" {'gap':>7} {'gap%':>6} {'launch/f':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for s in summaries:
        row = f"{s['trace']:<22} {s['steps']:>5} {s['wall_ms']:>7.3f} "
        row += " ".join(f"{s[f'{b}_ms']:>9.3f}" for b in BUCKETS)
        row += f" {s['gap_ms']:>7.3f} {s['gap_pct']:>5.1f}% "
        row += f"{sum(s[f'{b}_launch'] for b in BUCKETS):>8.1f}"
        print(row)
    print("\n单位 ms/forward;gap = 墙钟 - GPU 忙时(即 launch 空档与 CPU 阻塞)")


def print_group_table(group_rows: Dict) -> None:
    """每个阶段一张「分组 × adapter 档」表, 行按平均耗时降序.

    用来看两阶段的结构差异: 同一组在 prefill/decode 里的排名会换位.
    各列格式 = 单步 ms(括号内是单步次数); 行合计会略大于 gpu_busy_ms,
    因为少量 kernel 并行重叠(busy 取的是区间并集).
    """
    stages: Dict[str, List] = {}
    for (stage, tier), g in group_rows.items():
        stages.setdefault(stage, []).append((tier, g))
    for stage in sorted(stages):
        items = sorted(stages[stage])
        labels = sorted({k for _, g in items for k in g},
                        key=lambda k: -sum(g.get(k, [0.0])[0] for _, g in items))
        hdr = f"{'item':<16}" + \
            "".join(f"{('n=' + str(t)):>16}" for t, _ in items)
        print(f"\n=== {stage}: 单步 ms(次数/步), 按平均耗时降序 ===")
        print(hdr)
        print("-" * len(hdr))
        for k in labels:
            cells = "".join(
                f"{g[k][0]:>9.3f} ({g[k][1]:>4.0f})" if k in g else " " * 16
                for _, g in items)
            print(f"{k:<16}{cells}")
        print(f"{'Σ':<16}" + "".join(
            f"{sum(v[0] for v in g.values()):>9.3f} ({sum(v[1] for v in g.values()):>4.0f})"
            for _, g in items))


def write_csvs(summaries: List[Dict], details, group_rows, outdir: Path) -> None:
    with (outdir / "bucket_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        for s in summaries:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in s.items()})
    with (outdir / "bucket_detail.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace", "bucket", "kind", "name",
                   "ms_per_forward", "count_per_forward"])
        for trace, rows in details.items():
            for bucket, kind, name, us, cnt in rows:
                w.writerow([trace, bucket, kind, name,
                            round(us / 1000, 4), round(cnt, 1)])
    with (outdir / "kernel_group_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "adapters", "group",
                    "ms_per_forward", "count_per_forward"])
        for (stage, tier), g in sorted(group_rows.items()):
            for k, (ms, cnt) in sorted(g.items(), key=lambda x: -x[1][0]):
                w.writerow([stage, tier, k, round(ms, 4), round(cnt, 1)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="trace 目录, 或单个 .json/.json.gz")
    ap.add_argument("--max-steps", type=int, default=5,
                    help="派生 trace 保留前几次 forward(0=全部);数字永远基于全部 step")
    ap.add_argument("--no-derived", action="store_true",
                    help="只出数字, 不写派生 trace")
    args = ap.parse_args()

    target = Path(args.target)
    cand = sorted(list(target.glob("*.json")) + list(target.glob("*.json.gz"))) \
        if target.is_dir() else [target]
    # 跳过派生文件和 driver 写的 row_*.json(后者不是 trace,没有 traceEvents)
    files = [p for p in cand if ".bucketed" not in p.name
             and not p.name.startswith("row_")]
    skipped = [p.name for p in cand if ".bucketed" in p.name]
    if skipped:
        print(f"skip {len(skipped)} derived trace(s): {skipped[:3]}")
    if not files:
        print(f"no trace found under {target}")
        return 1

    summaries, details, group_rows = [], {}, {}
    for p in files:
        stem = p.name.replace(".json.gz", "").replace(".json", "")
        s, derived = analyze(p, args.max_steps)
        summaries.append(s)
        # 每个桶内部按耗时排名, kernel 名与 aten 算子名各一组
        rows = []
        for (b, name), (us, cnt) in sorted(
                s.pop("_detail").items(), key=lambda x: -x[1][0])[:12]:
            rows.append([b, "kernel", name, us, cnt / s["steps"]])
        for (b, name), (us, cnt) in sorted(
                s.pop("_op_detail").items(), key=lambda x: -x[1][0])[:12]:
            rows.append([b, "aten_op", name, us, cnt / s["steps"]])
        details[stem] = rows
        # 分组表: 从文件名取阶段与档(k64swap32 = 64 个 adapter、每步换 32 个),
        # 认不出来的归 stage=stem。第二元统一用 str,避免排序时 str/int 混比
        m = re.match(r"(prefill|plain|gplain|graph|decode|mix)_k(\d+)swap(\d+)", stem)
        key = ((m.group(1), f"k{m.group(2)}swap{m.group(3)}")
               if m else (stem, ""))
        group_rows[key] = s.pop("_groups")
        if not args.no_derived:
            dump_trace(derived, p.parent / f"{stem}.bucketed.json.gz")
            print(f"  derived -> {stem}.bucketed.json.gz")

    print_table(summaries)
    print_group_table(group_rows)
    write_csvs(summaries, details, group_rows, files[0].parent)
    print(f"\ncsv -> {files[0].parent}/bucket_summary.csv, "
          f"bucket_detail.csv, kernel_group_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
