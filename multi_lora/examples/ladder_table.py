#!/usr/bin/env python3
"""把各次跑的 bucket_summary.csv 拼成 Baseline/V1/V2 优化阶梯表.

行 = 一个实验配置,列 = 分桶耗时(ms/forward)。后续每做一版优化
(T16 合并 H2D、T17 copy stream...)加一行:跑一次 bench,新目录里的
trace 前缀对上下面的 LADDER 映射,本表自动多一行。

用法::

    python multi_lora/examples/ladder_table.py                    # 自动扫 traces/
    python multi_lora/examples/ladder_table.py traces/04_decode_n64/v7
    python multi_lora/examples/ladder_table.py traces/04_decode_n64/v6 traces/04_decode_n64/v7

行映射(trace 文件前缀 → 阶梯名):
    decode_*  → Baseline (Dynamic KV cache)
    gplain_*  → V1       (Static KV cache)
    graph_*   → V2       (Static KV cache + CUDA graph)
新版本优化改这里一行映射即可(如 "h2d_merged_" → "V3 合并H2D")。
"""

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# 前缀 → (阶梯名, 排序号);排序号相同的行按目录时间排
LADDER = {
    "decode_": ("Baseline Dynamic", 0),
    "gplain_": ("V1 Static", 1),
    "graph_": ("V2 Static+Graph", 2),
}

# 想追新版就加映射,例如:
#   "dynmerged_": ("V3 合并H2D", 3),
#   "copystream_": ("V4 CopyStream", 4),

COLS = [  # (表头, bucket_summary.csv 里的列名或派生)
    ("wall", "wall_ms"),
    ("base_gemm", "base_gemm_ms"),
    ("lora_ab_g", "lora_ab_gemm_ms"),
    ("adapter_h2d", "adapter_h2d_copy_ms"),
    ("other_gpu", "other_gpu_ms"),
    ("gap", "gap_ms"),
    ("gap%", "gap_pct"),
]

BATCH_SIZE = 64  # decode 每 step 产出 64 个 token(bench_decode_n64 的固定 batch)


def rows_from_dir(csv_path: Path):
    """一个 bucket_summary.csv → [(排序key, 阶梯名, trace路径, 单元格...)]."""
    out = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            stem = row["trace"].replace(".json.gz", "")
            hit = next(((v, k) for k, v in LADDER.items() if stem.startswith(k)), None)
            if not hit:
                continue  # prefill_/plain_/bucketed 等不进阶梯表
            (name, order), prefix = hit
            gz = csv_path.parent / row["trace"]
            try:
                tr = str(gz.relative_to(_ROOT))
            except ValueError:
                tr = str(gz)
            cells = [float(row[col]) for _, col in COLS]
            launch = sum(float(row[f"{b}_launch"])
                         for b in ("base_gemm", "lora_ab_gemm", "adapter_index",
                                   "adapter_h2d_copy", "other_gpu"))
            out.append(((order, str(csv_path.stat().st_mtime), stem),
                        name, tr, cells, launch))
    return out


def main() -> int:
    targets = [Path(a) for a in sys.argv[1:]] or sorted(
        {p.parent for p in (_ROOT / "traces").rglob("bucket_summary.csv")},
        key=lambda p: p.stat().st_mtime)
    if not targets:
        print("no bucket_summary.csv found under traces/")
        return 1

    rows = []
    for t in targets:
        csv_path = t / "bucket_summary.csv" if t.is_dir() else t
        if csv_path.is_file():
            rows += rows_from_dir(csv_path)
    if not rows:
        print(f"no ladder rows (decode_/gplain_/graph_) in {len(targets)} dir(s)")
        return 1

    rows.sort(key=lambda r: r[0])

    # throughput = 64 token/step ÷ wall(受 profiler 开销拖累,行间相对比较仍有意义)。
    # 提升百分比相对 Baseline 行;表里没有 Baseline 时退化为相对第一行
    thrs = [BATCH_SIZE * 1000.0 / r[3][0] for r in rows]
    ref_idx = next((i for i, r in enumerate(rows) if r[0][0] == 0), 0)
    ref_thr = thrs[ref_idx]

    hdr = (f"{'config':<20} {'trace(.json.gz)':<58} "
           + " ".join(f"{h:>10}" for h, _ in COLS)
           + f" {'launch/f':>9} {'tok/s':>8} {'vs-base%':>9}")
    print(hdr)
    print("-" * len(hdr))
    for (key, name, tr, cells, launch), thr in zip(rows, thrs):
        pct = (thr / ref_thr - 1.0) * 100.0
        line = f"{name:<20} {tr:<58} "
        line += " ".join(f"{c:>10.3f}" for c in cells)
        line += f" {launch:>9.0f} {thr:>8.1f} {pct:>+8.1f}%"
        print(line)
    if rows[ref_idx][0][0] != 0:
        print(f"\n注意: 表中没有 Baseline(Dynamic) 行,vs-base% 是相对"
              f"第一行({rows[ref_idx][1]})算的")
    print("\n单位 ms/forward;launch/f 含 Memcpy/Memset;"
          "wall 含宿主噪声,跨跑只比 GPU 列与比例;tok/s 同样受 profiler 开销影响,"
          "绝对值偏低,行间对比用")
    print("Perfetto 看时间线: 同目录下把 .json.gz 换成 .bucketed.json.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
