# 换入换出 + 算子融合 · 交接（2026-11-02）

> 本文件覆盖两条线：**换入**（T16-T20）和**算子融合**（T21-T23）。
> 长期任务表与物理边界推导在 `OPTIMIZATION_TODO.md`，这里只写「接了就能干活」的部分。
> 历史口径与内存修复记录保留在 §2 §3，别改，否则五档数据不可比。

---

## 1. 当前状态

```
换入线    T16-T19 结项（v12），T20 因显存暂缓
融合线    T21 结项（v14），T22 待做，T23 评估完挂起
剩余      T7 T8 T11 T12 T15 T22 T23
```

| 工况 | 当前最优 | wall | busy | tok/s | 卡在哪 |
|---|---|---|---|---|---|
| k64swap32（每步换 32 个） | v12 | 142.6ms | 138.9ms | 448.9 | **PCIe 硬顶**，H2D 108.9ms = 76% |
| k64swap0（纯计算） | v14 | 26.4ms | 24.5ms | 2424 | base_gemm 15.2ms（**访存受限**） |

两端的物理地板都探到了，剩下的项按性价比排：**T22（−0.2~0.3ms，安全）> T23（−3.0ms，但依赖私有不变量）> T15（只救 prefill）**。

---

## 2. 工况定义（不要改口径，否则五档不可比）

档位表在 `fix_batch_test_each_cost.py` 的 `TIERS`：

```python
TIERS = [(1, 1, 15), (8, 4, 15), (32, 16, 15), (64, 32, 15), (64, 0, 15)]
#        K  H steps  K=batch 内不同 adapter 数, H=每步换入=换出个数, H=0 是不换入的对照组
```

三条硬约束，都是一路踩出来的：

1. **槽位数必须等于 K**（`max_loras_per_batch=k`）。有空槽就不驱逐，测不到换出。
   `bench_decode_n64` 那档是 64 槽 / 64 adapter → 窗口内零换入，所以那个 bench 一直测不到这条路径。
2. **平均 rank 不能被换入序列破坏**。`TierRotation` 按 rank 分成 8/16/32/64 四组，
   每组各自滑动窗口：每步从每组取 `K/4` 个、前进 `H/4` 个 → batch 里四种 rank 恒各占 1/4，平均 rank 恒 30。
   `K=1` 特例只从 r32 组取（平均 32），跟历史 n=1 档对齐。
3. **每步都要真换入**：warmup 与计时阶段都从 rotation 取新的一组，`prefill()` / `decode()` /
   `plain_decode()` 里都调 `rotation.next_names()`。

每档注册到 manager 的 adapter 数（`used_names()`）与实测 PEFT 常驻：

| 档 | 注册数 | 平均 rank | 每步换入 | PEFT 常驻(bf16) | pool(94 MiB/槽) |
|---|---|---|---|---|---|
| k1swap1 | 2 | 32 | 1 | 90 MiB | 0.09 GiB |
| k8swap4 | 12 | 30 | 4 | 506 MiB | 0.75 GiB |
| k32swap16 | 48 | 30 | 16 | 2025 MiB | 3.0 GiB |
| k64swap32 | 96 | 30 | 32 | 4050 MiB | 6.0 GiB |
| k64swap0 | 64 | 30 | 0 | 2700 MiB | 6.0 GiB |

指标口径：表1（TTFT / TPOT / tok·s⁻¹，CUDA event，50 次）、
表2（CPU / wait / GPU / launch 每步）、表3（top-10 kernel）、
分桶五块（`base_gemm / lora_ab_gemm / adapter_index / adapter_h2d_copy / other_gpu`）。

**只信 `GPU busy` / `replay-only` / launch 数**。`wall` / `gap` 跨进程不可比：同配置 gplain
三次实测 113.5 / 139.6 / 273.3 ms，全是主机噪声。

---

## 3. 内存账（三个天花板，都已踩过）

显存 **23.5 GiB**；主机内存 **62.0 GiB**（看 `/sys/fs/cgroup/memory.max`，
`free -h` 的 503 GiB 是宿主机的，别信）。

### 3.1 显存 OOM（10-31，k64swap32 建池时崩）

- 根因：当时变体文件还是 **fp32**。PEFT 把注册的 adapter **按文件 dtype 原样常驻显存**，
  不 autocast；`convert_to_lora_layers` 之后才释放 → 峰值在 `from_pretrained` 途中。
- 修复：`make_qkvo_variants.py` 改存 **bf16**（磁盘 10.6 → 5.3 GiB）+ 每档只注册 `used_names()`。
- 实测（修复后 k64swap32）：base 7672 MiB → pool 建完 allocated 13432 / reserved 21632 MiB，
  峰值 ~17.4 GiB，稳态 13.4 GiB。

### 3.2 主机内存 cgroup kill（10-31，k64swap32 的 plain 窗口）

- 根因：每次 pageable H2D 产生 ~5 条 profiler 事件 × ~6 KB（`record_shapes=True`），
  k64swap32 每步 9216 次 H2D ≈ **1 GiB/步** 的 kineto 事件内存；
  **kineto 离开 `with` 块不还内存** → 25 步两窗口叠加 ~56 GiB + 基线 ~14 GiB > 62 GiB → `Killed`（无 traceback）。
- 修复三件套：窗口末尾 `del prof + gc.collect() + malloc_trim(0)`（实测要回 91%+）、
  记录步数 25 → 15、（后来证明）单进程即可，无需子进程。
- 实测（全量五档）：最高峰值 **29.7 GiB**，档间 RSS 只爬 0.7→2.8 GiB。

### 3.3 T18b 释放散张量（省 4.2GB 主机内存，GPU 显存不变）

`_bank_for` 拼完 pinned bank 后把 `weights.modules` 清空、置 `weights.flattened = True`。
之后**不能再访问 `store[name].modules`**——测试里要用同 seed 的独立副本校验 bank 内容
（`test_pool_write_many.py` 的 `store` / `ref` 分离就是这个原因）。
想保留散张量做对照，构造 `_GpuBuffers` 时传 `release_modules=False`。

---

## 4. 结果

### 4.1 换入五档（v12，2026-11-01，`traces/full5_v12.log`）

| tier | TTFT(ms) | TPOT(ms) | TPOT/st | graph(ms) | tok/s |
|---|---|---|---|---|---|
| k1swap1 | 62.97 | 58.69 | 64.93 | 30.71 | 1090.4 |
| k8swap4 | 87.07 | 83.35 | 116.01 | 41.45 | 767.9 |
| k32swap16 | 194.94 | 127.50 | 127.84 | 84.24 | 501.9 |
| k64swap32 | 250.01 | 261.73 | 201.50 | **144.21** | 244.5 |
| k64swap0 | 68.54 | 69.32 | 71.76 | **30.90** | 923.2 |

`graph(ms)` = 每步换入 + replay，含换入成本；k64swap32 的 144.21 就是这条线的目标指标。
表里的 `tok/s` 按 DynamicCache 的 TPOT 折算（64 / TPOT）；graph 口径是 64 / graph(ms)，
即 k64swap32 ≈ 448.9、k64swap0 ≈ 2071（v14 融合后 26.4ms → 2424）。

分桶（graph 窗口，ms/forward）：

| | wall | base_gemm | lora_ab | index | h2d_copy | other | gap | gap% |
|---|---|---|---|---|---|---|---|---|
| k64swap0 | 30.98 | 15.21 | 4.74 | 0.17 | 0.06 | 6.53 | 4.29 | 13.9% |
| k64swap32 | 146.85 | 15.22 | 4.75 | 3.55 | **108.87** | 6.52 | 7.95 | 5.4% |

### 4.2 换入优化链（k64swap32，graph 窗口，`traces/04_decode_n64/`）

| 版本 | 手段 | TPOT | CPU | gap | launch/f | tok/s |
|---|---|---|---|---|---|---|
| v6 | 逐条 pageable `copy_` | 861 | ~350 | ~480 | — | — |
| v8 | T16 装箱补零 | ~713(wall) | — | — | — | — |
| v9 | T17 pinned 逐条 + 异步 | 732 | 580 | — | — | — |
| v10 | T17b 按 (module,rank) 聚合 | 428 | 433 | 389 | 5324 | 120.0 |
| v11 | T18 整块 pinned bank | 169.8 | 57.2 | 80.9(35.9%) | 5360 | 283.7 |
| **v12** | **T19 散布融合单 kernel** | **142.7** | **5.3** | **3.7(2.6%)** | **3064** | **448.9** |

v12 细节：wall 142.58 / busy 138.94 / `replay-only` 26.42ms /
`copy_memset` 108.89ms（117 次）/ `lora_scatter` 3.39ms（8 次）。
v6→v12：TPOT **861→142.7（−83%，6.0 倍）**。

**换入成本对 H 严格线性**（五档实测，graph 窗口）：

| 档位 | H | adapter_h2d | 每 adapter | scatter 次数 |
|---|---|---|---|---|
| k1swap1 | 1 | 3.69 | 3.69 | 1 |
| k8swap4 | 4 | 13.67 | 3.42 | 4 |
| k32swap16 | 16 | 54.46 | 3.40 | 4 |
| k64swap32 | 32 | 108.87 | 3.40 | 8 |
| k64swap0 | 0 | 0.06 | — | 0 |

每 adapter 恒定 3.40ms（44MB ÷ 12.9GB/s）→ **T16-T19 对工况通用，不是 k64 特化**。
PCIe 硬顶：Gen4 ×8，理论 15.75GB/s，实测 12.86（82%）。判据：v10→v11 把 H2D 次数从
1152 降到 32，时间只从 112.7 变到 108.9ms —— **时间只跟字节走**，写法已不亏钱。

### 4.3 算子融合（k64swap0，`traces/04_decode_n64/v13|v14`）

v13 = 未融合（对照），v14 = `MULTI_LORA_FUSED_NORM=1`。

| 指标 | v13（关） | v14（开） | 变化 |
|---|---|---|---|
| graph wall | 28.058 | **26.399** | −1.66ms（−5.9%） |
| graph busy | 26.151 | **24.472** | −1.68ms |
| graph tok/s | 2281.0 | **2424.3** | +6.3% |
| replay-only | 25.89 | **24.27** | −1.62ms |
| other_gpu | 6.562 | 4.934 | −1.63ms |
| launch/步 | 3016 | **2001** | **−1015** |
| gplain wall | 113.500 | **90.541** | **−23.0ms（−20.2%）** |
| gplain tok/s | 563.9 | **706.9** | +25.4% |

145 个 norm = 36 层 × 4（input / post_attention / q_norm / k_norm）+ final。
原来每个 8 个 kernel，融合后 1 个（1.70µs/次，145 次共 0.246ms）。
无 graph 的 −23ms 里 **92% 来自少发 1015 次 launch**（≈20.5µs/次），只有 1.9ms 是 GPU 时间。

---

## 5. 现在怎么跑

**五档全量（跑完自动出分桶表）**：

```bash
cd /root/autodl-tmp/code/multi_lora-main
# 脚本 TRACE_DIR 写死，先备份上一轮
mv traces/02_fix_batch_each_cost traces/02_fix_batch_each_cost_$(date +%m%d_%H%M)
screen -L -Logfile traces/full5_$(date +%m%d_%H%M).log -S bench5 bash -c '
  cd /root/autodl-tmp/code/multi_lora-main &&
  TQDM_DISABLE=1 HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u multi_lora/examples/fix_batch_test_each_cost.py &&
  .venv/bin/python multi_lora/examples/analyze_trace.py traces/02_fix_batch_each_cost
'
```

**单档（快速迭代）**：

```bash
# 换入档
HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -u multi_lora/examples/bench_decode_n64.py graph 32

# 纯计算（融合 norm 默认开；要对照就加 MULTI_LORA_FUSED_NORM=0）
HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -u multi_lora/examples/bench_decode_n64.py graph 0
```

**正确性**：

```bash
HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m pytest multi_lora/tests -q
```

> 必须带 `HF_HOME`：`test_graph_decode.py` 加载 `sshleifer/tiny-gpt2`，
> 否则去 Hub 联网下载会卡死（缓存在 `/root/autodl-tmp/hf_cache`）。

保留的内存管理（不是调试，别删）：窗口末尾 `del prof + gc + malloc_trim`、
档末 `release_tier`、B-2 前 `empty_cache`、`StaticCache` 按 `steps` 开、TIERS 步数 15
（调回 25 = 主机内存重新顶爆，见 §3.2）。

环境开关：

| 变量 | 默认 | 作用 |
|---|---|---|
| `MULTI_LORA_FUSED_NORM=0` | **开** | 关掉融合 RMSNorm（T21）做 A/B 对照 |
| `MULTI_LORA_FUSED_QKV=1` | 关 | 合并 q/k/v 基座 GEMM（T7，仅 decode） |
| `LORA_SCATTER_IMPL=python` | `kernel` | 散布切回高级索引参考实现（T19） |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | — | bench 必带，否则 k64 档易 OOM |

---

## 6. 清理记录（别再找这些）

已删：`TIER`/`REPORT`/`RERUN` 分支、`run_tier_alone` + subprocess driver、`load_rows` + `rows/` 落盘、
`assign_names`、`SOLO_ADAPTER`、`ADAPTER_COUNTS`、`host_mem`/`gpu_mib` 及其打印、
`import pdb/os/subprocess/Counter`、`manager.py` 的 `_has_peft_layers` 打印、
trace 目录里的 `old_naming`/`old_fp32`/`old_25step`/`rows`。
`analyze_trace.py` 的分组正则已改成认 `k(\d+)swap(\d+)`（含 `graph`/`gplain` 阶段），
lora 正则同时匹配 `_gemv_` 与 `_sgemm_` 两套 kernel。

---

## 7. T 任务现状（长期表见 `OPTIMIZATION_TODO.md`）

| 编号 | 任务 | 状态 |
|---|---|---|
| T16 | 合并 H2D（装箱） | ✅ 被 T17 取代（补零使字节翻倍） |
| T17 | pinned + copy stream + non_blocking | ✅ v9/v10：TPOT 732→428 |
| T18 | 整块 pinned bank | ✅ v11：TPOT 428→169.8，CPU 433→57.2 |
| T18b | 释放散张量 | ✅ 省 4.2GB 主机内存（§3.3） |
| T19 | 散布融合单 kernel | ✅ v12：TPOT 169.8→142.7，CPU 57→5.3 |
| T20 | 预取重叠 | ⏸ 需 +1.4GB 显存，当前余量 2~3GB 且已见 `mapping failed` |
| T21 | RMSNorm 8 kernel → 1 | ✅ graph −1.66ms；非 graph −23ms |
| T22 | add + RMSNorm 融合 | ⬜ **下一个要做的**：graph −0.2~0.3ms，需重写 `Qwen3DecoderLayer.forward` |
| T23 | 消除 GQA `repeat_kv` | ⬜ 评估完挂起：3 行 patch 可 −3.0ms，但依赖 transformers 私有 `use_gqa_in_sdpa` |
| T7 | QKV / gate-up 合并 | ⬜ gate/up ≈0；qkv 上限 −0.5ms（CTA 占用），不建议 |
| T8 | 相同 adapter 合组 | ⬜ 代码在 `git stash`（「T8 合组」），需 batch > adapter 数 |
| T11 | segment 越界早退 | ⬜ 仅混合 batch |
| T12 | B store 补 `n_offset < N` mask | ⬜ **正确性**，换 out_dim 非 256 倍数的模型会踩内存 |
| T15 | autotune + `num_stages` | ⬜ 只救 prefill |

---

## 8. 这一轮踩过的坑（别再走一遍）

1. **`free -h` 的可用内存不是预算**，容器看 `/sys/fs/cgroup/memory.max`（这里 62 GiB）。
2. **PEFT 按文件 dtype 原样常驻注册的 adapter，不会 autocast**；峰值在 `from_pretrained` 途中。
3. **kineto 退出 `with` 块不还内存**，必须 `del prof` + `gc.collect()` + `malloc_trim(0)`。
4. **profiler 事件量 ∝ H2D 次数 × 5**，高 swap 档一个窗口二十多 GiB；这是记录成本，不是 GPU 问题。
5. **槽位数 = adapter 数 → 零换入**，任何换入优化在这种 bench 上都测不出来。
6. **`use_cache=False` 的 decode 窗口不是真实 decode**（不写 KV），与 plain/graph 不可直接比。
7. **首个 profiler 窗口贵 30~55 ms**，`wall`/`gap` 不能跨进程比，只能同进程内比。
8. **每次 pageable H2D 都是同步的**，`wait_ms` 就是它，别当成 CPU 在干活。
9. **跑 bench 用 `.venv/bin/python`**——系统 python 没有 torch/transformers。
10. **Triton 不支持 int→pointer cast**：散布 kernel 用「base 指针 + int64 元素偏移表」绕开，
    见 `kernels/scatter_bank.py`；B 段必须按 `row = offs // RANK, col = offs % RANK` 跨步写。
11. **pinned bank 的行必须连续**：`_ensure_stage` 先一维 `empty(_BANK_CHUNK * total)` 再 `view`，
    不能按 `[cap, max_total]` 分配后截列（行会不连续，散布写错位）。
12. **copy stream 必须 `wait_stream(计算流)`**，否则在上一轮的 scatter 还没读完 bank 时就覆盖。
13. **cat/stack 混用会触发 resize**：`torch.cat(list, 0, out=3维张量)` 形状不匹配，
    首装拼 bank 用 `torch.stack`。
14. **改完 graph 的任何 kernel 序列都要重新 capture**，否则 replay 的是旧图。
