# 换入换出基准 · 已收官（2026-10-31 夜）

> 本文件是这一轮工作的交接说明。四个步骤全部完成：内存问题修复 → 五档全量跑通 →
> 临时代码清理成一条命令 → trace 分桶出数。**下一步是 T16（合并 H2D）**，分母数据已齐。
> 长期任务表仍在 `OPTIMIZATION_TODO.md`，本文件只写「换入换出工况」这条线。

---

## 1. 最终状态

| 步 | 内容 | 状态 |
|---|---|---|
| 1 | 抓内存问题（显存 OOM + 主机 cgroup kill） | ✅ 定位并修复，见 §3 |
| 2 | 五档全量跑完 | ✅ 单进程连跑，无 FAILED/Killed，峰值 29.7 GiB |
| 3 | 删掉全部临时机制（TIER/REPORT/RERUN/rows/subprocess/调试打印） | ✅ 一条命令跑到底 |
| 4 | trace 分桶（analyze_trace） | ✅ 10 份 trace → 3 个 CSV + 10 份 bucketed |

核心结论（T16 的分母，详见 §4）：**每换入一个 adapter 的代价 ≈ 32.4 ms/步，
其中真实传输只占 3.7 ms，其余 ~29 ms 是 288 次逐条 pageable H2D 的 CPU 发起/同步开销。**

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

---

## 3. 内存账（两个死因，都已修复并实测验证）

两个天花板：显存 **23.5 GiB**；主机内存 **62.0 GiB**（看 `/sys/fs/cgroup/memory.max`，
`free -h` 的 503 GiB 是宿主机的，别信）。

### 3.1 显存 OOM（10-31 17:25，k64swap32 建池时崩）

- 根因：当时变体文件还是 **fp32**。PEFT 把注册的 adapter **按文件 dtype 原样常驻显存**，
  不 autocast；`convert_to_lora_layers` 之后才释放 → 峰值在 `from_pretrained` 途中。
- 修复：`make_qkvo_variants.py` 改存 **bf16**（磁盘 10.6 → 5.3 GiB）+ 每档只注册 `used_names()`。
- 实测（修复后 k64swap32）：base 7672 MiB → pool 建完 allocated 13432 / reserved 21632 MiB，
  峰值 ~17.4 GiB，稳态 13.4 GiB。

### 3.2 主机内存 cgroup kill（10-31 18:2x，k64swap32 的 plain 窗口）

- 根因：每次 pageable H2D 产生 ~5 条 profiler 事件 × ~6 KB（`record_shapes=True`），
  k64swap32 每步 9216 次 H2D ≈ **1 GiB/步** 的 kineto 事件内存；
  **kineto 离开 `with` 块不还内存** → 25 步两窗口叠加 ~56 GiB + 基线 ~14 GiB > 62 GiB → `Killed`（无 traceback）。
- 修复三件套：窗口末尾 `del prof + gc.collect() + malloc_trim(0)`（实测要回 91%+）、
  记录步数 25 → 15、（后来证明）单进程即可，无需子进程。
- 实测（全量五档）：最高峰值 **29.7 GiB**，档间 RSS 只爬 0.7→2.8 GiB，plain 窗口 29.7 vs
  prefill 29.4 GiB → 不叠加；每档 `模型加载完 allocated≈7680 MiB` → 上档释放干净。

---

## 4. 最终结果（2026-10-31 全量跑，数据在 `traces/02_fix_batch_each_cost/` + `multi_lora/tiers_swap.log`）

表1 延迟与吞吐（CUDA event，50 次）：

| tier | TTFT(ms) | TPOT(ms) | tok/s |
|---|---|---|---|
| k1swap1 | 93.5 | 88.4 | 723.7 |
| k8swap4 | 159.5 | 155.1 | 412.7 |
| k32swap16 | 424.1 | 421.2 | 151.9 |
| k64swap32 | 772.0 | **762.9** | 83.9 |
| k64swap0 | 66.8 | **62.4** | 1025.8 |

**k64swap32 vs k64swap0（同 64 adapter，唯一变量=每步换 32 个）**：TPOT 62.4 → 762.9（**12.2×**），
GPU% 30.5 → 15.5，wait 0.8 → 141.8 ms。

分桶（plain，ms/forward；算力项五档恒定，随 swap 增长的只有 h2d_copy + gap）：

| | wall | base_gemm | lora_ab | index | h2d_copy | other | gap | gap% |
|---|---|---|---|---|---|---|---|---|
| k64swap0 | 97.8 | 15.3 | 4.8 | 0.19 | 0.04 | 5.6 | 71.9 | 73.5% |
| k32swap16 | 598.0 | 15.5 | 4.1 | 0.19 | 59.0 | 8.7 | 510.6 | 85.4% |
| k64swap32 | 1135.8 | 15.3 | 4.9 | 0.19 | **118.1** | 11.7 | **985.7** | 86.8% |

- 换入单价 = (1135.8−97.8) / 32 = **32.4 ms/步/adapter**，其中 GPU 传输 118.1/32 = 3.7 ms，
  其余 ~28.7 ms 是 288 次逐条 pageable H2D 的 CPU 开销 → **T16 理论收益：762.9 → ~62+ε**。
- 验收全过：h2d_copy 次数/步 = 基线 77 + swap×288（k64swap32=9293，四档精确到个位）；
  TPOT 随 swap 单调；k64swap0=62.4 ∈ 历史 50~65 ms 区间；plain wait>0 且随 swap 线性；峰值 < 45 GiB。

---

## 5. 现在怎么跑（终态，一条命令）

```bash
cd /root/autodl-tmp/code/multi_lora-main
HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -u multi_lora/examples/fix_batch_test_each_cost.py 2>&1 | tee -a multi_lora/tiers_swap.log
```

- 无任何环境变量开关、无断点：单进程顺序跑五档，末尾打三张表（row 只在内存里）。
- 实测 ~35 min（k1 ~4、k8 ~5、k32 ~7、k64swap32 ~11、k64swap0 ~7）。
- 分桶：`.venv/bin/python multi_lora/examples/analyze_trace.py traces/02_fix_batch_each_cost`
  （会跳过自己上次生成的 `.bucketed` 文件，属正常）。产物：3 个 CSV + 每份 `.bucketed.json.gz`
  （Perfetto 打开，五桶五色）。
- 依赖顺序：先 `make_qkvo_variants.py` 生成 128 个 bf16 变体（脚本缺变体会启动即退出并提示）。

保留的内存管理（不是调试，别删）：窗口末尾 `del prof + gc + malloc_trim`、
档末 `release_tier`、B-2 前 `empty_cache`、`StaticCache` 按 `steps` 开、TIERS 步数 15
（调回 25 = 主机内存重新顶爆，见 §3.2）。

---

## 6. 清理记录（第 3 步已完成，别再找这些）

已删：`TIER`/`REPORT`/`RERUN` 分支、`run_tier_alone` + subprocess driver、`load_rows` + `rows/` 落盘、
`assign_names`、`SOLO_ADAPTER`、`ADAPTER_COUNTS`、`host_mem`/`gpu_mib` 及其打印、
`import pdb/os/subprocess/Counter`、`manager.py` 的 `_has_peft_layers` 打印、
trace 目录里的 `old_naming`/`old_fp32`/`old_25step`/`rows`。
`analyze_trace.py` 的分组正则已改成认 `k(\d+)swap(\d+)`（含 `graph`/`gplain` 阶段）。
`run_graph_tier` 由 `bench_decode_n64.py` 调用（graph 对比工况，见 §7）。

---

## 7. 与这条工况相关的 T 任务（长期表见 `OPTIMIZATION_TODO.md`）

| 编号 | 任务 | 状态/与本轮关系 |
|---|---|---|
| T9 | 搭换入工况 + 复测 T3 | ✅ **结项**：五档全量数据在 §4 |
| T3 | 换入不再全槽 `zero_()`，只清缺失模块 | ✅ 结项（代码+用例+性能数字齐） |
| T16 | **合并 H2D（下一步，必做）** | 分母已齐：32.4 ms/步/adapter，其中传输仅 3.7 ms；9216 次/步 → 合并到个位数 |
| T17 | copy stream + non_blocking + 双缓冲 | 排在 T16 之后 |
| — | **graph 下的换入换出对比** | ✅ 机制已实机验证（10-31 晚 `traces/04_decode_n64/v7`）：每步 graph 外 `_setup_forward_batch` + replay 跑通，graph/gplain 窗口的 kernel 序列与 GPU 时间逐位一致；Graph 省 launch ~110 ms/步（CPU 813 vs 913），但 ~730 ms 换入串行原样保留——**graph 吸收不了换入成本**。GPU 侧跨进程复现到毫秒（copy_memset 118.06 vs 118.10）。待做：replay-only 基准已加进 `run_graph_tier`（`tpot_replay_only`，下次跑出数）；"H2D 搬进 graph 内（pinned staging + 录制异步拷贝）"是后续设计点 |

已完成的：T1（去 `.item()`）、T2（常驻 metadata）、T3、T4（复用 `mid`）、
T5（真实 decode 的 CUDA Graph，wall 94.37 → 28.41）、T6（graph TPOT 30.21/30.22/30.21，极差 0.01 ms）。

---

## 8. 这一轮踩过的坑（别再走一遍）

1. **`free -h` 的可用内存不是预算**，容器看 `/sys/fs/cgroup/memory.max`（这里 62 GiB）。
2. **PEFT 按文件 dtype 原样常驻注册的 adapter，不会 autocast**；`convert_to_lora_layers` 之后才释放。
   所以显存峰值在 `from_pretrained` 途中，不在稳态。
3. **kineto 退出 `with` 块不还内存**，必须 `del prof` + `gc.collect()` + `malloc_trim(0)`。
4. **profiler 事件量 ∝ H2D 次数 × 5**，高 swap 档一个窗口就二十多 GiB；这不是 GPU 问题，是记录成本。
5. **槽位数 = adapter 数 → 零换入**，任何与换入相关的优化在这种 bench 上都测不出来。
6. **`use_cache=False` 的 decode 窗口不是真实 decode**（不写 KV），与 plain/graph 的 wall/busy 不可直接比。
7. **首个 profiler 窗口贵 30~55 ms**，`wall`/`gap` 不能跨进程比，只能同进程内比。
8. 每次 pageable H2D 都是同步的，`wait_ms` 就是它，别把它当成 CPU 在干活。
9. **跑 bench 用 `.venv/bin/python`**——系统 python 没有 torch/transformers。
10. 两轮"内存爆掉"是两个不同的病：17:25 显存（fp32 变体）、18:2x 主机（profiler 事件叠加），
    症状都出现在 k64swap32，别混为一谈。
