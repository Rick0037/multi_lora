# Multi-LoRA 待优化点（按当前代码核对）

> 本文按当前分支实际代码逐条核对，只保留**能在代码里定位到**的问题。
> 历史数据来自 `multi_lora/deprecated/optimization.md`，其中提到的
> `sgemm_lora_a_opti.py` / `sgemm_lora_a_opti_2.py` / `bench_multilora_workload.py`
> 在当前仓库中**已不存在**，相关 1.07–1.19x 收益无法复现，仅作参考。

## 0. 当前状态

- 生产 kernel 由 `multi_lora/kernels/__init__.py` 直接导出 **基础版** A/B kernel。
- A、B 两个 kernel 都没有 autotune、软件流水、`EVEN_K`、向量化提示。
- 每一步 forward 都会在 host 侧重建设备元数据、重新分配中间张量。
- 没有 CUDA Graph。

---

## 0.5 实测基线（batch=64 / 64 个 adapter / Qwen3-4B）

数据来源：`fix_batch_test_each_cost.py` + `analyze_trace.py`，产物在
`traces/02_fix_batch_each_cost/` 与 `traces/04_decode_n64/`。
单位 ms/forward，括号里的数字 = 每步该组的 kernel launch 次数。

| 分组 | decode | prefill |
|---|---|---|
| base_gemm | 11.77 (433) | 40.87 (253) |
| elementwise | 2.46 (1460) | 6.70 (1464) |
| lora_b | 2.75 (144) | 3.02 (144) |
| lora_a | 2.19 (144) | 2.34 (144) |
| cat_kv | 0.85 (145) | 1.06 (145) |
| attention | 0.78 (36) | 1.35 (36) |
| other | 0.56 (149) | 0.76 (150) |
| copy_memset | 0.034 (76) | 0.083 (149) |
| **合计 busy** | **20.6** | **56.0** |
| launch（不含 copy/memset） | 2512 | 2336 |

**从这张表能读出的三件事**：

1. **decode 是 launch-bound**：GPU busy 只有 20.6ms，但一次 forward 要发 2512 次 kernel，
   墙钟约 100ms → 约 80% 时间在等 CPU 发指令，不是 GPU 算得慢。
   平均每次 launch 摊到约 39μs CPU 时间（HF python + aten dispatch + triton launcher）。
2. **LoRA 的真实占比**：decode 里 `lora_a+lora_b = 4.94ms`（busy 的 24%），prefill 里 5.36ms（9.6%）。
   adapter 数 1→64 时 busy 只从 19.52 涨到 20.59（+5.5%），唯一随 n 增长的就是这两项。
3. **GPU 上的拷贝可以忽略**：`copy_memset` 每步 0.034ms（busy 的 0.16%）。
   想省“非推理耗时”不能盯 GPU 的 memcpy/memset，要盯 host 侧的同步与重建（见 P0-5）。

**比数时的口径约束**（都踩过坑）：

- `ProfilerStep*` 容器记录必须剔除：它的 device time = 整窗口跨度（含空档），
  会把 busy 重复计一遍（实测 50 步多算 4511ms）。
- 表2 的 `cpu_ms` 是 **CPU 算子净时间**，不是墙钟（墙钟 = cpu_ms + 容器空档）。
- **`wall` / `gap` 不能跨进程比**：每个进程的第一个 profiler 窗口固定贵 30~55ms
  （kineto/CUPTI 初始化 + 首次逐算子记录）。同一份 GPU 数据在三种跑法下 busy 为
  20.653 / 20.614 / 20.594（≤0.3%），而 wall 为 158.97 / 113.64 / 100.80。
- 端到端延迟（表1）当前报不住：同配置三次 TPOT = 71.83 / 86.75 / 90.01（±20%），
  至少重复 2~3 次取区间。

---

## P0：先做，收益确定或属于正确性问题

### P0-1 B kernel 的 store 缺 n 方向 mask（正确性）

**位置**：`multi_lora/kernels/sgemm_lora_b.py:88`

```python
output_mask = s_offset[:, None] < seg_len     # 缺少 n_offset < N
partial_sum += tl.load(output_ptr, mask=output_mask)
tl.store(output_ptr, partial_sum, mask=output_mask)
```

**问题**：grid 的 n 分块数是 `cdiv(out_dim, BLOCK_N=256)`。当 `out_dim % 256 != 0`，
最后一块的 `n_offset` 会超出 `out_dim`，此时 store 直接写到 `base_output` 的越界位置，
会踩到后面的内存。

**现状**：Qwen3-4B 的 4096 / 1024 / 2560 / 9728 都能被 256 整除，所以没暴露；
换模型（比如中间维度不是 256 倍数）就会出事。

**改法**：

```python
output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
```

A kernel 的 store 已经同时 mask 了 s 和 n，B 补齐即可。

---

### P0-2 A kernel 在 rank < max_lora_rank 时白算

**位置**：`multi_lora/kernels/sgemm_lora_a.py:137-146`、`42-56`

```python
BLOCK_M, BLOCK_K, BLOCK_N = 16, 256, 16
grid = (
    triton.cdiv(max_len, BLOCK_M) * triton.cdiv(N, BLOCK_N),  # N = max_lora_rank
    bs,
)
```

**问题**：grid 用的是 `weights.shape[-2] = max_lora_rank`（默认 64），
而 kernel 内部才 `N = tl.minimum(N, rank * stack_num)`。

- rank=16 时 `cdiv(64,16)=4` 个 n 分块，只有 1 块有效，另外 3 块数据全被 mask 成 0，
  但仍然完整跑完 K-loop 和 `tl.dot`。
- 实测配置里 adapter a 的 rank 就是 16，等于 A 阶段 **4 倍空转**。

**改法**（kernel 内早退，保持 grid 形状不变）：

```python
if pid_n * BLOCK_N >= rank * stack_num:
    return
```

**另一条更彻底的路子**：host 侧按 `rank` 建 grid（需要把所有 adapter 的 rank 传进来），
但会让 grid 依赖数据，和 CUDA Graph 冲突，不建议。

---

### P0-3 换入 adapter 时的冗余清零

**位置**：`multi_lora/pool.py:65-74`

```python
for key in self.A:
    self.A[key][slot].zero_()
    self.B[key][slot].zero_()
for key in weights.modules:
    self.A[key][slot, : weights.rank].copy_(...)
    self.B[key][slot, :, : weights.rank].copy_(...)
```

**问题**：换入一个 adapter 要清零**全部 module**（Qwen3-4B 是 252 个）的两个数组，
共 504 次 `zero_`。但 kernel 读权重时有 `n_offset < rank` mask，padding 区根本不会被读。

**改法**：直接删掉两个 `zero_()`。

**风险**：需要确认所有读取路径都受 rank mask 保护（A 的 `n_offset < N`、B 的 `n_offset < N`）。
当前两个 kernel 都满足。

---

### P0-4 A kernel 缺 segment 越界早退（混合 batch）

**位置**：`multi_lora/kernels/sgemm_lora_a.py:42-56`、`143-146`

**问题**：grid 第一维按**全局最长**`max_len` 展开，但只有 `rank == 0` 的早退，
没有 `pid_s` 越界判断。

混合 batch（例如 1 条 2048-token prefill + 31 条 decode）时，
decode 那 31 个 segment 会分到大量 `pid_s` 已超出自身 `seg_len` 的 program，
输入全 mask，但仍执行 load + MMA + K-loop。

**改法**：

```python
if pid_s * BLOCK_S >= seg_len:
    return
```

纯 decode（`max_len=1`）不会触发，只有混合 batch 才有收益。

---

### P0-5 每步重建设备端元数据（T1 已做一半，剩下常驻化待做）

**位置**：`multi_lora/batch_info.py`（`to_kernel_input_metadata`）、
`multi_lora/manager.py`（`_setup_forward_batch`）

**问题**：每个 forward step 都执行

```python
weight_indices = torch.tensor(pool_batch.weight_indices, device=device, ...)
lora_ranks     = torch.tensor(pool_batch.lora_ranks, ...)
scalings       = torch.tensor(pool_batch.scalings, ...)
seg_lens       = ...
seg_indptr     = torch.zeros(bs + 1, ...)
seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
```

外加 `pool.build_batch_info` 的 Python 循环和 `_setup_forward_batch` 遍历全部 252 个
wrapper 调 `set_lora_info`。全部是 host 操作，之后再把小张量搬上 GPU。

**改法**：初始化时按 `B_max` / `K` 预分配一次，后续原地更新前 `bs` 个元素：

```python
weight_indices_gpu = torch.empty(B_max, device=cuda, dtype=torch.int32)
lora_ranks_gpu     = torch.empty(K,     device=cuda, dtype=torch.int32)
scalings_gpu       = torch.empty(K,     device=cuda, dtype=dtype)
seg_lens_gpu       = torch.empty(B_max, device=cuda, dtype=torch.int32)
seg_indptr_gpu     = torch.empty(B_max + 1, device=cuda, dtype=torch.int32)
```

- `weight_indices` / `seg_lens` / `seg_indptr`：每 batch 只改前 `bs` 个。
- `lora_ranks` / `scalings`：只有换入或驱逐时才需要更新。

**额外收益**：地址稳定，是 CUDA Graph capture 的前置条件。

#### T1 已完成（已落到代码里）

`to_kernel_input_metadata` 里的段长、前缀和、最大值全部改在 **CPU 上算**：

- 删掉 `seg_lens.max().item()` → 每步少 **1 次 GPU->CPU 同步**（这是本条真正的收益，
  在 busy 上看不出来，只能从 墙钟 / 同步次数 上看）
- 删掉 `torch.cumsum` 与 `seg_lens.to(device)` 的 DtoD
- `seq_lens` 的参数语义改为 **CPU 整数序列**（`list` / CPU tensor）；
  传 CUDA tensor 仍兼容，但会多一次 `.tolist()` 同步

#### 遗留处理情况（已改完）

所有调用方现在都传 **CPU list**（`[1] * bs`），不再建 CUDA 张量：

| 文件 | 改动 |
|---|---|
| `manager.py` | `[seq_len] * batch_size`（替掉 `torch.full(..., device=...)`） |
| `examples/fix_batch_test_each_cost.py` | `decode_lens = [1] * BATCH_SIZE` |
| `examples/bench_add_batch_size.py` | 5 处（含 `grouped_decode` 里的 `[1] * idx.numel()`） |
| `examples/bench_qwen4b_e2e.py` | `decode_lens = [1] * batch` |
| `examples/inference_qwen4b_lora.py` | `decode_lens = [1] * ids.size(0)` |
| `examples/bench_one_decoder_layer.py` | `decode_lens = [1] * BATCH_SIZE` |

**不改**：`deprecated/profile_multilora.py`、`deprecated/bench_multilora_workload.py`
仍传 CUDA tensor，走 `to_kernel_input_metadata` 的 `.tolist()` 兼容分支。
已归档代码不追求收益，但要保证能跑（兼容分支因此不能删）。

---

## P1：kernel 本身

### P1-1 B kernel 没有任何优化

**位置**：`multi_lora/kernels/sgemm_lora_b.py:113-120`

固定 `BLOCK_S=16, BLOCK_N=256, BLOCK_R=16`，无 autotune、无 `num_stages`、无 `EVEN_K`。

B 的输出维度是 `out_dim`（Qwen3-4B 下 1024–9728），比 A 的 rank 维度大得多，
是单 kernel 里最贵的一个。可以把 A 计划加的优化整套搬过来。

### P1-2 A kernel 没有任何优化

**位置**：`multi_lora/kernels/sgemm_lora_a.py:137-170`

对比 `deprecated/optimization.md` 里记录的 11 项，当前一项都没接入：

| 项 | 现状 |
|---|---|
| autotune | 无 |
|`num_stages` /`num_warps` | 未传 |
|`EVEN_K` 编译期判断 | 无，K-loop 每块都带 mask |
|`tl.multiple_of` /`tl.max_contiguous` | 无 |
|`tl.assume(stride > 0)` | 无 |
| x load 用 `.cg` | 无 |
| 短段早退 | 只有 `rank == 0`（见 P0-4） |

### P1-3 每 step 每 module 重新分配中间张量

**位置**：`multi_lora/kernels/sgemm_lora_a.py:148`

```python
output = torch.empty((M, N), device=x.device, dtype=x.dtype)
```

decode 阶段 252 个 module × 每 step 一次分配，全是 host 开销。
可在 manager 侧按 `(M, max_rank)` 预分配复用。

### P1-4 q/k/v/o 没有 stacked LoRA

**位置**：`multi_lora/pool.py:112-117`（每个 `module_key` 一套 buffer）

Qwen3-4B 每 step 是 36 层 × 7 个 module × 2 个 kernel ≈ **504 次 launch**。
q/k/v 的 `in_dim` 相同，可以合并 buffer、一次 kernel 算三个投影。

---

## P2：架构级

### P2-1 CUDA Graph

当前没有任何 capture/replay。需要先完成 P0-5（地址稳定）。
decode 阶段形状固定，收益最大。

### P2-2 adapter 换入的 H2D 未隐藏

**位置**：`multi_lora/pool.py:65-74`

pageable 内存 + 默认流同步 `copy_`，且在 forward 之前串行执行。
可改 pinned 内存 + 独立 copy stream +`non_blocking=True` + event 等待，
并按下一 batch 的 adapter 集合预取（需要双缓冲，避免覆盖 in-flight kernel 正在读的权重）。

**⚠ 本条保留，不要降级。** 现在测不到收益只是**工况没搭好**：`POOL_SLOTS`=64 个 adapter
全部常驻，稳态每步没有任何权重 H2D（分桶里那 0.034ms 全是元数据小拷贝）。
真实场景显存不够、adapter 数远超槽位数时，这条是必经路径。

**要验证先把工况造出来**：把 `POOL_SLOTS` 调小（如 16）+ 用 adapter 轮换的请求序列，
让每步都有换入换出，否则任何异步改动都无法量化。

---

## 需要实验裁决（不要先下结论）

### E-1 decode 下 A/B kernel 全在算 padding —— 要不要写 GEMV

实测：decode 时 `max_len=1`，而

```python
BLOCK_M = 16                                # sgemm_lora_a.py:136
grid = (cdiv(max_len, BLOCK_M) * cdiv(N, BLOCK_N), bs)   # cdiv(1,16) = 1
```

→ **16 行里只有 1 行是真数据，另外 15 行被 mask 但仍完整走 `tl.dot`**；
B kernel 的 `BLOCK_S=16` 同理。再叠加 P0-2（rank16 时 n 方向 4 块只有 1 块有效），
`lora_a` 的 15.2μs/次、`lora_b` 的 19.1μs/次 里面很可能是 padding 开销而不是真计算。

**零成本裁决实验**（先不写新 kernel）：把 A 的 `BLOCK_M`、B 的 `BLOCK_S` 从 16 改成 4/2/1，
各跑一次 decode（`bench_decode_n64.py`），看 `lora_a`/`lora_b` 两个桶的 ms 能不能跌。

- 跌 → 证实 padding 主导，值得写 decode 专用 GEMV（`tl.sum` 归约，不走 `tl.dot`）
- 不跌 → 说明已到带宽/launch 下限，停在早退优化即可

### E-2 相同 adapter 的请求合组

现在 grid 第二维 =`bs`,64 行各开一份 program；把同 adapter 的行排在一起（16 个 adapter × 4 行）
能让 M 维真填满，launch 也从 288 降到 `2 × 16 × 分块数`。
**代价**：输入要 gather 成连续块（多一次 copy）。建议等 E-1 出结果再做 —— 两者解的是同一个
“计算强度不足”。

---

## 暂不建议做

| 项 | 原因 |
|---|---|
| A/B 两个 kernel 融合 | 中间量 `[S, max_rank]` 在 prefill 下 S 可达数万，进不了 shared memory；只能省一次 launch，性价比低 |
| base GEMM 与 LoRA 分 stream overlap | **实测反驳**：decode 的 80% 是 CPU 发不出指令（busy 20.6 vs 墙钟 100），不是 GPU 并行度不足；分 stream 只会多付 event 同步 |
|`torch.compile` | 自定义 triton kernel + KV cache 会变长 → graph break / 反复重编译；同样是为了消 launch 开销，CUDA Graph 更对症 |
| 手写小算子融合（norm/rope/silu/mul） |`elementwise` 确实有 1460 次/步、1.7μs/次，但 CPU 那部分 CUDA Graph 一次性就能拿到；先做 P2-1,再评估剩下的 2.46ms GPU 值不值得手写 |

> 已移除:“单独写 decode GEMV”不再归入本表，改由 E-1 实验裁决。

---

## 验证方法

**正确性**（每次改 kernel 必跑）：

```bash
.venv/bin/python -m pytest multi_lora/tests -q
```

**kernel 级**：`multi_lora/examples/bench_qwen4b_e2e.py`，分别看 prefill 与 decode，
不要只报最好 shape。

**端到端**（真实 4B 模型，prefill + decode）：

```bash
.venv/bin/python multi_lora/examples/bench_qwen4b_e2e.py --num-decode 32
.venv/bin/python multi_lora/examples/bench_qwen4b_e2e.py --num-decode 32 --backend torch
```

**数值一致性**（Triton vs PEFT 分组）：

```bash
.venv/bin/python multi_lora/examples/inference_qwen4b_triton.py --all-lora-modules --compare-peft
```

判据用 `mean_rel < 5e-3`、`max_rel < 5e-2`（bf16 场景）。

---

## 建议顺序

按“收益 ÷ 代价”，并以 0.5 的实测数据为前提：

1. **P0-1**（B 越界 mask）—— 正确性，一行的保险
2. **P0-2 + P0-4**（两个早退）—— 各加一行 `return`，rank<64 时直接省 4 倍空转
3. **P0-3**（删 504 次 `zero_`）—— 只删代码，无风险
4. **E-1 实验**（`BLOCK_M/BLOCK_S` 改 4/2/1 跑 decode）—— 不改结构，用来定要不要写 GEMV
5. **P0-5**（常驻 metadata）—— 兼两个作用：干掉每步的 `.item()` 同步与重建，
   同时是 CUDA Graph 的**硬前置**（地址必须稳定）
6. **P2-1 CUDA Graph** —— 本清单里收益最大的一招:decode 形状固定 + 2512 次 launch/步，
   基线就是 `bench_decode_n64.py`（busy 20.6 / 墙钟 100 / launch 2587），目标把墙钟压到 30ms 量级
7. **P1-1 / P1-2**（B、A 加 autotune 与流水）—— 主要救 prefill；且必须在 capture **之前** tune 完
8. **P1-3**（复用中间张量）—— 跟着 P0-5 一起做
9. **P1-4 stacked LoRA + 基座 QKV/gate-up 合并** —— 工作量最大（要改基座权重布局），
   但 base_gemm 在 decode 要 433 次 launch 而逻辑上只有 252 个 linear，收益实在
10. **E-2**（同 adapter 合组）、**P2-2**（异步 H2D，先把 `POOL_SLOTS` 换入换出工况搭起来）
