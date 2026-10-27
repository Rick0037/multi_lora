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

**问题**：grid 第一维按**全局最长** `max_len` 展开，但只有 `rank == 0` 的早退，
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

### P0-5 每步重建设备端元数据

**位置**：`multi_lora/batch_info.py:56-88`、`multi_lora/manager.py:177-198`

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
| `num_stages` / `num_warps` | 未传 |
| `EVEN_K` 编译期判断 | 无，K-loop 每块都带 mask |
| `tl.multiple_of` / `tl.max_contiguous` | 无 |
| `tl.assume(stride > 0)` | 无 |
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
可改 pinned 内存 + 独立 copy stream + `non_blocking=True` + event 等待，
并按下一 batch 的 adapter 集合预取（需要双缓冲，避免覆盖 in-flight kernel 正在读的权重）。

---

## 暂不建议做

| 项 | 原因 |
|---|---|
| A/B 两个 kernel 融合 | 中间量 `[S, max_rank]` 在 prefill 下 S 可达数万，进不了 shared memory；只能省一次 launch，性价比低 |
| 单独写 decode GEMV/BGMV | decode 时 `max_len=1`，现有 kernel 已退化成按 segment 的 GEMV；Punica/SGLang/vLLM 均未专门做 |
| base GEMM 与 LoRA 流 overlap | prefill 时 base 已占满 SM；decode 两者都很短，overlap 空间小 |

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

1. P0-1（B 越界 mask）—— 正确性
2. P0-2（A 的 n 分块早退）—— 一行判断，rank<64 时直接省 4 倍
3. P0-3（去掉 `zero_`）
4. P0-4（segment 早退）
5. P1-1 / P1-2（B、A kernel 加 autotune 与流水）
6. P1-3（复用中间张量）
7. P0-5（常驻 metadata）→ P2-1（CUDA Graph）
8. P1-4（stacked LoRA）
9. P2-2（异步 H2D, base model & lora model overlap）
