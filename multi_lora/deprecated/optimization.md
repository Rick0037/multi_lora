# 当前multi lora还存在的优化点汇总

> 本文整理截至 2026-06-29 的代码核对与已有 benchmark/profile 结果。标为“仍存在”的结论应在具体改动前再次以当前分支验证。

## 基线 benchmark 与结论

### H100 的模型推理结果：A 使用 opti2，B 使用原始 Triton

注意A为lora a gemm, B为lora b gemm，opti2为我已经优化了一波的`sgemm_lora_a_opti_2.py`，其具体的优化策略见下一节<<已有 A-kernel 优化及其适用性>>

工作负载为合成 Qwen2-7B（`hidden=3584`，q/k/v/o 共 28 层），benchmark脚本见`bench_multilora_workload.py`, 采用五大multi lora相关经典场景构造的workload特点，以完整测试multi lora推理性能：

- adapter 请求分布：uniform / distinct / skewed(Zipf)
- max_loras_per_batch（一个 batch 内并存的 distinct adapter 上限）
- prefill（变长 seq）/ decode（seq=1）
- 混合 rank（不同 adapter 不同 rank）
- base-only 基线（不挂 LoRA），用于量化 LoRA 额外开销。

以下结果是整模型 forward 的 p50/p99 延迟与吞吐：

| 场景 | A kernel | p50 (ms) | p99 (ms) | tok/s | opti/orig |
| :-- | :-- | --: | --: | --: | --: |
| prefill / uniform | orig → opti | 104.8 → 93.3 | 150 → 133 | 78.8k → 88.6k | 1.12x |
| prefill / distinct | orig → opti | 105.1 → 98.3 | 149 → 145 | 79.4k → 85.8k | 1.07x |
| prefill / skewed (zipf) | orig → opti | 55.9 → 46.9 | 107 → 96 | 131k → 154k | 1.19x |
| prefill / distinct-maxlpb8 | orig → opti | 137.3 → 128.9 | 208 → 200 | 57.9k → 61.5k | 1.06x |
| prefill / base-only | orig → opti | 24.6 → 24.5 | 28.6 → 26.6 | 338k | 1.00x |
| decode / uniform | orig → opti | 61.1 → 63.3 | 99 → 102 | 1053 → 1016 | 0.96x |
| decode / distinct | orig → opti | 59.5 → 61.6 | 105 → 108 | — | 0.96x |
| decode / skewed | orig → opti | 28.4 → 30.3 | 73 → 75 | 2358 → 2209 | 0.94x |
| decode / base-only | orig → opti | 10.7 → 12.5 | — | 6000 | 0.85x |

关键解读：

- sgemm_lora_a_opti_2.py带来的A kernel 的优化在 prefill 端到端可带来 **1.07–1.19x**；A 只是整条链路的一部分，所以对整条推理链路或整个模型的推理速度提升小于单独测该kernel 的 1.5–2.4x。
- decode 的 opti 略慢（0.94–0.96x）。序列长度为 1 时，大 block、`.cg` 与 autotune 没有足够收益，反而增加开销；不能把 prefill 的 kernel 配置直接用于 decode。
- prefill 的 base-only 约 1.00x，说明没有 LoRA 时 kernel 中 `rank == 0` 的early exit生效了；decode base-only 的差异主要是约 10 ms 的 launch-bound 路径中的 autotune 噪声。
- 真正的大头通常是 adapter 换入（后续会有针对性的优化建议），A kernel其实是小头了。例如 uniform 的 104.8 ms 相比 base-only 的 24.6 ms，多出的约 80 ms 主要来自每个 batch 将 adapter 从 CPU 重载到 GPU pool。Zipf 分布的热点 adapter 更少换入，因而接近快一倍；distinct/maxlpb8 的 adapter 多样性最大，最慢。

### 已有 A-kernel 优化及其适用性

`sgemm_lora_a_opti2.py` 的主要优化如下。它们目前主要服务 prefill；其中 group-M swizzle 在该 workload 未取得收益。

| 优化 | 代码体现 | 解决的问题 | 已有结论 |
| :-- | :-- | :-- | :-- |
| 自动调参 | `_build_configs()`、`@triton.autotune(key=["N", "K", "stack_num"])` | 固定块大小无法覆盖不同 shape | 核心收益来源 |
| 软件流水 | `num_stages=2~4` | 大 K 规约的访存延迟 | A100 上关键，可利用 `cp.async` |
| warp 调优 | `num_warps=4/8` | 占用率与寄存器资源平衡 | 应与块大小联合 autotune |
| 编译期 `EVEN_K` | `K % BLOCK_K == 0` | K loop 中的 mask/谓词 | 命中时可消除 K-loop mask |
| 编译期 K | `K: tl.constexpr` | 让 `EVEN_K` 可静态判断 | 配合上一项 |
| 向量化访存 | `tl.multiple_of`、`tl.max_contiguous` | 标量访存无法合并 | 促成宽访存 |
| 编译器约束 | `tl.assume(stride > 0)` | 不必要的分支与地址计算 | 简化生成代码 |
| L1 策略 | x load 使用 `.cg` | 低复用 x 污染 L1 | prefill 可减少缓存抖动 |
| fp32 累加 | `tl.zeros(..., tl.float32)` | 数值精度 | 保留原语义 |
| 短段早退 | `if pid >= num_pid_s * num_pid_n: return` | 越界 block “计算后再 mask” | 变长/decode 有效；已有 decode-bs512 +19% 结果 |
| group-M swizzle | `_swizzle_pid(..., GROUP_SIZE_S)` | L2 locality | 当前 autotune 一律选 1，暂不值得优先投入 |
| 稳定 benchmark | autotune `do_bench(rep=200)` | 相近 config 选择抖动 | 降低调参噪声 |

## P0级优化选择：multi lora调用gemm的路径、元数据与 adapter 换入

### P0.1 未优化的B kernel 在multi lora推理的关键路径

当前multi lora文件夹从 `kernels/__init__.py` 导入基础 `sgemm_lora_a.py` / `sgemm_lora_b.py`，而非 `sgemm_lora_a_opti.py` 或 `sgemm_lora_a_opti_2.py`。因此：

- 基础 A kernel 固定 `BLOCK_S=16`、`BLOCK_K=256`，没有调参和 `num_stages` 流水。
- 基础 A grid 按全 batch 的 `max_len` 展开，短 segment 的无效 block 会完整执行 K-loop 后才被 store mask 丢弃。
- 同时，B kernel 没有优化版本，是当前最大的单 kernel 候选收益点， 可以考虑迁移A kernel的优化策略到B kernel

### P0.2 混合 prefill/decode 的越界 program

基础 A kernel 的 grid 为：

```python
grid = (
    triton.cdiv(kernel_input_metadata.max_len, BLOCK_S) * triton.cdiv(R, BLOCK_R),
    kernel_input_metadata.bs,
)
```

每个 segment 都分到按全局最长 `max_len` 计算的 `pid_s`。现有 kernel 有 `rank == 0` early return，却没有 segment 边界判断。混合 batch 中，例如一条 2048-token prefill 加 31 条 decode，会产生大量 `pid_s` 已超出 decode segment 的 program；其输入全 mask，但仍执行 load、MMA 和 K-loop。

在计算出 `pid_s` 后应加入：

```python
if pid_s * BLOCK_S >= seg_len:
    return
```

纯 decode 时 `max_len=1` 不会暴露此问题；混合 batch就会， 这是一个bug fix的点。

### P0.3 每步重建 GPU 元数据

每个 forward step 都走 `_setup_forward_batch` → `build_kernel_input_metadata` → `kernel_input_metadata_from_pool_lists`新分配weight_indices,lora_ranks,scalings,seg_lens等python对象，这些全是 host 侧的操作（即CPU 操作），而后又搬到CUDA上面：

```python
w = torch.tensor(weight_indices, device=device, dtype=torch.int32)
ranks = torch.tensor(lora_ranks, device=device, dtype=torch.int32)
scal = torch.tensor(scalings, device=device, dtype=dtype)
```

建议：

- 预分配 GPU tensor, 初始化时按最大 batch B_max 和 max_loras_per_batch=K 创建一次
```
weight_indices_gpu = torch.empty(B_max, device="cuda", dtype=torch.int32)
lora_ranks_gpu = torch.empty(K, device="cuda", dtype=torch.int32)
scalings_gpu = torch.empty(K, device="cuda", dtype=dtype)
seg_lens_gpu = torch.empty(B_max, device="cuda", dtype=torch.int32)
seg_indptr_gpu = torch.empty(B_max + 1, device="cuda", dtype=torch.int32)
```
之后 KernelInputMetaData 引用这批稳定地址的 tensor，不用每个推理step都构造一个新的tensor，如果需要更新这些数据，比如：
- weight_indices：每 batch 会随请求 adapter 组合变化，但我只需要更新前 bs (< B_max) 个元素。seg_lens_gpu和seg_indptr_gpu同理
- lora_ranks/scalings：只有 pool slot 装入新 adapter 或发生驱逐时才需要更新；若 adapter 集合和 slot 映射未变，可完全复用。

以此，可消除 CUDA allocator、GPU对象创建，为后续加入CUDA Graph的capture提供静态地址，Graph replay 要求 kernel 读到的 tensor 地址稳定，如果每轮 build_kernel_input_metadata() 都分配新地址，不可能 capture，更别说replay了；采用预分配 tensor 后，Graph 可以在固定 bs、固定 shape 的 decode 阶段复用。

### P0.4 adapter H2D 切入带来的开销未被隐藏

当前换入流程在 forward 前串行执行，使用普通 pageable CPU 内存和默认流同步 `copy_`：

```python
a_gpu.zero_()
b_gpu.zero_()
a_gpu[: aw.rank].copy_(a_cpu.to(dtype=self.dtype))
b_gpu[:, : aw.rank].copy_(b_cpu.to(dtype=self.dtype))
```

建议使用 pinned CPU 权重、专用 copy stream、`non_blocking=True` 和 event：在真正读取该 slot 前由计算流等待 copy 完成。更进一步，依据下一 batch 的 adapter 集合预取；注意需要双缓冲或 slot/event 生命周期管理，避免覆盖仍被 in-flight kernel 读取的权重。

prefill 可把 copy 隐藏在当前或上一步的 attention/base GEMM 之下。decode 单步计算太短，单步内通常无法掩盖 H2D，应优先让热点 adapter 常驻或跨多个 step 提取。

## P1：kernel launch 次数与架构级吞吐

### P1.1 A→B 中间量与多余的launch（在lora场景正常）

每层每个投影都是 A 写 `mid[S, max_rank]` 到 HBM、B 再读回，形成两个 kernel launch 和一次中间量 HBM 往返。rank≤64 时可以评估 A/B 单 kernel 融合，让中间量保留在寄存器或 shared memory。大家可以想想为什么在这里算子融合A和B gemm不可取

### P1.2 q/k/v/o 没有 stacked LoRA

q/k/v/o 各自维护 buffer，独立执行一对 A/B kernel。decode 中这会形成大量小 kernel。对于相同 shape 的 q/k/v，可借鉴 SGLang 的 stacked LoRA，合并 buffer 和单次 kernel，降低 kernel launch 数量。

### P1.3 接入CUDA Graph

`use_cuda_graph` 字段虽已定义，但没有 capture/replay 逻辑。decode 的核心问题是 launch/host-bound，因此应以常驻、形状稳定的 metadata为前置条件 (P0.3)，捕获 decode cuda graph并replay。

## P2：细节开销、rank 与 decode 路径

### P2.1 冗余清零

kernel 已按 `rank` mask，padding 区不会被读取，因此每次 slot 换入前对整块 `a_gpu` / `b_gpu` 调用 `zero_()` 没有必要。可移除；不过也需通过测试确认所有读取都受 rank/边界 mask 保护。

相关代码见multi_lora/multi_lora/pool.py的_load_uid_to_slot()，这里 A_buffers[key][buffer_id] 的形状是 [max_lora_rank, in_dim]，B_buffers[key][buffer_id] 是 [out_dim, max_lora_rank]。当新 adapter 的实际 rank=16、但 pool 的 max_lora_rank=64 时，代码先把整个 64-rank slot 清零，再仅复制前 16 列/行的权重。
```
for key, (a_cpu, b_cpu) in aw.modules.items():
    if key not in self.A_buffers:
        self.finalize_buffers(weight_store)
    a_gpu = self.A_buffers[key][buffer_id]
    b_gpu = self.B_buffers[key][buffer_id]
    a_gpu.zero_()
    b_gpu.zero_()
    # 只拷贝有效 rank 区域，其余 padding 保持 0
    a_gpu[: aw.rank].copy_(a_cpu.to(dtype=self.dtype))
    b_gpu[:, : aw.rank].copy_(b_cpu.to(dtype=self.dtype))

self._slot_meta[buffer_id] = (aw.rank, aw.scaling)
```
因此应该可以如下修改:

1. 直接移除agpu.zero_()和bgpu.zeor_(), 反正后续的`: aw.rank`会保证读数据安全

### P2.2 decode阶段再写一个单独的 tall-skinny GEMV

当前 prefill/decode 共用 `_sgemm_lora_a_kernel`：decode 元数据为 `seg_lens=ones(bs)`，`BLOCK_S=16` 。考虑可以将其作为单独 workload 评估 grouped GEMV/BGMV + split-K。然而我看到sglang/vllm/punica都没有这样做，我猜测是收益不大，感兴趣可以试一下

Punica 的经验是：SGMV kernel实现以 segment 表达连续 token，适合某 adapter 有较多 token 的 prefill；BGMV kernel实现是每 token 一个 block，更适合每 token adapter 都可能不同的 decode

punica里面有一个cutlass的grouped gemm来完成lora gemm，且也有BGMV的代码，我看了一下，这个BGMV适合batch 里每个 token 都可能来自不同 LoRA的 decode 场景, 该grouped gemm适合某个 LoRA 有很多 token的prefill场景，可以考虑将本项目内的lora a gemm和lora b gemm在“不同 LoRA的 decode 场景”换成punica的BGMV，在“某个 LoRA 有很多 token的prefill场景”换成cutlass grouped gemm以获得这俩场景的性能提升

代码在https://github.com/punica-ai/punica/tree/master/csrc/bgmv，https://github.com/punica-ai/punica/blob/master/csrc/sgmv/sgmv_cutlass.cuh

### P2.3 base GEMM 与 LoRA stream overlap

当前 `y_base = base_layer(x)` 后才计算 LoRA delta，仍为串行。但是此项优化的优先级应最低，因为prefill 时 base GEMM 已接近占满 SM，另一流大多仍会排队

## 其他可借鉴的实现方向

### SGLang 的新版 chunked SGMV

旧版 SGMV 与当前实现类似。新版 `chunked_sgmv_shrink.py` / `chunked_sgmv_expand.py`（服务参数 `csgmv`）可作为后续参考：

- `permutation` 将逻辑 LoRA 行映射到物理行，无需令 x 中 token 按 LoRA 连续；
- host 先把长段切成固定大小 chunk；kernel 的 M 方向不再分块
- 使用 `pid_s >= num_segs`、`seg_start == seg_end` 等 early exit。
- 以 `(K, R, num_slices, chunk_size)` 查调优表，并缓存 Triton 编译结果。
- CUDA Graph 下固定 grid 以便 capture。

### S-LoRA / Punica / vLLM 在multi lora实现上的调研（需要了解）

- S-LoRA 的 MBGMM 可直接从 unified paging pool 分段读取不同 rank 的权重，减少统一 padding。
- vLLM：FP8 LoRA GEMM、fused MoE LoRA op。（如若进一步为此multi lora支持新的模型，建议从moe lora和fp8 lora入手更加有吸引力）

# torch profile验证以上优化点的存在

## 已有 timeline 数据

timeline获取脚本是 `profile_multilora.py`

### Prefill：bs=16、变长 seq、原始 kernel

注意：此profile数据在使用了我优化的`sgemm_lora_a_opti_2.py`的基础上而得

Self CUDA 1.10 s，Self CPU 2.20 s。

| 操作 | Self CUDA | 调用数 | 解释 |
| :-- | --: | --: | :-- |
| `aten::mm`（base GEMM） | 356 ms（32%） | 2240 | base 计算主体 |
| `_sgemm_lora_b_kernel` | 345 ms（31%） | 2240 | B 比 A 更贵且尚未优化 |
| `_sgemm_lora_a_kernel` | 290 ms（26%） | 2240 | 生产中仍是基础版 |
| `aten::copy_` | 62 ms | 24576（1229/step） | adapter H2D 重载 |

### Decode：bs=64、seq=1

- 冷池（每 iteration 更换活跃集）：Self CUDA 172 ms，Self CPU 1.55 s，CPU 时间约为 GPU 的 9 倍。
- warm pool（固定活跃集，无驱逐）：Self CUDA 98.9 ms（约 4.9 ms/step），Self CPU 488 ms（约 24 ms/step）。GPU 大部分时间空闲，明确为 launch/host-bound。
- warm 后 `aten::copy_` 从每 step 1229 次降至 8 次，支持“冷池 copy 主要来自驱逐重载”的判断。

## 逐项证据与判定

| 优化点 | 判定 | profile 证据 |
| :-- | :-- | :-- |
| B kernel 未优化 | 已坐实 | B 345 ms，高于 A 的 290 ms，接近 base GEMM |
| 逐步 host metadata 重建 | 已坐实 | `build_kernel_input_metadata` 约 0.87 ms/step；`aten::to` 628/step、`empty` 279/step、`reshape` 336/step |
| adapter切换的H2D操作未被 overlap | 已坐实 | 冷池 `aten::copy_` Self CPU 535 ms（34%）与 HtoD 51 ms，发生在 forward 前 |
| q/k/v/o 未合并 | 已坐实 | 每 step 224 次 Triton launch；20 step 合计 `cuLaunchKernel` 4480 次 |
| CUDA Graph 未使用 | 已坐实，优先级高 | warm decode GPU 4.9 ms/step 对 CPU 24 ms/step |
| `zero_()` 冗余 | 已坐实 | `zero_` 449–460/step，`fill_` 8980 次，CUDA 12.4 ms；padding 不被 rank mask 读取 |

## 建议的 profiler 验证方法

每项优化应在相同 workload、warmup 和 iteration 下对比改动前后的性能，并同时收集：

1. PyTorch profiler 的 CPU/CUDA activity、shape 与 stack，导出 Chrome trace。
2. `cuLaunchKernel` 数、每 token kernel launch 数、CPU self time。
3. HtoD memcpy 时间、次数和 copy stream 与 compute stream 的 timeline overlap。
4. Triton A/B 的 self CUDA time、base GEMM 时间、SM 利用率和显存峰值。
5. 数值对照：Triton 输出与 torch reference 的误差，以及整模型输出/生成一致性。

对 H2D overlap 的验收不仅仅是 `Memcpy HtoD` 总时长下降，而是 timeline图上 copy那一块与 base GEMM/attention真正地在时间线上overlap了起来

# 建议的优化路线

## 第一波：低风险、快速落地

1. 修复基础 A kernel 的 `pid_s * BLOCK_S >= seg_len` early return
2. 将 `sgemm_lora_a_opti_2.py` 的已验证能力接入生产 A kernel；建立 prefill/decode 分开 dispatch，避免 decode 使用不合适的 opti config。
3. 为 B kernel 增加等价的A kernel优化，比如 autotune、`EVEN_K`、向量化、软件流水与 segment 早退。B 是优先级最高的单 kernel 项。
4. 移除权重换入 `zero_()`，将 CPU adapter 权重置为 pinned，并引入 copy stream + event 的异步 H2D。

预期：prefill kernel 有明显收益，adapter 抖动下降。

## 第二波：decode 主路径

1. 常驻 GPU metadata 并原地更新，移除每 step Python/小张量创建。
2. 在稳定 metadata、buffer 地址与 batch 上限的条件下实施 CUDA Graph capture/replay。
3. 合并 same-shape 的 q/k/v LoRA（stacked LoRA），减少 per-layer 微 kernel launch。
4. 为 adapter 调度加入跨 batch prefetch、slot 生命周期与 double buffering。

预期：decode 的 CPU/launch 开销显著下降。风险中等，核心难点是 metadata 生命周期、动态 shape 回退与 graph-safe 内存地址。

## 第三波：高复杂度 kernel 深挖

1. cutlass grouped gemm和bgmv在相应场景的接入
2. 调研并试验 chunked SGMV 的 permutation/chunking，解决混合长度 batch 的全局 `max_len` 浪费。
3. 评估 FP8 LoRA GEMM、fused MoE LoRA以接入新的lora模型

## 每步的质量门槛

- 正确性：使用 `unittest/test_triton_sgemm.py` 对照 Triton 与 torch
- kernel 性能：使用 `examples/bench_sgemm_lora_a_opti.py`，分别报告 prefill 与 decode，不能只报告最好 shape。
- 端到端：报告 p50/p99、TTFT、TPOT、tok/s，以及 GPU/CPU self time 和显存峰值。

# 实验数据集与指标

## 推荐 workload

和文章开头讲的数据集一致

| 场景 | 变量 | 主要验证目标 |
| :-- | :-- | :-- |
| Decode | `bs ∈ {8, 32, 128}`；distinct LoRA `∈ {1, 2, 4, 8}`；热点/交替/随机 | Graph、metadata、launch 数、GEMV 路径 |
| Prefill | `seq_lens` 覆盖 512–8K；uniform、distinct、Zipf；segment 数与 token 数分开扫描 | A/B kernel、长短段不均、吞吐 |
| 混合 batch | 1 个长 prefill + 多个 decode；多种 `max_len / seg_len` 比 | `pid_s` 早退与 chunking |
| 驱逐/换入 | adapter 库 ≥100；每 batch 2–4 个与更高多样性；pool 容量扫点 | H2D、pinned/copy stream/prefetch、p99 |
| 端到端 | Qwen2-7B 合成层与可装入显存的 phase4 模型；PEFT grouped 对照 | 用户可感知吞吐与延迟 |
| rank 异构 | `rank ∈ {8, 16, 32, 64}`，单 rank 与混合 rank | padding、分桶、动态 grid |

## 指标定义

| 指标 | 含义 |
| :-- | :-- |
| TTFT | 首 token 延迟，重点观察 prefill、adapter miss 与换入 |
| TPOT | 相邻 token 延迟，重点观察 decode launch/host 路径 |
| tok/s | 端到端吞吐；应同时报告 request/token 配置 |
| p50 / p99 | 平均体验与 adapter churn 的尾延迟 |
| GPU util / SM 利用率 | 判断 compute-bound 与 launch-bound |
| H2D 时间与占比 | 包括 CPU self time、HtoD DMA 时间、是否与计算重叠 |
| kernel self CUDA time | 分开记录 base GEMM、LoRA A、LoRA B、zero/fill |
| kernel launches/token | CUDA Graph、融合与 qkv stack 的直接验收指标 |
| metadata CPU time | `build_kernel_input_metadata`、Python 循环、小张量创建 |
| peak allocated/reserved memory | 保证 phase4 在当前 GPU 显存范围内运行 |
| 数值误差 | 与 torch/原始实现的 max/mean error，以及端到端一致性 |

## benchmark脚本

`bench_multilora_workload.py`