# Multi-LoRA 优化任务表

> 只保留**能在代码里定位到**的项。历史数据来自 `multi_lora/deprecated/optimization.md`，
> 其中提到的 `sgemm_lora_a_opti*.py` / `bench_multilora_workload.py` 已不存在，仅作参考。
>
> 最近更新：记录 T23（GQA `repeat_kv` 评估，收益最大但依赖 transformers 私有判据，暂挂）；
> 修正 §5.4 对 `other_gpu` 的错误归因。

---

## 0. 当前进度

```
已完成 15 项：T1-T6 T9-T10 T13-T14 T16-T19 T21
剩余   7 项：T7  T8  T11  T12  T15  T22  T23
暂缓   1 项：T20
```

**两条路线的当前位置**（k64swap32 = 每步换 32 个，k64swap0 = 不换入）：

| 工况 | 当前最优 | wall | busy | tok/s | 卡在哪 |
|---|---|---|---|---|---|
| k64swap32（换入） | v12 | 142.6 | 138.9 | 448.9 | **PCIe 带宽硬顶**（H2D 108.9ms = 76%） |
| k64swap0（纯计算） | v14 | 26.4 | 24.5 | 2424 | base_gemm 15.2ms（**显存带宽受限**） |

两端都已到物理地板，剩下的可动项见 §3。

---

## 1. 测量口径（踩过的坑）

- `ProfilerStep*` 容器事件必须剔除：它的 device time = 整窗口跨度（含空档），会重复计 busy。
- **`wall` / `gap` 不能跨进程比**：每个进程首个 profiler 窗口固定贵 30~55ms。
  同一份 GPU 数据在三种跑法下 busy 为 20.653 / 20.614 / 20.594（≤0.3%），
  而 wall 为 158.97 / 113.64 / 100.80。
- **判断优化效果只信这三项**：`GPU busy`、`replay-only`、`launch 数`。
  实测同一配置两次运行差异 <0.01ms；而 `gplain wall` 同配置实测
  113.5 / 139.6 / 273.3ms（273.3 是主机抢占），完全不可比。
- `tok/s = batch(64) × 1000 ÷ wall`，wall 含 profiler 开销，绝对值偏低，只做行间对比。
- **两套 TPOT 口径**：DynamicCache（历史可比）与 StaticCache+采样（与 graph 同口径）。
  跨版本比较一律用后者，否则会出现 0.86× 这种口径错位假象。

## 1.1 优化前基线（历史参考，batch=64 / 64 adapter / Qwen3-4B）

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

当时 decode 是 **launch-bound**：busy 20.6ms 但墙钟约 100ms，80% 时间在等 CPU 发指令。
这条结论是后续所有优化的出发点。

## 1.2 T5 CUDA Graph 基线（v5，batch=64 / 64 adapter）

| 窗口 | wall | gpu_busy | gap | gap% | launch/步 |
|---|---|---|---|---|---|
| decode（非 graph） | 94.37 | 20.54 | 73.82 | 78.2% | 2582 |
| graph（replay） | **28.41** | 27.87 | **0.54** | **1.9%** | 2939 |

wall 降 66ms（3.3 倍）。**自此 busy ≈ wall，降 busy 就是降延迟**，这是后续 kernel 优化的前提。

---

## 2. 任务总表

状态：✅ 完成 / ⬜ 待做 / ⏸ 暂缓 / ➖ 合并
"收益"列均来自实测，未实测的标注"预估"。

| 编号 | 任务 | 组 | 状态 | 实测收益 | 生效工况 |
|---|---|---|---|---|---|
| T1 | 去 `.item()`，CPU 预算 lens/indptr/max_len | CPU | ✅ | decode wall 113.64→106.75 | 所有 decode |
| T2 | 常驻 metadata，原地改写 | CPU | ✅ | 0.5~1.5ms | 所有；graph capture 前置 |
| T3 | 换入不再全槽 `zero_()` | CPU | ✅ | 已随 T18 重写整体移除 | 换入 |
| T4 | 复用 A 阶段 `mid` 张量 | Graph 前置 | ✅ | `aten::empty` 295→151/步 | 所有 |
| T5 | CUDA Graph decode capture | Graph | ✅ | wall 94.37→28.41（−69.9%） | 稳态 decode |
| T6 | Graph 验收：TPOT 3 次取区间 | Graph | ✅ | 极差 0.01ms，对齐 3/3 | 稳态 decode |
| T7 | 基座 QKV / gate-up 合并 | launch | ⬜ | **gate/up ≈0；qkv 上限 −0.5ms** / 非 graph −2ms | graph 下仅 qkv 有收益 |
| T8 | 相同 adapter 请求合组 | launch | ⬜ | 取决于 batch 内重复度 | **需 batch > adapter 数** |
| T9 | 搭换入工况 + 复测 T3 | 工况 | ✅ | TIERS 五档已搭 | — |
| T10 | A kernel n 方向早退 | kernel | ✅ | 随 T14 一起（GEMV 路径） | decode |
| T11 | A/B segment 越界早退 | kernel | ⬜ | 0（纯 decode 不触发） | **仅混合 batch** |
| T12 | B store 补 `n_offset < N` mask | kernel | ⬜ | 0（正确性，非性能） | out_dim 非 256 倍数 |
| T13 | BLOCK_M/BLOCK_S 16→4/2/1 实验 | kernel | ➖ | 并入 T14 | — |
| T14 | decode 专用 GEMV | kernel | ✅ | lora_a+b 4.735→4.149（−12%）；BLOCK_N 64→16 再 −0.12 | 纯 decode |
| T15 | autotune + `num_stages` | kernel | ⬜ | decode ≈0 / **prefill 可能较多** | 主要 prefill |
| T16 | 合并 H2D（装箱） | 异步 | ✅ | 被 T17 取代（补零使字节翻倍） | 换入 |
| T17 | pinned + copy stream + non_blocking | 异步 | ✅ | v9/v10：TPOT 732→428 | 换入 |
| T18 | 整块 pinned bank | 异步 | ✅ | v11：TPOT 428→169.8，CPU 433→57.2 | 换入 |
| T19 | 散布融合单 kernel | 异步 | ✅ | v12：TPOT 169.8→142.7，CPU 57→5.3 | 换入 |
| T20 | 预取重叠 | 异步 | ⏸ | 上限 −26ms（预估） | 换入；**需 +1.4GB 显存** |
| T21 | RMSNorm 8 kernel → 1 | 算子融合 | ✅ | graph −1.66ms；**非 graph −23ms（−20%）** | 所有 |
| T22 | add + RMSNorm 融合 | 算子融合 | ⬜ | graph −0.2~0.3ms（预估） | 所有 |
| T23 | 消除 GQA `repeat_kv`（`enable_gqa`） | 算子/kernel | ⬜ | **−3.0ms（microbench 实测）** | decode；随 KV 长度线性放大 |

---

## 3. 剩余项详情

### T22 add + RMSNorm 融合（不依赖私有实现的首选项）

**位置**：`Qwen3DecoderLayer.forward`（transformers 侧），残差加与下一个 norm 相邻：

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)   # ← T21 已融合其内部 8 个 kernel
...
hidden_states = residual + hidden_states              # ← 仍是独立 kernel，72 次/步
```

**做法**：`fused_add_rms_norm(residual, x, weight, eps)` 一个 kernel 完成「加 + 归一化」，
TensorRT-LLM / vLLM 都是这个套路。需重写 `Qwen3DecoderLayer.forward`。

**收益**：graph 下预估 −0.2~0.3ms（launch 已免费，只剩省掉一次 HBM 写回+重读）；
非 graph 下 −72 次 launch ≈ −1.5ms。

**风险**：绑定 transformers 5.x 的内部结构，升级易失效。

### T23 消除 GQA `repeat_kv`（已评估，因依赖私有实现暂挂）

**现象**：v14 的 `other_gpu` 4.93ms 中，`copy_` 占 3.48ms，其中 **72 次（36 层 × k/v）是
`[64, 8, 4, 79, 128]` 的 GQA 展开**，单次 37µs，合计 **≈2.7ms/步**。

**根因**（transformers 5.20.2，`integrations/sdpa_attention.py`）：该文件已有 GQA 分支，但门槛是

```python
return attention_mask is None and key.shape[-1] == value.shape[-1] <= 256
```

decode 必须带 mask（`DecodeStep.attn_mask`，StaticCache 预分配格子里是 0，不 mask 会 attend 到未写入位置），
所以永远走 `repeat_kv`。

**四种走法实测**（B=64 / H=32 / KVH=8 / S=79 / D=128 / bf16）：

| 走法 | attention | repeat | 每层 | ×36 层 |
|---|---|---|---|---|
| repeat + mem_efficient + mask（现状） | 147.6µs | 74µs | 221.6µs | **7.98ms** |
| `enable_gqa` + mask（仍 mem_efficient） | 138.2µs | 0 | 138.2µs | **4.98ms** |
| `enable_gqa` + 无 mask（flash kernel） | 31.8µs | 0 | 31.8µs | **1.14ms** |

`enable_gqa` 与 repeat 两条路径输出 `max_abs_diff = 0.0`。

**两条路**：
- **A（3 行）**：patch `transformers.integrations.sdpa_attention.use_gqa_in_sdpa` 恒返回 True
  → −3.0ms，busy 24.5→21.5（−12%），tok/s 2424→≈2755。graph 需重新 capture。
- **C（几十行）**：换 `flash_attn_with_kvcache` + `cache_seqlens`，KV 布局
  `[B, KVH, L, D]` → `[B, L, KVH, D]`，牵动 `graph_decode.py` 的 capture 路径
  → 再省 3.84ms，attention 段降到 1.14ms（busy 17.6，−28%）。

**为什么暂不做**：
1. A 依赖 transformers 的私有函数名，版本升级即失效（T21 同样问题，但那只是替换实例 forward，不用私有判据）；
2. C 要改 `StaticCache` 布局，风险和改动量都上一个台阶；
3. 收益被换入稀释：k64swap32 只 −2.2%（108.9ms H2D 不动），只有 k64swap0 才有 −12%。
   真正值钱的场景是**长上下文**——repeat 量随 KV 长度线性涨，S=2048 时单项就 ~70ms。

### T15 autotune + `num_stages`（只救 prefill）

**位置**：`kernels/sgemm_lora_a.py` / `sgemm_lora_b.py`

当前固定 `BLOCK_K=256`、未传 `num_stages`/`num_warps`，K-loop 每块都带 mask。
decode 下 lora 已接近带宽地板（见 §5），收益主要来自 **prefill**（S 大、K-loop 长）。
做完同样要重跑 T5/T6 重新 capture。

### T12 B store 补 mask（正确性，优先级高于其余优化）

**位置**：`kernels/sgemm_lora_b.py`（GEMM 路径）

```python
output_mask = s_offset[:, None] < seg_len          # 现在
output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)   # 应为
```

grid 的 n 分块是 `cdiv(out_dim, BLOCK_N=256)`，当 `out_dim % 256 != 0` 时最后一块越界写。
Qwen3-4B 的 4096/1024/2560/9728 都能被 256 整除，所以没暴露；换模型会踩内存。

### T11 segment 越界早退（只有混合 batch 有效）

**位置**：`kernels/sgemm_lora_a.py`（GEMM 路径）

grid 第一维按全局 `max_len` 展开，混合 batch（1 条 2048-token prefill + 31 条 decode）时，
decode 段的大量 program 输入全 mask 但仍跑完 K-loop。加 `if pid_s*BLOCK_S >= seg_len: return`。
纯 decode（`max_len=1`）不触发。

### T7 QKV / gate-up 合并（评估后不建议）

**结论**：gate/up 合并 graph 下 ≈0；**qkv 合并上限 −0.5ms**（graph 下的纯 GPU 收益），
非 graph 再加 −2ms（108 次 launch）。

**单次口径实测**（v14 gplain，M=64；地板 = 权重字节 ÷ 1.008TB/s）：

| GEMM | 单次权重 | 单次地板 | 单次实测 | 效率 | ×36 层/步 |
|---|---|---|---|---|---|
| gate | 49.8MB | 49.4µs | 62.1µs | 80% | 2.24ms |
| up | 49.8MB | 49.4µs | 62.1µs | 80% | 2.24ms |
| down | 49.8MB | 49.4µs | 60.5µs | 82% | 2.18ms |
| o_proj | 21.0MB | 20.8µs | 36.2µs | **57%** | 1.30ms |
| q_proj | 21.0MB | 20.8µs | 32.8µs | **63%** | 1.18ms |
| k / v（各） | 5.2MB | 5.2µs | 10.3µs | **50%** | 0.74ms（k+v） |
| lm_head | 778MB | 772µs | 883.7µs | 87% | 0.88ms |

实测 = 该 `aten::mm` 发出的所有 GPU kernel（含 split-K 的 reduce）之和；效率 = 地板 / 实测。

**分两档**：
- gate/up/down/lm_head 在 80~87%，已到顶。gate/up 的 N=9728 已有 152 个 CTA（128 SM 饱和），
  合并成 N=19456 不会提升，**零收益**；
- q/o/k/v 只有 50~63%，根因是 M=64 时 CTA 不足：k/v 的 N=1024 → 16 CTA，q/o 的 N=4096 → 64 CTA。

**qkv 合并的账**：

```
现在      q 1.18 + kv 0.74 = 1.92ms
地板      31.4MB/层 × 36 / 1.008TB/s = 1.12ms
按 80%    1.40ms                      → 省 ≈0.5ms（−27%）
```

**为什么仍不建议**：
1. 0.5ms 要重写 `Qwen3Attention.forward`：`q_norm`/`k_norm` 得按 `[q|k|v]` 分段后作用在 head_dim 上，
   RoPE 要拆两段正余弦，`past_key_values.update` 要按新 head 布局切 —— 全部绑死 transformers 5.x；
2. `o_proj` 依赖 attention 输出，吃不到这 0.5ms；
3. **FLOPs 与权重字节都不变**，收益只来自 CTA 占用率，天花板就是那 0.5ms；
4. 同样量级的投入，fp8 量化能拿 −4ms 以上、batch 提到 82 能拿 −3.5ms，都不是「拼 GEMM」。

### T8 相同 adapter 合组（需换工况）

**现状**：代码已实现（perm + 分组 metadata，见 git stash「T8 合组」），但被 stash 未合入。

**为什么现在没用**：`TierRotation` 让 64 行覆盖 64 个**不同** adapter（`per = k/4 = 16`，
四个 rank 组各 16 个），是为了把换入压力拉满。没有重复行可合。

**什么时候有用**：batch 行数 > adapter 数（例如 batch=128 × 64 adapter = 每 adapter 2 行），
此时合组能把 lora 的 M 从 1 提到 2，同时 launch 从 288 降到 `2×组数×分块`。
和 **增大 batch** 这个方向是同一条路上的。

### T20 预取重叠（显存不足，暂缓）

把下一步的 H2D 藏进本轮 replay 的 26ms 里，双缓冲需要 **+1.4GB 显存**。
k64swap32 档当前余量约 2~3GB，且已多次出现 `expandable_segments: mapping failed` 警告，
风险过高。收益上限 26ms（等于 replay 时间），实际取决于能否完全重叠。

---

## 4. 版本账

### 4.1 换入优化（k64swap32，graph 窗口，`traces/04_decode_n64/`）

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

每 adapter 恒定 3.40ms（44MB ÷ 12.9GB/s）。**这说明 T16-T19 对工况通用，不是 k64 特化。**

**PCIe 硬顶**：RTX 4090 D，`gen=4`、`width=8` → 理论 15.75GB/s，实测 12.86（82%）。
判据：v10→v11 把 H2D 次数从 1152 降到 32，时间只从 112.7 变到 108.9ms——
**时间只跟字节走**，写法已不亏钱。

### 4.2 算子融合（k64swap0，无换入，`traces/04_decode_n64/`）

v13 = 未融合（对照），v14 = `MULTI_LORA_FUSED_NORM=1`（原 v16，最优一次）。
> 注：该开关现已**默认开**，`=0` 才是关闭。

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

**145 = 36 层 × 4 个 norm（input / post_attention / q_norm / k_norm）+ 1 个 final norm**。
每个 norm 原来 8 个 kernel：bf16→fp32 cast、pow(2)、mean(-1)、+eps、rsqrt、mul×2、fp32→bf16 cast。
融合后每实例 1 个 `_rmsnorm_kernel`（1.70µs/次，145 次共 0.246ms）。

**收益结构完全不同**：
- 无 graph：−23ms 里 **92% 来自少发 1015 次 launch**（≈20.5µs/次 Python 派发），只有 1.9ms 是 GPU 时间；
- 有 graph：launch 已免费，只剩 −1.62ms 纯 GPU 收益。

---

## 5. 为什么剩下这些不动了（物理边界）

### 5.1 k64swap32：PCIe 带宽

```
一步 busy 138.94 = H2D 108.9（76%）+ replay 26.4 + scatter 3.4
wall 142.58，gap 只剩 3.7ms（2.6%），GPU 已无空转
```

108.9ms = 1.4GB ÷ 12.86GB/s，链路是 Gen4 ×8，已用 82%。**再改写法不会更快。**

### 5.2 k64swap0：base_gemm 是访存受限，不是算力不足

| 界限 | 算法 | 数值 |
|---|---|---|
| 显存带宽 | 7.26GB 权重 ÷ 1.008TB/s | **7.2ms** |
| 算力 | 515 GFLOP ÷ 82.6 TFLOPS | 6.2ms |
| 实测 GEMM | | **10.7ms** |

带宽下限 > 算力下限 → **访存受限，已达 roofline 的 67%**。
根因：机器平衡点 = 82.6 TFLOPS ÷ 1.008 TB/s = **82 FLOP/byte**，
而 decode 的算术强度恰好等于 batch 行数 **M**。当前 M=64 < 82，仍在线性区。

**想真正提升，只有两条路（都不在 T 表里，因为属于变更而非优化）：**

| 手段 | batch=64 | batch=82（拐点） | batch=128 |
|---|---|---|---|
| GEMM 时间 | 10.7ms | 7.2ms | 12.5ms |
| 吞吐 | 6.0 tok/ms | **11.4 tok/ms** | 10.2 tok/ms |
| 受限类型 | 带宽 | 平衡 | 算力 |

1. **增大 batch（最有效，需改 workload）**：M 从 64 → 82 吞吐翻倍。
   附带好处：batch=128 时 64 个 adapter 各占 2 行，**T8 合组随之生效**。
2. **量化权重（不需要改 workload）**：fp8 把 7.26GB → 3.63GB，
   M=64 下就能从 10.7ms 降到约 6.5ms（转为算力受限），busy 24.5 → ~20.3（−17%）。
   代价是重新量化 + `torch._scaled_mm` 适配。

### 5.3 lora 部分已到带宽地板

每步读取 ≈ 64 slot × avg_rank 30 × in_dim × 36 层 × 4 module ≈ 2.8GB → 1TB/s 下约 2.8ms。
实测 lora_a+b = 4.15ms，其中 2.8ms 是不可避免的权重读取。

### 5.4 other_gpu 剩下的不是 norm，是 attention 前处理

v14 的 4.93ms 构成：

- `elementwise_kernel` 363 次共 3.93ms，其中 **`aten::copy_` 112 次就占 3.48ms**
  —— 72 次是 GQA 的 `repeat_kv`（见 T23），其余是 mask 拷贝；
- `vectorized_elementwise` 476 次 0.57ms（`fill_` mask、silu、add）；
- `_rmsnorm_kernel` 145 次 0.245ms（T21 已融合）；
- `CatArrayBatchedCopy` 72 次 0.188ms（KV cache 的 `torch.cat`）。

纯 elementwise 的 silu/mul/add 加起来只有 ~0.5ms，再融合收益 <0.2ms。
真正的大头是把 GQA 展开和 mask 重建从图里拿掉（T23）。

---

## 6. 已否决的方案（附证据）

| 项 | 否决理由 |
|---|---|
| A/B 两个 kernel 融合 | 中间量 `[S, max_rank]` 在 prefill 下 S 可达数万，进不了 shared memory；只能省一次 launch |
| base GEMM 与 LoRA 分 stream overlap | 实测反驳：decode 80% 是 CPU 发不出指令（busy 20.6 vs 墙钟 100），分 stream 只会多付 event 同步 |
| `torch.compile` | 自定义 triton kernel + 变长 KV → graph break / 反复重编译；消 launch 用 CUDA Graph 更对症 |
| 手写 A/B kernel 的 tile 调优 | cuBLAS/cutlass 已把 base_gemm 打到访存 roofline 的 67%，手写难超越 |
| QKV / gate-up 合并（T7） | 见 §3 T7：gate/up 已饱和；qkv 上限 0.5ms（CTA 占用），要重写 attention 前处理，投入产出不成比例 |
| 单独调 `BLOCK_M/BLOCK_S`（T13） | 已被 T14 的 GEMV 路径取代 |

> 注：早期把「手写 norm/silu 融合」列入否决，理由是"graph 已消 launch"。
> **T21 实测推翻了这一半**——graph 下仍有 1.62ms 纯 GPU 收益，非 graph 下更有 −23ms。

---

## 7. 验证方法

**正确性（改 kernel 必跑）**：

```bash
HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m pytest multi_lora/tests -q
```

> 必须带 `HF_HOME`：`test_graph_decode.py` 要加载 `sshleifer/tiny-gpt2`，
> 否则会去 HuggingFace Hub 联网下载而卡死（缓存在 `/root/autodl-tmp/hf_cache`）。

| 用例 | 覆盖 |
|---|---|
| `test_graph_decode.py` | graph replay 与直接跑的 logits 逐步对齐；capture 前后常驻张量地址不变 |
| `test_pool_write_many.py` | CPU：布局/bank/tables 不变量、散张量释放 |
| `test_pool_write_many_gpu.py` | GPU：kernel 与 python 两条散布路径各 5 轮逐位对比 |
| `test_triton_sgemm.py` | GEMV 路径（`seq_lens=[1]*8`）与 torch 后端对比 |
| `test_fused_rmsnorm.py` | 融合 RMSNorm 与 transformers 原实现对比（hidden 128/2560，2/3 维输入） |

**性能（单档迭代）**：

```bash
# k64swap32（换入）
HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -u multi_lora/examples/bench_decode_n64.py graph 32

# k64swap0（纯计算；融合 RMSNorm 默认开，对照时加 MULTI_LORA_FUSED_NORM=0）
HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -u multi_lora/examples/bench_decode_n64.py graph 0

# 阶梯表
.venv/bin/python multi_lora/examples/ladder_table.py traces/04_decode_n64/v13 traces/04_decode_n64/v14
```

**五档全量**（每个窗口都产 trace）：

```bash
TQDM_DISABLE=1 HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -u multi_lora/examples/fix_batch_test_each_cost.py
```

**数值一致性（Triton vs PEFT 分组）**：

```bash
.venv/bin/python multi_lora/examples/inference_qwen4b_triton.py --all-lora-modules --compare-peft
```

判据 `mean_rel < 5e-3`、`max_rel < 5e-2`（bf16）。

---

## 8. 环境开关

| 变量 | 默认 | 作用 |
|---|---|---|
| `MULTI_LORA_FUSED_NORM=0` | **开** | 关掉融合 RMSNorm（T21）做 A/B 对照 |
| `MULTI_LORA_FUSED_QKV=1` | 关 | 合并 q/k/v 基座 GEMM（T7，仅 decode；待实测） |
| `LORA_SCATTER_IMPL=python` | `kernel` | 散布切回高级索引参考实现（T19） |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | — | bench 必带，否则 k64 档易 OOM |

---

## 9. 相关文件

| 位置 | 职责 |
|---|---|
| `multi_lora/pool.py` | 槽位池、pinned bank、H2D、GPU 内散布 |
| `multi_lora/kernels/scatter_bank.py` | T19 散布融合 kernel |
| `multi_lora/kernels/rmsnorm.py` | T21 融合 RMSNorm + `fuse_rmsnorm` |
| `multi_lora/kernels/sgemm_lora_a.py` | A 阶段：GEMM + GEMV 双路径 |
| `multi_lora/kernels/sgemm_lora_b.py` | B 阶段：GEMM + GEMV 双路径 |
| `multi_lora/graph_decode.py` | `DecodeStep` + `capture_graph` |
| `multi_lora/examples/fix_batch_test_each_cost.py` | 五档全量 bench（prefill/decode/graph/gplain 四窗口） |
| `multi_lora/examples/bench_decode_n64.py` | 单档迭代（`graph [swap]`） |
| `multi_lora/examples/analyze_trace.py` | trace 分桶 + `tok/s` 列 |
| `multi_lora/examples/ladder_table.py` | Baseline/V1/V2 阶梯表 |
