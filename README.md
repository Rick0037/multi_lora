# multi_lora

单进程 multi-LoRA serving 的分阶段实现：同一服务进程内为不同请求绑定不同 LoRA adapter。四个目录是同一条演进路线上的四个快照，**每个都是自包含的 Python 包，可独立安装与运行**（各自 README 内有 独立安装与运行 章节）。

| 目录 | 阶段 | 新增能力 | README |
|------|------|----------|--------|
| `multi_lora_phase1/` | 第一步 | HF + PEFT 串行推理，单 adapter 逐个加载生成 | multi_lora_phase1/README.md |
| `multi_lora_phase2/` | 第二步 | 一个 `PeftModel` 挂多 adapter，同一 batch 内按 adapter 分组 forward | multi_lora_phase2/README.md |
| `multi_lora_phase3/` | 第三步 | registry / CPU 权重库 / GPU 槽位池 / 驱逐 / `MultiLoRAManager` | multi_lora_phase3/README.md |
| `multi_lora/` | 第四步（当前主目录） | `KernelInputMetaData`、可插拔 Torch/Triton 后端、整模型 segment GEMM | multi_lora/README.md |

第四步与第三步的逐项对比见multi_lora/PHASE4_vs_PHASE3.md
