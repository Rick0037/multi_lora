"""LoRA 线性层:基座 Linear 的输出 + LoRA delta."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_lora.backend.base import LoRABackend

# 进程级开关:关掉后 LinearWithLoRA 只跑基座 GEMM,用于测「不含 LoRA」的基线耗时
LORA_ENABLED = True


def set_lora_enabled(enabled: bool) -> None:
    """全局开/关 LoRA delta 计算;同一进程同一模型,只跳过 delta 那一步."""
    global LORA_ENABLED
    LORA_ENABLED = enabled


def lora_delta(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    *,
    rank: int,
) -> torch.Tensor:
    """朴素 PyTorch 版LoRA delta = scaling * x @ A^T @ B^T(单测 / 数值对照用)."""
    a = lora_a[:rank]  # [rank, in]
    b = lora_b[:, :rank]  # [out, rank]
    flat = x.reshape(-1, x.shape[-1])
    delta = (flat @ a.T @ b.T) * scaling

    out_shape = list(x.shape[:-1])
    out_shape.append(b.shape[0])
    return delta.reshape(out_shape)


class LinearWithLoRA(nn.Module):
    """替换 PEFT 的 LoraLayer;forward = 基座 GEMM + 后端算的 LoRA delta."""

    # baseline * x + x * a * b
    #! 这里的a_buffer, b_buffer 实际上是那个组大 batch 出来的 padding的大矩阵
    #! backend 计算的时候还需要根据索引去分组算
    def __init__(self, base_layer: nn.Module) -> None:
        """base_layer 是 PEFT 包裹下的原始 Linear;A/B 由 manager 每步注入."""
        super().__init__()
        self.base_layer = base_layer
        self.backend: Optional[LoRABackend] = None
        self.A_buffer: Optional[torch.Tensor] = None
        self.B_buffer: Optional[torch.Tensor] = None
        # 参与 QKV 融合时指向共享组,否则 None(见 fuse_qkv_projections)
        self.qkv_group: Optional["QKVProjGroup"] = None
        self.qkv_slice: Optional[tuple] = None

    def set_lora_info(
        self,
        backend: LoRABackend,
        a_buffer: torch.Tensor,
        b_buffer: torch.Tensor,
    ) -> None:
        """每个 forward step 前由 manager 注入:后端 + 按槽位排列的权重 [K, ...]."""
        self.backend = backend
        self.A_buffer = a_buffer
        self.B_buffer = b_buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """基座输出 + LoRA delta;QKV 融合时分派给共享组的合并 GEMM."""
        if self.qkv_group is not None:
            return self.qkv_group.forward(self, x)
        return self.add_delta(x, self.base_layer(x))

    def add_delta(self, x: torch.Tensor, y_base: torch.Tensor) -> torch.Tensor:
        """基座输出已算好(可能是合并 GEMM 的切片),只叠 LoRA delta."""
        if not LORA_ENABLED:
            return y_base
        if self.backend is None or self.A_buffer is None or self.B_buffer is None:
            return y_base
        # 切片出来的 y_base 行 stride 是 6144 而不是 out_dim,reshape 仍是零拷贝 view;
        # 不 contiguous —— B kernel 按 output_stride 写,强转反而多一次 108 次/步的 copy
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y_base.reshape(-1, y_base.shape[-1])
        y_flat = self.backend.apply_lora(
            x_flat, self.A_buffer, self.B_buffer, y_flat)
        return y_flat.reshape(y_base.shape)


class QKVProjGroup:
    """同一 attention 下 q/k/v 三层的基座 GEMM 合并成一次 mm.

    decode 时三层吃同一个 hidden_states,各自做一次 [tokens, in] @ [in, out]。
    合并后 hidden_states 只读一遍,且 N=1024 的 k/v 并进 N=4096 的 q,
    CTA 从 16 提到 96,tile 占用改善(实测每步省 ~0.5ms)。
    三层的 weight 改成同一块 storage 的切片,不额外占显存。
    """

    def __init__(self, wrappers: list) -> None:
        """wrappers 必须按 q/k/v 顺序传入,拼接顺序即切片顺序."""
        if any(w.base_layer.bias is not None for w in wrappers):
            raise RuntimeError("QKV 融合要求 bias=None")
        dims = [int(w.base_layer.weight.shape[0]) for w in wrappers]
        self.weight = torch.cat([w.base_layer.weight for w in wrappers], dim=0)
        offset = 0
        for wrapper, dim in zip(wrappers, dims):
            wrapper.base_layer.weight = nn.Parameter(
                self.weight[offset:offset + dim], requires_grad=False)
            wrapper.qkv_slice = (offset, offset + dim)
            wrapper.qkv_group = self
            offset += dim
        self._cached_x: Optional[torch.Tensor] = None
        self._cached_out: Optional[torch.Tensor] = None

    def forward(self, wrapper: LinearWithLoRA, x: torch.Tensor) -> torch.Tensor:
        """第一次调用算完整 qkv,后两次直接切片(prefill 不合并)."""
        if x.shape[-2] != 1:
            return wrapper.add_delta(x, wrapper.base_layer(x))
        # 同一个 step 内 q/k/v 传进来的是同一个张量对象,用 is 判断即可
        if self._cached_x is not x:
            self._cached_x = x
            self._cached_out = F.linear(x, self.weight)
        start, end = wrapper.qkv_slice
        return wrapper.add_delta(x, self._cached_out[..., start:end])


def fuse_qkv_projections(wrappers: dict) -> int:
    """把同一 attention 下的 q/k/v 三层归组,返回融合的组数."""
    by_prefix: dict = {}
    for key, wrapper in wrappers.items():
        for suffix in ("q_proj", "k_proj", "v_proj"):
            if key.endswith("." + suffix):
                by_prefix.setdefault(key[: -len(suffix)], {})[suffix] = wrapper
                break
    fused = 0
    for parts in by_prefix.values():
        if len(parts) != 3:
            continue
        QKVProjGroup([parts["q_proj"], parts["k_proj"], parts["v_proj"]])
        fused += 1
    return fused
