"""Phase-2 multi-LoRA batching: one logical batch, different LoRA adapter per row.

Keeps a single ``PeftModel`` with multiple adapters (``load_adapter`` / ``set_adapter``).
Rows that use the base model go through ``disable_adapter``. Rows sharing the same
adapter name are grouped into sub-batches; logits are scattered back with
``index_copy_``. When sequences do not attend across rows, this matches per-row
forwards.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from peft import PeftModel


# 每行绑定的 PEFT adapter 名；``None`` 表示使用基座（``disable_adapter``）。
LoraKey = Optional[str]


def group_row_indices(lora_ids: Sequence[LoraKey]) -> "OrderedDict[LoraKey, List[int]]":
    """按 adapter 名分组 batch 行号，保持各 key 首次出现的顺序。

    Returns:
        ``adapter_key -> [row indices]``；``None`` 表示基座（无 LoRA）。
    """
    groups: "OrderedDict[LoraKey, List[int]]" = OrderedDict()
    for i, key in enumerate(lora_ids):
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    return groups


def validate_lora_batch(
    lora_ids: Sequence[LoraKey],
    *,
    max_loras_per_batch: int,
) -> bool:
    """Return whether distinct adapters in one batch are within the configured limit.

    ``None`` counts as one logical slot (base model, no LoRA).
    """
    distinct = set(lora_ids)
    return len(distinct) <= max_loras_per_batch


def validate_lora_set(
    lora_ids: Sequence[LoraKey],
    *,
    max_loras_per_batch: int,
) -> bool:
    """Alias of ``validate_lora_batch`` (kept for call sites that use this name)."""
    return validate_lora_batch(lora_ids, max_loras_per_batch=max_loras_per_batch)


@torch.inference_mode()
def forward_logits_grouped(
    model: PeftModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    lora_ids: Sequence[LoraKey],
) -> torch.Tensor:
    """多 LoRA 分组前向：按 ``lora_ids`` 分子 batch 前向，再 ``index_copy_`` 拼回 logits。

    Args:
        model: 已 ``load_adapter`` 多个权重的 ``PeftModel``。
        input_ids: ``[B, T]`` token ids。
        attention_mask: 可选 ``[B, T]``，与 ``input_ids`` 对齐。
        lora_ids: 长度 ``B``；元素为 adapter 名或 ``None``（基座）。

    Returns:
        ``[B, T, V]`` logits，与逐行独立前向在因果 LM 行独立时一致。
    """
    # input_ids 形状必须是 [batch, seq_len]；因果 LM 一次前向对应整段 prompt。
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be [batch, seq]")
    b = input_ids.shape[0]
    # lora_ids[i] 表示第 i 条序列用哪个 adapter；长度必须与 batch 维一致。
    if len(lora_ids) != b:
        raise ValueError("lora_ids length must match batch size")

    device = input_ids.device
    # 最终拼回的 logits 缓冲；首次子 batch 前向后再按形状分配。
    out_logits: Optional[torch.Tensor] = None

    # 按 adapter 分组：例如 lora_ids=["a", None, "a"] -> { "a":[0,2], None:[1] }。
    # 同一组内行共享同一套 LoRA（或同为基座），可合并成一次子 batch forward。
    for key, rows in group_row_indices(lora_ids).items():
        # rows 是原 batch 中的行号列表；转成张量供 index_select / index_copy_ 使用。
        row_idx = torch.tensor(rows, device=device, dtype=torch.long)
        # 从整批 input_ids 中抽出本组行，得到子 batch [G, seq_len]（G = 本组行数）。
        sub_ids = input_ids.index_select(0, row_idx)
        # padding 场景下同步抽出对应 attention_mask，避免 pad 位参与错误注意力。
        sub_mask = (
            attention_mask.index_select(0, row_idx) if attention_mask is not None else None
        )

        if key is None:
            # None 表示该行走纯基座：临时关闭 PEFT 注入的 LoRA 分支。
            with model.disable_adapter():
                logits = model(
                    input_ids=sub_ids,
                    attention_mask=sub_mask,
                    use_cache=False,  # 仅算 logits，不维护 KV cache（与单步 decode 场景解耦）。
                    return_dict=True,
                ).logits  # [G, seq_len, vocab_size]
        else:
            # 切换到已 load_adapter 的同名权重，再对本子 batch 前向。
            model.set_adapter(key)
            logits = model(
                input_ids=sub_ids,
                attention_mask=sub_mask,
                use_cache=False,
                return_dict=True,
            ).logits

        if out_logits is None:
            # 用第一组输出的 T、V 推断全批输出形状；device/dtype 与子 batch 一致。
            out_logits = torch.empty(
                b,
                logits.size(1),
                logits.size(2),
                device=logits.device,
                dtype=logits.dtype,
            )
        # scatter
        # 把当前iter下的子 batch 的 logits 写回原 batch 的对应行（按 row_idx 散射）。
        out_logits.index_copy_(0, row_idx, logits)

    assert out_logits is not None
    return out_logits


@torch.inference_mode()
def forward_logits_rowwise_reference(
    model: PeftModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    lora_ids: Sequence[LoraKey],
) -> torch.Tensor:
    """逐行对照前向：每行 ``batch_size=1`` 单独 ``set_adapter`` / ``disable_adapter``。

    用于单测与 smoke 中验证 ``forward_logits_grouped`` 的数值正确性。
    """
    b = input_ids.shape[0]
    # 逐行保存每次 bs=1 前向的 logits，最后在 batch 维拼接成 [B, T, V]。
    pieces: List[torch.Tensor] = []
    for i in range(b):
        # 只取第 i 行，保持二维 [1, seq_len]，与 HF 单条推理接口一致。
        row_ids = input_ids[i : i + 1]
        row_mask = attention_mask[i : i + 1] if attention_mask is not None else None
        key = lora_ids[i]
        if key is None:
            # 第 i 行使用基座：关闭 adapter 后单独前向（逐行对照实现）。
            with model.disable_adapter():
                log = model(
                    input_ids=row_ids,
                    attention_mask=row_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits  # [1, seq_len, vocab_size]
        else:
            # 第 i 行切换到指定 adapter，再 bs=1 前向。
            model.set_adapter(key)
            log = model(
                input_ids=row_ids,
                attention_mask=row_mask,
                use_cache=False,
                return_dict=True,
            ).logits
        pieces.append(log)
    # dim=0 拼接 B 个 [1,T,V] -> [B,T,V]，应与 grouped 版 index_copy_ 结果一致。
    return torch.cat(pieces, dim=0)


def load_peft_multi_adapter(
    base_model: nn.Module,
    adapters: Mapping[str, str],
    *,
    first: Optional[str] = None,
) -> PeftModel:
    """在同一基座上加载多个 LoRA checkpoint，供 ``set_adapter`` 切换。

    Args:
        base_model: 未包 PEFT 的因果 LM。
        adapters: ``adapter_name -> Hugging Face ID 或本地目录``。
        first: 首个挂载的 adapter 名；默认取 ``adapters`` 键序第一个。

    Returns:
        含全部 adapter 的 ``PeftModel``。
    """
    if not adapters:
        raise ValueError("adapters must be non-empty")

    names = list(adapters.keys())
    primary = first if first is not None else names[0]

    model = PeftModel.from_pretrained(
        base_model,
        adapters[primary],
        adapter_name=primary,
        is_trainable=False,
    )
    for name in names:
        if name == primary:
            continue
        model.load_adapter(adapters[name], adapter_name=name)
    return model


def logits_close(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> bool:
    """判断两组 logits 是否在给定 ``rtol`` / ``atol`` 下逐元素接近。"""
    return torch.allclose(a, b, rtol=rtol, atol=atol)
