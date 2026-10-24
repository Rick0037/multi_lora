"""按行分配 adapter 的批量 forward(PEFT 分组实现,作为 golden 参考).

做法:把用同一个 adapter 的行合成一个子 batch,各跑一次 forward,再把 logits 拼回去。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from peft import PeftModel

from multi_lora.registry import LoraName


def group_row_indices(
    lora_ids: Sequence[LoraName],
) -> "OrderedDict[LoraName, List[int]]":
    """按 adapter 把行号分组,保持每组第一次出现的顺序.

    例如 ["a", None, "a"] -> {"a": [0, 2], None: [1]}
    """
    groups: "OrderedDict[LoraName, List[int]]" = OrderedDict()
    for i, key in enumerate(lora_ids):
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    return groups


def validate_lora_batch(
    lora_ids: Sequence[LoraName],
    *,
    max_loras_per_batch: int,
) -> bool:
    """本 batch 里不同的 adapter 数是否超过槽位上限(None 也算一个)."""
    return len(set(lora_ids)) <= max_loras_per_batch


@torch.inference_mode()
def forward_logits_grouped(
    model: PeftModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    lora_ids: Sequence[LoraName],
) -> torch.Tensor:
    """把行按 adapter 分组,各组跑一次,再按原行号写回 logits."""
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be [batch, seq]")
    batch_size = input_ids.shape[0]
    if len(lora_ids) != batch_size:
        raise ValueError("lora_ids length must match batch size")

    out_logits: Optional[torch.Tensor] = None
    groups = group_row_indices(lora_ids)

    for adapter, rows in groups.items():
        # 取出本组的行,组成子 batch
        row_idx = torch.tensor(rows, device=input_ids.device, dtype=torch.long)
        sub_ids = input_ids.index_select(0, row_idx)

        if attention_mask is None:
            sub_mask = None
        else:
            sub_mask = attention_mask.index_select(0, row_idx)

        if adapter is None:
            # 基座:临时关掉 LoRA
            with model.disable_adapter():
                logits = model(
                    input_ids=sub_ids,
                    attention_mask=sub_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits
        else:
            # 换成该 adapter 的权重再算
            model.set_adapter(adapter)
            logits = model(
                input_ids=sub_ids,
                attention_mask=sub_mask,
                use_cache=False,
                return_dict=True,
            ).logits

        if out_logits is None:
            # 用第一组的输出形状建 buffer:[batch, seq, vocab]
            out_logits = torch.empty(
                batch_size,
                logits.shape[1],
                logits.shape[2],
                device=logits.device,
                dtype=logits.dtype,
            )

        # 把本组结果写回原来的行
        out_logits.index_copy_(0, row_idx, logits)

    assert out_logits is not None
    return out_logits


@torch.inference_mode()
def forward_logits_rowwise_reference(
    model: PeftModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    lora_ids: Sequence[LoraName],
) -> torch.Tensor:
    """参考实现:每行单独跑一次 forward,再在 batch 维拼起来."""
    pieces: List[torch.Tensor] = []

    for i in range(input_ids.shape[0]):
        row_ids = input_ids[i : i + 1]

        if attention_mask is None:
            row_mask = None
        else:
            row_mask = attention_mask[i : i + 1]

        adapter = lora_ids[i]
        if adapter is None:
            with model.disable_adapter():
                log = model(
                    input_ids=row_ids,
                    attention_mask=row_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits
        else:
            model.set_adapter(adapter)
            log = model(
                input_ids=row_ids,
                attention_mask=row_mask,
                use_cache=False,
                return_dict=True,
            ).logits

        pieces.append(log)

    return torch.cat(pieces, dim=0)


def load_peft_multi_adapter(
    base_model: nn.Module,
    adapters: Mapping[str, str],
    *,
    first: Optional[str] = None,
) -> PeftModel:
    """在一个 base 模型上挂多个 LoRA,之后可用 set_adapter 切换."""
    if not adapters:
        raise ValueError("adapters must be non-empty")

    names = list(adapters)
    if first is None:
        primary = names[0]
    else:
        primary = first

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
    """两组 logits 是否在给定容差内一致(单测用)."""
    return torch.allclose(a, b, rtol=rtol, atol=atol)
