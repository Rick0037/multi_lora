"""Hugging Face + PEFT helpers for multi_lora_phase2 examples and tests."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    """单次 smoke 用例的模型配置（基座 ID、可选 LoRA 路径、日志标签）。"""

    base_model: str
    lora_path: Optional[str] = None
    label: str = ""


def pick_device() -> torch.device:
    """选择计算设备：有 CUDA 则用 ``cuda``，否则 ``cpu``。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """按设备选择推理 dtype（GPU 优先 bfloat16/float16，CPU 为 float32）。"""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(model_id: str):
    """加载 Hugging Face tokenizer；若无 pad_token 则设为 eos_token。"""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    base_model: str,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    lora_path: Optional[str] = None,
    low_vram: bool = False,
    gpu_id: int = 0,
) -> torch.nn.Module:
    """加载因果语言模型；若 ``lora_path`` 非空则包装为 ``PeftModel`` 并 ``eval()``。

    ``low_vram=True`` 时通过 ``device_map="auto"`` 将部分层 offload 到 CPU。
    """
    device = device or pick_device()
    dtype = dtype or pick_dtype(device)

    if low_vram:
        # Fit ~7B on a ~10GB-free GPU by offloading excess layers to CPU.
        max_memory = {gpu_id: "9GiB", "cpu": "64GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
        )
    elif device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
        )

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)

    model.eval()
    return model


def model_input_device(model: torch.nn.Module) -> torch.device:
    """返回模型参数所在设备（用于将 ``input_ids`` 搬到正确 device）。"""
    return next(model.parameters()).device


def release_model(model: torch.nn.Module) -> None:
    """释放模型引用，并执行 ``gc.collect()`` / ``empty_cache()``。"""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> str:
    """对单条 prompt 调用 ``model.generate``，返回新生成文本（不含 prompt 部分）。"""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = model_input_device(model)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
