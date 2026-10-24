"""Hugging Face + PEFT 辅助;加载模型、选设备 dtype、简单 generate."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    """示例 / smoke 测试用的模型与 LoRA 路径描述."""

    base_model: str
    lora_path: Optional[str] = None
    label: str = ""


def pick_device() -> torch.device:
    """优先 CUDA,否则 CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """GPU 上优先 bfloat16(若支持),否则 float16;CPU 用 float32."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_tokenizer(model_id: str):
    """加载 HF tokenizer,并确保 pad_token 存在."""
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
    """加载 Causal LM,可选挂载单个 LoRA adapter."""
    if device is None:
        device = pick_device()
    if dtype is None:
        dtype = pick_dtype(device)

    if low_vram:
        # 约 10GB 显存场景;部分层 offload 到 CPU
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            device_map="auto",
            max_memory={gpu_id: "9GiB", "cpu": "64GiB"},
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
    """返回模型参数所在设备,用于把 input_ids 搬到同一 device."""
    return next(model.parameters()).device


def release_model(model: torch.nn.Module) -> None:
    """释放模型引用并清理 GPU 缓存."""
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
    """对单条 prompt 做 generate 并解码新生成 token."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = model_input_device(model)

    # 把输入搬到模型所在设备
    for key in inputs:
        inputs[key] = inputs[key].to(input_device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    # 只解码 prompt 之后的新 token
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
