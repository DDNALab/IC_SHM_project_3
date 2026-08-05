from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

from utils import get_compute_dtype, local_rank

VISION_MARKERS = ("visual", "vision", "image_encoder", "vision_tower", "vision_model")
OUTPUT_MARKERS = ("lm_head", "embed_tokens", "output_layer")


def build_processor(model_id: str, min_pixels: int, max_pixels: int) -> Any:
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return processor


def _linear_module_types() -> tuple[type, ...]:
    module_types: list[type] = [torch.nn.Linear]
    try:
        import bitsandbytes as bnb

        module_types.extend([bnb.nn.Linear4bit, bnb.nn.Linear8bitLt])
    except ImportError:
        pass
    return tuple(module_types)


def find_lora_target_modules(model: torch.nn.Module, scope: str = "language") -> list[str]:
    """Return exact linear-module names, excluding the vision tower by default."""
    if scope not in {"language", "all"}:
        raise ValueError("lora scope must be 'language' or 'all'.")

    module_types = _linear_module_types()
    targets: list[str] = []
    for name, module in model.named_modules():
        lower = name.lower()
        if not isinstance(module, module_types):
            continue
        if any(marker in lower for marker in OUTPUT_MARKERS):
            continue
        if scope == "language" and any(marker in lower for marker in VISION_MARKERS):
            continue
        targets.append(name)

    if not targets:
        raise RuntimeError("No LoRA target modules were found. Inspect model.named_modules().")
    return targets


def build_quantization_config(dtype: torch.dtype) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def load_qlora_model(
    model_id: str,
    precision: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_scope: str,
    gradient_checkpointing: bool,
) -> tuple[torch.nn.Module, list[str]]:
    dtype = get_compute_dtype(precision)
    device_map = {"": local_rank()} if torch.cuda.is_available() else None
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        dtype=dtype,
        quantization_config=build_quantization_config(dtype),
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )
    targets = find_lora_target_modules(model, scope=lora_scope)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)
    return model, targets


def load_adapter_for_inference(
    model_id: str,
    adapter_path: str | Path,
    precision: str = "bf16",
    load_in_4bit: bool = True,
) -> torch.nn.Module:
    dtype = get_compute_dtype(precision)
    device_map = "auto" if torch.cuda.is_available() else None
    quantization_config = build_quantization_config(dtype) if load_in_4bit else None
    base_model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        dtype=dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    return model


def trainable_parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
    }
