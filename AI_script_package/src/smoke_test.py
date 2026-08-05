from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import DamageDataCollator, DamageDataset
from model import build_processor, load_qlora_model, trainable_parameter_summary
from paths import CONFIG_DIR, PROJECT_ROOT, TABLES_DIR, ensure_output_dirs
from utils import load_yaml, save_json, set_seed


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one QLoRA forward/backward pass before full training.")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "qwen35_4b_24gb.yaml")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this smoke test.")

    ensure_output_dirs()
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 2026)))
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    run_name = str(config.get("run_name", "qwen35_4b_shm"))

    dataset = DamageDataset(resolve_project_path(data_cfg["train_csv"]))
    if len(dataset) == 0:
        raise RuntimeError("Training split is empty.")

    processor = build_processor(
        model_cfg["model_id"],
        min_pixels=int(model_cfg["min_pixels"]),
        max_pixels=int(model_cfg["max_pixels"]),
    )
    model, targets = load_qlora_model(
        model_id=model_cfg["model_id"],
        precision=model_cfg.get("precision", "bf16"),
        lora_rank=int(model_cfg.get("lora_rank", 16)),
        lora_alpha=int(model_cfg.get("lora_alpha", 32)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
        lora_scope=model_cfg.get("lora_scope", "language"),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
    )
    collator = DamageDataCollator(
        processor=processor,
        max_length=int(data_cfg.get("max_length", 1024)),
        use_assistant_mask=bool(data_cfg.get("use_assistant_mask", True)),
    )

    torch.cuda.reset_peak_memory_stats()
    batch = collator([dataset[0]])
    device = next(model.parameters()).device
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    model.train()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    report = {
        "status": "passed",
        "loss": float(loss.detach().cpu()),
        "input_shape": list(batch["input_ids"].shape),
        "trainable_token_count": int(batch["labels"].ne(-100).sum().item()),
        "peak_allocated_gb": round(peak_allocated, 3),
        "peak_reserved_gb": round(peak_reserved, 3),
        "lora_target_module_count": len(targets),
        **trainable_parameter_summary(model),
    }
    output_path = TABLES_DIR / f"{run_name}_smoke_test.json"
    save_json(report, output_path)
    print(report)
    print(f"Smoke-test report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
