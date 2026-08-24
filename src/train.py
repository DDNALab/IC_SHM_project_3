from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import Trainer, TrainingArguments

from dataset import DamageDataCollator, DamageDataset
from model import (
    build_processor,
    load_qlora_model,
    trainable_parameter_summary,
)
from paths import (
    CHECKPOINTS_DIR,
    CONFIG_DIR,
    PROJECT_ROOT,
    TABLES_DIR,
    ensure_output_dirs,
)
from utils import (
    load_yaml,
    save_json,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen3.5-4B on Project 3."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "qwen35_4b_24gb.yaml",
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Setup
    # --------------------------------------------------

    ensure_output_dirs()

    config = load_yaml(
        args.config
    )

    seed = int(
        config.get(
            "seed",
            2026,
        )
    )

    set_seed(seed)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    run_name = str(
        config.get(
            "run_name",
            "qwen35_4b_shm",
        )
    )

    # --------------------------------------------------
    # Resolve project paths
    # --------------------------------------------------

    train_csv = Path(
        data_cfg.get(
            "train_csv",
            TABLES_DIR / "train.csv",
        )
    )

    val_csv = Path(
        data_cfg.get(
            "val_csv",
            TABLES_DIR / "val.csv",
        )
    )

    run_dir = Path(
        train_cfg.get(
            "output_dir",
            CHECKPOINTS_DIR / run_name,
        )
    )

    train_csv = (
        train_csv
        if train_csv.is_absolute()
        else PROJECT_ROOT / train_csv
    )

    val_csv = (
        val_csv
        if val_csv.is_absolute()
        else PROJECT_ROOT / val_csv
    )

    run_dir = (
        run_dir
        if run_dir.is_absolute()
        else PROJECT_ROOT / run_dir
    )

    final_adapter_dir = (
        run_dir / "final_adapter"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Processor
    # --------------------------------------------------

    print("\nLoading processor...")

    processor = build_processor(
        model_cfg["model_id"],
        min_pixels=int(
            model_cfg["min_pixels"]
        ),
        max_pixels=int(
            model_cfg["max_pixels"]
        ),
    )

    # --------------------------------------------------
    # QLoRA model
    # --------------------------------------------------

    print("\nLoading QLoRA model...")

    model, lora_targets = load_qlora_model(
        model_id=model_cfg[
            "model_id"
        ],
        precision=model_cfg.get(
            "precision",
            "bf16",
        ),
        lora_rank=int(
            model_cfg.get(
                "lora_rank",
                16,
            )
        ),
        lora_alpha=int(
            model_cfg.get(
                "lora_alpha",
                32,
            )
        ),
        lora_dropout=float(
            model_cfg.get(
                "lora_dropout",
                0.05,
            )
        ),
        lora_scope=model_cfg.get(
            "lora_scope",
            "language",
        ),
        gradient_checkpointing=bool(
            train_cfg.get(
                "gradient_checkpointing",
                True,
            )
        ),
    )

    # --------------------------------------------------
    # Parameter summary
    # --------------------------------------------------

    parameter_summary = (
        trainable_parameter_summary(
            model
        )
    )

    parameter_summary[
        "lora_target_module_count"
    ] = len(
        lora_targets
    )

    parameter_summary[
        "lora_target_modules"
    ] = lora_targets

    save_json(
        parameter_summary,
        TABLES_DIR
        / f"{run_name}_model_parameters.json",
    )

    model.print_trainable_parameters()

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = DamageDataset(
        train_csv
    )

    eval_dataset = DamageDataset(
        val_csv
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(eval_dataset),
    )

    # --------------------------------------------------
    # Data collator
    # --------------------------------------------------

    collator = DamageDataCollator(
        processor=processor,
        max_length=int(
            data_cfg.get(
                "max_length",
                1024,
            )
        ),
        use_assistant_mask=bool(
            data_cfg.get(
                "use_assistant_mask",
                True,
            )
        ),
        prompt_variant=str(
            data_cfg.get(
                "prompt_variant",
                "baseline",
            )
        ),
    )

    # --------------------------------------------------
    # Precision
    # --------------------------------------------------

    precision = (
        model_cfg.get(
            "precision",
            "bf16",
        )
        .lower()
    )

    use_bf16 = (
        precision == "bf16"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    use_fp16 = (
        torch.cuda.is_available()
        and not use_bf16
    )

    print(
        "\nBF16:",
        use_bf16,
    )

    print(
        "FP16:",
        use_fp16,
    )

    # --------------------------------------------------
    # TrainingArguments
    # --------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(
            run_dir
        ),

        run_name=run_name,

        num_train_epochs=float(
            train_cfg.get(
                "num_train_epochs",
                3,
            )
        ),

        per_device_train_batch_size=int(
            train_cfg.get(
                "per_device_train_batch_size",
                1,
            )
        ),

        per_device_eval_batch_size=int(
            train_cfg.get(
                "per_device_eval_batch_size",
                1,
            )
        ),

        gradient_accumulation_steps=int(
            train_cfg.get(
                "gradient_accumulation_steps",
                16,
            )
        ),

        learning_rate=float(
            train_cfg.get(
                "learning_rate",
                2e-4,
            )
        ),

        weight_decay=float(
            train_cfg.get(
                "weight_decay",
                0.01,
            )
        ),

        # Transformers 5.x compatibility:
        # warmup_ratio was removed from TrainingArguments.
        # A value below 1 for warmup_steps is interpreted
        # as a fraction of total training steps.
        warmup_steps=float(
            train_cfg.get(
                "warmup_ratio",
                0.05,
            )
        ),

        lr_scheduler_type=str(
            train_cfg.get(
                "lr_scheduler_type",
                "cosine",
            )
        ),

        logging_steps=int(
            train_cfg.get(
                "logging_steps",
                5,
            )
        ),

        eval_strategy="steps",

        eval_steps=int(
            train_cfg.get(
                "eval_steps",
                50,
            )
        ),

        save_strategy="steps",

        save_steps=int(
            train_cfg.get(
                "save_steps",
                50,
            )
        ),

        save_total_limit=int(
            train_cfg.get(
                "save_total_limit",
                2,
            )
        ),

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        bf16=use_bf16,

        fp16=use_fp16,

        tf32=(
            bool(
                train_cfg.get(
                    "tf32",
                    True,
                )
            )
            and torch.cuda.is_available()
        ),

        gradient_checkpointing=bool(
            train_cfg.get(
                "gradient_checkpointing",
                True,
            )
        ),

        optim=str(
            train_cfg.get(
                "optim",
                "paged_adamw_8bit",
            )
        ),

        max_grad_norm=float(
            train_cfg.get(
                "max_grad_norm",
                1.0,
            )
        ),

        report_to=list(
            train_cfg.get(
                "report_to",
                [
                    "tensorboard"
                ],
            )
        ),

        remove_unused_columns=False,

        dataloader_num_workers=int(
            train_cfg.get(
                "dataloader_num_workers",
                0,
            )
        ),

        seed=seed,

        data_seed=seed,

        ddp_find_unused_parameters=False,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\nStarting training...\n")

    train_result = trainer.train(
        resume_from_checkpoint=
            args.resume_from_checkpoint
    )

    # --------------------------------------------------
    # Save final adapter
    # --------------------------------------------------

    trainer.save_model(
        str(
            final_adapter_dir
        )
    )

    processor.save_pretrained(
        str(
            final_adapter_dir
        )
    )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    trainer.save_metrics(
        "train",
        train_result.metrics,
    )

    trainer.save_state()

    print(
        "\nRunning final evaluation..."
    )

    eval_metrics = (
        trainer.evaluate()
    )

    trainer.save_metrics(
        "eval",
        eval_metrics,
    )

    # --------------------------------------------------
    # Training log
    # --------------------------------------------------

    log_frame = pd.DataFrame(
        trainer.state.log_history
    )

    log_frame.to_csv(
        TABLES_DIR
        / f"{run_name}_training_log.csv",
        index=False,
    )

    # --------------------------------------------------
    # Summary JSON
    # --------------------------------------------------

    save_json(
        {
            "config":
                config,

            "train_metrics":
                train_result.metrics,

            "eval_metrics":
                eval_metrics,

            "final_adapter":
                str(
                    final_adapter_dir.resolve()
                ),
        },

        TABLES_DIR
        / f"{run_name}_training_summary.json",
    )

    print(
        "\nFinal adapter:"
    )

    print(
        final_adapter_dir.resolve()
    )


if __name__ == "__main__":
    main()