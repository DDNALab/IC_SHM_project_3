from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image, ImageOps

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "week5_resnet18_baseline"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DAMAGE ONTOLOGY
# ============================================================

CATEGORIES = [
    "crack",
    "void",
    "honeycomb",
    "looseness",
    "spalling",
    "exposed_rebar",
    "corrosion",
    "efflorescence",
    "pothole",
]

NUM_CLASSES = len(
    CATEGORIES
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    torch.backends.cudnn.deterministic = (
        True
    )

    torch.backends.cudnn.benchmark = (
        False
    )


# ============================================================
# LABEL PARSING
# ============================================================

def parse_categories(
    value,
) -> list[str]:

    if isinstance(
        value,
        list,
    ):
        return [
            str(item)
            for item in value
        ]

    if pd.isna(
        value
    ):
        return []

    text = str(
        value
    ).strip()

    try:
        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            list,
        ):
            return [
                str(item)
                for item in parsed
            ]

    except json.JSONDecodeError:
        pass

    return []


def encode_labels(
    categories: list[str],
) -> np.ndarray:

    target = np.zeros(
        NUM_CLASSES,
        dtype=np.float32,
    )

    for category in categories:

        if category in CATEGORIES:

            index = CATEGORIES.index(
                category
            )

            target[
                index
            ] = 1.0

    return target


# ============================================================
# DATASET
# ============================================================

class StructuralDamageDataset(
    Dataset
):

    def __init__(
        self,
        csv_path: Path,
        transform,
    ) -> None:

        self.frame = (
            pd.read_csv(
                csv_path
            )
            .reset_index(
                drop=True
            )
        )

        self.transform = (
            transform
        )

        required = {
            "image_id",
            "image_path",
            "damage_categories",
        }

        missing = (
            required
            - set(
                self.frame.columns
            )
        )

        if missing:

            raise ValueError(
                f"{csv_path} missing columns: "
                f"{sorted(missing)}"
            )

    def __len__(
        self,
    ) -> int:

        return len(
            self.frame
        )

    def __getitem__(
        self,
        index: int,
    ):

        row = self.frame.iloc[
            index
        ]

        image_path = Path(
            row[
                "image_path"
            ]
        )

        with Image.open(
            image_path
        ) as image:

            image = (
                ImageOps.exif_transpose(
                    image
                )
                .convert(
                    "RGB"
                )
            )

            image = (
                self.transform(
                    image
                )
            )

        categories = (
            parse_categories(
                row[
                    "damage_categories"
                ]
            )
        )

        labels = torch.tensor(
            encode_labels(
                categories
            ),
            dtype=torch.float32,
        )

        return (
            image,
            labels,
            str(
                row[
                    "image_id"
                ]
            ),
        )


# ============================================================
# TRANSFORMS
# ============================================================

def build_transforms():

    weights = (
        ResNet18_Weights.DEFAULT
    )

    normalize = transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    )

    train_transform = (
        transforms.Compose(
            [
                transforms.Resize(
                    256
                ),

                transforms.RandomResizedCrop(
                    224,
                    scale=(
                        0.80,
                        1.0,
                    ),
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.RandomRotation(
                    degrees=8
                ),

                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.05,
                    hue=0.02,
                ),

                transforms.ToTensor(),

                normalize,
            ]
        )
    )

    val_transform = (
        transforms.Compose(
            [
                transforms.Resize(
                    256
                ),

                transforms.CenterCrop(
                    224
                ),

                transforms.ToTensor(),

                normalize,
            ]
        )
    )

    return (
        train_transform,
        val_transform,
        weights,
    )


# ============================================================
# CLASS IMBALANCE
# ============================================================

def compute_pos_weight(
    csv_path: Path,
) -> torch.Tensor:

    frame = pd.read_csv(
        csv_path
    )

    positives = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    for value in frame[
        "damage_categories"
    ]:

        labels = encode_labels(
            parse_categories(
                value
            )
        )

        positives += labels

    total = len(
        frame
    )

    negatives = (
        total
        - positives
    )

    pos_weight = (
        negatives
        / np.maximum(
            positives,
            1.0,
        )
    )

    # Avoid extreme instability for exceptionally rare classes.
    pos_weight = np.clip(
        pos_weight,
        1.0,
        20.0,
    )

    print()
    print(
        "=== POSITIVE CLASS WEIGHTS ==="
    )

    for category, count, weight in zip(
        CATEGORIES,
        positives,
        pos_weight,
    ):

        print(
            f"{category:15s} "
            f"positive={int(count):4d} "
            f"pos_weight={weight:.4f}"
        )

    return torch.tensor(
        pos_weight,
        dtype=torch.float32,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:

    y_pred = (
        y_prob
        >= threshold
    ).astype(
        np.int32
    )

    exact = accuracy_score(
        y_true,
        y_pred,
    )

    (
        micro_p,
        micro_r,
        micro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    (
        macro_p,
        macro_r,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "exact_label_set_accuracy":
            float(
                exact
            ),

        "micro_precision":
            float(
                micro_p
            ),

        "micro_recall":
            float(
                micro_r
            ),

        "micro_f1":
            float(
                micro_f1
            ),

        "macro_precision":
            float(
                macro_p
            ),

        "macro_recall":
            float(
                macro_r
            ),

        "macro_f1":
            float(
                macro_f1
            ),
    }


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
):

    candidates = np.arange(
        0.10,
        0.91,
        0.05,
    )

    results = []

    for threshold in candidates:

        metrics = calculate_metrics(
            y_true,
            y_prob,
            float(
                threshold
            ),
        )

        results.append(
            metrics
        )

    frame = pd.DataFrame(
        results
    )

    # Primary threshold-selection criterion:
    # validation micro-F1.
    #
    # Tie breakers:
    # macro-F1, exact accuracy.
    frame = frame.sort_values(
        [
            "micro_f1",
            "macro_f1",
            "exact_label_set_accuracy",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    best = frame.iloc[
        0
    ].to_dict()

    return (
        float(
            best[
                "threshold"
            ]
        ),
        frame,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    device,
):

    model.eval()

    all_true = []
    all_prob = []
    all_ids = []

    total_loss = 0.0
    total_samples = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.inference_mode():

        for (
            images,
            labels,
            image_ids,
        ) in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            probabilities = (
                torch.sigmoid(
                    logits
                )
            )

            batch_size = (
                labels.shape[0]
            )

            total_loss += (
                float(
                    loss.item()
                )
                * batch_size
            )

            total_samples += (
                batch_size
            )

            all_true.append(
                labels
                .detach()
                .cpu()
                .numpy()
            )

            all_prob.append(
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

            all_ids.extend(
                list(
                    image_ids
                )
            )

    y_true = np.concatenate(
        all_true,
        axis=0,
    )

    y_prob = np.concatenate(
        all_prob,
        axis=0,
    )

    average_loss = (
        total_loss
        / max(
            total_samples,
            1,
        )
    )

    return (
        average_loss,
        y_true,
        y_prob,
        all_ids,
    )


# ============================================================
# TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    device = torch.device(
        "cuda"
    )

    print()
    print(
        "Device:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Categories:",
        CATEGORIES,
    )

    train_csv = (
        TABLES_DIR
        / "train.csv"
    )

    val_csv = (
        TABLES_DIR
        / "val.csv"
    )

    (
        train_transform,
        val_transform,
        weights,
    ) = build_transforms()

    train_dataset = (
        StructuralDamageDataset(
            train_csv,
            train_transform,
        )
    )

    val_dataset = (
        StructuralDamageDataset(
            val_csv,
            val_transform,
        )
    )

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print()
    print(
        "Training images:",
        len(
            train_dataset
        ),
    )

    print(
        "Validation images:",
        len(
            val_dataset
        ),
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print()
    print(
        "Loading ImageNet-pretrained ResNet-18..."
    )

    model = models.resnet18(
        weights=weights
    )

    input_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES,
    )

    model = model.to(
        device
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Trainable parameters:",
        trainable_parameters,
    )

    print(
        "Total parameters:",
        total_parameters,
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    pos_weight = (
        compute_pos_weight(
            train_csv
        )
        .to(
            device
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    # --------------------------------------------------
    # Training state
    # --------------------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR
        / (
            f"resnet18_seed"
            f"{args.seed}_best.pt"
        )
    )

    training_log_path = (
        TABLES_DIR
        / (
            f"week5_resnet18_seed"
            f"{args.seed}_training_log.csv"
        )
    )

    best_micro_f1 = (
        -1.0
    )

    best_epoch = 0

    epochs_without_improvement = 0

    log_rows = []

    torch.cuda.reset_peak_memory_stats()

    training_start = (
        time.perf_counter()
    )

    # ==================================================
    # EPOCH LOOP
    # ==================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        running_loss = 0.0
        samples_seen = 0

        epoch_start = (
            time.perf_counter()
        )

        for (
            images,
            labels,
            _,
        ) in train_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                labels.shape[0]
            )

            running_loss += (
                float(
                    loss.item()
                )
                * batch_size
            )

            samples_seen += (
                batch_size
            )

        train_loss = (
            running_loss
            / max(
                samples_seen,
                1,
            )
        )

        (
            val_loss,
            y_true,
            y_prob,
            _,
        ) = evaluate_model(
            model,
            val_loader,
            device,
        )

        (
            threshold,
            threshold_results,
        ) = find_best_threshold(
            y_true,
            y_prob,
        )

        metrics = calculate_metrics(
            y_true,
            y_prob,
            threshold,
        )

        micro_f1 = (
            metrics[
                "micro_f1"
            ]
        )

        scheduler.step(
            micro_f1
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        current_lr = (
            optimizer
            .param_groups[0][
                "lr"
            ]
        )

        print()
        print(
            f"Epoch {epoch:02d}/{args.epochs}"
        )

        print(
            f"Train loss: {train_loss:.6f}"
        )

        print(
            f"Val loss:   {val_loss:.6f}"
        )

        print(
            f"Threshold:  {threshold:.2f}"
        )

        print(
            f"Exact:      "
            f"{metrics['exact_label_set_accuracy']:.6f}"
        )

        print(
            f"Micro F1:   "
            f"{metrics['micro_f1']:.6f}"
        )

        print(
            f"Macro F1:   "
            f"{metrics['macro_f1']:.6f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_seconds:.2f}s"
        )

        log_rows.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "threshold":
                    threshold,

                "exact_label_set_accuracy":
                    metrics[
                        "exact_label_set_accuracy"
                    ],

                "micro_f1":
                    metrics[
                        "micro_f1"
                    ],

                "macro_f1":
                    metrics[
                        "macro_f1"
                    ],

                "learning_rate":
                    current_lr,

                "epoch_seconds":
                    epoch_seconds,
            }
        )

        pd.DataFrame(
            log_rows
        ).to_csv(
            training_log_path,
            index=False,
        )

        # --------------------------------------------------
        # Best checkpoint
        # --------------------------------------------------

        if (
            micro_f1
            > best_micro_f1
        ):

            best_micro_f1 = (
                micro_f1
            )

            best_epoch = (
                epoch
            )

            epochs_without_improvement = (
                0
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "seed":
                        args.seed,

                    "epoch":
                        epoch,

                    "threshold":
                        threshold,

                    "categories":
                        CATEGORIES,

                    "micro_f1":
                        metrics[
                            "micro_f1"
                        ],

                    "macro_f1":
                        metrics[
                            "macro_f1"
                        ],

                    "exact_label_set_accuracy":
                        metrics[
                            "exact_label_set_accuracy"
                        ],
                },
                checkpoint_path,
            )

            threshold_results.to_csv(
                TABLES_DIR
                / (
                    f"week5_resnet18_seed"
                    f"{args.seed}_threshold_search.csv"
                ),
                index=False,
            )

            print(
                "Saved new best checkpoint."
            )

        else:

            epochs_without_improvement += (
                1
            )

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    total_training_seconds = (
        time.perf_counter()
        - training_start
    )

    peak_allocated_gb = (
        torch.cuda
        .max_memory_allocated()
        / 1024**3
    )

    peak_reserved_gb = (
        torch.cuda
        .max_memory_reserved()
        / 1024**3
    )

    summary = {
        "model":
            "ResNet-18",

        "task":
            "multi-label classification",

        "pretraining":
            "ImageNet",

        "seed":
            args.seed,

        "best_epoch":
            best_epoch,

        "best_validation_micro_f1":
            best_micro_f1,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "trainable_parameters":
            trainable_parameters,

        "total_parameters":
            total_parameters,

        "training_seconds":
            total_training_seconds,

        "peak_allocated_gb":
            float(
                peak_allocated_gb
            ),

        "peak_reserved_gb":
            float(
                peak_reserved_gb
            ),
    }

    summary_path = (
        TABLES_DIR
        / (
            f"week5_resnet18_seed"
            f"{args.seed}_training_summary.json"
        )
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        "======================================"
    )

    print(
        "RESNET-18 TRAINING COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation micro F1:",
        round(
            best_micro_f1,
            6,
        ),
    )

    print(
        "Peak allocated GB:",
        round(
            peak_allocated_gb,
            3,
        ),
    )

    print(
        "Peak reserved GB:",
        round(
            peak_reserved_gb,
            3,
        ),
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        checkpoint_path.resolve()
    )

    print()
    print(
        "Training summary:"
    )

    print(
        summary_path.resolve()
    )


if __name__ == "__main__":
    main()