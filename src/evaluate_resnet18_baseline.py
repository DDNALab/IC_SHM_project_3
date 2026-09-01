from __future__ import annotations

import argparse
import json
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

from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "week5_resnet18_baseline"
    / "resnet18_seed2026_best.pt"
)

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

NUM_CLASSES = len(CATEGORIES)


def parse_categories(value):

    if isinstance(value, list):
        return [str(x) for x in value]

    if pd.isna(value):
        return []

    try:
        parsed = json.loads(str(value))

        if isinstance(parsed, list):
            return [str(x) for x in parsed]

    except json.JSONDecodeError:
        pass

    return []


def encode_labels(categories):

    target = np.zeros(
        NUM_CLASSES,
        dtype=np.float32,
    )

    for category in categories:

        if category in CATEGORIES:
            target[
                CATEGORIES.index(category)
            ] = 1.0

    return target


class DamageDataset(Dataset):

    def __init__(
        self,
        csv_path,
        transform,
    ):

        self.df = pd.read_csv(
            csv_path
        )

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[index]

        image_path = Path(
            row["image_path"]
        )

        with Image.open(
            image_path
        ) as image:

            image = (
                ImageOps.exif_transpose(
                    image
                )
                .convert("RGB")
            )

            image = self.transform(
                image
            )

        true_categories = parse_categories(
            row["damage_categories"]
        )

        labels = torch.tensor(
            encode_labels(
                true_categories
            ),
            dtype=torch.float32,
        )

        return (
            image,
            labels,
            str(row["image_id"]),
        )


def build_transform():

    weights = (
        ResNet18_Weights.DEFAULT
    )

    normalize = transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    )

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            DEFAULT_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=str(
            TABLES_DIR / "val.csv"
        ),
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="week5_resnet18",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    device = torch.device(
        "cuda"
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    csv_path = Path(
        args.csv
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    threshold = float(
        checkpoint["threshold"]
    )

    saved_categories = checkpoint[
        "categories"
    ]

    if saved_categories != CATEGORIES:

        raise RuntimeError(
            "Checkpoint category order does not "
            "match evaluator category order."
        )

    print()
    print(
        "Device:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Evaluation CSV:",
        csv_path,
    )

    print(
        "Threshold:",
        threshold,
    )

    transform = build_transform()

    dataset = DamageDataset(
        csv_path,
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    true_rows = []
    prob_rows = []
    image_ids = []

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    with torch.inference_mode():

        for (
            images,
            labels,
            ids,
        ) in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(
                images
            )

            probabilities = torch.sigmoid(
                logits
            )

            true_rows.append(
                labels.numpy()
            )

            prob_rows.append(
                probabilities
                .cpu()
                .numpy()
            )

            image_ids.extend(
                list(ids)
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    y_true = np.concatenate(
        true_rows,
        axis=0,
    )

    y_prob = np.concatenate(
        prob_rows,
        axis=0,
    )

    y_pred = (
        y_prob >= threshold
    ).astype(
        np.int32
    )

    exact_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    (
        micro_precision,
        micro_recall,
        micro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    (
        per_precision,
        per_recall,
        per_f1,
        per_support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    prediction_rows = []

    for index, image_id in enumerate(
        image_ids
    ):

        true_categories = [
            CATEGORIES[i]
            for i in range(
                NUM_CLASSES
            )
            if y_true[
                index,
                i
            ] == 1
        ]

        predicted_categories = [
            CATEGORIES[i]
            for i in range(
                NUM_CLASSES
            )
            if y_pred[
                index,
                i
            ] == 1
        ]

        row = {
            "image_id":
                image_id,

            "true_categories":
                json.dumps(
                    true_categories
                ),

            "predicted_categories":
                json.dumps(
                    predicted_categories
                ),

            "exact_match":
                bool(
                    true_categories
                    == predicted_categories
                ),
        }

        for i, category in enumerate(
            CATEGORIES
        ):

            row[
                f"prob_{category}"
            ] = float(
                y_prob[
                    index,
                    i
                ]
            )

        prediction_rows.append(
            row
        )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    predictions_path = (
        TABLES_DIR
        / f"{args.prefix}_val_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    metrics = {
        "model":
            "ResNet-18",

        "split":
            "validation",

        "n":
            int(
                len(dataset)
            ),

        "threshold":
            threshold,

        "exact_label_set_accuracy":
            float(
                exact_accuracy
            ),

        "micro_precision":
            float(
                micro_precision
            ),

        "micro_recall":
            float(
                micro_recall
            ),

        "micro_f1":
            float(
                micro_f1
            ),

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "total_inference_seconds":
            float(
                total_seconds
            ),

        "mean_inference_seconds_per_image":
            float(
                total_seconds
                / len(dataset)
            ),

        "images_per_second":
            float(
                len(dataset)
                / total_seconds
            ),

        "peak_allocated_gb":
            float(
                torch.cuda
                .max_memory_allocated()
                / 1024**3
            ),

        "peak_reserved_gb":
            float(
                torch.cuda
                .max_memory_reserved()
                / 1024**3
            ),

        "trainable_parameters":
            int(
                trainable_parameters
            ),

        "total_parameters":
            int(
                total_parameters
            ),
    }

    metrics_path = (
        TABLES_DIR
        / f"{args.prefix}_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        TABLES_DIR
        / f"{args.prefix}_metrics.csv",
        index=False,
    )

    per_class_rows = []

    for i, category in enumerate(
        CATEGORIES
    ):

        per_class_rows.append(
            {
                "category":
                    category,

                "precision":
                    float(
                        per_precision[i]
                    ),

                "recall":
                    float(
                        per_recall[i]
                    ),

                "f1":
                    float(
                        per_f1[i]
                    ),

                "support":
                    int(
                        per_support[i]
                    ),
            }
        )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    per_class_path = (
        TABLES_DIR
        / f"{args.prefix}_per_class_metrics.csv"
    )

    per_class_df.to_csv(
        per_class_path,
        index=False,
    )

    print()
    print(
        "========================================"
    )

    print(
        "RESNET-18 VALIDATION EVALUATION COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "Images:",
        len(dataset),
    )

    print(
        "Threshold:",
        round(
            threshold,
            4,
        ),
    )

    print(
        "Exact accuracy:",
        round(
            exact_accuracy,
            6,
        ),
    )

    print(
        "Micro F1:",
        round(
            micro_f1,
            6,
        ),
    )

    print(
        "Macro F1:",
        round(
            macro_f1,
            6,
        ),
    )

    print(
        "Inference seconds:",
        round(
            total_seconds,
            4,
        ),
    )

    print(
        "Seconds/image:",
        round(
            total_seconds
            / len(dataset),
            6,
        ),
    )

    print(
        "Images/second:",
        round(
            len(dataset)
            / total_seconds,
            3,
        ),
    )

    print(
        "Peak allocated GB:",
        round(
            torch.cuda
            .max_memory_allocated()
            / 1024**3,
            3,
        ),
    )

    print(
        "Peak reserved GB:",
        round(
            torch.cuda
            .max_memory_reserved()
            / 1024**3,
            3,
        ),
    )

    print()
    print(
        "Predictions saved to:"
    )

    print(
        predictions_path.resolve()
    )

    print()
    print(
        "Per-class metrics saved to:"
    )

    print(
        per_class_path.resolve()
    )


if __name__ == "__main__":
    main()