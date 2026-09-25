from pathlib import Path

import numpy as np
import pandas as pd

from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from data_processing import CATEGORIES
from utils import parse_list_field


TABLES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

OUTPUT = (
    TABLES
    / "multi_damage_results.csv"
)


# ============================================================
# SAME METEOR LOGIC AS src/evaluate.py
# ============================================================

def corpus_meteor(
    references,
    hypotheses,
):

    scores = []

    for reference, hypothesis in zip(
        references,
        hypotheses,
    ):

        try:

            score = meteor_score(
                [
                    str(reference).split()
                ],
                str(hypothesis).split(),
            )

        except LookupError as error:

            raise RuntimeError(
                "NLTK WordNet resources are missing."
            ) from error

        scores.append(
            float(score)
        )

    if not scores:
        return np.nan

    return float(
        np.mean(scores)
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    frame,
):

    true_sets = [
        parse_list_field(value)
        for value
        in frame[
            "true_categories"
        ]
    ]

    predicted_sets = [
        parse_list_field(value)
        for value
        in frame[
            "predicted_categories"
        ]
    ]

    binarizer = MultiLabelBinarizer(
        classes=CATEGORIES
    )

    y_true = (
        binarizer.fit_transform(
            true_sets
        )
    )

    y_pred = (
        binarizer.transform(
            predicted_sets
        )
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

    exact = accuracy_score(
        y_true,
        y_pred,
    )

    return {
        "exact_label_set_accuracy":
            float(exact),

        "micro_precision":
            float(micro_p),

        "micro_recall":
            float(micro_r),

        "micro_f1":
            float(micro_f1),

        "macro_precision":
            float(macro_p),

        "macro_recall":
            float(macro_r),

        "macro_f1":
            float(macro_f1),
    }


# ============================================================
# FILTER MULTI-DAMAGE IMAGES
# ============================================================

def filter_multi_damage(
    frame,
):

    frame = frame.copy()

    frame[
        "true_damage_count"
    ] = frame[
        "true_categories"
    ].apply(
        lambda value:
            len(
                parse_list_field(
                    value
                )
            )
    )

    return (
        frame[
            frame[
                "true_damage_count"
            ] >= 2
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# GENERIC VLM / TEMPLATE MODEL
# ============================================================

def evaluate_text_model(
    model_name,
    path,
):

    frame = pd.read_csv(
        path
    ).fillna("")

    subset = filter_multi_damage(
        frame
    )

    metrics = classification_metrics(
        subset
    )

    meteor = corpus_meteor(
        subset[
            "reference_description"
        ].astype(str).tolist(),

        subset[
            "generated_description"
        ].astype(str).tolist(),
    )

    if "json_valid" in subset.columns:

        json_valid_rate = float(
            subset[
                "json_valid"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
            .mean()
        )

    else:

        json_valid_rate = np.nan

    return {
        "model":
            model_name,

        "n_multi_damage_images":
            int(
                len(subset)
            ),

        **metrics,

        "meteor":
            float(meteor),

        "json_valid_rate":
            json_valid_rate,
    }


# ============================================================
# RESNET CLASSIFICATION-ONLY MODEL
# ============================================================

def evaluate_resnet(
    path,
):

    frame = pd.read_csv(
        path
    ).fillna("")

    subset = filter_multi_damage(
        frame
    )

    metrics = classification_metrics(
        subset
    )

    return {
        "model":
            "ResNet-18",

        "n_multi_damage_images":
            int(
                len(subset)
            ),

        **metrics,

        # ResNet does not generate descriptions.
        "meteor":
            np.nan,

        "json_valid_rate":
            np.nan,
    }


# ============================================================
# EVALUATE ALL FIVE WEEK 5 MODELS
# ============================================================

rows = []


rows.append(
    evaluate_text_model(
        "Majority/Template Baseline",
        TABLES
        / "week5_majority_val_scored_predictions.csv",
    )
)


rows.append(
    evaluate_resnet(
        TABLES
        / "week5_resnet18_val_predictions.csv",
    )
)


rows.append(
    evaluate_text_model(
        "Zero-shot Qwen3.5-4B",
        TABLES
        / "zero_shot_scored_predictions.csv",
    )
)


rows.append(
    evaluate_text_model(
        "Qwen3.5-4B V1-direct QLoRA",
        TABLES
        / "week3_v1_direct_free_text_scored_predictions.csv",
    )
)


rows.append(
    evaluate_text_model(
        "Qwen3.5-4B Fixed-Crop QLoRA",
        TABLES
        / "week4_fixed_crops_scored_predictions.csv",
    )
)


# ============================================================
# CREATE FINAL TABLE
# ============================================================

result = pd.DataFrame(
    rows
)


result.to_csv(
    OUTPUT,
    index=False,
)


print()

print(
    "========================================"
)

print(
    "WEEK 5 MULTI-DAMAGE RESULTS COMPLETE"
)

print(
    "========================================"
)

print()

print(
    result.to_string(
        index=False
    )
)

print()

print(
    "Saved to:"
)

print(
    OUTPUT.resolve()
)