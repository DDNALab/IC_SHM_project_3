from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES = PROJECT_ROOT / "outputs" / "tables"

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from data_processing import CATEGORIES
from utils import parse_list_field


OUTPUT = (
    TABLES
    / "resolution_subset_results.csv"
)

MEDIAN_PIXELS = 262144


def classification_metrics(frame):

    true_sets = [
        parse_list_field(value)
        for value
        in frame["true_categories"]
    ]

    predicted_sets = [
        parse_list_field(value)
        for value
        in frame["predicted_categories"]
    ]

    binarizer = MultiLabelBinarizer(
        classes=CATEGORIES
    )

    y_true = binarizer.fit_transform(
        true_sets
    )

    y_pred = binarizer.transform(
        predicted_sets
    )

    exact = accuracy_score(
        y_true,
        y_pred,
    )

    (
        _,
        _,
        micro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    (
        _,
        _,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "exact_label_set_accuracy":
            float(exact),

        "micro_f1":
            float(micro_f1),

        "macro_f1":
            float(macro_f1),
    }


metadata = pd.read_csv(
    TABLES / "image_metadata.csv"
)

metadata["pixels"] = (
    metadata["width"]
    * metadata["height"]
)

metadata["resolution_group"] = np.where(
    metadata["pixels"] < MEDIAN_PIXELS,
    "low_resolution",
    "high_resolution",
)


MODELS = {
    "ResNet-18":
        "week5_resnet18_val_predictions.csv",

    "Qwen3.5-4B V1-direct QLoRA":
        "week3_v1_direct_free_text_scored_predictions.csv",

    "Qwen3.5-4B Fixed-Crop QLoRA":
        "week4_fixed_crops_scored_predictions.csv",
}


rows = []


for model_name, filename in MODELS.items():

    predictions = pd.read_csv(
        TABLES / filename
    ).fillna("")

    merged = predictions.merge(
        metadata[
            [
                "image_id",
                "width",
                "height",
                "pixels",
                "resolution_group",
            ]
        ],
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if merged[
        "resolution_group"
    ].isna().any():

        raise RuntimeError(
            f"Missing image metadata for {model_name}"
        )

    for group_name in [
        "low_resolution",
        "high_resolution",
    ]:

        subset = (
            merged[
                merged[
                    "resolution_group"
                ] == group_name
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        metrics = classification_metrics(
            subset
        )

        meteor = np.nan

        if (
            "meteor" in subset.columns
            and model_name != "ResNet-18"
        ):

            meteor = float(
                pd.to_numeric(
                    subset["meteor"],
                    errors="coerce",
                ).mean()
            )

        rows.append(
            {
                "model":
                    model_name,

                "resolution_group":
                    group_name,

                "threshold_pixels":
                    MEDIAN_PIXELS,

                "n_images":
                    int(
                        len(subset)
                    ),

                "median_pixels_in_group":
                    float(
                        subset[
                            "pixels"
                        ].median()
                    ),

                **metrics,

                "meteor":
                    meteor,
            }
        )


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
    "WEEK 5 RESOLUTION ANALYSIS COMPLETE"
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