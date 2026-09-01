from __future__ import annotations

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

sys.path.insert(0, "src")

from data_processing import CATEGORIES
from utils import parse_list_field


TABLES_DIR = Path("outputs/tables")

EXPERIMENTS = {
    "week3_v1_direct_free_text": (
        TABLES_DIR
        / "week3_v1_direct_free_text_scored_predictions.csv"
    ),

    "week4_fixed_crops": (
        TABLES_DIR
        / "week4_fixed_crops_scored_predictions.csv"
    ),

    "week4_selected_crops": (
        TABLES_DIR
        / "week4_selected_crops_scored_predictions.csv"
    ),
}

OUTPUT_FILE = (
    TABLES_DIR
    / "fine_damage_subset_metrics.csv"
)


def corpus_meteor(
    references: list[str],
    hypotheses: list[str],
) -> float:

    scores = []

    for reference, hypothesis in zip(
        references,
        hypotheses,
    ):
        try:
            score = meteor_score(
                [reference.split()],
                hypothesis.split(),
            )
        except LookupError as error:
            raise RuntimeError(
                "NLTK WordNet resources are missing. "
                "Run: python -m nltk.downloader "
                "wordnet omw-1.4"
            ) from error

        scores.append(
            float(score)
        )

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )


def evaluate_subset(
    frame: pd.DataFrame,
) -> dict:

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

    exact_match = accuracy_score(
        y_true,
        y_pred,
    )

    meteor = corpus_meteor(
        frame[
            "reference_description"
        ]
        .astype(str)
        .tolist(),

        frame[
            "generated_description"
        ]
        .astype(str)
        .tolist(),
    )

    json_valid_rate = float(
        frame[
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

    return {
        "n_images": len(frame),

        "exact_label_set_accuracy":
            float(
                exact_match
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

        "meteor":
            float(
                meteor
            ),

        "json_valid_rate":
            json_valid_rate,
    }


def main() -> None:

    rows = []

    reference_ids = None

    for experiment_name, path in (
        EXPERIMENTS.items()
    ):

        if not path.exists():
            raise FileNotFoundError(
                f"Missing prediction file: {path}"
            )

        frame = (
            pd.read_csv(
                path
            )
            .fillna("")
        )

        # --------------------------------------------------
        # Fine-damage proxy:
        # validation samples whose ground truth contains crack
        # --------------------------------------------------

        mask = frame[
            "true_categories"
        ].apply(
            lambda value:
            "crack"
            in parse_list_field(
                value
            )
        )

        subset = (
            frame[
                mask
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        current_ids = set(
            subset[
                "image_id"
            ]
            .astype(str)
            .tolist()
        )

        # --------------------------------------------------
        # Verify every experiment uses same subset
        # --------------------------------------------------

        if reference_ids is None:
            reference_ids = (
                current_ids
            )

        elif (
            current_ids
            != reference_ids
        ):
            raise RuntimeError(
                "Fine-damage subsets do not match "
                "between experiments."
            )

        metrics = evaluate_subset(
            subset
        )

        rows.append(
            {
                "experiment":
                    experiment_name,

                "subset_definition":
                    (
                        "validation samples "
                        "with ground-truth "
                        "category containing crack"
                    ),

                **metrics,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "=== FINE-DAMAGE SUBSET METRICS ==="
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
        OUTPUT_FILE.resolve()
    )


if __name__ == "__main__":
    main()