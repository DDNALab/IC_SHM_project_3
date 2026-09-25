import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
TABLES = PROJECT_ROOT / "outputs" / "tables"


def parse_categories(value):
    import json

    if isinstance(value, list):
        return value

    text = str(value).strip()

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return parsed

    except Exception:
        pass

    return []


train = pd.read_csv(
    TABLES / "train.csv"
).fillna("")

val = pd.read_csv(
    TABLES / "val.csv"
).fillna("")


category_counts = {}

for value in train["damage_categories"]:

    for category in parse_categories(value):

        category_counts[category] = (
            category_counts.get(category, 0)
            + 1
        )


majority_category = max(
    category_counts,
    key=category_counts.get,
)

print(
    "\nMajority training category:",
    majority_category,
)


description_candidates = [
    "description",
    "reference_description",
    "damage_description",
    "target_description",
]

description_column = None

for candidate in description_candidates:

    if candidate in val.columns:

        description_column = candidate
        break


if description_column is None:

    print(
        "\nVAL COLUMNS:"
    )

    print(
        val.columns.tolist()
    )

    raise RuntimeError(
        "Could not automatically identify "
        "the description column."
    )


baseline = pd.DataFrame(
    {
        "image_id":
            val["image_id"],

        "true_categories":
            val["damage_categories"],

        "predicted_categories":
            [
                str(
                    [majority_category]
                )
            ]
            * len(val),

        "reference_description":
            val[
                description_column
            ].astype(str),

        "generated_description":
            [
                f"The structure shows "
                f"{majority_category.replace('_', ' ')} "
                f"damage."
            ]
            * len(val),

        "json_valid":
            [True]
            * len(val),
    }
)


baseline_path = (
    TABLES
    / "week5_majority_val_predictions.csv"
)

baseline.to_csv(
    baseline_path,
    index=False,
)


subprocess.run(
    [
        sys.executable,
        "src/evaluate.py",
        "--predictions",
        str(baseline_path),
        "--output-prefix",
        "week5_majority_val",
    ],
    check=True,
)


print(
    "\nWEEK 5 MAJORITY/TEMPLATE "
    "VALIDATION BASELINE COMPLETE"
)

print(
    "Predictions:",
    baseline_path.resolve(),
)