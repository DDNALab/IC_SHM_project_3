import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

test = pd.read_csv(
    TABLES / "test.csv"
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

    if candidate in test.columns:

        description_column = candidate
        break


if description_column is None:

    print(
        "\nTEST COLUMNS:"
    )

    print(
        test.columns.tolist()
    )

    raise RuntimeError(
        "Could not automatically identify "
        "the description column."
    )


baseline = pd.DataFrame(
    {
        "image_id":
            test["image_id"],

        "true_categories":
            test["damage_categories"],

        "predicted_categories":
            [
                str(
                    [majority_category]
                )
            ]
            * len(test),

        "reference_description":
            test[
                description_column
            ].astype(str),

        "generated_description":
            [
                f"The structure shows "
                f"{majority_category.replace('_', ' ')} "
                f"damage."
            ]
            * len(test),

        "json_valid":
            [True]
            * len(test),
    }
)


baseline_path = (
    TABLES
    / "week5_majority_test_predictions.csv"
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
        "week5_majority_test",
    ],
    check=True,
)


print(
    "\nWEEK 5 MAJORITY/TEMPLATE "
    "TEST BASELINE COMPLETE"
)

print(
    "Predictions:",
    baseline_path.resolve(),
)