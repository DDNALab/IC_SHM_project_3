from pathlib import Path
import json
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES = PROJECT_ROOT / "outputs" / "tables"

OUTPUT = (
    TABLES
    / "category_description_consistency.csv"
)


# ============================================================
# CATEGORY LANGUAGE ALIASES
# ============================================================

CATEGORY_PATTERNS = {
    "crack": [
        r"\bcrack\b",
        r"\bcracks\b",
        r"\bcracked\b",
        r"\bcracking\b",
    ],

    "void": [
        r"\bvoid\b",
        r"\bvoids\b",
        r"\bcavity\b",
        r"\bcavities\b",
    ],

    "honeycomb": [
        r"\bhoneycomb\b",
        r"\bhoneycombing\b",
    ],

    "looseness": [
        r"\blooseness\b",
        r"\bloose\b",
        r"\bloosened\b",
    ],

    "spalling": [
        r"\bspall\b",
        r"\bspalls\b",
        r"\bspalled\b",
        r"\bspalling\b",
    ],

    "exposed_rebar": [
        r"\bexposed rebar\b",
        r"\bvisible rebar\b",
        r"\bexposed reinforcement\b",
        r"\bvisible reinforcement\b",
        r"\bexposed reinforcing\b",
        r"\bexposed steel\b",
    ],

    "corrosion": [
        r"\bcorrosion\b",
        r"\bcorroded\b",
        r"\bcorroding\b",
        r"\brust\b",
        r"\brusted\b",
        r"\brusting\b",
        r"\brusty\b",
    ],

    "efflorescence": [
        r"\befflorescence\b",
        r"\bsalt deposit\b",
        r"\bsalt deposits\b",
        r"\bwhite deposit\b",
        r"\bwhite deposits\b",
    ],

    "pothole": [
        r"\bpothole\b",
        r"\bpotholes\b",
    ],
}


def parse_categories(value):

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    text = str(value).strip()

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return [
                str(item)
                for item in parsed
            ]

    except Exception:
        pass

    return []


def categories_mentioned_in_description(
    description,
):

    text = str(
        description
    ).lower()

    mentioned = []

    for category, patterns in (
        CATEGORY_PATTERNS.items()
    ):

        found = any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            is not None
            for pattern in patterns
        )

        if found:
            mentioned.append(
                category
            )

    return mentioned


def analyze_model(
    model_name,
    filename,
):

    frame = pd.read_csv(
        TABLES / filename
    ).fillna("")

    rows = []

    for _, row in frame.iterrows():

        predicted = parse_categories(
            row[
                "predicted_categories"
            ]
        )

        mentioned = (
            categories_mentioned_in_description(
                row[
                    "generated_description"
                ]
            )
        )

        predicted_set = set(
            predicted
        )

        mentioned_set = set(
            mentioned
        )

        missing_from_description = sorted(
            predicted_set
            - mentioned_set
        )

        extra_in_description = sorted(
            mentioned_set
            - predicted_set
        )

        fully_consistent = (
            len(
                missing_from_description
            ) == 0
            and len(
                extra_in_description
            ) == 0
        )

        rows.append(
            {
                "model":
                    model_name,

                "image_id":
                    row[
                        "image_id"
                    ],

                "predicted_categories":
                    json.dumps(
                        predicted
                    ),

                "description_mentions":
                    json.dumps(
                        mentioned
                    ),

                "missing_predicted_categories_in_description":
                    json.dumps(
                        missing_from_description
                    ),

                "extra_categories_mentioned_in_description":
                    json.dumps(
                        extra_in_description
                    ),

                "fully_consistent":
                    fully_consistent,

                "generated_description":
                    row[
                        "generated_description"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ANALYZE TWO FINAL QWEN VARIANTS
# ============================================================

direct = analyze_model(
    "Qwen3.5-4B V1-direct QLoRA",
    "week3_v1_direct_free_text_scored_predictions.csv",
)

fixed = analyze_model(
    "Qwen3.5-4B Fixed-Crop QLoRA",
    "week4_fixed_crops_scored_predictions.csv",
)


combined = pd.concat(
    [
        direct,
        fixed,
    ],
    ignore_index=True,
)


combined.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for model_name, group in combined.groupby(
    "model",
    sort=False,
):

    fully_consistent = int(
        group[
            "fully_consistent"
        ].sum()
    )

    missing_count = int(
        group[
            "missing_predicted_categories_in_description"
        ]
        .apply(
            lambda value:
                len(
                    json.loads(
                        value
                    )
                ) > 0
        )
        .sum()
    )

    extra_count = int(
        group[
            "extra_categories_mentioned_in_description"
        ]
        .apply(
            lambda value:
                len(
                    json.loads(
                        value
                    )
                ) > 0
        )
        .sum()
    )

    summary_rows.append(
        {
            "model":
                model_name,

            "n_images":
                int(
                    len(group)
                ),

            "fully_consistent_images":
                fully_consistent,

            "consistency_rate":
                fully_consistent
                / len(group),

            "images_with_missing_description_category":
                missing_count,

            "images_with_extra_description_category":
                extra_count,
        }
    )


summary = pd.DataFrame(
    summary_rows
)

summary_path = (
    TABLES
    / "category_description_consistency_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


print()

print(
    "========================================"
)

print(
    "WEEK 5 CATEGORY-DESCRIPTION "
    "CONSISTENCY COMPLETE"
)

print(
    "========================================"
)

print()

print(
    summary.to_string(
        index=False
    )
)

print()

print(
    "Detailed results:"
)

print(
    OUTPUT.resolve()
)

print()

print(
    "Summary:"
)

print(
    summary_path.resolve()
)

print()

print(
    "NOTE: This is a keyword/alias-based consistency "
    "analysis, not a semantic-language metric."
)