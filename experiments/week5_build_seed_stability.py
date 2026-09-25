from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES = PROJECT_ROOT / "outputs" / "tables"

OUTPUT = TABLES / "seed_stability_results.csv"


def read_single_row(path):
    frame = pd.read_csv(path)

    if len(frame) != 1:
        raise RuntimeError(
            f"Expected exactly one metrics row in {path}, "
            f"but found {len(frame)}."
        )

    return frame.iloc[0]


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

resnet_2026 = read_single_row(
    TABLES / "week5_resnet18_metrics.csv"
)

resnet_2027 = read_single_row(
    TABLES / "week5_resnet18_seed2027_metrics.csv"
)

qwen_2026 = read_single_row(
    TABLES / "week3_v1_direct_free_text_metrics.csv"
)

qwen_2027 = read_single_row(
    TABLES / "week5_v1_direct_seed2027_metrics.csv"
)


# ============================================================
# NORMALIZE METRICS
# ============================================================

records = [
    {
        "model": "ResNet-18",
        "seed": 2026,
        "exact_label_set_accuracy":
            float(
                resnet_2026[
                    "exact_label_set_accuracy"
                ]
            ),
        "micro_f1":
            float(
                resnet_2026[
                    "micro_f1"
                ]
            ),
        "macro_f1":
            float(
                resnet_2026[
                    "macro_f1"
                ]
            ),
        "meteor": np.nan,
        "json_valid_rate": np.nan,
    },

    {
        "model": "ResNet-18",
        "seed": 2027,
        "exact_label_set_accuracy":
            float(
                resnet_2027[
                    "exact_label_set_accuracy"
                ]
            ),
        "micro_f1":
            float(
                resnet_2027[
                    "micro_f1"
                ]
            ),
        "macro_f1":
            float(
                resnet_2027[
                    "macro_f1"
                ]
            ),
        "meteor": np.nan,
        "json_valid_rate": np.nan,
    },

    {
        "model": "Qwen3.5-4B V1-direct QLoRA",
        "seed": 2026,
        "exact_label_set_accuracy":
            float(
                qwen_2026[
                    "exact_label_set_accuracy"
                ]
            ),
        "micro_f1":
            float(
                qwen_2026[
                    "micro_f1"
                ]
            ),
        "macro_f1":
            float(
                qwen_2026[
                    "macro_f1"
                ]
            ),
        "meteor":
            float(
                qwen_2026[
                    "meteor"
                ]
            ),
        "json_valid_rate":
            float(
                qwen_2026[
                    "json_valid_rate"
                ]
            ),
    },

    {
        "model": "Qwen3.5-4B V1-direct QLoRA",
        "seed": 2027,
        "exact_label_set_accuracy":
            float(
                qwen_2027[
                    "exact_label_set_accuracy"
                ]
            ),
        "micro_f1":
            float(
                qwen_2027[
                    "micro_f1"
                ]
            ),
        "macro_f1":
            float(
                qwen_2027[
                    "macro_f1"
                ]
            ),
        "meteor":
            float(
                qwen_2027[
                    "meteor"
                ]
            ),
        "json_valid_rate":
            float(
                qwen_2027[
                    "json_valid_rate"
                ]
            ),
    },
]


seed_df = pd.DataFrame(
    records
)


# ============================================================
# MODEL SUMMARY
# ============================================================

metric_columns = [
    "exact_label_set_accuracy",
    "micro_f1",
    "macro_f1",
    "meteor",
    "json_valid_rate",
]

summary_rows = []

for model_name, group in seed_df.groupby(
    "model",
    sort=False,
):

    summary = {
        "model": model_name,
        "seed": "mean",
    }

    std_summary = {
        "model": model_name,
        "seed": "std",
    }

    for column in metric_columns:

        values = pd.to_numeric(
            group[column],
            errors="coerce",
        )

        valid_values = values.dropna()

        if len(valid_values) == 0:

            summary[column] = np.nan
            std_summary[column] = np.nan

        else:

            summary[column] = float(
                valid_values.mean()
            )

            # Population SD across the two experimental seeds.
            std_summary[column] = float(
                valid_values.std(
                    ddof=0
                )
            )

    summary_rows.append(
        summary
    )

    summary_rows.append(
        std_summary
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# FINAL TABLE
# ============================================================

final_df = pd.concat(
    [
        seed_df,
        summary_df,
    ],
    ignore_index=True,
)


final_df.to_csv(
    OUTPUT,
    index=False,
)


print()
print(
    "========================================"
)

print(
    "WEEK 5 SEED STABILITY TABLE COMPLETE"
)

print(
    "========================================"
)

print()

print(
    final_df.to_string(
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