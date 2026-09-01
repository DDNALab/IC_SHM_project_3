from pathlib import Path

import pandas as pd


ROOT = Path(".")
TABLES = ROOT / "outputs" / "tables"

OUTPUT = TABLES / "week5_val_test_comparison.csv"


rows = [
    {
        "model": "Majority/template",
        "split": "validation",
        "exact_label_set_accuracy": 0.000000,
        "micro_f1": 0.380952,
        "macro_f1": 0.078144,
        "meteor": 0.062349,
        "json_valid_rate": 1.000000,
    },
    {
        "model": "Majority/template",
        "split": "test",
        "exact_label_set_accuracy": 0.000000,
        "micro_f1": 0.387640,
        "macro_f1": 0.079038,
        "meteor": 0.062247,
        "json_valid_rate": 1.000000,
    },

    {
        "model": "ResNet-18",
        "split": "validation",
        "exact_label_set_accuracy": 0.872881,
        "micro_f1": 0.952596,
        "macro_f1": 0.842340,
        "meteor": None,
        "json_valid_rate": None,
    },
    {
        "model": "ResNet-18",
        "split": "test",
        "exact_label_set_accuracy": 0.760000,
        "micro_f1": 0.920430,
        "macro_f1": 0.822070,
        "meteor": None,
        "json_valid_rate": None,
    },

    {
        "model": "Zero-shot Qwen3.5-4B",
        "split": "validation",
        "exact_label_set_accuracy": 0.330508,
        "micro_f1": 0.656934,
        "macro_f1": 0.510702,
        "meteor": 0.173194,
        "json_valid_rate": 0.991525,
    },
    {
        "model": "Zero-shot Qwen3.5-4B",
        "split": "test",
        "exact_label_set_accuracy": 0.304000,
        "micro_f1": 0.630385,
        "macro_f1": 0.508069,
        "meteor": 0.180772,
        "json_valid_rate": 1.000000,
    },

    {
        "model": "V1-direct Qwen3.5-4B QLoRA",
        "split": "validation",
        "exact_label_set_accuracy": 0.864407,
        "micro_f1": 0.939326,
        "macro_f1": 0.818730,
        "meteor": 0.590538,
        "json_valid_rate": 1.000000,
    },
    {
        "model": "V1-direct Qwen3.5-4B QLoRA",
        "split": "test",
        "exact_label_set_accuracy": 0.768000,
        "micro_f1": 0.876356,
        "macro_f1": 0.742141,
        "meteor": 0.540067,
        "json_valid_rate": 1.000000,
    },

    {
        "model": "Fixed-crop Qwen3.5-4B QLoRA",
        "split": "validation",
        "exact_label_set_accuracy": 0.838983,
        "micro_f1": 0.922374,
        "macro_f1": 0.800841,
        "meteor": 0.598817,
        "json_valid_rate": 1.000000,
    },
    {
        "model": "Fixed-crop Qwen3.5-4B QLoRA",
        "split": "test",
        "exact_label_set_accuracy": 0.768000,
        "micro_f1": 0.880694,
        "macro_f1": 0.769518,
        "meteor": 0.537439,
        "json_valid_rate": 1.000000,
    },
]


df = pd.DataFrame(rows)


# ------------------------------------------------------------
# Add generalization gaps
# validation - test
# ------------------------------------------------------------

gap_rows = []

for model in df["model"].unique():

    model_df = df[
        df["model"] == model
    ].set_index("split")

    if (
        "validation" not in model_df.index
        or "test" not in model_df.index
    ):
        continue

    val = model_df.loc["validation"]
    test = model_df.loc["test"]

    gap_rows.append(
        {
            "model": model,

            "exact_gap_val_minus_test":
                (
                    val["exact_label_set_accuracy"]
                    - test["exact_label_set_accuracy"]
                ),

            "micro_f1_gap_val_minus_test":
                (
                    val["micro_f1"]
                    - test["micro_f1"]
                ),

            "macro_f1_gap_val_minus_test":
                (
                    val["macro_f1"]
                    - test["macro_f1"]
                ),

            "meteor_gap_val_minus_test":
                (
                    val["meteor"]
                    - test["meteor"]
                    if pd.notna(val["meteor"])
                    and pd.notna(test["meteor"])
                    else None
                ),
        }
    )


gap_df = pd.DataFrame(
    gap_rows
)


df.to_csv(
    OUTPUT,
    index=False,
)


GAP_OUTPUT = (
    TABLES
    / "week5_generalization_gaps.csv"
)


gap_df.to_csv(
    GAP_OUTPUT,
    index=False,
)


print()
print(
    "========================================"
)
print(
    "WEEK 5 VALIDATION VS TEST COMPARISON"
)
print(
    "========================================"
)
print()

print(
    df.to_string(
        index=False
    )
)

print()

print(
    "Generalization gaps "
    "(validation - test):"
)

print()

print(
    gap_df.to_string(
        index=False
    )
)

print()

print(
    "Saved:"
)

print(
    OUTPUT.resolve()
)

print(
    GAP_OUTPUT.resolve()
)