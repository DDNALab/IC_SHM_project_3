from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES = PROJECT_ROOT / "outputs" / "tables"

OUTPUT = TABLES / "main_results.csv"


def load_metrics(filename):
    path = TABLES / filename

    frame = pd.read_csv(path)

    if len(frame) != 1:
        raise RuntimeError(
            f"Expected one metrics row in {path}, "
            f"found {len(frame)}."
        )

    return frame.iloc[0]


# ============================================================
# LOAD FIVE REQUIRED MODELS
# ============================================================

majority = load_metrics(
    "week5_majority_val_metrics.csv"
)

resnet = load_metrics(
    "week5_resnet18_metrics.csv"
)

zero_shot = load_metrics(
    "zero_shot_metrics.csv"
)

qwen_direct = load_metrics(
    "week3_v1_direct_free_text_metrics.csv"
)

qwen_fixed = load_metrics(
    "week4_fixed_crops_metrics.csv"
)


# ============================================================
# HELPER
# ============================================================

def metric(row, name, default=np.nan):

    if name not in row.index:
        return default

    value = row[name]

    if pd.isna(value):
        return default

    return float(value)


# ============================================================
# MAIN RESULTS
# ============================================================

rows = [
    {
        "model":
            "Majority/Template Baseline",

        "model_family":
            "Rule-based baseline",

        "visual_input":
            "none",

        "exact_label_set_accuracy":
            metric(
                majority,
                "exact_label_set_accuracy",
            ),

        "micro_f1":
            metric(
                majority,
                "micro_f1",
            ),

        "macro_f1":
            metric(
                majority,
                "macro_f1",
            ),

        "meteor":
            metric(
                majority,
                "meteor",
            ),

        "json_valid_rate":
            metric(
                majority,
                "json_valid_rate",
            ),
    },

    {
        "model":
            "ResNet-18",

        "model_family":
            "Conventional CNN",

        "visual_input":
            "single image",

        "exact_label_set_accuracy":
            metric(
                resnet,
                "exact_label_set_accuracy",
            ),

        "micro_f1":
            metric(
                resnet,
                "micro_f1",
            ),

        "macro_f1":
            metric(
                resnet,
                "macro_f1",
            ),

        # ResNet performs classification only.
        "meteor":
            np.nan,

        "json_valid_rate":
            np.nan,
    },

    {
        "model":
            "Zero-shot Qwen3.5-4B",

        "model_family":
            "Vision-language model",

        "visual_input":
            "single image",

        "exact_label_set_accuracy":
            metric(
                zero_shot,
                "exact_label_set_accuracy",
            ),

        "micro_f1":
            metric(
                zero_shot,
                "micro_f1",
            ),

        "macro_f1":
            metric(
                zero_shot,
                "macro_f1",
            ),

        "meteor":
            metric(
                zero_shot,
                "meteor",
            ),

        "json_valid_rate":
            metric(
                zero_shot,
                "json_valid_rate",
            ),
    },

    {
        "model":
            "Qwen3.5-4B V1-direct QLoRA",

        "model_family":
            "Fine-tuned vision-language model",

        "visual_input":
            "single image",

        "exact_label_set_accuracy":
            metric(
                qwen_direct,
                "exact_label_set_accuracy",
            ),

        "micro_f1":
            metric(
                qwen_direct,
                "micro_f1",
            ),

        "macro_f1":
            metric(
                qwen_direct,
                "macro_f1",
            ),

        "meteor":
            metric(
                qwen_direct,
                "meteor",
            ),

        "json_valid_rate":
            metric(
                qwen_direct,
                "json_valid_rate",
            ),
    },

    {
        "model":
            "Qwen3.5-4B Fixed-Crop QLoRA",

        "model_family":
            "Enhanced vision-language model",

        "visual_input":
            "full image + 2 fixed crops",

        "exact_label_set_accuracy":
            metric(
                qwen_fixed,
                "exact_label_set_accuracy",
            ),

        "micro_f1":
            metric(
                qwen_fixed,
                "micro_f1",
            ),

        "macro_f1":
            metric(
                qwen_fixed,
                "macro_f1",
            ),

        "meteor":
            metric(
                qwen_fixed,
                "meteor",
            ),

        "json_valid_rate":
            metric(
                qwen_fixed,
                "json_valid_rate",
            ),
    },
]


result = pd.DataFrame(rows)

result.to_csv(
    OUTPUT,
    index=False,
)


print()
print(
    "========================================"
)

print(
    "WEEK 5 MAIN RESULTS TABLE COMPLETE"
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