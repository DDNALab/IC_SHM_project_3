from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES = PROJECT_ROOT / "outputs" / "tables"

OUTPUT = TABLES / "per_class_results.csv"


FILES = {
    "ResNet-18":
        "week5_resnet18_per_class_metrics.csv",

    "Zero-shot Qwen3.5-4B":
        "zero_shot_per_class_metrics.csv",

    "Qwen3.5-4B V1-direct QLoRA":
        "week3_v1_direct_free_text_per_class_metrics.csv",

    "Qwen3.5-4B Fixed-Crop QLoRA":
        "week4_fixed_crops_per_class_metrics.csv",
}


CATEGORY_ORDER = [
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


frames = []

for model_name, filename in FILES.items():

    frame = pd.read_csv(
        TABLES / filename
    ).copy()

    frame["model"] = model_name

    frames.append(
        frame
    )


combined = pd.concat(
    frames,
    ignore_index=True,
)


combined["category"] = pd.Categorical(
    combined["category"],
    categories=CATEGORY_ORDER,
    ordered=True,
)


combined = combined.sort_values(
    [
        "category",
        "model",
    ]
).reset_index(
    drop=True
)


# ============================================================
# ADD SUPPORT INTERPRETATION
# ============================================================

def support_group(value):

    value = int(value)

    if value == 0:
        return "no validation samples"

    if value < 10:
        return "very low support"

    if value <= 20:
        return "low support"

    return "moderate/high support"


combined[
    "support_group"
] = combined[
    "support"
].apply(
    support_group
)


# ============================================================
# SAVE LONG-FORM TABLE
# ============================================================

combined.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# DISPLAY F1 MATRIX
# ============================================================

f1_matrix = combined.pivot(
    index="category",
    columns="model",
    values="f1",
)


support_series = (
    combined
    .drop_duplicates(
        subset=["category"]
    )
    .set_index(
        "category"
    )[
        "support"
    ]
)


f1_matrix.insert(
    0,
    "support",
    support_series,
)


print()
print(
    "========================================"
)

print(
    "WEEK 5 PER-CLASS RESULTS COMPLETE"
)

print(
    "========================================"
)

print()

print(
    f1_matrix.to_string()
)

print()

print(
    "Saved to:"
)

print(
    OUTPUT.resolve()
)