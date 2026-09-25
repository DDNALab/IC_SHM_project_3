from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TABLES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

FIGURES = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)

INPUT = (
    TABLES
    / "main_results.csv"
)

OUTPUT = (
    FIGURES
    / "main_model_comparison.png"
)


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(
    INPUT
)


# Short labels make the figure easier to read.
name_map = {
    "Majority/Template Baseline":
        "Majority\nBaseline",

    "ResNet-18":
        "ResNet-18",

    "Zero-shot Qwen3.5-4B":
        "Zero-shot\nQwen",

    "Qwen3.5-4B V1-direct QLoRA":
        "V1-direct\nQwen",

    "Qwen3.5-4B Fixed-Crop QLoRA":
        "Fixed-Crop\nQwen",
}


df["display_name"] = (
    df["model"]
    .map(name_map)
)


# ============================================================
# METRICS TO DISPLAY
# ============================================================

metrics = [
    (
        "exact_label_set_accuracy",
        "Exact-set Accuracy",
    ),
    (
        "micro_f1",
        "Micro-F1",
    ),
    (
        "macro_f1",
        "Macro-F1",
    ),
    (
        "meteor",
        "METEOR",
    ),
]


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9),
)

axes = axes.flatten()


for ax, (
    column,
    title,
) in zip(
    axes,
    metrics,
):

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    x = np.arange(
        len(df)
    )

    bars = ax.bar(
        x,
        values.fillna(0),
    )

    ax.set_title(
        title,
        fontsize=13,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        df["display_name"],
        fontsize=9,
    )

    ax.set_ylabel(
        "Score"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )


    # Add numeric values above bars.
    for i, (
        bar,
        value,
    ) in enumerate(
        zip(
            bars,
            values,
        )
    ):

        if pd.isna(
            value
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                0.03,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        else:

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height()
                + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )


fig.suptitle(
    "Week 5 Main Model Comparison",
    fontsize=16,
)


fig.text(
    0.5,
    0.01,
    (
        "All results use the same 118-image validation split. "
        "ResNet-18 is classification-only, so METEOR is not applicable."
    ),
    ha="center",
    fontsize=9,
)


fig.tight_layout(
    rect=[
        0,
        0.04,
        1,
        0.96,
    ]
)


fig.savefig(
    OUTPUT,
    dpi=220,
    bbox_inches="tight",
)


plt.close(
    fig
)


print()

print(
    "========================================"
)

print(
    "WEEK 5 MAIN MODEL COMPARISON COMPLETE"
)

print(
    "========================================"
)

print()

print(
    "Saved to:"
)

print(
    OUTPUT.resolve()
)