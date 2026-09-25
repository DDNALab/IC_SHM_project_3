from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

sys.path.insert(0, "src")

from dataset import (
    generate_overlapping_crops,
    select_informative_crops,
)
from utils import parse_list_field


TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    FIGURES_DIR
    / "fine_damage_examples.png"
)

VAL_FILE = (
    TABLES_DIR
    / "val.csv"
)

BASELINE_FILE = (
    TABLES_DIR
    / "week3_v1_direct_free_text_scored_predictions.csv"
)

FIXED_FILE = (
    TABLES_DIR
    / "week4_fixed_crops_scored_predictions.csv"
)

SELECTED_FILE = (
    TABLES_DIR
    / "week4_selected_crops_scored_predictions.csv"
)


# --------------------------------------------------
# Load files
# --------------------------------------------------

val = pd.read_csv(
    VAL_FILE
).fillna("")

baseline = pd.read_csv(
    BASELINE_FILE
).fillna("")

fixed = pd.read_csv(
    FIXED_FILE
).fillna("")

selected = pd.read_csv(
    SELECTED_FILE
).fillna("")


# --------------------------------------------------
# Merge predictions
# --------------------------------------------------

baseline = baseline[
    [
        "image_id",
        "true_categories",
        "predicted_categories",
        "meteor",
    ]
].rename(
    columns={
        "predicted_categories":
            "baseline_prediction",
        "meteor":
            "baseline_meteor",
    }
)

fixed = fixed[
    [
        "image_id",
        "predicted_categories",
        "meteor",
    ]
].rename(
    columns={
        "predicted_categories":
            "fixed_prediction",
        "meteor":
            "fixed_meteor",
    }
)

selected = selected[
    [
        "image_id",
        "predicted_categories",
        "meteor",
    ]
].rename(
    columns={
        "predicted_categories":
            "selected_prediction",
        "meteor":
            "selected_meteor",
    }
)

merged = (
    val.merge(
        baseline,
        on="image_id",
        how="inner",
    )
    .merge(
        fixed,
        on="image_id",
        how="inner",
    )
    .merge(
        selected,
        on="image_id",
        how="inner",
    )
)


# --------------------------------------------------
# Keep crack-containing validation examples
# --------------------------------------------------

crack_rows = merged[
    merged[
        "true_categories"
    ].apply(
        lambda value:
        "crack"
        in parse_list_field(
            value
        )
    )
].copy()


# --------------------------------------------------
# Prefer examples where fixed crops improve METEOR
# --------------------------------------------------

crack_rows[
    "fixed_gain"
] = (
    crack_rows[
        "fixed_meteor"
    ]
    - crack_rows[
        "baseline_meteor"
    ]
)

crack_rows = crack_rows.sort_values(
    "fixed_gain",
    ascending=False,
)


# --------------------------------------------------
# Select three representative examples
# --------------------------------------------------

examples = crack_rows.head(
    3
).copy()

if len(examples) < 3:
    raise RuntimeError(
        "Could not find at least "
        "3 crack-containing validation images."
    )


# --------------------------------------------------
# Utility
# --------------------------------------------------

def open_rgb(path):
    with Image.open(path) as image:
        image = (
            ImageOps.exif_transpose(
                image
            )
            .convert("RGB")
        )
        return image.copy()


def shorten_list(value):
    categories = parse_list_field(
        value
    )

    if not categories:
        return "None"

    return ", ".join(
        categories
    )


# --------------------------------------------------
# Build figure
# --------------------------------------------------

fig, axes = plt.subplots(
    nrows=3,
    ncols=4,
    figsize=(16, 12),
)

column_titles = [
    "Original Image",
    "Fixed Local Crops",
    "Automatically Selected Crops",
    "Model Comparison",
]

for column_index, title in enumerate(
    column_titles
):
    axes[
        0,
        column_index,
    ].set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=10,
    )


for row_index, (_, row) in enumerate(
    examples.iterrows()
):

    image_path = Path(
        row[
            "image_path"
        ]
    )

    image = open_rgb(
        image_path
    )

    crops = generate_overlapping_crops(
        image,
        crop_fraction=0.65,
    )

    # --------------------------------------------------
    # Fixed crops used during Week 4
    # --------------------------------------------------

    fixed_names = [
        "top_left",
        "bottom_right",
    ]

    fixed_preview = image.copy()

    fixed_draw = ImageDraw.Draw(
        fixed_preview
    )

    for name in fixed_names:

        width, height = image.size

        crop_width = int(
            width * 0.65
        )

        crop_height = int(
            height * 0.65
        )

        if name == "top_left":
            box = (
                0,
                0,
                crop_width,
                crop_height,
            )

        elif name == "bottom_right":
            box = (
                width - crop_width,
                height - crop_height,
                width,
                height,
            )

        fixed_draw.rectangle(
            box,
            outline="white",
            width=max(
                2,
                width // 150,
            ),
        )

    # --------------------------------------------------
    # Selected crops
    # --------------------------------------------------

    selected_images = (
        select_informative_crops(
            crops,
            top_k=2,
        )
    )

    selected_preview = image.copy()

    selected_draw = ImageDraw.Draw(
        selected_preview
    )

    crop_items = list(
        crops.items()
    )

    for selected_crop in selected_images:

        matched_name = None

        for name, crop in crop_items:

            if (
                crop.size
                == selected_crop.size
                and list(
                    crop.getdata()
                )[:50]
                == list(
                    selected_crop.getdata()
                )[:50]
            ):
                matched_name = name
                break

        if matched_name is None:
            continue

        width, height = image.size

        crop_width = int(
            width * 0.65
        )

        crop_height = int(
            height * 0.65
        )

        boxes = {
            "top_left": (
                0,
                0,
                crop_width,
                crop_height,
            ),
            "top_right": (
                width - crop_width,
                0,
                width,
                crop_height,
            ),
            "bottom_left": (
                0,
                height - crop_height,
                crop_width,
                height,
            ),
            "bottom_right": (
                width - crop_width,
                height - crop_height,
                width,
                height,
            ),
        }

        selected_draw.rectangle(
            boxes[
                matched_name
            ],
            outline="white",
            width=max(
                2,
                width // 150,
            ),
        )

    # --------------------------------------------------
    # Original image
    # --------------------------------------------------

    axes[
        row_index,
        0,
    ].imshow(
        image
    )

    axes[
        row_index,
        0,
    ].axis(
        "off"
    )

    axes[
        row_index,
        0,
    ].set_ylabel(
        str(
            row[
                "image_id"
            ]
        ),
        fontsize=11,
        fontweight="bold",
    )

    # --------------------------------------------------
    # Fixed preview
    # --------------------------------------------------

    axes[
        row_index,
        1,
    ].imshow(
        fixed_preview
    )

    axes[
        row_index,
        1,
    ].axis(
        "off"
    )

    # --------------------------------------------------
    # Selected preview
    # --------------------------------------------------

    axes[
        row_index,
        2,
    ].imshow(
        selected_preview
    )

    axes[
        row_index,
        2,
    ].axis(
        "off"
    )

    # --------------------------------------------------
    # Text comparison
    # --------------------------------------------------

    axes[
        row_index,
        3,
    ].axis(
        "off"
    )

    true_text = shorten_list(
        row[
            "true_categories"
        ]
    )

    baseline_text = shorten_list(
        row[
            "baseline_prediction"
        ]
    )

    fixed_text = shorten_list(
        row[
            "fixed_prediction"
        ]
    )

    selected_text = shorten_list(
        row[
            "selected_prediction"
        ]
    )

    comparison_text = (
        f"Ground truth:\n"
        f"{true_text}\n\n"
        f"Single image:\n"
        f"{baseline_text}\n"
        f"METEOR: "
        f"{row['baseline_meteor']:.3f}\n\n"
        f"Fixed crops:\n"
        f"{fixed_text}\n"
        f"METEOR: "
        f"{row['fixed_meteor']:.3f}\n\n"
        f"Selected crops:\n"
        f"{selected_text}\n"
        f"METEOR: "
        f"{row['selected_meteor']:.3f}"
    )

    axes[
        row_index,
        3,
    ].text(
        0.0,
        0.95,
        comparison_text,
        ha="left",
        va="top",
        fontsize=10,
        transform=(
            axes[
                row_index,
                3,
            ].transAxes
        ),
    )


fig.suptitle(
    (
        "Fine-Damage Qualitative Examples: "
        "Single Image vs Local-Crop Inputs"
    ),
    fontsize=16,
    fontweight="bold",
    y=0.99,
)

plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.97,
    ]
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "Selected examples:"
)

for _, row in examples.iterrows():
    print(
        row[
            "image_id"
        ],
        "fixed METEOR gain =",
        round(
            row[
                "fixed_gain"
            ],
            4,
        ),
    )

print()
print(
    "Saved fine-damage figure to:"
)

print(
    OUTPUT_PATH.resolve()
)