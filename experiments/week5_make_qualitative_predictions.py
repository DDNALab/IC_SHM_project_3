from pathlib import Path
import json
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT = (
    FIGURES
    / "qualitative_predictions.png"
)


EXAMPLES = [
    {
        "image_id": "20.jpg",
        "title": "Correct Prediction",
    },
    {
        "image_id": "xiu_213.jpg",
        "title": "Partial Success",
    },
    {
        "image_id": "24.jpg",
        "title": "Difficult Failure",
    },
]


predictions = pd.read_csv(
    TABLES
    / "week3_v1_direct_free_text_scored_predictions.csv"
).fillna("")


rows = []

for example in EXAMPLES:

    match = predictions[
        predictions[
            "image_id"
        ].astype(str)
        == example["image_id"]
    ]

    if len(match) != 1:
        raise RuntimeError(
            f"Expected exactly one row for "
            f"{example['image_id']}, "
            f"found {len(match)}."
        )

    row = match.iloc[0].copy()

    row["panel_title"] = (
        example["title"]
    )

    rows.append(
        row
    )


fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(12, 13),
)


for index, row in enumerate(rows):

    image_path = Path(
        row["image_path"]
    )

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        axes[
            index,
            0
        ].imshow(
            image
        )

    axes[
        index,
        0
    ].axis(
        "off"
    )

    axes[
        index,
        0
    ].set_title(
        f"{row['panel_title']}\n"
        f"{row['image_id']}",
        fontsize=12,
    )


    true_categories = (
        row[
            "true_categories"
        ]
    )

    predicted_categories = (
        row[
            "predicted_categories"
        ]
    )

    reference_description = str(
        row[
            "reference_description"
        ]
    )

    generated_description = str(
        row[
            "generated_description"
        ]
    )

    meteor = float(
        row[
            "meteor"
        ]
    )


    text = (
        f"True categories:\n"
        f"{true_categories}\n\n"
        f"Predicted categories:\n"
        f"{predicted_categories}\n\n"
        f"Reference description:\n"
        f"{textwrap.fill(reference_description, 65)}\n\n"
        f"Generated description:\n"
        f"{textwrap.fill(generated_description, 65)}\n\n"
        f"METEOR: {meteor:.3f}"
    )


    axes[
        index,
        1
    ].axis(
        "off"
    )

    axes[
        index,
        1
    ].text(
        0.0,
        1.0,
        text,
        ha="left",
        va="top",
        fontsize=10,
        wrap=True,
    )


fig.suptitle(
    "Week 5 Qualitative Prediction Examples",
    fontsize=15,
    y=0.995,
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.98,
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
    "WEEK 5 QUALITATIVE FIGURE COMPLETE"
)

print(
    "========================================"
)

print()

for row in rows:

    print(
        f"{row['panel_title']}: "
        f"{row['image_id']} "
        f"(METEOR={float(row['meteor']):.3f})"
    )

print()

print(
    "Saved to:"
)

print(
    OUTPUT.resolve()
)