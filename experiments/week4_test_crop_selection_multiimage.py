from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageDraw


IMAGE_DIR = Path("dataset/image")
OUTPUT_DIR = Path("outputs/week4_crop_selection_test")
TABLES_DIR = Path("outputs/tables")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Test images
# --------------------------------------------------

TEST_IMAGES = [
    "crack_0690.jpg",
    "20.jpg",
    "xiu_046.jpg",
    "00020.jpg",
]

TOP_K = 2


# --------------------------------------------------
# Scoring functions
# --------------------------------------------------

def compute_entropy(gray_array):
    hist, _ = np.histogram(
        gray_array.flatten(),
        bins=256,
        range=(0, 256),
    )

    probabilities = hist / hist.sum()
    probabilities = probabilities[
        probabilities > 0
    ]

    return float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )


def compute_edge_density(gray_image):
    edges = gray_image.filter(
        ImageFilter.FIND_EDGES
    )

    array = np.array(
        edges,
        dtype=np.float32,
    )

    return float(
        (array > 40).sum()
        / array.size
    )


def compute_laplacian_response(gray_array):
    center = gray_array[1:-1, 1:-1]
    up = gray_array[:-2, 1:-1]
    down = gray_array[2:, 1:-1]
    left = gray_array[1:-1, :-2]
    right = gray_array[1:-1, 2:]

    laplacian = (
        4 * center
        - up
        - down
        - left
        - right
    )

    return float(
        np.mean(
            np.abs(laplacian)
        )
    )


def create_crops(image):
    width, height = image.size

    crop_width = int(width * 0.65)
    crop_height = int(height * 0.65)

    return {
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


all_rows = []


# --------------------------------------------------
# Process test images
# --------------------------------------------------

for image_name in TEST_IMAGES:

    image_path = (
        IMAGE_DIR
        / image_name
    )

    if not image_path.exists():
        print(
            "WARNING: missing image:",
            image_path,
        )
        continue

    image = Image.open(
        image_path
    ).convert("RGB")

    boxes = create_crops(
        image
    )

    image_rows = []

    for crop_name, box in boxes.items():

        crop = image.crop(
            box
        )

        gray = crop.convert("L")

        gray_array = np.array(
            gray,
            dtype=np.float32,
        )

        entropy = compute_entropy(
            gray_array
        )

        edge_density = compute_edge_density(
            gray
        )

        laplacian_response = (
            compute_laplacian_response(
                gray_array
            )
        )

        row = {
            "image_id": image_name,
            "crop_name": crop_name,
            "entropy": entropy,
            "edge_density": edge_density,
            "laplacian_response": laplacian_response,
            "box": box,
        }

        image_rows.append(
            row
        )

    df = pd.DataFrame(
        image_rows
    )

    # Normalize within each image
    for column in [
        "entropy",
        "edge_density",
        "laplacian_response",
    ]:

        minimum = df[column].min()
        maximum = df[column].max()

        if maximum > minimum:
            df[f"{column}_norm"] = (
                df[column] - minimum
            ) / (
                maximum - minimum
            )
        else:
            df[f"{column}_norm"] = 0.0

    df["combined_score"] = (
        df["entropy_norm"]
        + df["edge_density_norm"]
        + df["laplacian_response_norm"]
    ) / 3.0

    df = df.sort_values(
        "combined_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    df["rank"] = (
        np.arange(
            1,
            len(df) + 1
        )
    )

    all_rows.append(
        df.copy()
    )

    # --------------------------------------------------
    # Create visual selection image
    # --------------------------------------------------

    preview = image.copy()

    draw = ImageDraw.Draw(
        preview
    )

    selected = df.head(
        TOP_K
    )

    for _, row in selected.iterrows():

        box = row["box"]

        draw.rectangle(
            box,
            outline="white",
            width=max(
                2,
                image.width // 150,
            ),
        )

        x1, y1, _, _ = box

        draw.text(
            (
                x1 + 5,
                y1 + 5,
            ),
            f"Rank {int(row['rank'])}",
            fill="white",
        )

    output_preview = (
        OUTPUT_DIR
        / f"{Path(image_name).stem}_selection.jpg"
    )

    preview.save(
        output_preview,
        quality=95,
    )

    print()
    print(
        "IMAGE:",
        image_name,
    )

    print(
        df[
            [
                "rank",
                "crop_name",
                "combined_score",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "Preview:",
        output_preview,
    )


# --------------------------------------------------
# Save combined CSV
# --------------------------------------------------

if all_rows:

    combined = pd.concat(
        all_rows,
        ignore_index=True,
    )

    output_csv = (
        TABLES_DIR
        / "week4_multiimage_crop_scores.csv"
    )

    combined.to_csv(
        output_csv,
        index=False,
    )

    print()
    print(
        "Combined scores saved to:"
    )

    print(
        output_csv.resolve()
    )