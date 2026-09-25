from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


CROP_DIR = Path("outputs/week4_fixed_crops")
OUTPUT_CSV = Path("outputs/tables/week4_fixed_crop_scores.csv")

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)


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

    entropy = -np.sum(
        probabilities
        * np.log2(probabilities)
    )

    return float(entropy)


def compute_edge_density(gray_image):
    edges = gray_image.filter(
        ImageFilter.FIND_EDGES
    )

    edge_array = np.array(
        edges,
        dtype=np.float32,
    )

    threshold = 40.0

    edge_pixels = (
        edge_array > threshold
    ).sum()

    total_pixels = (
        edge_array.size
    )

    return float(
        edge_pixels / total_pixels
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

    response = np.mean(
        np.abs(laplacian)
    )

    return float(response)


rows = []


for image_path in sorted(
    CROP_DIR.glob("*.jpg")
):

    image = Image.open(
        image_path
    ).convert("RGB")

    gray = image.convert("L")

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

    laplacian_response = compute_laplacian_response(
        gray_array
    )

    rows.append(
        {
            "crop_name": image_path.name,
            "entropy": entropy,
            "edge_density": edge_density,
            "laplacian_response": laplacian_response,
        }
    )


df = pd.DataFrame(rows)


# --------------------------------------------------
# Normalize scores to 0-1
# --------------------------------------------------

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


# --------------------------------------------------
# Combined informativeness score
# --------------------------------------------------

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


df.to_csv(
    OUTPUT_CSV,
    index=False,
)


print("=== FIXED CROP SCORES ===")
print()

print(
    df[
        [
            "rank",
            "crop_name",
            "entropy",
            "edge_density",
            "laplacian_response",
            "combined_score",
        ]
    ]
    .round(6)
    .to_string(index=False)
)

print()
print("Best crop:")
print(
    df.iloc[0]["crop_name"]
)

print()
print("Saved to:")
print(
    OUTPUT_CSV.resolve()
)