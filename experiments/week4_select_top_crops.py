from pathlib import Path
import shutil

import pandas as pd


SCORES_FILE = Path(
    "outputs/tables/week4_fixed_crop_scores.csv"
)

CROP_DIR = Path(
    "outputs/week4_fixed_crops"
)

OUTPUT_DIR = Path(
    "outputs/week4_selected_crops"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Load crop rankings
# --------------------------------------------------

scores = pd.read_csv(
    SCORES_FILE
)

scores = scores.sort_values(
    "rank"
)


# --------------------------------------------------
# Select top 2 crops
# --------------------------------------------------

selected = scores.head(2)


print("=== SELECTED CROPS ===")
print()


for _, row in selected.iterrows():

    crop_name = row["crop_name"]

    source = (
        CROP_DIR
        / crop_name
    )

    destination = (
        OUTPUT_DIR
        / crop_name
    )

    shutil.copy2(
        source,
        destination,
    )

    print(
        f"Rank {int(row['rank'])}: "
        f"{crop_name} "
        f"(score={row['combined_score']:.4f})"
    )


print()
print("Selected crop folder:")
print(
    OUTPUT_DIR.resolve()
)