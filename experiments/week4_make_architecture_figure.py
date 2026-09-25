from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUTPUT_PATH = Path(
    "outputs/figures/proposed_method_architecture.png"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def add_box(
    ax,
    x,
    y,
    width,
    height,
    text,
    fontsize=10,
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        fill=False,
    )

    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def add_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.5,
    )

    ax.add_patch(arrow)


fig, ax = plt.subplots(
    figsize=(16, 8)
)

ax.set_xlim(
    0,
    16,
)

ax.set_ylim(
    0,
    8,
)

ax.axis("off")


# --------------------------------------------------
# Title
# --------------------------------------------------

ax.text(
    8,
    7.55,
    "Global-Local Multi-Image Damage Diagnosis Framework",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
)


# --------------------------------------------------
# Input image
# --------------------------------------------------

add_box(
    ax,
    0.5,
    3.0,
    2.0,
    1.3,
    "Structural\nDamage Image",
    fontsize=11,
)


# --------------------------------------------------
# Global branch
# --------------------------------------------------

add_box(
    ax,
    3.3,
    5.2,
    2.2,
    1.1,
    "Global View\nFull Image",
)


# --------------------------------------------------
# Crop generation
# --------------------------------------------------

add_box(
    ax,
    3.3,
    2.0,
    2.2,
    1.3,
    "Overlapping Local\nCrop Generation\n(4 candidates)",
)


# --------------------------------------------------
# Fixed crop path
# --------------------------------------------------

add_box(
    ax,
    6.2,
    3.5,
    2.4,
    1.1,
    "Fixed Crop Path\nPredetermined Tiles",
)


# --------------------------------------------------
# Automatic crop scoring
# --------------------------------------------------

add_box(
    ax,
    6.2,
    1.1,
    2.4,
    1.6,
    "Automatic Crop Scoring\n\nEntropy\nEdge Density\nLaplacian Response",
    fontsize=9,
)


# --------------------------------------------------
# Selected crops
# --------------------------------------------------

add_box(
    ax,
    9.2,
    1.3,
    2.2,
    1.2,
    "Top-2 Informative\nLocal Crops",
)


# --------------------------------------------------
# Multi-image fusion input
# --------------------------------------------------

add_box(
    ax,
    9.2,
    4.2,
    2.2,
    1.5,
    "Multi-Image Input\n\nFull Image\n+\n2 Local Crops",
)


# --------------------------------------------------
# Model
# --------------------------------------------------

add_box(
    ax,
    12.1,
    4.1,
    2.3,
    1.7,
    "Qwen3.5-4B\n+\nQLoRA Adaptation",
    fontsize=11,
)


# --------------------------------------------------
# Output
# --------------------------------------------------

add_box(
    ax,
    12.1,
    1.5,
    2.3,
    1.5,
    "Structured Output\n\nDamage Categories\n+\nDescription",
    fontsize=10,
)


# --------------------------------------------------
# Arrows
# --------------------------------------------------

# Input -> global
add_arrow(
    ax,
    2.5,
    3.9,
    3.3,
    5.55,
)

# Input -> crop generation
add_arrow(
    ax,
    2.5,
    3.5,
    3.3,
    2.65,
)

# Crop generation -> fixed
add_arrow(
    ax,
    5.5,
    2.8,
    6.2,
    4.0,
)

# Crop generation -> automatic scoring
add_arrow(
    ax,
    5.5,
    2.4,
    6.2,
    1.9,
)

# Automatic scoring -> top 2
add_arrow(
    ax,
    8.6,
    1.9,
    9.2,
    1.9,
)

# Global -> multi-image input
add_arrow(
    ax,
    5.5,
    5.75,
    9.2,
    5.1,
)

# Fixed path -> multi-image input
add_arrow(
    ax,
    8.6,
    4.05,
    9.2,
    4.65,
)

# Selected crops -> multi-image input
add_arrow(
    ax,
    10.3,
    2.5,
    10.3,
    4.2,
)

# Multi-image input -> model
add_arrow(
    ax,
    11.4,
    4.95,
    12.1,
    4.95,
)

# Model -> output
add_arrow(
    ax,
    13.25,
    4.1,
    13.25,
    3.0,
)


# --------------------------------------------------
# Ablation annotation
# --------------------------------------------------

ax.text(
    7.3,
    6.8,
    (
        "Week 4 Ablation:\n"
        "Single Image vs Fixed Local Crops vs "
        "Automatically Selected Local Crops"
    ),
    ha="center",
    va="center",
    fontsize=10,
)


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "Saved architecture figure to:"
)

print(
    OUTPUT_PATH.resolve()
)