from pathlib import Path

from PIL import Image


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_PATH = Path("dataset/image/crack_0690.jpg")

OUTPUT_DIR = Path("outputs/week4_fixed_crops")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

width, height = image.size

print("Image:", IMAGE_PATH)
print("Original size:", width, "x", height)


# --------------------------------------------------
# CREATE 4 OVERLAPPING CROPS
# --------------------------------------------------

crop_width = int(width * 0.65)
crop_height = int(height * 0.65)

positions = {
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


# --------------------------------------------------
# SAVE CROPS
# --------------------------------------------------

for name, box in positions.items():

    crop = image.crop(box)

    output_path = (
        OUTPUT_DIR
        / f"{IMAGE_PATH.stem}_{name}.jpg"
    )

    crop.save(
        output_path,
        quality=95,
    )

    print(
        name,
        "->",
        crop.size,
        "saved to",
        output_path,
    )


print()
print("Fixed crop generation complete.")
print("Output folder:")
print(OUTPUT_DIR.resolve())