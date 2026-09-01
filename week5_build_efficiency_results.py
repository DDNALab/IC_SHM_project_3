from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
TABLES = PROJECT_ROOT / "outputs" / "tables"


# ------------------------------------------------------------
# ResNet efficiency
# ------------------------------------------------------------

resnet_metrics = pd.read_csv(
    TABLES / "week5_resnet18_metrics.csv"
).iloc[0]


# ------------------------------------------------------------
# Qwen V1-direct efficiency
# ------------------------------------------------------------

with open(
    TABLES
    / "week5_v1_direct_efficiency_predictions_efficiency.json",
    "r",
    encoding="utf-8",
) as file:
    qwen_direct_eff = json.load(file)

with open(
    TABLES
    / "week3_v1_direct_free_text_model_parameters.json",
    "r",
    encoding="utf-8",
) as file:
    qwen_direct_params = json.load(file)


# ------------------------------------------------------------
# Qwen fixed-crop efficiency
# ------------------------------------------------------------

with open(
    TABLES
    / "week5_fixed_crops_efficiency_predictions_efficiency.json",
    "r",
    encoding="utf-8",
) as file:
    qwen_fixed_eff = json.load(file)

with open(
    TABLES
    / "week4_fixed_crops_model_parameters.json",
    "r",
    encoding="utf-8",
) as file:
    qwen_fixed_params = json.load(file)


# ------------------------------------------------------------
# Build table
# ------------------------------------------------------------

rows = [
    {
        "model": "ResNet-18",
        "visual_input": "single image",
        "n_images": int(
            resnet_metrics["n"]
        ),
        "total_inference_seconds": float(
            resnet_metrics[
                "total_inference_seconds"
            ]
        ),
        "seconds_per_image": float(
            resnet_metrics[
                "mean_inference_seconds_per_image"
            ]
        ),
        "images_per_second": float(
            resnet_metrics[
                "images_per_second"
            ]
        ),
        "peak_allocated_gb": float(
            resnet_metrics[
                "peak_allocated_gb"
            ]
        ),
        "peak_reserved_gb": float(
            resnet_metrics[
                "peak_reserved_gb"
            ]
        ),
        "trainable_parameters": int(
            resnet_metrics[
                "trainable_parameters"
            ]
        ),
        "total_parameters": int(
            resnet_metrics[
                "total_parameters"
            ]
        ),
    },

    {
        "model":
            "Qwen3.5-4B V1-direct QLoRA",
        "visual_input":
            "single image",
        "n_images": int(
            qwen_direct_eff[
                "n_images"
            ]
        ),
        "total_inference_seconds": float(
            qwen_direct_eff[
                "total_generation_seconds"
            ]
        ),
        "seconds_per_image": float(
            qwen_direct_eff[
                "mean_generation_seconds_per_image"
            ]
        ),
        "images_per_second": float(
            qwen_direct_eff[
                "images_per_second"
            ]
        ),
        "peak_allocated_gb": float(
            qwen_direct_eff[
                "peak_allocated_gb"
            ]
        ),
        "peak_reserved_gb": float(
            qwen_direct_eff[
                "peak_reserved_gb"
            ]
        ),
        "trainable_parameters": int(
            qwen_direct_params[
                "trainable_parameters"
            ]
        ),
        "total_parameters": int(
            qwen_direct_params[
                "total_parameters"
            ]
        ),
    },

    {
        "model":
            "Qwen3.5-4B Fixed-Crop QLoRA",
        "visual_input":
            "full image + 2 fixed crops",
        "n_images": int(
            qwen_fixed_eff[
                "n_images"
            ]
        ),
        "total_inference_seconds": float(
            qwen_fixed_eff[
                "total_generation_seconds"
            ]
        ),
        "seconds_per_image": float(
            qwen_fixed_eff[
                "mean_generation_seconds_per_image"
            ]
        ),
        "images_per_second": float(
            qwen_fixed_eff[
                "images_per_second"
            ]
        ),
        "peak_allocated_gb": float(
            qwen_fixed_eff[
                "peak_allocated_gb"
            ]
        ),
        "peak_reserved_gb": float(
            qwen_fixed_eff[
                "peak_reserved_gb"
            ]
        ),
        "trainable_parameters": int(
            qwen_fixed_params[
                "trainable_parameters"
            ]
        ),
        "total_parameters": int(
            qwen_fixed_params[
                "total_parameters"
            ]
        ),
    },
]


result = pd.DataFrame(
    rows
)

output_path = (
    TABLES
    / "efficiency_results.csv"
)

result.to_csv(
    output_path,
    index=False,
)


print()
print(
    "========================================"
)

print(
    "WEEK 5 EFFICIENCY TABLE COMPLETE"
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
    output_path.resolve()
)