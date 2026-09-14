from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path

import torch

from paths import DESCRIPTION_PATH, IMAGE_DIR, TABLES_DIR, ensure_output_dirs
from utils import save_json


PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "pandas",
    "scikit-learn",
]


def recommendation(vram_gb: float) -> str:
    """
    Return a practical GPU-memory recommendation.

    Consumer/professional GPUs advertised as 16 GB may report slightly
    less than 16 GiB to PyTorch. Treat GPUs reporting at least 15.5 GiB
    as the 16 GB class.

    The final project configuration was successfully trained and tested
    on an NVIDIA RTX 2000 Ada Generation GPU reporting approximately
    16 GB of VRAM.
    """

    if vram_gb < 12:
        return (
            "Not recommended. Use a cloud GPU with at least "
            "16 GB, preferably 24 GB."
        )

    if vram_gb < 15.5:
        return (
            "Possible with aggressive pixel reduction and memory-saving "
            "techniques; a 16 GB or larger GPU is recommended."
        )

    if vram_gb < 24:
        return (
            "Suitable for the tested low-memory QLoRA configuration: "
            "batch size 1, gradient checkpointing, and reduced max_pixels."
        )

    if vram_gb < 48:
        return (
            "Recommended QLoRA configuration: batch size 1 with "
            "higher-resolution image settings where appropriate."
        )

    return (
        "Comfortable for higher-resolution QLoRA or "
        "non-quantized LoRA experiments."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check GPU and project paths."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TABLES_DIR / "environment_report.json",
    )
    args = parser.parse_args()

    ensure_output_dirs()

    package_versions = {}

    for package in PACKAGES:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions,
        "description_path": str(DESCRIPTION_PATH.resolve()),
        "description_exists": DESCRIPTION_PATH.is_file(),
        "image_dir": str(IMAGE_DIR.resolve()),
        "image_dir_exists": IMAGE_DIR.is_dir(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        vram_gb = properties.total_memory / 1024**3

        report.update(
            {
                "gpu_name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "vram_gb": round(vram_gb, 2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "recommendation": recommendation(vram_gb),
            }
        )

    else:
        report["recommendation"] = (
            "CUDA GPU not detected; training is not practical on CPU."
        )

    save_json(report, args.output)

    print(report)
    print(f"Report written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()