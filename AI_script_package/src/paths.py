from __future__ import annotations

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGE_DIR = DATASET_DIR / "image"
DESCRIPTION_PATH = DATASET_DIR / "description.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
CONFIG_DIR = PROJECT_ROOT / "configs"


def ensure_output_dirs() -> None:
    for directory in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, CHECKPOINTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
