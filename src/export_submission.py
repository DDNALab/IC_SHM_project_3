from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from paths import TABLES_DIR, ensure_output_dirs
from utils import parse_list_field


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prediction CSV to the competition JSON schema.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=TABLES_DIR / "submission.json")
    args = parser.parse_args()

    ensure_output_dirs()
    frame = pd.read_csv(args.predictions).fillna("")
    required = {"image_id", "predicted_categories", "generated_description"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    submission = [
        {
            "image_id": str(row.image_id),
            "damage_categories": parse_list_field(row.predicted_categories),
            "description": str(row.generated_description).strip(),
        }
        for row in frame.itertuples(index=False)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(submission, file, indent=2, ensure_ascii=False)
    print(f"Submission JSON written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
