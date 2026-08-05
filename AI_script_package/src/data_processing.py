from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from paths import DESCRIPTION_PATH, IMAGE_DIR, TABLES_DIR, ensure_output_dirs
from utils import save_json

CATEGORIES = [
    "crack",
    "void",
    "honeycomb",
    "looseness",
    "spalling",
    "exposed_rebar",
    "corrosion",
    "efflorescence",
    "pothole",
]

CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "crack": (r"\bcracks?\b", r"\bfissures?\b", r"\bcracked\b"),
    "void": (r"\bvoids?\b",),
    "honeycomb": (r"\bhoneycomb(?:ing)?\b",),
    "looseness": (r"\blooseness\b", r"\bloosen(?:ing|ed)?\b", r"\bloose\b"),
    "spalling": (r"\bspall(?:ing|ed)?\b",),
    "exposed_rebar": (
        r"\bexposed (?:rebar|reinforcement|reinforcing bars?|steel bars?)\b",
        r"\b(?:rebar|reinforcement|reinforcing bars?|steel bars?) (?:is|are|has been|have been)?\s*exposed\b",
    ),
    "corrosion": (r"\bcorrosion\b", r"\bcorrod(?:e|ed|ing)\b", r"\brust(?:ed|ing|y)?\b"),
    "efflorescence": (r"\befflorescence\b",),
    "pothole": (r"\bpotholes?\b",),
}

MATERIAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "steel": (r"\bsteel\b", r"\bweld\b"),
    "asphalt": (r"\basphalt\b", r"\broad\b", r"\bpavement\b", r"\bpothole\b"),
    "tile_or_masonry": (r"\btile\b", r"\bwall\b", r"\bmasonry\b"),
    "concrete": (r"\bconcrete\b", r"\brebar\b", r"\breinforcement\b"),
}


def parse_concatenated_json(path: str | Path) -> list[dict[str, Any]]:
    """Parse valid JSON arrays and the supplied array of adjacent JSON objects."""
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("description.json must contain a list of records.")
        return parsed
    except json.JSONDecodeError:
        pass

    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("Malformed description file is not enclosed in an array.")

    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    index = 1
    while index < len(text) - 1:
        while index < len(text) - 1 and text[index] in " \t\r\n,":
            index += 1
        if index >= len(text) - 1:
            break
        record, next_index = decoder.raw_decode(text, index)
        if not isinstance(record, dict):
            raise ValueError(f"Expected object near character {index}.")
        records.append(record)
        index = next_index
    return records


def choose_record_pair(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not records:
        raise ValueError("No records supplied for image.")
    ordered = sorted(records, key=lambda item: len(str(item.get("prompt", ""))))
    coarse = ordered[0]
    detailed_candidates = [
        item for item in ordered if re.search(r"describe|characteristic|feature", str(item.get("prompt", "")), re.I)
    ]
    detailed = max(detailed_candidates or ordered, key=lambda item: len(str(item.get("prompt", ""))))
    return coarse, detailed


def infer_categories(coarse_label: str, detailed_label: str) -> list[str]:
    text = f"{coarse_label} {detailed_label}".lower()
    categories = [
        category
        for category in CATEGORIES
        if any(re.search(pattern, text, flags=re.I) for pattern in CATEGORY_PATTERNS[category])
    ]

    # The supplied annotations occasionally describe a generic "hole" for the primary void class.
    if "void" not in categories and re.search(r"\bvoids?/holes?\b", coarse_label, flags=re.I):
        categories.append("void")

    # Preserve category ordering for deterministic targets and evaluation.
    return [category for category in CATEGORIES if category in set(categories)]


def infer_primary_category(coarse_label: str) -> str:
    """Infer the organizer-style primary class from the short answer only."""
    text = coarse_label.lower()
    priority = [
        "efflorescence",
        "pothole",
        "honeycomb",
        "looseness",
        "void",
        "spalling",
        "corrosion",
        "crack",
    ]
    for category in priority:
        if any(re.search(pattern, text, flags=re.I) for pattern in CATEGORY_PATTERNS[category]):
            return category
    if re.search(r"\bholes?\b", text, flags=re.I):
        return "void"
    return "none"


def infer_material(coarse_label: str, detailed_label: str) -> str:
    text = f"{coarse_label} {detailed_label}".lower()
    for material, patterns in MATERIAL_PATTERNS.items():
        if any(re.search(pattern, text, flags=re.I) for pattern in patterns):
            return material
    return "unknown"


def make_group_id(image_id: str, block_size: int) -> str:
    stem = Path(image_id).stem.lower()
    match = re.match(r"^(.*?)(\d+)$", stem)
    if not match:
        return stem
    prefix = match.group(1).rstrip("_-") or "numeric"
    number = int(match.group(2))
    return f"{prefix}_{number // block_size:05d}"


def build_manifest(
    description_path: str | Path,
    image_dir: str | Path,
    group_block_size: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = parse_concatenated_json(description_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        image_reference = str(record.get("img", "")).replace("\\", "/")
        image_id = Path(image_reference).name
        if not image_id:
            continue
        grouped[image_id].append(record)

    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    non_pair_counts: dict[str, int] = {}

    for image_id, image_records in sorted(grouped.items()):
        coarse, detailed = choose_record_pair(image_records)
        coarse_label = str(coarse.get("label", "")).strip()
        description = str(detailed.get("label", "")).strip()
        categories = infer_categories(coarse_label, description)
        image_path = Path(image_dir) / image_id
        exists = image_path.is_file()
        if not exists:
            missing_images.append(image_id)
        if len(image_records) != 2:
            non_pair_counts[image_id] = len(image_records)

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "image_exists": exists,
                "damage_categories": json.dumps(categories),
                "primary_category": infer_primary_category(coarse_label),
                "material": infer_material(coarse_label, description),
                "coarse_label": coarse_label,
                "reference_description": description,
                "group_id": make_group_id(image_id, group_block_size),
                "annotation_count": len(image_records),
            }
        )

    dataframe = pd.DataFrame(rows)
    report = {
        "raw_record_count": len(records),
        "unique_image_count": len(dataframe),
        "missing_image_count": len(missing_images),
        "missing_images": missing_images,
        "images_with_annotation_count_not_equal_to_two": non_pair_counts,
        "empty_category_count": int((dataframe["damage_categories"] == "[]").sum()) if not dataframe.empty else 0,
        "category_counts": dict(Counter(category for value in dataframe["damage_categories"] for category in json.loads(value))),
    }
    return dataframe, report


def assign_group_splits(
    dataframe: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Create approximately stratified splits without filename-group leakage.

    For the default 80/10/10 design, samples are first assigned to ten
    stratified group folds. The validation/test fold pair is then selected by
    minimizing class-distribution and sample-size error while requiring every
    observed class to remain represented in training.
    """
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.")
    if dataframe["group_id"].nunique() < 3:
        raise ValueError("At least three groups are required for train/validation/test splitting.")
    if abs(val_fraction - test_fraction) > 1e-8:
        raise ValueError("The stratified group splitter currently requires equal validation and test fractions.")

    n_splits = round(1.0 / val_fraction)
    if n_splits < 3 or abs(1.0 / n_splits - val_fraction) > 1e-6:
        raise ValueError("Validation/test fractions must be reciprocal integer folds, e.g. 0.10 or 0.20.")

    result = dataframe.copy().reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_ids = pd.Series(-1, index=result.index, dtype=int)
    for fold_id, (_, holdout_idx) in enumerate(
        splitter.split(result, y=result["primary_category"], groups=result["group_id"])
    ):
        fold_ids.iloc[holdout_idx] = fold_id
    if (fold_ids < 0).any():
        raise RuntimeError("Some rows were not assigned a stratified group fold.")

    categories = sorted(result["primary_category"].unique())
    total_class_counts = result["primary_category"].value_counts().to_dict()
    target_fractions = {"train": train_fraction, "val": val_fraction, "test": test_fraction}

    best: tuple[float, int, int] | None = None
    for val_fold in range(n_splits):
        for test_fold in range(n_splits):
            if test_fold == val_fold:
                continue
            split = pd.Series("train", index=result.index)
            split[fold_ids == val_fold] = "val"
            split[fold_ids == test_fold] = "test"

            train_classes = set(result.loc[split == "train", "primary_category"])
            if train_classes != set(categories):
                continue

            score = 0.0
            category_group_counts = result.groupby("primary_category")["group_id"].nunique().to_dict()
            for split_name, target_fraction in target_fractions.items():
                subset = result.loc[split == split_name]
                actual_fraction = len(subset) / len(result)
                score += 8.0 * (actual_fraction - target_fraction) ** 2
                counts = subset["primary_category"].value_counts().to_dict()
                for category in categories:
                    denominator = max(total_class_counts[category], 1)
                    actual_class_fraction = counts.get(category, 0) / denominator
                    score += (actual_class_fraction - target_fraction) ** 2
                    # Prefer validation/test coverage whenever multiple independent
                    # filename groups exist, without forcing leakage for very rare classes.
                    if split_name in {"val", "test"} and category_group_counts.get(category, 0) >= 3:
                        if counts.get(category, 0) == 0:
                            score += 0.75
            candidate = (score, val_fold, test_fold)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        raise RuntimeError("Could not find group-aware splits retaining every class in training.")

    _, val_fold, test_fold = best
    result["fold_id"] = fold_ids
    result["split"] = "train"
    result.loc[result["fold_id"] == val_fold, "split"] = "val"
    result.loc[result["fold_id"] == test_fold, "split"] = "test"

    leakage = result.groupby("group_id")["split"].nunique()
    if leakage.max() != 1:
        raise RuntimeError("Group leakage detected across splits.")
    return result


def write_summary_tables(dataframe: pd.DataFrame, output_dir: Path) -> None:
    category_rows: list[dict[str, Any]] = []
    for split_name in ["all", "train", "val", "test"]:
        subset = dataframe if split_name == "all" else dataframe[dataframe["split"] == split_name]
        counts = Counter(category for value in subset["damage_categories"] for category in json.loads(value))
        for category in CATEGORIES:
            category_rows.append({"split": split_name, "category": category, "count": counts.get(category, 0)})
    pd.DataFrame(category_rows).to_csv(output_dir / "category_counts.csv", index=False)

    split_summary = (
        dataframe.groupby("split")
        .agg(images=("image_id", "count"), groups=("group_id", "nunique"), missing_images=("image_exists", lambda x: int((~x).sum())))
        .reset_index()
    )
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IC-SHM Project 3 image-text data.")
    parser.add_argument("--description", type=Path, default=DESCRIPTION_PATH)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--group-block-size", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Keep manifest generation running when referenced images are missing. Missing rows are excluded from split CSV files.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataframe, report = build_manifest(args.description, args.image_dir, args.group_block_size)
    dataframe.to_csv(args.output_dir / "manifest_all_references.csv", index=False)
    save_json(report, args.output_dir / "data_quality_report.json")

    if report["missing_image_count"] and not args.allow_missing_images:
        raise FileNotFoundError(
            f"{report['missing_image_count']} referenced images are missing. "
            f"See {args.output_dir / 'data_quality_report.json'} or rerun with --allow-missing-images."
        )

    usable = dataframe[dataframe["image_exists"]].reset_index(drop=True)
    if usable.empty:
        raise RuntimeError("No referenced images were found.")
    usable = assign_group_splits(
        usable,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    usable.to_csv(args.output_dir / "manifest.csv", index=False)
    for split_name in ("train", "val", "test"):
        usable[usable["split"] == split_name].to_csv(args.output_dir / f"{split_name}.csv", index=False)
    write_summary_tables(usable, args.output_dir)

    split_primary_counts = (
        usable.groupby(["split", "primary_category"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    report["split_primary_category_counts"] = split_primary_counts
    report["categories_missing_by_split"] = {
        split_name: sorted(set(usable["primary_category"]) - set(usable.loc[usable["split"] == split_name, "primary_category"]))
        for split_name in ("train", "val", "test")
    }
    save_json(report, args.output_dir / "data_quality_report.json")

    print(f"Prepared {len(usable)} images.")
    print(usable["split"].value_counts().to_string())
    print(f"Tables written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
