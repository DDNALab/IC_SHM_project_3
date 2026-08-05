from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from data_processing import CATEGORIES
from paths import TABLES_DIR, ensure_output_dirs
from utils import parse_list_field, save_json


def corpus_meteor(references: list[str], hypotheses: list[str]) -> tuple[float, list[float]]:
    scores: list[float] = []
    for reference, hypothesis in zip(references, hypotheses):
        try:
            score = meteor_score([reference.split()], hypothesis.split())
        except LookupError as error:
            raise RuntimeError(
                "NLTK WordNet resources are missing. Run: "
                "python -m nltk.downloader wordnet omw-1.4"
            ) from error
        scores.append(float(score))
    return float(np.mean(scores)) if scores else 0.0, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Project 3 predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, default=None)
    args = parser.parse_args()

    ensure_output_dirs()
    frame = pd.read_csv(args.predictions).fillna("")
    prefix = args.output_prefix or args.predictions.stem
    true_sets = [parse_list_field(value) for value in frame["true_categories"]]
    predicted_sets = [parse_list_field(value) for value in frame["predicted_categories"]]

    binarizer = MultiLabelBinarizer(classes=CATEGORIES)
    y_true = binarizer.fit_transform(true_sets)
    y_pred = binarizer.transform(predicted_sets)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    exact_match = accuracy_score(y_true, y_pred)
    meteor, meteor_rows = corpus_meteor(
        frame["reference_description"].astype(str).tolist(),
        frame["generated_description"].astype(str).tolist(),
    )
    json_valid_rate = float(frame["json_valid"].astype(str).str.lower().isin(["true", "1"]).mean())

    summary = {
        "n_images": len(frame),
        "exact_label_set_accuracy": float(exact_match),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "meteor": float(meteor),
        "json_valid_rate": json_valid_rate,
    }
    save_json(summary, TABLES_DIR / f"{prefix}_metrics.json")
    pd.DataFrame([summary]).to_csv(TABLES_DIR / f"{prefix}_metrics.csv", index=False)

    per_class = pd.DataFrame(
        {
            "category": CATEGORIES,
            "precision": per_p,
            "recall": per_r,
            "f1": per_f1,
            "support": support,
        }
    )
    per_class.to_csv(TABLES_DIR / f"{prefix}_per_class_metrics.csv", index=False)
    scored = frame.copy()
    scored["meteor"] = meteor_rows
    scored.to_csv(TABLES_DIR / f"{prefix}_scored_predictions.csv", index=False)

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Metrics written under: {TABLES_DIR.resolve()}")


if __name__ == "__main__":
    main()
