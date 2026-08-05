from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paths import FIGURES_DIR, TABLES_DIR, ensure_output_dirs


def plot_category_counts(counts_csv: Path) -> None:
    frame = pd.read_csv(counts_csv)
    subset = frame[frame["split"] == "all"].sort_values("count", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(subset["category"], subset["count"])
    axis.set_xlabel("Number of images")
    axis.set_ylabel("Damage category")
    axis.set_title("Project 3 category distribution")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "category_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_split_counts(counts_csv: Path) -> None:
    frame = pd.read_csv(counts_csv)
    frame = frame[frame["split"].isin(["train", "val", "test"])]
    pivot = frame.pivot(index="category", columns="split", values="count").fillna(0)
    pivot = pivot.reindex(columns=[column for column in ["train", "val", "test"] if column in pivot.columns])
    axis = pivot.plot(kind="bar", figsize=(10, 5))
    axis.set_xlabel("Damage category")
    axis.set_ylabel("Number of images")
    axis.set_title("Category distribution by split")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "split_category_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_log(log_csv: Path) -> None:
    if not log_csv.is_file():
        return
    frame = pd.read_csv(log_csv)
    figure, axis = plt.subplots(figsize=(8, 5))
    if "loss" in frame:
        train = frame.dropna(subset=["loss"])
        axis.plot(train.get("step", train.index), train["loss"], label="train loss")
    if "eval_loss" in frame:
        evaluation = frame.dropna(subset=["eval_loss"])
        axis.plot(evaluation.get("step", evaluation.index), evaluation["eval_loss"], label="validation loss")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Loss")
    axis.set_title("Training history")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "training_loss.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_per_class_f1(metrics_csv: Path) -> None:
    if not metrics_csv.is_file():
        return
    frame = pd.read_csv(metrics_csv).sort_values("f1", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(frame["category"], frame["f1"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("F1 score")
    axis.set_ylabel("Damage category")
    axis.set_title("Per-category validation/test F1")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "per_class_f1.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tables-derived figures.")
    parser.add_argument("--run-name", type=str, default="qwen35_4b_shm")
    parser.add_argument("--metrics-prefix", type=str, default=None)
    args = parser.parse_args()

    ensure_output_dirs()
    plot_category_counts(TABLES_DIR / "category_counts.csv")
    plot_split_counts(TABLES_DIR / "category_counts.csv")
    plot_training_log(TABLES_DIR / f"{args.run_name}_training_log.csv")
    if args.metrics_prefix:
        plot_per_class_f1(TABLES_DIR / f"{args.metrics_prefix}_per_class_metrics.csv")
    print(f"Figures written to: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
