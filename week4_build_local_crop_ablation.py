from pathlib import Path

import pandas as pd


TABLES_DIR = Path("outputs/tables")

EXPERIMENTS = [
    {
        "experiment": "week3_v1_direct_free_text",
        "method": "single_image",
        "visual_inputs": "full image only",
        "metrics_file": (
            TABLES_DIR
            / "week3_v1_direct_free_text_metrics.csv"
        ),
    },
    {
        "experiment": "week4_fixed_crops",
        "method": "global_plus_fixed",
        "visual_inputs": (
            "full image + 2 fixed overlapping crops"
        ),
        "metrics_file": (
            TABLES_DIR
            / "week4_fixed_crops_metrics.csv"
        ),
    },
    {
        "experiment": "week4_selected_crops",
        "method": "global_plus_selected",
        "visual_inputs": (
            "full image + top-2 automatically selected crops"
        ),
        "metrics_file": (
            TABLES_DIR
            / "week4_selected_crops_metrics.csv"
        ),
    },
]

OUTPUT_FILE = (
    TABLES_DIR
    / "local_crop_ablation.csv"
)


def main():

    rows = []

    for experiment in EXPERIMENTS:

        metrics_path = (
            experiment["metrics_file"]
        )

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing metrics file: "
                f"{metrics_path}"
            )

        metrics = pd.read_csv(
            metrics_path
        )

        if len(metrics) != 1:
            raise RuntimeError(
                f"Expected one metrics row in "
                f"{metrics_path}, "
                f"found {len(metrics)}"
            )

        row = metrics.iloc[0]

        rows.append(
            {
                "experiment":
                    experiment["experiment"],

                "method":
                    experiment["method"],

                "visual_inputs":
                    experiment[
                        "visual_inputs"
                    ],

                "n_images":
                    int(
                        row["n_images"]
                    ),

                "exact_label_set_accuracy":
                    float(
                        row[
                            "exact_label_set_accuracy"
                        ]
                    ),

                "micro_precision":
                    float(
                        row[
                            "micro_precision"
                        ]
                    ),

                "micro_recall":
                    float(
                        row[
                            "micro_recall"
                        ]
                    ),

                "micro_f1":
                    float(
                        row[
                            "micro_f1"
                        ]
                    ),

                "macro_precision":
                    float(
                        row[
                            "macro_precision"
                        ]
                    ),

                "macro_recall":
                    float(
                        row[
                            "macro_recall"
                        ]
                    ),

                "macro_f1":
                    float(
                        row[
                            "macro_f1"
                        ]
                    ),

                "meteor":
                    float(
                        row["meteor"]
                    ),

                "json_valid_rate":
                    float(
                        row[
                            "json_valid_rate"
                        ]
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    # --------------------------------------------------
    # Add deltas against the Week 3 single-image baseline
    # --------------------------------------------------

    baseline = result.iloc[0]

    for metric in [
        "exact_label_set_accuracy",
        "micro_f1",
        "macro_f1",
        "meteor",
    ]:

        result[
            f"delta_{metric}"
        ] = (
            result[metric]
            - baseline[metric]
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "=== LOCAL CROP ABLATION ==="
    )
    print()

    display_columns = [
        "experiment",
        "method",
        "exact_label_set_accuracy",
        "micro_f1",
        "macro_f1",
        "meteor",
        "delta_exact_label_set_accuracy",
        "delta_micro_f1",
        "delta_macro_f1",
        "delta_meteor",
    ]

    print(
        result[
            display_columns
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE.resolve()
    )


if __name__ == "__main__":
    main()