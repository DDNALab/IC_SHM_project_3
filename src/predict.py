from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from dataset import make_messages, open_rgb_image
from model import (
    build_processor,
    load_adapter_for_inference,
    load_base_model_for_inference,
)
from paths import CONFIG_DIR, TABLES_DIR, ensure_output_dirs
from utils import (
    extract_json_object,
    load_yaml,
    normalize_category,
    parse_list_field,
)


VALID_CATEGORIES = {
    "crack",
    "void",
    "honeycomb",
    "looseness",
    "spalling",
    "exposed_rebar",
    "corrosion",
    "efflorescence",
    "pothole",
}


def parse_prediction(
    raw_output: str,
) -> tuple[list[str], str, bool]:

    parsed = extract_json_object(raw_output)

    if parsed is None:
        return [], raw_output.strip(), False

    categories = parsed.get(
        "damage_categories",
        [],
    )

    if isinstance(categories, str):
        categories = [categories]

    normalized = []

    for item in (
        categories
        if isinstance(categories, list)
        else []
    ):

        category = normalize_category(
            str(item)
        )

        if (
            category in VALID_CATEGORIES
            and category not in normalized
        ):
            normalized.append(category)

    description = str(
        parsed.get(
            "description",
            "",
        )
    ).strip()

    return (
        normalized,
        description,
        True,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Qwen3.5-4B zero-shot "
            "or LoRA-adapter predictions."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR
        / "qwen35_4b_24gb.yaml",
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    ensure_output_dirs()

    config = load_yaml(
        args.config
    )

    model_cfg = config[
        "model"
    ]

    data_cfg = config[
        "data"
    ]

    run_name = str(
        config.get(
            "run_name",
            "qwen35_4b_shm",
        )
    )

    input_csv = (
        args.input_csv
        or (
            TABLES_DIR
            / f"{args.split}.csv"
        )
    )

    output_csv = (
        args.output_csv
        or (
            TABLES_DIR
            / f"{run_name}_{args.split}_predictions.csv"
        )
    )

    # --------------------------------------------------
    # Build processor
    # --------------------------------------------------

    print("\nLoading processor...")

    processor = build_processor(
        model_cfg["model_id"],
        min_pixels=int(
            model_cfg[
                "min_pixels"
            ]
        ),
        max_pixels=int(
            model_cfg[
                "max_pixels"
            ]
        ),
    )

    processor.tokenizer.padding_side = (
        "left"
    )

    # --------------------------------------------------
    # Choose inference mode
    # --------------------------------------------------

    if args.adapter is None:

        print(
            "\nInference mode: "
            "ZERO-SHOT base model"
        )

        model = (
            load_base_model_for_inference(
                model_id=model_cfg[
                    "model_id"
                ],
                precision=model_cfg.get(
                    "precision",
                    "bf16",
                ),
                load_in_4bit=True,
            )
        )

    else:

        print(
            "\nInference mode: "
            "LoRA adapter"
        )

        print(
            "Adapter:",
            args.adapter.resolve(),
        )

        if not args.adapter.exists():
            raise FileNotFoundError(
                "LoRA adapter path does "
                "not exist: "
                f"{args.adapter}"
            )

        model = (
            load_adapter_for_inference(
                model_id=model_cfg[
                    "model_id"
                ],
                adapter_path=
                    args.adapter,
                precision=model_cfg.get(
                    "precision",
                    "bf16",
                ),
                load_in_4bit=True,
            )
        )

    # --------------------------------------------------
    # Load input split
    # --------------------------------------------------

    print(
        "\nReading input CSV:"
    )

    print(
        input_csv.resolve()
    )

    frame = pd.read_csv(
        input_csv
    )

    print(
        "Images to predict:",
        len(frame),
    )

    # --------------------------------------------------
    # Generate predictions
    # --------------------------------------------------

    rows = []

    total = len(frame)

    for index, (_, row) in enumerate(
        frame.iterrows(),
        start=1,
    ):

        print(
            f"[{index}/{total}] "
            f"{row['image_id']}"
        )

        image = open_rgb_image(
            row["image_path"]
        )

        messages = make_messages(
            image,
            target=None,
            prompt_variant=str(
                data_cfg.get(
                    "prompt_variant",
                    "baseline",
                )
            ),
        )

        inputs = (
            processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        )

        model_device = next(
            model.parameters()
        ).device

        inputs = {
            key: (
                value.to(
                    model_device
                )
                if torch.is_tensor(
                    value
                )
                else value
            )
            for key, value
            in inputs.items()
        }

        # ----------------------------------------------
        # Inference
        # ----------------------------------------------

        with torch.inference_mode():

            generated = (
                model.generate(
                    **inputs,
                    max_new_tokens=
                        args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=
                        processor.tokenizer
                        .pad_token_id,
                    eos_token_id=
                        processor.tokenizer
                        .eos_token_id,
                )
            )

        # ----------------------------------------------
        # Decode only newly generated tokens
        # ----------------------------------------------

        prompt_length = (
            inputs[
                "input_ids"
            ].shape[1]
        )

        new_tokens = (
            generated[
                0,
                prompt_length:
            ]
        )

        raw_output = (
            processor.decode(
                new_tokens,
                skip_special_tokens=True,
            )
            .strip()
        )

        # ----------------------------------------------
        # Parse structured prediction
        # ----------------------------------------------

        (
            predicted_categories,
            generated_description,
            json_valid,
        ) = parse_prediction(
            raw_output
        )

        rows.append(
            {
                "image_id":
                    row[
                        "image_id"
                    ],

                "image_path":
                    row[
                        "image_path"
                    ],

                "true_categories":
                    json.dumps(
                        parse_list_field(
                            row[
                                "damage_categories"
                            ]
                        )
                    ),

                "predicted_categories":
                    json.dumps(
                        predicted_categories
                    ),

                "reference_description":
                    row[
                        "reference_description"
                    ],

                "generated_description":
                    generated_description,

                "json_valid":
                    json_valid,

                "raw_output":
                    raw_output,
            }
        )

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        output_csv,
        index=False,
    )

    print(
        "\nPredictions written to:"
    )

    print(
        output_csv.resolve()
    )

    print(
        "\nPrediction run complete."
    )


if __name__ == "__main__":
    main()