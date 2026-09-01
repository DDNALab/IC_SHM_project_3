from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset

from prompts import build_target, get_prompts
from utils import parse_list_field


# ============================================================
# DATASET
# ============================================================

class DamageDataset(Dataset):
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path)

        required = {
            "image_id",
            "image_path",
            "damage_categories",
            "reference_description",
        }

        missing = required - set(self.frame.columns)

        if missing:
            raise ValueError(
                f"{self.csv_path} is missing columns: {sorted(missing)}"
            )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]

        return {
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
            "damage_categories": parse_list_field(
                row["damage_categories"]
            ),
            "reference_description": str(
                row["reference_description"]
            ),
        }


# ============================================================
# IMAGE LOADING
# ============================================================

def open_rgb_image(
    path: str | Path,
) -> Image.Image:
    with Image.open(path) as image:
        image = (
            ImageOps.exif_transpose(image)
            .convert("RGB")
        )

        return image.copy()


# ============================================================
# WEEK 4 LOCAL CROP GENERATION
# ============================================================

def generate_overlapping_crops(
    image: Image.Image,
    crop_fraction: float = 0.65,
) -> dict[str, Image.Image]:
    """
    Generate four overlapping local crops from an image.

    The crop size is crop_fraction of the full image width
    and height.

    Returned positions:
    - top_left
    - top_right
    - bottom_left
    - bottom_right
    """

    width, height = image.size

    crop_width = max(
        1,
        int(width * crop_fraction),
    )

    crop_height = max(
        1,
        int(height * crop_fraction),
    )

    boxes = {
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

    return {
        name: image.crop(box)
        for name, box in boxes.items()
    }


# ============================================================
# WEEK 4 CROP SCORING
# ============================================================

def compute_entropy(
    gray_array: np.ndarray,
) -> float:

    histogram, _ = np.histogram(
        gray_array.flatten(),
        bins=256,
        range=(0, 256),
    )

    total = histogram.sum()

    if total == 0:
        return 0.0

    probabilities = (
        histogram / total
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    return float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )


def compute_edge_density(
    gray_image: Image.Image,
) -> float:

    edges = gray_image.filter(
        ImageFilter.FIND_EDGES
    )

    edge_array = np.array(
        edges,
        dtype=np.float32,
    )

    threshold = 40.0

    return float(
        (
            edge_array > threshold
        ).sum()
        / edge_array.size
    )


def compute_laplacian_response(
    gray_array: np.ndarray,
) -> float:

    if (
        gray_array.shape[0] < 3
        or gray_array.shape[1] < 3
    ):
        return 0.0

    center = (
        gray_array[
            1:-1,
            1:-1,
        ]
    )

    up = (
        gray_array[
            :-2,
            1:-1,
        ]
    )

    down = (
        gray_array[
            2:,
            1:-1,
        ]
    )

    left = (
        gray_array[
            1:-1,
            :-2,
        ]
    )

    right = (
        gray_array[
            1:-1,
            2:,
        ]
    )

    laplacian = (
        4 * center
        - up
        - down
        - left
        - right
    )

    return float(
        np.mean(
            np.abs(laplacian)
        )
    )


def select_informative_crops(
    crops: dict[str, Image.Image],
    top_k: int = 2,
) -> list[Image.Image]:
    """
    Rank crops using:
    - entropy
    - edge density
    - Laplacian response

    Each metric is min-max normalized within the image.
    Final score is the average of the three normalized metrics.
    """

    rows: list[dict[str, Any]] = []

    for name, crop in crops.items():

        gray = crop.convert("L")

        gray_array = np.array(
            gray,
            dtype=np.float32,
        )

        rows.append(
            {
                "name": name,
                "image": crop,
                "entropy": compute_entropy(
                    gray_array
                ),
                "edge_density": compute_edge_density(
                    gray
                ),
                "laplacian_response": (
                    compute_laplacian_response(
                        gray_array
                    )
                ),
            }
        )

    frame = pd.DataFrame(
        [
            {
                "name": row["name"],
                "entropy": row["entropy"],
                "edge_density": row["edge_density"],
                "laplacian_response": row[
                    "laplacian_response"
                ],
            }
            for row in rows
        ]
    )

    for column in [
        "entropy",
        "edge_density",
        "laplacian_response",
    ]:

        minimum = frame[column].min()
        maximum = frame[column].max()

        if maximum > minimum:
            frame[
                f"{column}_norm"
            ] = (
                frame[column] - minimum
            ) / (
                maximum - minimum
            )

        else:
            frame[
                f"{column}_norm"
            ] = 0.0

    frame["combined_score"] = (
        frame["entropy_norm"]
        + frame["edge_density_norm"]
        + frame["laplacian_response_norm"]
    ) / 3.0

    frame = frame.sort_values(
        "combined_score",
        ascending=False,
    )

    selected_names = (
        frame.head(top_k)[
            "name"
        ].tolist()
    )

    crop_lookup = {
        row["name"]: row["image"]
        for row in rows
    }

    return [
        crop_lookup[name]
        for name in selected_names
    ]


# ============================================================
# WEEK 4 VISUAL INPUT BUILDER
# ============================================================

def build_visual_inputs(
    image: Image.Image,
    visual_input_mode: str = "single_image",
    num_local_crops: int = 2,
    crop_fraction: float = 0.65,
) -> list[Image.Image]:
    """
    Build the image list supplied to the VLM.

    Modes:
    - single_image
        Full image only.

    - global_plus_fixed
        Full image plus predetermined overlapping crops.

    - global_plus_selected
        Full image plus automatically ranked informative crops.
    """

    mode = str(
        visual_input_mode
    ).strip().lower()

    if mode == "single_image":
        return [
            image
        ]

    crops = generate_overlapping_crops(
        image,
        crop_fraction=crop_fraction,
    )

    num_local_crops = max(
        1,
        min(
            int(num_local_crops),
            len(crops),
        ),
    )

    if mode == "global_plus_fixed":

        # Spread fixed selections across the image instead
        # of simply taking adjacent quadrants.
        fixed_order = [
            "top_left",
            "bottom_right",
            "top_right",
            "bottom_left",
        ]

        selected_names = (
            fixed_order[
                :num_local_crops
            ]
        )

        selected = [
            crops[name]
            for name in selected_names
        ]

        return [
            image,
            *selected,
        ]

    if mode == "global_plus_selected":

        selected = (
            select_informative_crops(
                crops,
                top_k=num_local_crops,
            )
        )

        return [
            image,
            *selected,
        ]

    raise ValueError(
        f"Unknown visual_input_mode: {visual_input_mode}. "
        "Expected 'single_image', "
        "'global_plus_fixed', or "
        "'global_plus_selected'."
    )


# ============================================================
# MULTIMODAL MESSAGE CONSTRUCTION
# ============================================================

def make_messages(
    image: Image.Image | list[Image.Image],
    target: str | None = None,
    prompt_variant: str = "baseline",
) -> list[dict[str, Any]]:
    """
    Build the multimodal conversation.

    image may be:
    - one PIL image;
    - a list containing the global image and local crops.

    Prompt variants are handled by get_prompts().
    """

    system_prompt, user_prompt = (
        get_prompts(
            prompt_variant
        )
    )

    if isinstance(
        image,
        list,
    ):
        images = image
    else:
        images = [
            image
        ]

    user_content: list[
        dict[str, Any]
    ] = []

    for current_image in images:
        user_content.append(
            {
                "type": "image",
                "image": current_image,
            }
        )

    user_content.append(
        {
            "type": "text",
            "text": user_prompt,
        }
    )

    messages: list[
        dict[str, Any]
    ] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    if target is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": target,
                    }
                ],
            }
        )

    return messages


# ============================================================
# DATA COLLATOR
# ============================================================

@dataclass
class DamageDataCollator:
    processor: Any
    max_length: int = 1024
    use_assistant_mask: bool = True

    prompt_variant: str = (
        "baseline"
    )

    # Week 4 options
    visual_input_mode: str = (
        "single_image"
    )

    num_local_crops: int = 2

    crop_fraction: float = 0.65

    def _processor_kwargs(
        self,
    ) -> dict[str, Any]:

        return {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "enable_thinking": False,
        }

    def __call__(
        self,
        examples: list[
            dict[str, Any]
        ],
    ) -> dict[str, torch.Tensor]:

        full_conversations: list[
            list[dict[str, Any]]
        ] = []

        prompt_conversations: list[
            list[dict[str, Any]]
        ] = []

        for example in examples:

            image = open_rgb_image(
                example["image_path"]
            )

            visual_inputs = (
                build_visual_inputs(
                    image,
                    visual_input_mode=(
                        self.visual_input_mode
                    ),
                    num_local_crops=(
                        self.num_local_crops
                    ),
                    crop_fraction=(
                        self.crop_fraction
                    ),
                )
            )

            target = build_target(
                example[
                    "damage_categories"
                ],
                example[
                    "reference_description"
                ],
            )

            full_conversations.append(
                make_messages(
                    visual_inputs,
                    target=target,
                    prompt_variant=(
                        self.prompt_variant
                    ),
                )
            )

            prompt_conversations.append(
                make_messages(
                    visual_inputs,
                    target=None,
                    prompt_variant=(
                        self.prompt_variant
                    ),
                )
            )

        common = (
            self._processor_kwargs()
        )

        full_inputs = None
        assistant_mask = None

        if self.use_assistant_mask:
            try:

                full_inputs = (
                    self.processor.apply_chat_template(
                        full_conversations,
                        add_generation_prompt=False,
                        return_assistant_tokens_mask=True,
                        **common,
                    )
                )

                assistant_mask = (
                    full_inputs.pop(
                        "assistant_masks",
                        None,
                    )
                )

                if (
                    assistant_mask
                    is not None
                    and int(
                        assistant_mask.sum()
                    )
                    == 0
                ):
                    assistant_mask = None

            except (
                TypeError,
                ValueError,
                KeyError,
            ):
                full_inputs = None
                assistant_mask = None

        if full_inputs is None:

            full_inputs = (
                self.processor.apply_chat_template(
                    full_conversations,
                    add_generation_prompt=False,
                    **common,
                )
            )

        labels = (
            full_inputs[
                "input_ids"
            ].clone()
        )

        if assistant_mask is not None:

            labels[
                assistant_mask.to(
                    dtype=torch.bool
                )
                == 0
            ] = -100

        else:

            prompt_inputs = (
                self.processor.apply_chat_template(
                    prompt_conversations,
                    add_generation_prompt=True,
                    **common,
                )
            )

            prompt_lengths = (
                prompt_inputs[
                    "attention_mask"
                ]
                .sum(dim=1)
                .tolist()
            )

            for (
                row_index,
                prompt_length,
            ) in enumerate(
                prompt_lengths
            ):

                labels[
                    row_index,
                    : int(
                        prompt_length
                    ),
                ] = -100

        labels[
            full_inputs[
                "attention_mask"
            ]
            == 0
        ] = -100

        if not torch.any(
            labels.ne(-100)
        ):
            raise RuntimeError(
                "The batch contains no trainable assistant tokens. "
                "Increase max_length, reduce max_pixels, "
                "or reduce the number of local crops."
            )

        full_inputs[
            "labels"
        ] = labels

        return full_inputs