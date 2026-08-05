from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from prompts import SYSTEM_PROMPT, USER_PROMPT, build_target
from utils import parse_list_field


class DamageDataset(Dataset):
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path)
        required = {"image_id", "image_path", "damage_categories", "reference_description"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
            "damage_categories": parse_list_field(row["damage_categories"]),
            "reference_description": str(row["reference_description"]),
        }


def open_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image.copy()


def make_messages(image: Image.Image, target: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    if target is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            }
        )
    return messages


@dataclass
class DamageDataCollator:
    processor: Any
    max_length: int = 1024
    use_assistant_mask: bool = True

    def _processor_kwargs(self) -> dict[str, Any]:
        return {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "enable_thinking": False,
        }

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_conversations: list[list[dict[str, Any]]] = []
        prompt_conversations: list[list[dict[str, Any]]] = []
        for example in examples:
            image = open_rgb_image(example["image_path"])
            target = build_target(example["damage_categories"], example["reference_description"])
            full_conversations.append(make_messages(image, target=target))
            prompt_conversations.append(make_messages(image, target=None))

        common = self._processor_kwargs()
        full_inputs = None
        assistant_mask = None

        if self.use_assistant_mask:
            try:
                full_inputs = self.processor.apply_chat_template(
                    full_conversations,
                    add_generation_prompt=False,
                    return_assistant_tokens_mask=True,
                    **common,
                )
                assistant_mask = full_inputs.pop("assistant_masks", None)
                if assistant_mask is not None and int(assistant_mask.sum()) == 0:
                    assistant_mask = None
            except (TypeError, ValueError, KeyError):
                full_inputs = None
                assistant_mask = None

        if full_inputs is None:
            full_inputs = self.processor.apply_chat_template(
                full_conversations,
                add_generation_prompt=False,
                **common,
            )

        labels = full_inputs["input_ids"].clone()
        if assistant_mask is not None:
            labels[assistant_mask.to(dtype=torch.bool) == 0] = -100
        else:
            prompt_inputs = self.processor.apply_chat_template(
                prompt_conversations,
                add_generation_prompt=True,
                **common,
            )
            prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
            for row_index, prompt_length in enumerate(prompt_lengths):
                labels[row_index, : int(prompt_length)] = -100

        labels[full_inputs["attention_mask"] == 0] = -100
        if not torch.any(labels.ne(-100)):
            raise RuntimeError(
                "The batch contains no trainable assistant tokens. Increase max_length or reduce max_pixels."
            )
        full_inputs["labels"] = labels
        return full_inputs
