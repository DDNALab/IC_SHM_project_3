from __future__ import annotations

import ast
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, float) and np.isnan(value):
        return []

    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def normalize_category(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "cracks": "crack",
        "holes": "void",
        "hole": "void",
        "voids": "void",
        "loose": "looseness",
        "loosening": "looseness",
        "spall": "spalling",
        "corroded": "corrosion",
        "rust": "corrosion",
        "exposed_reinforcement": "exposed_rebar",
        "exposed_reinforcing_bar": "exposed_rebar",
        "potholes": "pothole",
    }
    return aliases.get(normalized, normalized)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from a model response."""
    text = text.strip()
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S)
    candidates.extend(fenced)

    first = text.find("{")
    if first >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(first, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[first : index + 1])
                    break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def get_compute_dtype(preference: str = "bf16") -> torch.dtype:
    preference = preference.lower()
    if preference == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if preference in {"bf16", "fp16"} and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))
