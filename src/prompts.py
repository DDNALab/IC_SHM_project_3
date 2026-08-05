from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are a structural inspection assistant. Identify the visible damage types and describe "
    "their observable characteristics. Return valid JSON only."
)

USER_PROMPT = """Analyze the structural image.

Return exactly one JSON object with this schema:
{
  "damage_categories": ["category_1", "category_2"],
  "description": "A concise description of visible damage characteristics."
}

Use only categories from:
crack, void, honeycomb, looseness, spalling, exposed_rebar, corrosion, efflorescence, pothole.

Requirements:
- Include every visible damage category supported by the image.
- Use an empty list only when no visible damage is present.
- Describe visible material/component, orientation, morphology, distribution, extent, and associated damage when observable.
- Do not include markdown, commentary, or reasoning outside the JSON object.
"""


def build_target(categories: list[str], description: str) -> str:
    return json.dumps(
        {
            "damage_categories": categories,
            "description": description.strip(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
