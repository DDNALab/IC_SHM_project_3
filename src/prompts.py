from __future__ import annotations

import json


# ============================================================
# V1 BASELINE PROMPT
# Strict JSON prompt.
# ============================================================

BASELINE_SYSTEM_PROMPT = (
    "You are a structural inspection assistant. Identify the visible damage types and describe "
    "their observable characteristics. Return valid JSON only."
)


BASELINE_USER_PROMPT = """Analyze the structural image.

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


# ============================================================
# DIRECT / FREE-FORM INSTRUCTION PROMPT
#
# Week 3 prompt-style ablation.
#
# This deliberately removes the detailed checklist and
# category-first reasoning structure.
#
# IMPORTANT:
# The JSON output format is still retained so that prompt-style
# experiments use the same training target and evaluator.
# ============================================================

DIRECT_FREE_TEXT_SYSTEM_PROMPT = (
    "You are a structural inspection assistant. "
    "Inspect the image and report the visible structural damage."
)


DIRECT_FREE_TEXT_USER_PROMPT = """Inspect this structural image and identify the visible damage.

Use only these damage categories:
crack, void, honeycomb, looseness, spalling, exposed_rebar, corrosion, efflorescence, pothole.

Briefly describe what you can visibly observe.

Return your answer as:
{
  "damage_categories": ["category_1", "category_2"],
  "description": "Brief description of the visible damage."
}

Return only the JSON object.
"""


# ============================================================
# V3 STRUCTURED CATEGORY-FIRST PROMPT
# Original long structured prompt used in the completed V3 run.
# Keep this unchanged for reproducibility.
# ============================================================

STRUCTURED_SYSTEM_PROMPT = (
    "You are a structural damage inspection assistant. "
    "Analyze only visible evidence in the image. "
    "Determine the damage categories first, then write a concise description "
    "that is consistent with those selected categories. "
    "Return valid JSON only."
)


STRUCTURED_USER_PROMPT = """Analyze the structural image using a category-first inspection procedure.

Step 1 - Damage categories:
Select every visibly supported damage category from this fixed ontology only:

crack
void
honeycomb
looseness
spalling
exposed_rebar
corrosion
efflorescence
pothole

Category rules:
- Select a category only when there is visible evidence for it.
- Include all visible damage categories.
- Do not infer hidden or unsupported damage.
- Do not use categories outside the fixed ontology.
- Do not duplicate categories.
- Use an empty list only when no listed damage category is visibly supported.

Step 2 - Description:
After selecting the categories, write one concise description of the visible damage.

When observable, describe:
- material or structural component;
- damage morphology or appearance;
- orientation;
- distribution;
- extent or severity;
- relationships among multiple visible damage types.

Description rules:
- Keep the description consistent with the selected damage_categories.
- Do not introduce an additional damage type in the description unless it is also included in damage_categories.
- Describe only visible characteristics.
- Avoid speculation about causes or hidden conditions.
- Keep the description concise and factual.

Return exactly one JSON object using this schema:
{
  "damage_categories": ["category_1", "category_2"],
  "description": "Concise description of the visible structural damage."
}

Do not include markdown, explanations, reasoning, headings, or text outside the JSON object.
"""


# ============================================================
# COMPACT STRUCTURED CATEGORY-FIRST PROMPT
# Shorter structured prompt used for high-resolution experiments.
# ============================================================

COMPACT_STRUCTURED_SYSTEM_PROMPT = (
    "You are a structural damage inspection assistant. "
    "Identify visible damage categories first, then write a concise description. "
    "Use only visible evidence and return valid JSON only."
)


COMPACT_STRUCTURED_USER_PROMPT = """Analyze the structural image.

First select all visibly supported categories from:
crack, void, honeycomb, looseness, spalling, exposed_rebar, corrosion, efflorescence, pothole.

Rules:
- Include all visible categories.
- Do not add unsupported categories.
- Do not duplicate categories.
- Use an empty list only if none are visible.

Then write one concise description consistent with the selected categories.
When visible, mention material/component, morphology, orientation, distribution, extent, and relationships among damage types.
Do not speculate about hidden causes.

Return exactly:
{
  "damage_categories": ["category_1", "category_2"],
  "description": "Concise visible-damage description."
}

Return JSON only. No markdown, reasoning, or extra text.
"""


# ============================================================
# PROMPT SELECTOR
# ============================================================

def get_prompts(
    variant: str = "baseline",
) -> tuple[str, str]:
    """
    Return the system and user prompts for an experiment.

    Supported variants:
    - baseline
    - direct_free_text
    - structured_category_first
    - structured_category_first_compact
    """

    variant = str(
        variant
    ).strip().lower()

    if variant == "baseline":
        return (
            BASELINE_SYSTEM_PROMPT,
            BASELINE_USER_PROMPT,
        )

    if variant == "direct_free_text":
        return (
            DIRECT_FREE_TEXT_SYSTEM_PROMPT,
            DIRECT_FREE_TEXT_USER_PROMPT,
        )

    if variant == "structured_category_first":
        return (
            STRUCTURED_SYSTEM_PROMPT,
            STRUCTURED_USER_PROMPT,
        )

    if variant == "structured_category_first_compact":
        return (
            COMPACT_STRUCTURED_SYSTEM_PROMPT,
            COMPACT_STRUCTURED_USER_PROMPT,
        )

    raise ValueError(
        f"Unknown prompt variant: {variant}. "
        "Expected 'baseline', "
        "'direct_free_text', "
        "'structured_category_first', or "
        "'structured_category_first_compact'."
    )


# ============================================================
# BACKWARD-COMPATIBLE DEFAULTS
# ============================================================

# Existing code importing SYSTEM_PROMPT and USER_PROMPT
# continues to use the original V1 baseline prompt.
SYSTEM_PROMPT = BASELINE_SYSTEM_PROMPT
USER_PROMPT = BASELINE_USER_PROMPT


# ============================================================
# TRAINING TARGET
# ============================================================

def build_target(
    categories: list[str],
    description: str,
) -> str:
    return json.dumps(
        {
            "damage_categories": categories,
            "description": description.strip(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )