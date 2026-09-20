from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import torch

from dataset import (
    build_visual_inputs,
    make_messages,
)
from model import (
    build_processor,
    load_adapter_for_inference,
)
from predict import parse_prediction
from utils import load_yaml


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "final_model.yaml"
)

ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "final_model"
    / "final_adapter"
)


# ============================================================
# Load final model configuration
# ============================================================

print("\n========================================")
print(" SHM Structural Damage Analysis Demo")
print("========================================")

print("\nLoading configuration...")

config = load_yaml(CONFIG_PATH)

model_cfg = config["model"]
data_cfg = config["data"]

visual_input_mode = str(
    data_cfg.get(
        "visual_input_mode",
        "single_image",
    )
)

num_local_crops = int(
    data_cfg.get(
        "num_local_crops",
        2,
    )
)

crop_fraction = float(
    data_cfg.get(
        "crop_fraction",
        0.65,
    )
)

prompt_variant = str(
    data_cfg.get(
        "prompt_variant",
        "direct_free_text",
    )
)


# ============================================================
# Load processor
# ============================================================

print("Loading processor...")

processor = build_processor(
    model_cfg["model_id"],
    min_pixels=int(
        model_cfg["min_pixels"]
    ),
    max_pixels=int(
        model_cfg["max_pixels"]
    ),
)

processor.tokenizer.padding_side = "left"


# ============================================================
# Load FINAL language-adapted LoRA model
# ============================================================

print("Loading final Qwen3.5-4B adapter...")

if not ADAPTER_PATH.exists():
    raise FileNotFoundError(
        f"Final adapter not found: {ADAPTER_PATH}"
    )

model = load_adapter_for_inference(
    model_id=model_cfg["model_id"],
    adapter_path=ADAPTER_PATH,
    precision=model_cfg.get(
        "precision",
        "bf16",
    ),
    load_in_4bit=True,
)

model.eval()

model_device = next(
    model.parameters()
).device

print("Model loaded successfully.")
print("Device:", model_device)


# ============================================================
# Display helpers
# ============================================================

DISPLAY_NAMES = {
    "crack": "Crack",
    "void": "Void",
    "honeycomb": "Honeycomb",
    "looseness": "Looseness",
    "spalling": "Spalling",
    "exposed_rebar": "Exposed Rebar",
    "corrosion": "Corrosion",
    "efflorescence": "Efflorescence",
    "pothole": "Pothole",
}


def format_categories(
    categories: list[str],
) -> str:

    if not categories:
        return "No valid damage category detected"

    return ", ".join(
        DISPLAY_NAMES.get(
            category,
            category.replace("_", " ").title(),
        )
        for category in categories
    )


# ============================================================
# Inference function
# ============================================================

def analyze_image(image):

    if image is None:
        return (
            "Please upload an image.",
            "",
            "",
        )

    try:

        visual_inputs = build_visual_inputs(
            image,
            visual_input_mode=visual_input_mode,
            num_local_crops=num_local_crops,
            crop_fraction=crop_fraction,
        )

        messages = make_messages(
            visual_inputs,
            target=None,
            prompt_variant=prompt_variant,
        )

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        inputs = {
            key: (
                value.to(model_device)
                if torch.is_tensor(value)
                else value
            )
            for key, value in inputs.items()
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():

            generated = model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    processor.tokenizer.pad_token_id
                ),
                eos_token_id=(
                    processor.tokenizer.eos_token_id
                ),
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_seconds = (
            time.perf_counter()
            - start_time
        )

        prompt_length = (
            inputs["input_ids"].shape[1]
        )

        new_tokens = generated[
            0,
            prompt_length:
        ]

        raw_output = (
            processor.decode(
                new_tokens,
                skip_special_tokens=True,
            )
            .strip()
        )

        (
            categories,
            description,
            json_valid,
        ) = parse_prediction(
            raw_output
        )

        category_text = format_categories(
            categories
        )

        if json_valid:
            status = (
                f"Analysis completed in "
                f"{inference_seconds:.2f} seconds"
            )
        else:
            status = (
                f"Analysis completed in "
                f"{inference_seconds:.2f} seconds "
                f"(unstructured model response)"
            )

        return (
            category_text,
            description,
            status,
        )

    except Exception as error:

        print("\nInference error:")
        print(error)

        return (
            "Analysis failed",
            str(error),
            "Please check the terminal for details.",
        )


# ============================================================
# Gradio interface
# ============================================================

CSS = """
body {
    background: linear-gradient(
        135deg,
        #dbeafe 0%,
        #e0f2fe 25%,
        #ecfdf5 55%,
        #f5f3ff 100%
    ) !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding: 30px !important;
    background: transparent !important;
}

#title {
    text-align: center;
    margin-bottom: 4px;
    font-weight: 800;
    color: #123b63;
    font-size: 38px;
}

#subtitle {
    text-align: center;
    color: #356a78;
    margin-bottom: 26px;
    font-size: 17px;
}

#result-box,
#description-box {
    background: rgba(255, 255, 255, 0.96) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(22, 119, 255, 0.15) !important;
    box-shadow: 0 6px 18px rgba(30, 60, 90, 0.08) !important;
}

#result-box {
    min-height: 72px;
}

#description-box {
    min-height: 165px;
}

button.primary {
    background: linear-gradient(
        90deg,
        #1677ff,
        #14b8a6
    ) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    min-height: 48px !important;
}

button.primary:hover {
    filter: brightness(1.06);
    transform: translateY(-1px);
}

textarea,
input {
    background: rgba(255, 255, 255, 0.96) !important;
}

footer {
    display: none !important;
}
.image-container img {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
}

[data-testid="image"] img {
    object-fit: contain !important;
}

"""


with gr.Blocks(
    title="Structural Damage Analysis",
    css=CSS,
) as demo:

    gr.Markdown(
        "# Structural Damage Analysis System",
        elem_id="title",
    )

    gr.Markdown(
        "**Qwen3.5-4B • Language-Adapted QLoRA • Structural Health Monitoring**",
        elem_id="subtitle",
    )

    with gr.Row():

        with gr.Column(scale=1):

            input_image = gr.Image(
                type="pil",
                label="Structural Image",
                height=430,
		image_mode="RGB",
    		show_label=True,
    		container=True,
            )

            analyze_button = gr.Button(
                "Analyze Damage",
                variant="primary",
                size="lg",
            )

            clear_button = gr.ClearButton(
                [
                    input_image,
                ],
                value="Clear",
            )

        with gr.Column(scale=1):

            damage_output = gr.Textbox(
                label="Predicted Damage Categories",
                interactive=False,
                elem_id="result-box",
            )

            description_output = gr.Textbox(
                label="Damage Description",
                interactive=False,
                lines=6,
                elem_id="description-box",
            )

            status_output = gr.Textbox(
                label="Inference Status",
                interactive=False,
            )

            gr.Markdown(
                """
                **Supported damage categories:**  
                Crack • Void • Honeycomb • Looseness • Spalling •
                Exposed Rebar • Corrosion • Efflorescence • Pothole
                """
            )

    analyze_button.click(
        fn=analyze_image,
        inputs=[
            input_image,
        ],
        outputs=[
            damage_output,
            description_output,
            status_output,
        ],
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        inbrowser=True,
        show_error=True,
    )