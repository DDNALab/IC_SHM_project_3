# SHM Project 3 — Structural Damage Diagnosis

This project contains the reproducible pipeline for structural damage diagnosis using image-text multimodal data.

The final system uses Qwen3.5-4B with QLoRA fine-tuning to predict:

- image_id
- damage_categories
- description

The final primary model is the validation-selected V1-direct single-image Qwen3.5-4B QLoRA model.

## Project Structure

```text
release_candidate_1/
├── configs/
│   └── final_model.yaml
├── dataset/
│   ├── description.json
│   └── image/
├── model_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── chat_template.jinja
│   ├── processor_config.json
│   ├── README.md
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_args.bin
├── outputs/
│   └── tables/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── final_predictions.csv
│       ├── final_metrics.json
│       ├── submission.json
│       ├── week6_final_model_freeze_manifest.json
│       └── week6_final_run_summary.json
├── reproduction_metadata/
│   ├── all_results.json
│   ├── eval_results.json
│   ├── trainer_state.json
│   ├── train_results.json
│   └── PORTABLE_SPLITS_NOTE.md
├── src/
│   ├── check_environment.py
│   ├── data_processing.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── evaluate_resnet18_baseline.py
│   ├── evaluate_resnet18_test.py
│   ├── export_submission.py
│   ├── model.py
│   ├── paths.py
│   ├── predict.py
│   ├── prompts.py
│   ├── smoke_test.py
│   ├── train.py
│   ├── train_resnet18_baseline.py
│   ├── utils.py
│   ├── visualize.py
│   └── __init__.py
├── README.md
├── requirements.txt
└── requirements_frozen.txt
```

## Dataset Split

The leakage-resistant dataset split contains:

```text
Training:   957 images
Validation: 118 images
Test:       125 images
Total:      1200 unique images
```

The final ontology contains nine structural-damage categories:

```text
crack
void
honeycomb
looseness
spalling
exposed_rebar
corrosion
efflorescence
pothole
```

The packaged split files are:

```text
outputs/tables/train.csv
outputs/tables/val.csv
outputs/tables/test.csv
```

The image paths inside the release-candidate split files use portable relative paths such as:

```text
dataset/image/00002.jpg
```

The original final-run split CSV files were frozen and hashed before the final training run. Only the copies inside the release package were rewritten to replace absolute Windows paths with portable relative paths.

Labels, image IDs, split membership, and annotations were not changed.

See:

```text
reproduction_metadata/PORTABLE_SPLITS_NOTE.md
```

for details.

## Final Model

```text
Backbone: Qwen/Qwen3.5-4B
Method: QLoRA
Precision: bf16
LoRA scope: language-side
LoRA rank: 8
LoRA alpha: 16
LoRA dropout: 0.05
Prompt: direct_free_text
Epochs: 3
Batch size: 1
Gradient accumulation: 32
Learning rate: 0.0002
Seed: 2026
```

Locked configuration:

```text
configs/final_model.yaml
```

Packaged final adapter:

```text
model_adapter
```

The adapter package contains the LoRA adapter, tokenizer, processor configuration, and chat template required for inference.

## Hardware Used

```text
GPU: NVIDIA RTX 2000 Ada Generation
VRAM: 16 GB
Python: 3.11.15
CUDA available: Yes
```

Final smoke-test peak GPU memory:

```text
Peak allocated: 5.833 GB
Peak reserved: 6.004 GB
```

Final internal-test inference:

```text
Approximately 5.02 seconds/image
Peak allocated GPU memory: 3.294 GB
Peak reserved GPU memory: 3.379 GB
```

## Environment Setup

Python 3.11 is recommended.

The final successful environment used:

```text
Python 3.11.15
CUDA 12.8 compatible PyTorch build
```

### Step 1 — Install CUDA-enabled PyTorch

For the environment used in the final Week 6 run, install:

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

This step should be completed before installing the frozen project dependencies.

### Step 2 — Install Remaining Frozen Dependencies

Run:

```bash
pip install -r requirements_frozen.txt
```

The exact remaining package versions are stored in:

```text
requirements_frozen.txt
```

### Alternative General Installation

For a less strict installation, project dependencies can also be installed using:

```bash
pip install -r requirements.txt
```

The frozen requirements are recommended for reproducing the reported final results.

## Exact Final Environment

The final successful model run used:

```text
Python           3.11.15
torch            2.11.0+cu128
torchvision      0.26.0+cu128
transformers     5.15.0
accelerate       1.14.0
peft             0.20.0
bitsandbytes     0.50.1
safetensors      0.8.0
pandas           3.0.5
numpy            2.4.6
Pillow           12.3.0
scikit-learn     1.9.0
PyYAML           6.0.3
matplotlib       3.11.1
nltk             3.10.3
tensorboard      2.21.0
```

## Environment Check

From the root of the release package, run:

```bash
python src/check_environment.py
```

The successful package test confirmed:

```text
Python: 3.11.15
GPU: NVIDIA RTX 2000 Ada Generation
VRAM: 16 GB
CUDA available: True
BF16 supported: True
description.json found: True
dataset/image found: True
```

The environment checker writes:

```text
outputs/tables/environment_report.json
```

Some machines may display a conservative recommendation from the environment checker for 16 GB VRAM. The frozen final configuration nevertheless trained and ran successfully on the NVIDIA RTX 2000 Ada Generation 16 GB GPU used for the final experiment.

## Dataset

The release package contains:

```text
dataset/description.json
dataset/image/
```

The image directory contains the complete packaged dataset required by the provided split manifests.

The split manifests reference exactly 1,200 images.

The packaged-path verification confirmed:

```text
checked = 1200
missing = 0
first_missing = none
```

## Data Processing

The preprocessing pipeline can be run with:

```bash
python src/data_processing.py
```

The final project uses the locked leakage-resistant split.

For exact reproduction of the reported final experiment, use the provided split manifests:

```text
outputs/tables/train.csv
outputs/tables/val.csv
outputs/tables/test.csv
```

Do not regenerate or modify the split after the final model freeze when attempting to reproduce the reported results.

## Final Smoke Test

Run:

```bash
python src/smoke_test.py --config configs/final_model.yaml
```

Expected output:

```text
outputs/tables/final_model_smoke_test.json
```

The successful final smoke test reported:

```text
status: passed
loss: 1.857167
prompt_variant: direct_free_text
visual_input_mode: single_image
visual_image_count: 1
trainable_parameters: 16,232,448
total_parameters: 2,606,326,272
trainable_percent: 0.6228%
peak_allocated_gb: 5.833
peak_reserved_gb: 6.004
```

## Final Training

To reproduce the final training run:

```bash
python src/train.py --config configs/final_model.yaml
```

The original final training run produced the final adapter under:

```text
outputs/checkpoints/final_model/final_adapter
```

In this release candidate, the same final adapter is packaged at:

```text
model_adapter
```

The final training metadata is available under:

```text
reproduction_metadata/
```

including:

```text
all_results.json
eval_results.json
trainer_state.json
train_results.json
```

The final training summary was:

```text
Epochs: 3.0
Training loss: 0.509571
Validation loss: 0.263942
Training runtime: 5348.7394 seconds
Validation runtime: 56.5009 seconds
```

## Validation Prediction Using Packaged Adapter

On Windows CMD:

```cmd
python src\predict.py --config configs\final_model.yaml --split val --adapter model_adapter --output-csv outputs\tables\week6_final_val_predictions.csv
```

On Linux or WSL2:

```bash
python src/predict.py --config configs/final_model.yaml --split val --adapter model_adapter --output-csv outputs/tables/week6_final_val_predictions.csv
```

## Validation Evaluation

On Windows CMD:

```cmd
python src\evaluate.py --predictions outputs\tables\week6_final_val_predictions.csv --output-prefix week6_final_val
```

On Linux or WSL2:

```bash
python src/evaluate.py --predictions outputs/tables/week6_final_val_predictions.csv --output-prefix week6_final_val
```

Final validation results:

```text
Exact-set accuracy: 0.864407
Micro-F1:           0.939326
Macro-F1:           0.818730
METEOR:             0.590538
JSON validity:      1.000000
```

The final Week 6 retraining reproduced the selected V1-direct validation results exactly.

## Final Test Prediction Using Packaged Adapter

On Windows CMD:

```cmd
python src\predict.py --config configs\final_model.yaml --split test --adapter model_adapter --output-csv outputs\tables\final_predictions.csv
```

On Linux or WSL2:

```bash
python src/predict.py --config configs/final_model.yaml --split test --adapter model_adapter --output-csv outputs/tables/final_predictions.csv
```

The internal-test split contains:

```text
125 images
```

## Final Test Evaluation

On Windows CMD:

```cmd
python src\evaluate.py --predictions outputs\tables\final_predictions.csv --output-prefix final
```

On Linux or WSL2:

```bash
python src/evaluate.py --predictions outputs/tables/final_predictions.csv --output-prefix final
```

Final internal-test results:

```text
Exact-set accuracy: 0.760000
Micro-F1:           0.884120
Macro-F1:           0.763974
METEOR:             0.543396
JSON validity:      1.000000
```

The internal-test results are evaluation evidence only.

They must not be used for:

- prompt tuning
- threshold tuning
- model selection
- crop-selection tuning
- architecture changes
- hyperparameter optimization

## Package-Level Inference Smoke Test

The release candidate was tested directly from inside its own folder.

A one-image validation CSV was created and inference was run using:

```cmd
python src\predict.py --config configs\final_model.yaml --input-csv outputs\tables\package_smoke.csv --adapter model_adapter --output-csv outputs\tables\package_smoke_predictions.csv
```

The packaged model successfully:

- loaded the configuration
- found the packaged dataset image
- loaded the packaged LoRA adapter
- loaded the Qwen3.5-4B backbone
- generated the prediction
- saved the prediction CSV
- saved the inference-efficiency report

The successful package-level inference test reported:

```text
Images to predict: 1
Generation seconds: 4.9511
Seconds/image: 4.951104
Images/second: 0.202
Peak allocated GPU memory: 3.290 GB
Peak reserved GPU memory: 3.379 GB
```

This confirms that the release package can perform inference using its packaged files rather than relying on the original development directory.

## Export Competition Submission

On Windows CMD:

```cmd
python src\export_submission.py --predictions outputs\tables\final_predictions.csv --output outputs\tables\submission.json
```

On Linux or WSL2:

```bash
python src/export_submission.py --predictions outputs/tables/final_predictions.csv --output outputs/tables/submission.json
```

Final submission file:

```text
outputs/tables/submission.json
```

## Submission Validation

The final submission JSON was validated successfully.

Results:

```text
records = 125
unique_ids = 125
duplicate_ids = 0
bad_field_records = 0
bad_category_records = 0
empty_descriptions = 0
```

Every record contains exactly:

```json
{
  "image_id": "...",
  "damage_categories": ["..."],
  "description": "..."
}
```

The legal damage-category ontology is:

```text
crack
void
honeycomb
looseness
spalling
exposed_rebar
corrosion
efflorescence
pothole
```

## Important Final Artifacts

```text
configs/final_model.yaml

model_adapter/

dataset/description.json
dataset/image/

outputs/tables/train.csv
outputs/tables/val.csv
outputs/tables/test.csv
outputs/tables/final_predictions.csv
outputs/tables/final_metrics.json
outputs/tables/submission.json
outputs/tables/week6_final_model_freeze_manifest.json
outputs/tables/week6_final_run_summary.json

reproduction_metadata/all_results.json
reproduction_metadata/eval_results.json
reproduction_metadata/trainer_state.json
reproduction_metadata/train_results.json
reproduction_metadata/PORTABLE_SPLITS_NOTE.md

requirements.txt
requirements_frozen.txt
README.md
```

## Final Model Freeze

Before the final Week 6 model run, the following components were frozen:

- training split
- validation split
- internal-test split
- damage ontology
- prompt
- model configuration
- dataset-processing logic
- prediction code
- evaluation code

SHA-256 fingerprints of the original frozen files were stored in:

```text
outputs/tables/week6_final_model_freeze_manifest.json
```

The freeze manifest corresponds to the original final-run files used during the final experiment.

The packaged split CSV files contain identical:

- image IDs
- labels
- annotations
- split membership

Only their image-path representation was changed from absolute Windows paths to portable relative paths.

Example:

```text
dataset/image/00002.jpg
```

See:

```text
reproduction_metadata/PORTABLE_SPLITS_NOTE.md
```

for details.

## Reproducibility Rule

The internal-test set must not be used for:

- prompt tuning
- threshold tuning
- model selection
- crop-selection tuning
- architecture changes
- hyperparameter optimization

The final test results are evaluation evidence only.

## Notes

Some environments may display non-fatal warnings related to:

```text
triton not found
use_cache with gradient checkpointing
processor kwargs
checkpoint reentrant behavior
Hugging Face unauthenticated requests
```

These warnings did not prevent successful final training or inference.

The Qwen backbone is loaded through Hugging Face when it is not already available in the local model cache. An authenticated Hugging Face token is optional but may provide higher download rate limits.

Internet access may therefore be required the first time the base Qwen3.5-4B model is downloaded.

The packaged `model_adapter` contains the fine-tuned LoRA adapter and associated tokenizer/processor configuration, but not a second full copy of the multi-gigabyte Qwen3.5-4B backbone.

## Final System Summary

The final complete-task system is:

```text
Qwen3.5-4B
+ language-side QLoRA
+ direct free-text prompting
+ single-image visual input
```

The system performs:

```text
multi-label structural-damage classification
+
natural-language structural-damage description generation
```

and produces valid competition-format output.

The final adapter was:

- retrained using the frozen final configuration
- reloaded from disk
- evaluated on the validation split
- evaluated once on the internal-test split
- packaged inside the release candidate
- successfully tested from inside the release package