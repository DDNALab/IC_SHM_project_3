# Qwen3.5-4B QLoRA pipeline for IC-SHM Project 3

## Expected project layout

```text
project_root/
├── src/
├── configs/
├── dataset/
│   ├── description.json
│   └── image/
└── outputs/
    ├── checkpoints/
    ├── figures/
    └── tables/
```

The scripts resolve paths relative to the project root, not the current shell directory. The supplied `description.json` may contain adjacent JSON objects without commas; `src/data_processing.py` handles both valid JSON and that malformed format.

## Recommended hardware

- Recommended starting point: one 24 GB NVIDIA GPU (RTX 3090/4090, RTX A5000, L4, A10/A10G).
- 16 GB can work only after lowering `max_pixels` and possibly `max_length`.
- 48 GB permits higher-resolution QLoRA or non-quantized LoRA.

Three starter configurations are provided: `qwen35_4b_16gb.yaml`, `qwen35_4b_24gb.yaml`, and `qwen35_4b_48gb.yaml`. The 16 GB configuration is a fallback, not the preferred scientific setup, because its lower image budget can suppress fine cracks.
- Full-parameter fine-tuning is not recommended for this 1,200-image dataset.

The default configuration freezes the vision tower and applies QLoRA to language-side linear layers. Visual-tower adaptation should be evaluated later as an ablation.

## Environment

Install a CUDA-enabled PyTorch build for the machine, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

Qwen3.5 is a recent architecture. Keep `transformers` current if model-loading errors occur.
For better Gated DeltaNet performance on supported Linux/CUDA systems, optionally install `causal-conv1d` and `flash-linear-attention`; without them Transformers falls back to slower, more memory-hungry PyTorch kernels.

## Run in stages

```bash
python src/check_environment.py
python src/data_processing.py
python src/visualize.py
python src/smoke_test.py --config configs/qwen35_4b_24gb.yaml
python src/train.py --config configs/qwen35_4b_24gb.yaml
python src/predict.py --config configs/qwen35_4b_24gb.yaml --split test
python src/evaluate.py \
  --predictions outputs/tables/qwen35_4b_shm_test_predictions.csv
python src/export_submission.py \
  --predictions outputs/tables/qwen35_4b_shm_test_predictions.csv \
  --output outputs/tables/submission.json
python src/visualize.py \
  --run-name qwen35_4b_shm \
  --metrics-prefix qwen35_4b_shm_test_predictions
```

Training can be resumed with:

```bash
python src/train.py \
  --config configs/qwen35_4b_24gb.yaml \
  --resume-from-checkpoint outputs/checkpoints/qwen35_4b_shm/checkpoint-100
```

## Principal outputs

`outputs/tables/` contains the parsed manifest, group-aware splits, data-quality report, training log, predictions, aggregate metrics, and per-class metrics. `outputs/figures/` contains class-distribution, split-distribution, loss-history, and per-class-F1 plots. Adapter checkpoints are kept separately in `outputs/checkpoints/` because they are neither figures nor tables.

## Important modeling notes

1. The split is group-aware using filename blocks to reduce leakage from near-duplicate image sequences.
2. Categories are weakly reconstructed from the provided text. Audit the generated `manifest.csv` before final competition training.
3. The initial ontology is: crack, void, honeycomb, looseness, spalling, exposed_rebar, corrosion, efflorescence, pothole.
4. The loss is applied only to assistant output tokens. The collator first tries the model's assistant-token mask and uses a prompt-length fallback.
5. The default image budget is intentionally conservative for 24 GB. Fine-crack performance should later be tested with larger pixel budgets and/or defect-focused crops.
