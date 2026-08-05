#!/usr/bin/env bash
set -euo pipefail

# Run this script from the project root.
python src/check_environment.py
python src/data_processing.py
python src/visualize.py
python src/smoke_test.py --config configs/qwen35_4b_24gb.yaml
python src/train.py --config configs/qwen35_4b_24gb.yaml
python src/predict.py --config configs/qwen35_4b_24gb.yaml --split test
python src/evaluate.py --predictions outputs/tables/qwen35_4b_shm_test_predictions.csv
python src/visualize.py --run-name qwen35_4b_shm --metrics-prefix qwen35_4b_shm_test_predictions
