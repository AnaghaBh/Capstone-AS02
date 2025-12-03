#!/bin/bash

# Evaluation script for misinformation classifier

echo "Starting evaluation..."

cd "$(dirname "$0")/.."

python src/evaluate.py \
    --model_path results/best_model.pt \
    --data_path data/raw/sample_data.json \
    --output_dir results \
    "$@"

echo "Evaluation completed!"