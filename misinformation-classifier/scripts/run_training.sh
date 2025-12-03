#!/bin/bash

# Training script for misinformation classifier

echo "Starting training..."

cd "$(dirname "$0")/.."

python src/train.py \
    --data_path data/raw/sample_data.json \
    --output_dir results \
    "$@"

echo "Training completed!"