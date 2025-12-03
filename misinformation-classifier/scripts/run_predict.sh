#!/bin/bash

# Prediction script for misinformation classifier

echo "Starting prediction..."

cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 --text 'Your headline here' [other options]"
    echo "   or: $0 --input_file path/to/file.csv [other options]"
    exit 1
fi

python src/predict.py \
    --model_path results/best_model.pt \
    "$@"

echo "Prediction completed!"