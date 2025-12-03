# Misinformation Classifier

A DistilBERT-based multi-label classifier for detecting psychological mechanisms in misinformation headlines.

## Overview

This project implements a transformer-based classification system that analyses misinformation headlines and predicts the presence of five psychological mechanisms:

**Framework 1 — Elaboration Likelihood Model:**
- `central_route_present` (0/1)
- `peripheral_route_present` (0/1)

**Framework 2 — Cognitive Biases:**
- `naturalness_bias` (0/1)
- `availability_bias` (0/1)
- `illusory_correlation` (0/1)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AnaghaBh/Capstone-AS02
cd misinformation-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Train the model using the training script:

```bash
python src/train.py --data_path data/raw/sample_data.csv --output_dir results
```

Or use the shell script:
```bash
bash scripts/run_training.sh
```

**Training Parameters:**
- `--data_path`: Path to training data (CSV or JSONL)
- `--test_path`: Optional separate test file
- `--output_dir`: Directory to save results (default: results)

### Evaluation

Evaluate a trained model:

```bash
python src/evaluate.py --model_path results/best_model.pt --data_path data/raw/sample_data.csv
```

Or use the shell script:
```bash
bash scripts/run_eval.sh
```

### Prediction

**Single text prediction:**
```bash
python src/predict.py --model_path results/best_model.pt --text "Your headline here"
```

**Batch prediction:**
```bash
python src/predict.py --model_path results/best_model.pt --input_file data/new_headlines.csv
```

Or use the shell script:
```bash
bash scripts/run_predict.sh --text "Your headline here"
```

## Project Structure

```
misinformation-classifier/
├── data/
│   ├── raw/                    # Raw input data
│   └── processed/              # Processed data
├── src/
│   ├── config.py              # Configuration settings
│   ├── dataset.py             # Data loading and preprocessing
│   ├── model.py               # Model architecture
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── predict.py             # Prediction script
│   └── utils.py               # Utility functions
├── notebooks/
│   └── exploratory.ipynb      # Data exploration
├── results/                   # Model outputs and metrics
├── scripts/                   # Shell scripts
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Model Architecture

- **Base Model:** DistilBERT (distilbert-base-uncased)
- **Task:** Multi-label classification
- **Loss Function:** BCEWithLogitsLoss
- **Optimizer:** AdamW with linear warmup
- **Output:** 5 binary labels (one for each psychological mechanism)

## Configuration

Key hyperparameters can be modified in `src/config.py`:

```python
@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 5
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
```

## Evaluation Metrics

The system computes:
- Per-label precision, recall, and F1-score
- Micro and macro averaged metrics
- ROC-AUC scores
- Confusion matrices per label

Results are saved to `results/metrics.json`.

## Example Output

```
PREDICTION RESULTS
==================================================
Text: Scientists discover miracle cure that doctors don't want you to know about

Probabilities:
  central_route_present    : 0.1234
  peripheral_route_present : 0.8765
  naturalness_bias         : 0.9123
  availability_bias        : 0.2345
  illusory_correlation     : 0.3456

Predictions (threshold=0.50):
   central_route_present    : 0
   peripheral_route_present : 1
   naturalness_bias         : 1
   availability_bias        : 0
   illusory_correlation     : 0
```

## Psychological Frameworks

### Elaboration Likelihood Model (ELM)
- **Central Route:** Systematic processing of argument quality and evidence
- **Peripheral Route:** Reliance on superficial cues (authority, emotion, etc.)

### Cognitive Biases
- **Naturalness Bias:** Preference for "natural" solutions over artificial ones
- **Availability Bias:** Overestimating likelihood based on easily recalled examples
- **Illusory Correlation:** Perceiving relationships between unrelated variables


### Running Tests

```bash
# Run exploratory analysis
jupyter notebook notebooks/exploratory.ipynb

## Common problems

1. **CUDA out of memory:** Reduce `batch_size` in config
2. **File not found:** Ensure data paths are correct
3. **Import errors:** Check that all dependencies are installed

**Performance:**

- Use GPU for faster training (`torch.cuda.is_available()`)
- Adjust `max_length` based on your text lengths
- Increase `batch_size` if you have sufficient memory

```
