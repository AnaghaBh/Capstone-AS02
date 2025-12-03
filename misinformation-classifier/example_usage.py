#!/usr/bin/env python3
"""
Example usage script for the misinformation classifier.
Demonstrates training, evaluation, and prediction workflows.
"""

import os
import sys
sys.path.append('src')

from src.config import Config
from src.dataset import create_datasets
from src.model import build_model
from src.train import train_epoch, evaluate_model
from src.predict import predict_single, load_trained_model
from src.utils import set_seed, setup_logging

def main():
    """Run example workflow."""
    print("Misinformation Classifier - Example Usage")
    print("=" * 50)
    
    # Setup
    setup_logging()
    set_seed(42)
    config = Config()
    
    # Check if sample data exists
    data_path = "data/raw/sample_data.json"
    if not os.path.exists(data_path):
        print(f"Sample data not found at {data_path}")
        print("Please ensure the sample data file exists before running this example.")
        return
    
    print(f"Using sample data: {data_path}")
    print(f"Model: {config.model_name}")
    print(f"Labels: {config.label_names}")
    print()
    
    # Example 1: Data loading
    print("1. Loading and exploring data...")
    try:
        train_dataset, val_dataset, test_dataset, tokenizer = create_datasets(
            data_path, config
        )
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    # Example 2: Model building
    print("\n2. Building model...")
    try:
        model = build_model(config)
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        print(f"   Error building model: {e}")
        return
    
    # Example 3: Training (minimal example)
    print("\n3. Training example...")
    print("   To train the full model, run:")
    print("   python src/train.py --data_path data/raw/sample_data.csv")
    print("   or")
    print("   bash scripts/run_training.sh")
    
    # Example 4: Prediction (if model exists)
    print("\n4. Prediction example...")
    model_path = "results/best_model.pt"
    
    if os.path.exists(model_path):
        print("   Loading trained model...")
        try:
            model, config, tokenizer = load_trained_model(model_path, 'cpu')
            
            # Example predictions
            examples = [
                "Scientists discover miracle cure that doctors don't want you to know about",
                "New research confirms link between diet and health outcomes",
                "This one weird trick will solve all your problems naturally"
            ]
            
            for text in examples:
                result = predict_single(text, model, tokenizer, config, 'cpu')
                print(f"\n   Text: {text}")
                print("   Predictions:")
                for label, pred in result['predictions'].items():
                    prob = result['probabilities'][label]
                    status = "✓" if pred else "✗"
                    print(f"     {status} {label}: {prob:.3f}")
                    
        except Exception as e:
            print(f"   Error loading model: {e}")
            print("   Train the model first using the training script.")
    else:
        print("   No trained model found. Train first using:")
        print("   python src/train.py --data_path data/raw/sample_data.csv")
    
    # Example 5: CLI usage
    print("\n5. Command-line usage examples:")
    print("   # Training")
    print("   python src/train.py --data_path data/raw/sample_data.json")
    print()
    print("   # Evaluation")
    print("   python src/evaluate.py --model_path results/best_model.pt --data_path data/raw/sample_data.json")
    print()
    print("   # Single prediction")
    print('   python src/predict.py --model_path results/best_model.pt --text "Your headline here"')
    print()
    print("   # Batch prediction")
    print("   python src/predict.py --model_path results/best_model.pt --input_file data/new_headlines.csv")
    
    print("\n" + "=" * 50)
    print("Example completed! Check the README.md for detailed usage instructions.")

if __name__ == "__main__":
    main()