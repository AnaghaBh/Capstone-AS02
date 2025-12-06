"""Training script for misinformation classifier."""

import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

from config import Config
from dataset import create_datasets
from model import build_model
from utils import set_seed, setup_logging, compute_metrics, save_metrics

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device, config):
    """Evaluate model and return metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs['loss'].item()
            
            # Apply sigmoid to get probabilities
            predictions = torch.sigmoid(outputs['logits'])
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    metrics = compute_metrics(predictions, labels, config.label_names)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Train misinformation classifier')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--test_path', type=str, help='Path to test data (optional)')
    parser.add_argument('--config_path', type=str, help='Path to config file (optional)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    set_seed(42)
    
    # Load config
    config = Config()
    config.results_dir = args.output_dir
    config.model_save_path = os.path.join(args.output_dir, 'best_model')
    
    # Device (prefer MPS on Mac, then CUDA, then CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset, tokenizer = create_datasets(
        args.data_path, config, args.test_path
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Build model
    model = build_model(config).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = 0
    
    for epoch in range(config.num_epochs):
        logging.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logging.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, device, config)
        logging.info(f"Val loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['macro_f1']:.4f}")
        
        # Save best model
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'tokenizer': tokenizer
            }, config.model_save_path + '.pt')
            logging.info(f"New best model saved with F1: {best_val_f1:.4f}")
    
    # Final evaluation on test set
    logging.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, config)
    
    # Save metrics
    save_metrics(test_metrics, os.path.join(config.results_dir, 'test_metrics.json'))
    
    logging.info("Training completed!")
    logging.info(f"Test F1: {test_metrics['macro_f1']:.4f}")

if __name__ == "__main__":
    main()