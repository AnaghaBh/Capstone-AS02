"""Utility functions for the misinformation classifier."""

import json
import logging
import random
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from typing import Dict, List, Tuple

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def compute_metrics(predictions: np.ndarray, labels: np.ndarray, 
                   label_names: List[str], threshold: float = 0.5) -> Dict:
    """Compute multi-label classification metrics."""
    # Apply threshold
    pred_binary = (predictions > threshold).astype(int)
    
    # Per label metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_binary, average=None, zero_division=0
    )
    
    # Micro and macro averages
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, pred_binary, average='micro', zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, pred_binary, average='macro', zero_division=0
    )
    
    # ROC AUC (handle plossible errors)
    try:
        roc_auc = roc_auc_score(labels, predictions, average='macro')
    except ValueError:
        roc_auc = 0.0
    
    metrics = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'roc_auc': roc_auc
    }
    
    # Add per-label metrics
    for i, label in enumerate(label_names):
        metrics[f'{label}_precision'] = precision[i] if isinstance(precision, np.ndarray) else precision
        metrics[f'{label}_recall'] = recall[i] if isinstance(recall, np.ndarray) else recall
        metrics[f'{label}_f1'] = f1[i] if isinstance(f1, np.ndarray) else f1
    
    return metrics

def save_metrics(metrics: Dict, filepath: str):
    """Save metrics to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(filepath: str) -> Dict:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)