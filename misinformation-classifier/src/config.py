"""Configuration settings for the misinformation classifier."""

import os
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # Model settings
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 5
    max_length: int = 128
    
    # Training settings
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Data settings
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Paths
    data_dir: str = "data"
    results_dir: str = "results"
    model_save_path: str = "results/best_model"
    
    # Labels
    label_names: List[str] = None
    
    def __post_init__(self):
        if self.label_names is None:
            self.label_names = [
                "central_route_present",
                "peripheral_route_present", 
                "naturalness_bias",
                "availability_bias",
                "illusory_correlation"
            ]
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)