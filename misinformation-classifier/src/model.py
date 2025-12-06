"""Model definition for misinformation classification."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional

class MisinformationClassifier(nn.Module):
    """DistilBERT-based multi-label classifier for psychological mechanisms."""
    
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        
        # Load DistilBERT configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.distilbert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None):
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }

def build_model(config) -> MisinformationClassifier:
    """Factory function to build the model."""
    dropout_rate = getattr(config, 'dropout_rate', 0.1)
    return MisinformationClassifier(
        model_name=config.model_name,
        num_labels=config.num_labels,
        dropout_rate=dropout_rate
    )