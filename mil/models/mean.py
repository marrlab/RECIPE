import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregator import BaseAggregator

class Mean(BaseAggregator):
    def __init__(self, input_dim, num_classes, latent_dim=512, dropout_p=0.0):
        super(BaseAggregator, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Optional: a linear projection to latent_dim before classification
        self.project = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )

        # Head dropout (p=0.0 -> identity) enables MC-dropout uncertainty when >0.
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, D]
        Returns:
            latent: Tensor of shape [B, latent_dim]
            logits: Tensor of shape [B, num_classes]
        """
        # Mean pooling over instances (N)
        pooled = x.mean(dim=1)  # [B, D]
        latent = self.project(pooled)  # [B, latent_dim]
        logits = self.classifier(latent)  # [B, num_classes]
        return latent, logits