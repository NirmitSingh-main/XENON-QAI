"""
Waveform Transformer Encoder with Positional Encodings.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for waveform time bins."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class WaveformTransformerEncoder(nn.Module):
    """
    Self-attention encoder for 1D pulse waveforms (S1 or S2).
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        patch_size: int = 10
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, total_bins)
        batch_size, total_bins = x.shape
        num_patches = total_bins // self.patch_size
        x_patched = x[:, :num_patches * self.patch_size].view(batch_size, num_patches, self.patch_size)
        
        h = self.proj(x_patched)
        h = self.pos_encoder(h)
        h = self.transformer(h)
        h = self.norm(h)
        
        # Global pooling across sequence
        # (batch_size, num_patches, d_model) -> (batch_size, d_model)
        pooled = h.mean(dim=1)
        return pooled
