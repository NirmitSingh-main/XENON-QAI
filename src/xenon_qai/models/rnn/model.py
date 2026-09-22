"""
Recurrent Neural Network (RNN / GRU / LSTM) for XENON-QAI.

Processes temporal S1 and S2 waveform sequences with Bidirectional GRUs.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from xenon_qai.data.types import (
    S1_MAX_LEN,
    S2_TARGET_LEN,
    TOP_PMT_COUNT,
    BOT_PMT_COUNT,
    PHYSICS_FEATURE_DIM,
    MULTIMODAL_DIM
)
from xenon_qai.models.cnn.model import PMTSpatialMLP, PhysicsFeatureMLP


class WaveformRNNEncoder(nn.Module):
    """Bidirectional GRU encoder for 1D pulse waveforms."""
    
    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.norm = nn.LayerNorm(self.out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        _, h_n = self.gru(x)
        # Concatenate forward and backward final hidden states
        if self.gru.bidirectional:
            h_out = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_out = h_n[-1]
        return self.norm(h_out)


class XenonRNNModel(nn.Module):
    """Multi-modal architecture using RNN temporal encoders on waveforms."""
    
    def __init__(
        self,
        latent_dim: int = 32,
        num_classes: int = 1,
        hidden_dim: int = 64,
        include_physics_features: bool = True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.include_physics_features = include_physics_features
        
        self.s1_rnn = WaveformRNNEncoder(in_dim=1, hidden_dim=32, num_layers=2)
        self.s2_rnn = WaveformRNNEncoder(in_dim=1, hidden_dim=64, num_layers=2)
        self.pmt_encoder = PMTSpatialMLP(out_features=64)
        
        fusion_dim = 64 + 128 + 64  # 256
        if include_physics_features:
            self.physics_encoder = PhysicsFeatureMLP(out_features=32)
            fusion_dim += 32
        else:
            self.physics_encoder = None
            
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.head = nn.Linear(latent_dim, num_classes)
        
    def _slice_multimodal_vector(self, multimodal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = multimodal[:, :S1_MAX_LEN]
        s2 = multimodal[:, S1_MAX_LEN : S1_MAX_LEN + S2_TARGET_LEN]
        pmts = multimodal[:, S1_MAX_LEN + S2_TARGET_LEN : MULTIMODAL_DIM]
        return s1, s2, pmts
    
    def forward(
        self,
        x: torch.Tensor,
        physics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s1, s2, pmts = self._slice_multimodal_vector(x)
        
        s1_h = self.s1_rnn(s1)
        s2_h = self.s2_rnn(s2)
        pmt_h = self.pmt_encoder(pmts)
        
        reps = [s1_h, s2_h, pmt_h]
        if self.include_physics_features and physics is not None:
            reps.append(self.physics_encoder(physics))
            
        fused = torch.cat(reps, dim=-1)
        z = self.fusion(fused)
        logits = self.head(z)
        return logits, z
