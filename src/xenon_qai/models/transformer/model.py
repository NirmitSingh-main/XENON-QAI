"""
Multi-Modal Waveform Transformer for XENON-QAI.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from xenon_qai.data.types import (
    S1_MAX_LEN,
    S2_TARGET_LEN,
    TOP_PMT_COUNT,
    BOT_PMT_COUNT,
    PHYSICS_FEATURE_DIM,
    MULTIMODAL_DIM
)
from xenon_qai.models.transformer.encoder import WaveformTransformerEncoder
from xenon_qai.models.cnn.model import PMTSpatialMLP, PhysicsFeatureMLP


class XenonTransformerModel(nn.Module):
    """
    Multi-modal Transformer architecture with self-attention on waveforms.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        num_classes: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        include_physics_features: bool = True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.include_physics_features = include_physics_features
        
        self.s1_transformer = WaveformTransformerEncoder(
            in_channels=1, d_model=d_model, nhead=nhead, num_layers=num_layers, patch_size=8
        )
        self.s2_transformer = WaveformTransformerEncoder(
            in_channels=1, d_model=d_model, nhead=nhead, num_layers=num_layers, patch_size=20
        )
        self.pmt_encoder = PMTSpatialMLP(out_features=64)
        
        fusion_dim = d_model + d_model + 64
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
        
        s1_emb = self.s1_transformer(s1)
        s2_emb = self.s2_transformer(s2)
        pmt_emb = self.pmt_encoder(pmts)
        
        reps = [s1_emb, s2_emb, pmt_emb]
        if self.include_physics_features and physics is not None:
            reps.append(self.physics_encoder(physics))
            
        fused = torch.cat(reps, dim=-1)
        z = self.fusion(fused)
        logits = self.head(z)
        return logits, z
