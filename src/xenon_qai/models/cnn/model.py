"""
Multi-Modal Deep Convolutional Neural Network (CNN) for XENON-QAI.

Implements the multi-branch deep architecture specified in dataanalysis.txt:
- S1 1D-CNN branch: Captures prompt scintillation pulse shape (160 bins)
- S2 1D-CNN branch: Captures ionization drift pulse shape (500 bins)
- PMT spatial MLP branch: Captures top (127) and bottom (121) PMT hit patterns
- Physics feature MLP branch: Incorporates 13 expert domain features
- Fusion Bottleneck: Produces low-dimensional latent representation z in R^{latent_dim}
  (specifically designed for Quantum ML hybrid layer interface)
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


class S1WaveformCNN(nn.Module):
    """
    1D-CNN encoder for S1 prompt scintillation waveforms (160 bins).
    Uses multi-scale temporal convolutions to resolve sub-nanosecond peak physics.
    """
    
    def __init__(self, out_features: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 160) or (batch_size, 1, 160)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return F.gelu(self.fc(x))


class S2WaveformCNN(nn.Module):
    """
    1D-CNN encoder for S2 ionization drift waveforms (500 bins).
    Uses dilated convolutions to capture extended electron diffusion envelopes.
    """
    
    def __init__(self, out_features: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=11, stride=2, padding=5)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 500) or (batch_size, 1, 500)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.pool(x).squeeze(-1)
        return F.gelu(self.fc(x))


class PMTSpatialMLP(nn.Module):
    """
    MLP encoder for PMT hit arrays (127 top + 121 bottom = 248 PMTs).
    Learns spatial energy distribution and top/bottom asymmetry.
    """
    
    def __init__(self, in_features: int = TOP_PMT_COUNT + BOT_PMT_COUNT, out_features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsFeatureMLP(nn.Module):
    """
    MLP encoder for 13 hand-crafted physics features.
    """
    
    def __init__(self, in_features: int = PHYSICS_FEATURE_DIM, out_features: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class XenonMultimodalCNN(nn.Module):
    """
    Complete Multi-Modal Neural Network with Bottleneck Latent Space.
    
    Accepts:
      - S1 waveform (160)
      - S2 waveform (500)
      - PMT hits (248)
      - Physics features (13)
      OR standard 908D multimodal vector + optional 13D physics features.
    
    Outputs:
      - Logits for classification (Binary or Multiclass)
      - Latent representation z in R^{latent_dim} (e.g. 32D, 16D, 8D for Quantum ML)
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        num_classes: int = 1,  # 1 for binary (BCEWithLogitsLoss), >1 for multiclass
        include_physics_features: bool = True,
        dropout: float = 0.15
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.include_physics_features = include_physics_features
        
        # Encoders
        self.s1_encoder = S1WaveformCNN(out_features=64)
        self.s2_encoder = S2WaveformCNN(out_features=128)
        self.pmt_encoder = PMTSpatialMLP(out_features=64)
        
        fusion_in_dim = 64 + 128 + 64  # 256
        if include_physics_features:
            self.physics_encoder = PhysicsFeatureMLP(out_features=32)
            fusion_in_dim += 32  # 288
        else:
            self.physics_encoder = None
        
        # Fusion and bottleneck projection
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Classification Head
        self.head = nn.Linear(latent_dim, num_classes)
        
    def _slice_multimodal_vector(
        self,
        multimodal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slices 908D vector into [S1 (160) | S2 (500) | PMTs (248)]."""
        s1 = multimodal[:, :S1_MAX_LEN]
        s2 = multimodal[:, S1_MAX_LEN : S1_MAX_LEN + S2_TARGET_LEN]
        pmts = multimodal[:, S1_MAX_LEN + S2_TARGET_LEN : MULTIMODAL_DIM]
        return s1, s2, pmts
    
    def encode(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        pmts: torch.Tensor,
        physics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extracts latent representation z in R^{latent_dim}."""
        s1_rep = self.s1_encoder(s1)
        s2_rep = self.s2_encoder(s2)
        pmt_rep = self.pmt_encoder(pmts)
        
        reps = [s1_rep, s2_rep, pmt_rep]
        if self.include_physics_features and physics is not None:
            phys_rep = self.physics_encoder(physics)
            reps.append(phys_rep)
        
        fused = torch.cat(reps, dim=-1)
        z = self.fusion(fused)
        return z
    
    def forward(
        self,
        x: torch.Tensor,
        physics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: If dim is (batch, 908), automatically slices into components.
            physics: Optional (batch, 13) physics feature tensor.
            
        Returns:
            Tuple of (logits, latent_z)
        """
        if x.shape[-1] == MULTIMODAL_DIM:
            s1, s2, pmts = self._slice_multimodal_vector(x)
        else:
            raise ValueError(f"Expected input shape (*, {MULTIMODAL_DIM}), got {x.shape}")
        
        z = self.encode(s1, s2, pmts, physics)
        logits = self.head(z)
        return logits, z
    
    def predict_candidate_score(
        self,
        x: torch.Tensor,
        physics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes calibrated dark matter candidate compatibility score in [0, 1].
        """
        logits, _ = self.forward(x, physics)
        if self.num_classes == 1:
            return torch.sigmoid(logits).squeeze(-1)
        else:
            return torch.softmax(logits, dim=-1)
