"""
Unit tests for Transformer and RNN temporal models.
"""

import pytest
import torch
from xenon_qai.data.types import MULTIMODAL_DIM, PHYSICS_FEATURE_DIM
from xenon_qai.models.transformer import XenonTransformerModel
from xenon_qai.models.rnn import XenonRNNModel


def test_transformer_forward():
    batch_size = 4
    x_multi = torch.randn(batch_size, MULTIMODAL_DIM)
    x_phys = torch.randn(batch_size, PHYSICS_FEATURE_DIM)
    
    model = XenonTransformerModel(latent_dim=16, num_classes=1, d_model=32, num_layers=1)
    logits, z = model(x_multi, x_phys)
    
    assert logits.shape == (batch_size, 1)
    assert z.shape == (batch_size, 16)


def test_rnn_forward():
    batch_size = 4
    x_multi = torch.randn(batch_size, MULTIMODAL_DIM)
    x_phys = torch.randn(batch_size, PHYSICS_FEATURE_DIM)
    
    model = XenonRNNModel(latent_dim=16, num_classes=1, hidden_dim=32)
    logits, z = model(x_multi, x_phys)
    
    assert logits.shape == (batch_size, 1)
    assert z.shape == (batch_size, 16)
