"""
End-to-End Pipeline Integration Tests for XENON-QAI.
"""

from pathlib import Path
import pytest
import numpy as np
import torch
from xenon_qai.data import create_dataloaders
from xenon_qai.models.cnn import XenonMultimodalCNN, train_xenon_cnn
from xenon_qai.models.baselines import XenonRandomForest
from xenon_qai.evaluation import compare_models, format_comparison_table


@pytest.mark.skipif(not Path("data/extracted").exists(), reason="Dataset directory not found")
def test_end_to_end_training_and_evaluation():
    data_dir = Path("data/extracted")
    
    # 1. Create DataLoaders with small subsample for fast integration testing
    train_loader, val_loader, test_loader, preprocessor, splits = create_dataloaders(
        data_dir=data_dir,
        batch_size=16,
        mode="both",
        max_events_per_class=10,  # 10 events per class -> fast execution
        random_seed=42
    )
    
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    
    # 2. Train XenonMultimodalCNN for 2 epochs
    cnn_model = XenonMultimodalCNN(latent_dim=16, num_classes=1, include_physics_features=True)
    cnn_results = train_xenon_cnn(
        cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=2,
        lr=1e-3,
        verbose=False
    )
    
    assert "test_metrics" in cnn_results
    assert "roc_auc" in cnn_results["test_metrics"]
    
    # 3. Model comparison table test
    dummy_classical = {"roc_auc": 0.85, "accuracy": 0.82}
    dummy_quantum = {"roc_auc": 0.86, "accuracy": 0.83}
    comp = compare_models(dummy_classical, dummy_quantum)
    table_str = format_comparison_table(comp)
    assert "Classical vs Quantum Performance Benchmark" in table_str
