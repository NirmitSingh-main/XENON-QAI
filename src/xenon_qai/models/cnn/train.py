"""
Training pipeline for XenonMultimodalCNN.

Features:
- AdamW optimizer with Cosine Annealing learning rate schedule
- Checkpoint saving of best validation loss & best ROC-AUC
- Evaluates binary and multi-class metrics
- Exports latent representations for downstream Quantum ML
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from xenon_qai.data.types import MULTIMODAL_DIM
from xenon_qai.models.cnn.model import XenonMultimodalCNN
from xenon_qai.evaluation.metrics import compute_binary_metrics, compute_multiclass_metrics


def _unpack_batch(
    batch: Any,
    task: str = "binary",
    has_physics_features: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Helper to unpack batches in all dataset modes."""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 4:  # (x_multi, x_phys, y_sig, y_cls)
            x_multi, x_phys, y_sig, y_cls = batch
            y = y_sig if task == "binary" else y_cls
        elif len(batch) == 3:  # (x, y_sig, y_cls)
            x_item, y_sig, y_cls = batch
            y = y_sig if task == "binary" else y_cls
            if x_item.shape[-1] == MULTIMODAL_DIM:
                x_multi = x_item
                x_phys = None
            else:
                x_multi = None
                x_phys = x_item
        elif len(batch) == 2:
            x_multi, y = batch
            x_phys = None
        else:
            raise ValueError(f"Unexpected batch tuple length: {len(batch)}")
    elif isinstance(batch, dict):
        x_multi = batch.get("multimodal")
        x_phys = batch.get("physics")
        y = batch["is_signal"] if task == "binary" else batch.get("class_idx", batch.get("label"))
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")
        
    return x_multi, x_phys if has_physics_features else None, y


def train_epoch(
    model: XenonMultimodalCNN,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0,
    has_physics_features: bool = True
) -> float:
    """Trains model for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    task = "binary" if model.num_classes == 1 else "multiclass"
    
    for batch in dataloader:
        x_multi, x_phys, y = _unpack_batch(batch, task=task, has_physics_features=has_physics_features)
        
        x_multi = x_multi.to(device)
        if x_phys is not None:
            x_phys = x_phys.to(device)
            
        if model.num_classes == 1:
            y_target = y.to(device).float().unsqueeze(1)
        else:
            y_target = y.to(device).long()
            
        optimizer.zero_grad()
        logits, _ = model(x_multi, x_phys)
        loss = criterion(logits, y_target)
        loss.backward()
        
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
        optimizer.step()
        
        batch_size = x_multi.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
    return total_loss / max(1, total_samples)


def evaluate_model(
    model: XenonMultimodalCNN,
    dataloader: DataLoader,
    device: torch.device,
    task: str = "binary",
    has_physics_features: bool = True
) -> Dict[str, Any]:
    """Evaluates model on dataloader."""
    model.eval()
    all_scores: list = []
    all_preds: list = []
    all_targets: list = []
    all_latents: list = []
    
    with torch.no_grad():
        for batch in dataloader:
            x_multi, x_phys, y = _unpack_batch(batch, task=task, has_physics_features=has_physics_features)
            
            x_multi = x_multi.to(device)
            if x_phys is not None:
                x_phys = x_phys.to(device)
                
            logits, z = model(x_multi, x_phys)
            
            all_latents.append(z.cpu().numpy())
            all_targets.append(y.numpy())
            
            if task == "binary":
                probs = torch.sigmoid(logits).cpu().numpy().squeeze(1)
                all_scores.append(probs)
                all_preds.append((probs >= 0.5).astype(int))
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_scores.append(probs)
                all_preds.append(np.argmax(probs, axis=-1))
                
    y_true = np.concatenate(all_targets, axis=0)
    y_score = np.concatenate(all_scores, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    latents = np.concatenate(all_latents, axis=0)
    
    if task == "binary":
        metrics = compute_binary_metrics(y_true, y_score, y_pred)
    else:
        metrics = compute_multiclass_metrics(y_true, y_pred)
        
    metrics["latents"] = latents
    return metrics


def train_xenon_cnn(
    model: XenonMultimodalCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    save_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Complete training orchestrator for XenonMultimodalCNN."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    
    task = "binary" if model.num_classes == 1 else "multiclass"
    criterion = nn.BCEWithLogitsLoss() if task == "binary" else nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_auc = -1.0
    best_state = None
    history = {"train_loss": [], "val_auc": [], "val_acc": []}
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, dev,
            has_physics_features=model.include_physics_features
        )
        scheduler.step()
        
        val_metrics = evaluate_model(
            model, val_loader, dev, task=task,
            has_physics_features=model.include_physics_features
        )
        val_auc = val_metrics.get("roc_auc", val_metrics.get("macro_f1", 0.0))
        val_acc = val_metrics.get("accuracy", 0.0)
        
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_dir / "best_model.pt")
                
        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | ({elapsed:.1f}s)")
            
    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate_model(
            model, test_loader, dev, task=task,
            has_physics_features=model.include_physics_features
        )
        
    return {
        "model": model,
        "history": history,
        "best_val_auc": best_val_auc,
        "test_metrics": test_metrics
    }
