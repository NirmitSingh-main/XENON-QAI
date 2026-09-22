"""
Model Checkpointing and Artifact Management.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages saving and restoring model weights and training metadata.
    """
    
    def __init__(self, checkpoint_dir: Path, max_to_keep: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.saved_checkpoints: list = []
        
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        filename: Optional[str] = None
    ) -> Path:
        """Saves a checkpoint bundle."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"
        path = self.checkpoint_dir / filename
        
        payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "metrics": metrics or {}
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
            
        torch.save(payload, path)
        self.saved_checkpoints.append(path)
        return path
    
    def load_best(self, model: nn.Module, filename: str = "best_model.pt") -> Dict[str, Any]:
        """Loads best model checkpoint."""
        path = self.checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        return checkpoint
