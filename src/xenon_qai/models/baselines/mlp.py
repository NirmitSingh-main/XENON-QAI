"""
Multilayer Perceptron (MLP) Baseline for XENON-QAI.

Implements PyTorch-based neural network for tabular physics features (13D)
and concatenated multimodal representations (908D).
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

from xenon_qai.evaluation.metrics import compute_binary_metrics, compute_multiclass_metrics


class XenonMLPModule(nn.Module):
    """
    Feedforward neural network with LayerNorm, GELU/ReLU, and Dropout.
    Can extract intermediate latent representations (e.g. 32D / 64D).
    """
    
    def __init__(
        self,
        input_dim: int = 13,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        output_dim: int = 1,
        dropout: float = 0.1,
        use_layernorm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        layers: List[nn.Module] = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        self.latent_dim = prev_dim
        self.head = nn.Linear(prev_dim, output_dim)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent feature representation before classification head."""
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        latent = self.encoder(x)
        return self.head(latent)


class XenonMLP:
    """
    High-level scikit-learn style wrapper around XenonMLPModule.
    Handles training loop, early stopping, evaluation, and latent extraction.
    """
    
    def __init__(
        self,
        input_dim: int = 13,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        num_classes: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 50,
        device: Optional[str] = None,
        task: str = "binary"  # "binary" or "multiclass"
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.task = task
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        output_dim = 1 if task == "binary" else num_classes
        self.network = XenonMLPModule(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        ).to(self.device)
        
        self.is_fitted: bool = False
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> "XenonMLP":
        """Trains the MLP model."""
        self.network.train()
        
        X_t = torch.tensor(X_train, dtype=torch.float32)
        if self.task == "binary":
            y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            y_t = torch.tensor(y_train, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
        
        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        best_loss = float("inf")
        best_state = None
        
        for epoch in range(self.epochs):
            self.network.train()
            total_loss = 0.0
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.network(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_x)
            
            scheduler.step()
            avg_train_loss = total_loss / len(dataset)
            
            if X_val is not None and y_val is not None:
                val_loss = self._compute_val_loss(X_val, y_val, criterion)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
        
        if best_state is not None:
            self.network.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        
        self.is_fitted = True
        return self
    
    def _compute_val_loss(self, X_val: np.ndarray, y_val: np.ndarray, criterion: nn.Module) -> float:
        self.network.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            if self.task == "binary":
                y_t = torch.tensor(y_val, dtype=torch.float32, device=self.device).unsqueeze(1)
            else:
                y_t = torch.tensor(y_val, dtype=torch.long, device=self.device)
            logits = self.network(x_t)
            loss = criterion(logits, y_t)
            return float(loss.item())
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class probabilities."""
        self.network.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.network(x_t)
            if self.task == "binary":
                probs_pos = torch.sigmoid(logits).cpu().numpy().squeeze(1)
                probs_neg = 1.0 - probs_pos
                return np.column_stack([probs_neg, probs_pos])
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns hard class predictions."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def extract_latent(self, X: np.ndarray) -> np.ndarray:
        """Extracts latent embeddings from the encoder layer."""
        self.network.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            latent = self.network.extract_features(x_t)
            return latent.cpu().numpy()
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluates using physics-relevant metrics."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        if self.task == "binary":
            y_score = y_proba[:, 1]
            return compute_binary_metrics(y_test, y_score, y_pred)
        else:
            metrics = compute_multiclass_metrics(y_test, y_pred, class_names)
            try:
                metrics["roc_auc_ovr"] = float(roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="weighted"
                ))
            except ValueError:
                metrics["roc_auc_ovr"] = 0.0
            return metrics
    
    def save(self, filepath: Path) -> None:
        """Saves model state and configuration."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.network.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "task": self.task
        }, filepath)
    
    @classmethod
    def load(cls, filepath: Path, device: Optional[str] = None) -> "XenonMLP":
        """Loads model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            num_classes=checkpoint["num_classes"],
            dropout=checkpoint["dropout"],
            task=checkpoint["task"],
            device=device
        )
        model.network.load_state_dict(checkpoint["state_dict"])
        model.is_fitted = True
        return model
