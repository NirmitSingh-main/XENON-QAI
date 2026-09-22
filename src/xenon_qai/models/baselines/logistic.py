"""
Logistic Regression Baseline for XENON-QAI.

Simplest possible classifier as a lower-bound benchmark.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from xenon_qai.evaluation.metrics import compute_binary_metrics, compute_multiclass_metrics


class XenonLogisticRegression:
    """
    Logistic Regression baseline for XENON-QAI physics features.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
        task: str = "binary"
    ):
        self.task = task
        solver = "lbfgs"
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver
        )
        self.is_fitted: bool = False
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XenonLogisticRegression":
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[list] = None
    ) -> Dict[str, Any]:
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
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Path) -> "XenonLogisticRegression":
        with open(filepath, "rb") as f:
            return pickle.load(f)
