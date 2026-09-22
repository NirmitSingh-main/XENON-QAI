"""
Random Forest Baseline for XENON-QAI.

Per dataanalysis.txt Section 9: First model on 13D physics features.
Reported baseline: ROC-AUC ~ 0.8485 (all-signal vs all-background).
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from xenon_qai.evaluation.metrics import compute_binary_metrics, compute_multiclass_metrics


class XenonRandomForest:
    """
    Random Forest baseline classifier for XENON-QAI physics features.
    Supports both binary (signal vs background) and multi-class classification.
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        task: str = "binary"  # "binary" or "multiclass"
    ):
        self.task = task
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.is_fitted: bool = False
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XenonRandomForest":
        """Fits the Random Forest on training data."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns hard class predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities."""
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Evaluates the model on test data using physics-relevant metrics.
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        if self.task == "binary":
            y_score = y_proba[:, 1]  # Probability of signal class
            return compute_binary_metrics(y_test, y_score, y_pred)
        else:
            metrics = compute_multiclass_metrics(y_test, y_pred, class_names)
            # Add OVR AUC for multiclass
            try:
                metrics["roc_auc_ovr"] = float(roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="weighted"
                ))
            except ValueError:
                metrics["roc_auc_ovr"] = 0.0
            return metrics
    
    def feature_importances(self) -> np.ndarray:
        """Returns feature importance scores (Gini importance)."""
        return self.model.feature_importances_
    
    def save(self, filepath: Path) -> None:
        """Saves the fitted model."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Path) -> "XenonRandomForest":
        """Loads a saved model."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
