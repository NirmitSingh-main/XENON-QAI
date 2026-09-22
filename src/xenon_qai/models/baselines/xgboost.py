"""
XGBoost Baseline Model for XENON-QAI.

Gradient Boosted Decision Trees on 13D physics features and multimodal vectors.
Per dataanalysis.txt Section 9: Fast tabular benchmark with feature importance.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import pickle
import numpy as np

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import roc_auc_score
from xenon_qai.evaluation.metrics import compute_binary_metrics, compute_multiclass_metrics


class XenonXGBoost:
    """
    Gradient Boosted Decision Tree classifier for XENON-QAI.
    Uses XGBoost if available, with HistGradientBoostingClassifier fallback.
    """
    
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        task: str = "binary"  # "binary" or "multiclass"
    ):
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.is_fitted: bool = False
        
        if HAS_XGBOOST:
            if task == "binary":
                self.model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    scale_pos_weight=scale_pos_weight or 1.0,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    eval_metric="logloss"
                )
            else:
                self.model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    eval_metric="mlogloss"
                )
        else:
            if task == "binary":
                self.model = HistGradientBoostingClassifier(
                    max_iter=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=random_state
                )
            else:
                self.model = HistGradientBoostingClassifier(
                    max_iter=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=random_state
                )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[List[tuple]] = None
    ) -> "XenonXGBoost":
        """Fits the GBDT model."""
        if HAS_XGBOOST and eval_set is not None:
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
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
    
    def feature_importances(self) -> Optional[np.ndarray]:
        """Returns feature importance array if available."""
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None
    
    def save(self, filepath: Path) -> None:
        """Saves fitted model."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Path) -> "XenonXGBoost":
        """Loads saved model."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
