"""
Dimensionality Reduction for Learned Latent Embeddings.
"""

from typing import Optional, Dict, Any
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class LatentReducer:
    """
    Reduces high-dimensional latent vectors z in R^D to 2D/3D for visualization.
    """
    
    def __init__(self, method: str = "pca", n_components: int = 2, random_state: int = 42):
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        
        if self.method == "pca":
            self.model = PCA(n_components=n_components, random_state=random_state)
        elif self.method == "tsne":
            self.model = TSNE(n_components=n_components, random_state=random_state, perplexity=30.0)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
            
        self.is_fitted: bool = False
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        embedding = self.model.fit_transform(X)
        self.is_fitted = True
        return embedding
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == "pca":
            return self.model.transform(X)
        else:
            raise NotImplementedError("t-SNE does not support transform on unseen data.")
