"""
Candidate Event Ranking and Priority Ordering Engine.

Per README.md and scientific objective:
Ranks liquid xenon events by their dark matter compatibility and anomaly scores,
producing a prioritized audit list for detector physicists.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd


class CandidateRanker:
    """
    Ranks events by combined dark matter compatibility and anomaly scores.
    """
    
    def __init__(self, weight_candidate: float = 0.7, weight_anomaly: float = 0.3):
        self.weight_candidate = weight_candidate
        self.weight_anomaly = weight_anomaly
        
    def rank_events(
        self,
        event_ids: List[Union[str, int]],
        candidate_scores: np.ndarray,
        anomaly_scores: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Produces a ranked DataFrame of candidate events sorted by Priority Score.
        """
        n_events = len(event_ids)
        c_scores = np.asarray(candidate_scores)
        
        if anomaly_scores is not None:
            a_scores = np.asarray(anomaly_scores)
            priority_score = self.weight_candidate * c_scores + self.weight_anomaly * a_scores
        else:
            a_scores = np.zeros(n_events)
            priority_score = c_scores
            
        data = {
            "rank": np.zeros(n_events, dtype=int),
            "event_id": event_ids,
            "priority_score": priority_score,
            "candidate_score": c_scores,
            "anomaly_score": a_scores
        }
        
        if metadata is not None:
            for key in metadata[0].keys() if metadata else []:
                data[key] = [m.get(key) for m in metadata]
                
        df = pd.DataFrame(data)
        df = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        
        if top_k is not None:
            df = df.head(top_k)
            
        return df
