"""
Device Selection Utility.
"""

from typing import Optional
import torch


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Returns appropriate torch.device (CUDA GPU, MPS, or CPU).
    """
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
