"""
XENON-QAI Data Types and Constants Specification.
Based on research findings in dataanalysis.txt.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# All 12 classes in the XENON-QAI simulated dataset
ALL_CLASSES: List[str] = [
    "AC",
    "CNNS",
    "ER",
    "RN",
    "WALL",
    "mDM10GeV",
    "mDM20GeV",
    "mDM35GeV",
    "mDM50GeV",
    "mDM100GeV",
    "mDM200GeV",
    "mDM500GeV"
]

# Physical background categories
BACKGROUND_CLASSES: List[str] = [
    "AC",
    "CNNS",
    "ER",
    "RN",
    "WALL"
]

# Dark matter WIMP signal categories
WIMP_CLASSES: List[str] = [
    "mDM10GeV",
    "mDM20GeV",
    "mDM35GeV",
    "mDM50GeV",
    "mDM100GeV",
    "mDM200GeV",
    "mDM500GeV"
]

# Numerical class mappings (0..11)
CLASS_TO_IDX: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(ALL_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls_name for i, cls_name in enumerate(ALL_CLASSES)}

# 6-class high-level physics mapping
PHYSICS_CLASS_MAP: Dict[str, int] = {
    "AC": 0,
    "CNNS": 1,
    "ER": 2,
    "RN": 3,
    "WALL": 4,
    "mDM10GeV": 5,
    "mDM20GeV": 5,
    "mDM35GeV": 5,
    "mDM50GeV": 5,
    "mDM100GeV": 5,
    "mDM200GeV": 5,
    "mDM500GeV": 5,
}

PHYSICS_CLASS_NAMES: List[str] = ["AC", "CNNS", "ER", "RN", "WALL", "WIMP"]

# Verified input dimensions per dataanalysis.txt
S1_MAX_LEN: int = 160        # Padded fixed length for peakS1_onlyhits (observed max length: 155)
S2_TARGET_LEN: int = 500     # Resampled/interpolated length for peakS2_onlyhits
TOP_PMT_COUNT: int = 127     # HITS_top physical sensor array count
BOT_PMT_COUNT: int = 121     # HITS_bot physical sensor array count
TOTAL_PMT_COUNT: int = 248   # Total physical PMTs (127 + 121)

# Multimodal vector total dimension: 160 + 500 + 127 + 121 = 908
MULTIMODAL_DIM: int = S1_MAX_LEN + S2_TARGET_LEN + TOP_PMT_COUNT + BOT_PMT_COUNT  # 908

# 13 Physic-derived tabular feature names
PHYSICS_FEATURE_NAMES: List[str] = [
    "s1_area_hits",    # Integrated charge of S1 pulse (PE)
    "s2_area_hits",    # Integrated charge of S2 pulse (PE)
    "s1_area_raw",     # Integrated charge of raw S1 pulse (PE)
    "s2_area_raw",     # Integrated charge of raw S2 pulse (PE)
    "log_s2_s1",       # log10((S2_area + 1e-3) / (S1_area + 1e-3))
    "s1_len",          # Temporal bin duration of S1
    "s2_len",          # Temporal bin duration of S2
    "s1_max",          # Peak amplitude of S1 pulse
    "s2_max",          # Peak amplitude of S2 pulse
    "top_sum",         # Total integrated charge on top PMT array
    "bot_sum",         # Total integrated charge on bottom PMT array
    "total_pmt_hits",  # top_sum + bot_sum
    "top_bot_asym",    # (top_sum - bot_sum) / (total_pmt_hits + 1e-6)
]
PHYSICS_FEATURE_DIM: int = len(PHYSICS_FEATURE_NAMES)  # 13

@dataclass
class DatasetConfig:
    data_dir: Path = Path("data/extracted")
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    random_seed: int = 42
    s1_max_len: int = S1_MAX_LEN
    s2_target_len: int = S2_TARGET_LEN
    classes: List[str] = field(default_factory=lambda: list(ALL_CLASSES))
    mode: str = "multimodal"  # 'multimodal' (908D), 'physics' (13D), or 'both'
    normalize: bool = True
    use_onlyhits: bool = True
