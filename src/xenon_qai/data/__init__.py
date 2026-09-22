"""
XENON-QAI Data Pipeline Package.
"""

from xenon_qai.data.types import (
    ALL_CLASSES,
    BACKGROUND_CLASSES,
    WIMP_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    PHYSICS_FEATURE_NAMES,
    PHYSICS_FEATURE_DIM,
    MULTIMODAL_DIM,
    S1_MAX_LEN,
    S2_TARGET_LEN,
    TOP_PMT_COUNT,
    BOT_PMT_COUNT,
    DatasetConfig
)
from xenon_qai.data.features import compute_physics_features, extract_features_from_hdf5_event
from xenon_qai.data.preprocessing import (
    pad_s1_waveform,
    resample_s2_waveform,
    build_multimodal_vector,
    XenonPreprocessor
)
from xenon_qai.data.splitting import (
    get_sorted_event_keys,
    get_hdf5_file_for_class,
    create_stratified_split,
    save_split_manifest,
    load_split_manifest
)
from xenon_qai.data.dataset import XenonHDF5Dataset
from xenon_qai.data.loader import create_dataloaders
from xenon_qai.data.validation import validate_split_integrity, validate_batch

__all__ = [
    "ALL_CLASSES",
    "BACKGROUND_CLASSES",
    "WIMP_CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "PHYSICS_FEATURE_NAMES",
    "PHYSICS_FEATURE_DIM",
    "MULTIMODAL_DIM",
    "S1_MAX_LEN",
    "S2_TARGET_LEN",
    "TOP_PMT_COUNT",
    "BOT_PMT_COUNT",
    "DatasetConfig",
    "compute_physics_features",
    "extract_features_from_hdf5_event",
    "pad_s1_waveform",
    "resample_s2_waveform",
    "build_multimodal_vector",
    "XenonPreprocessor",
    "get_sorted_event_keys",
    "get_hdf5_file_for_class",
    "create_stratified_split",
    "save_split_manifest",
    "load_split_manifest",
    "XenonHDF5Dataset",
    "create_dataloaders",
    "validate_split_integrity",
    "validate_batch"
]
