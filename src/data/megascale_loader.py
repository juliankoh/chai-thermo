"""Shared MegaScale data loading utilities.

Provides consistent data loading across all training scripts with support for:
- HuggingFace MegaScale dataset
- Local parquet files
- ThermoMPNN official splits
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset

from src.data.dataset import MutationSample, parse_mutation, get_wt_sequence

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

DEFAULT_MEGASCALE_PATH = "data/megascale.parquet"


# =============================================================================
# Global Caches (avoid reloading large datasets)
# =============================================================================

_HF_DATASET_CACHE = None
_PARQUET_CACHE = None


def _get_hf_dataset():
    """Get cached HuggingFace dataset, loading once if needed."""
    global _HF_DATASET_CACHE
    if _HF_DATASET_CACHE is None:
        logger.info("Loading MegaScale dataset from HuggingFace (one-time)...")
        _HF_DATASET_CACHE = load_dataset("RosettaCommons/MegaScale", "dataset3_single_cv")
        logger.info(f"Dataset loaded. Splits: {list(_HF_DATASET_CACHE.keys())}")
    return _HF_DATASET_CACHE


def _get_parquet_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load MegaScale data from local parquet file (cached)."""
    global _PARQUET_CACHE

    if _PARQUET_CACHE is not None:
        return _PARQUET_CACHE

    path = Path(data_path) if data_path else Path(DEFAULT_MEGASCALE_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"MegaScale data not found at {path}. "
            f"Run 'python scripts/download_megascale.py' first."
        )

    logger.info(f"Loading MegaScale data from {path}...")
    _PARQUET_CACHE = pd.read_parquet(path)
    logger.info(
        f"Loaded {len(_PARQUET_CACHE)} mutations from "
        f"{_PARQUET_CACHE['WT_name'].nunique()} proteins"
    )

    return _PARQUET_CACHE


def clear_caches():
    """Clear all global caches (useful for testing or memory management)."""
    global _HF_DATASET_CACHE, _PARQUET_CACHE
    _HF_DATASET_CACHE = None
    _PARQUET_CACHE = None


# =============================================================================
# ThermoMPNN Splits Loading
# =============================================================================


def load_thermompnn_splits(splits_file: str | Path) -> dict:
    """Load the ThermoMPNN splits pickle file."""
    with open(splits_file, "rb") as f:
        return pickle.load(f)


def get_split_proteins(
    splits: dict,
    split: str,
    cv_fold: Optional[int] = None,
) -> set[str]:
    """
    Get protein names for a ThermoMPNN split.

    Args:
        splits: Loaded splits dict from load_thermompnn_splits()
        split: One of 'train', 'val', 'test'
        cv_fold: If provided, use cv_train_{fold}, cv_val_{fold}, cv_test_{fold}
                If None, use the main train/val/test split

    Returns:
        Set of protein names (without .pdb extension)
    """
    if cv_fold is not None:
        split_key = f"cv_{split}_{cv_fold}"
    else:
        split_key = split

    if split_key not in splits:
        available = list(splits.keys())
        raise ValueError(f"Split '{split_key}' not found. Available: {available}")

    # Normalize protein names (remove .pdb extension, handle numpy arrays)
    target_proteins = set()
    for p in splits[split_key]:
        if hasattr(p, "item"):
            p = p.item()
        target_proteins.add(p.replace(".pdb", ""))

    return target_proteins


# =============================================================================
# ThermoMPNN Split Dataset (using HuggingFace)
# =============================================================================


class ThermoMPNNSplitDatasetHF(Dataset):
    """
    MegaScale dataset using ThermoMPNN's official splits.

    Uses HuggingFace dataset as the data source. This is the original
    implementation from train.py.
    """

    def __init__(
        self,
        splits_file: str | Path,
        split: str = "train",
        cv_fold: Optional[int] = None,
        _preloaded_splits: Optional[dict] = None,
    ):
        """
        Load MegaScale filtered to ThermoMPNN's splits.

        Args:
            splits_file: Path to mega_splits.pkl from ThermoMPNN
            split: One of 'train', 'val', 'test'
            cv_fold: If provided, use CV splits; if None, use main split
            _preloaded_splits: Pre-loaded splits dict (for efficiency)
        """
        self.splits_file = Path(splits_file)
        self.split = split

        # Load splits
        if _preloaded_splits is not None:
            splits = _preloaded_splits
        else:
            splits = load_thermompnn_splits(splits_file)

        # Get target proteins
        target_proteins = get_split_proteins(splits, split, cv_fold)
        split_key = f"cv_{split}_{cv_fold}" if cv_fold is not None else split
        logger.info(f"Split '{split_key}': {len(target_proteins)} proteins")

        # Get cached HuggingFace dataset
        ds = _get_hf_dataset()

        # Collect all mutations for target proteins (deduplicated)
        all_rows = []
        seen = set()

        for hf_split in ds.keys():
            for row in ds[hf_split]:
                key = (row["WT_name"], row["mut_type"])
                if key not in seen:
                    seen.add(key)
                    wt_name = row["WT_name"].replace(".pdb", "")
                    if wt_name in target_proteins:
                        all_rows.append(row)

        self.data = all_rows
        logger.info(f"Found {len(self.data)} mutations for {split_key}")

        # Build WT sequence cache
        self._wt_cache: dict[str, str] = {}
        self._build_wt_cache()

    def _build_wt_cache(self) -> None:
        """Build cache of WT sequences for each protein."""
        for row in self.data:
            wt_name = row["WT_name"]
            if wt_name not in self._wt_cache:
                self._wt_cache[wt_name] = get_wt_sequence(row["aa_seq"], row["mut_type"])

    def get_wt_sequence(self, wt_name: str) -> str:
        return self._wt_cache[wt_name]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MutationSample:
        row = self.data[idx]
        wt_aa, pos_1idx, mut_aa = parse_mutation(row["mut_type"])
        ddg_standard = -row["ddG_ML"]  # Flip sign to standard convention

        return MutationSample(
            wt_name=row["WT_name"],
            wt_sequence=self.get_wt_sequence(row["WT_name"]),
            mut_sequence=row["aa_seq"],
            position=pos_1idx - 1,
            wt_residue=wt_aa,
            mut_residue=mut_aa,
            ddg=ddg_standard,
        )

    @property
    def unique_proteins(self) -> list[str]:
        return list(self._wt_cache.keys())

    @property
    def wt_sequences(self) -> dict[str, str]:
        return dict(self._wt_cache)

    def encode_residue(self, aa: str) -> int:
        return AA_TO_IDX.get(aa, 20)


# =============================================================================
# ThermoMPNN Split Dataset (using Parquet)
# =============================================================================


class ThermoMPNNSplitDatasetParquet(Dataset):
    """
    MegaScale dataset using ThermoMPNN's official splits.

    Uses local parquet file as the data source. Faster than HuggingFace
    for repeated access.
    """

    def __init__(
        self,
        splits_file: str | Path,
        split: str = "train",
        cv_fold: Optional[int] = None,
        data_path: str | Path | None = None,
    ):
        """
        Load MegaScale filtered to ThermoMPNN's splits.

        Args:
            splits_file: Path to mega_splits.pkl from ThermoMPNN
            split: One of 'train', 'val', 'test'
            cv_fold: If provided, use CV splits; if None, use main split
            data_path: Path to local parquet file (default: data/megascale.parquet)
        """
        self.splits_file = Path(splits_file)
        self.split = split

        # Load splits
        splits = load_thermompnn_splits(splits_file)

        # Get target proteins
        target_proteins = get_split_proteins(splits, split, cv_fold)
        split_key = f"cv_{split}_{cv_fold}" if cv_fold is not None else split
        logger.info(f"Split '{split_key}': {len(target_proteins)} proteins")

        # Load from local parquet file
        df = _get_parquet_data(data_path)

        # Filter to target proteins
        df_filtered = df[
            df["WT_name"].str.replace(".pdb", "", regex=False).isin(target_proteins)
        ]

        # Convert to list of dicts for compatibility
        self.data = df_filtered.to_dict("records")
        logger.info(f"Found {len(self.data)} mutations for {split_key}")

        # Build WT sequence cache
        self._wt_cache: dict[str, str] = {}
        self._build_wt_cache()

    def _build_wt_cache(self) -> None:
        for row in self.data:
            wt_name = row["WT_name"]
            if wt_name not in self._wt_cache:
                self._wt_cache[wt_name] = get_wt_sequence(row["aa_seq"], row["mut_type"])

    def get_wt_sequence(self, wt_name: str) -> str:
        return self._wt_cache[wt_name]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MutationSample:
        row = self.data[idx]
        wt_aa, pos_1idx, mut_aa = parse_mutation(row["mut_type"])
        ddg_standard = -row["ddG_ML"]

        return MutationSample(
            wt_name=row["WT_name"],
            wt_sequence=self.get_wt_sequence(row["WT_name"]),
            mut_sequence=row["aa_seq"],
            position=pos_1idx - 1,
            wt_residue=wt_aa,
            mut_residue=mut_aa,
            ddg=ddg_standard,
        )

    @property
    def unique_proteins(self) -> list[str]:
        return list(self._wt_cache.keys())

    @property
    def wt_sequences(self) -> dict[str, str]:
        return dict(self._wt_cache)

    def encode_residue(self, aa: str) -> int:
        return AA_TO_IDX.get(aa, 20)


# =============================================================================
# Convenience Functions
# =============================================================================


def load_thermompnn_mutations(
    splits_file: Path,
    split: str,
    cv_fold: Optional[int],
    data_path: Path | None = None,
) -> list[MutationSample]:
    """
    Load mutations for a ThermoMPNN split as a list of MutationSample.

    This is useful for datasets that need raw samples rather than a Dataset object.

    Args:
        splits_file: Path to mega_splits.pkl
        split: One of 'train', 'val', 'test'
        cv_fold: CV fold number (0-4) or None for main split
        data_path: Path to local parquet file (uses parquet if provided)

    Returns:
        List of MutationSample objects
    """
    if data_path is not None:
        dataset = ThermoMPNNSplitDatasetParquet(
            splits_file, split, cv_fold, data_path
        )
    else:
        dataset = ThermoMPNNSplitDatasetHF(splits_file, split, cv_fold)

    return [dataset[i] for i in range(len(dataset))]
