"""MegaScale dataset loading and preprocessing for stability prediction.

NOTE ON SIGN CONVENTION:
    MegaScale raw data uses INVERTED sign: ΔΔG = ΔG(WT) - ΔG(mutant)
        - Positive = stabilizing (Stabilizing_mut=True has positive ddG)
        - Negative = destabilizing

    This module converts to STANDARD convention: ΔΔG = ΔG(mutant) - ΔG(WT)
        - Positive = destabilizing
        - Negative = stabilizing

    We flip the sign when loading so downstream code uses standard convention.
"""

from dataclasses import dataclass
from typing import Iterator

import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset, DataLoader


@dataclass
class MutationSample:
    """A single mutation sample."""

    wt_name: str  # protein identifier
    wt_sequence: str  # wild-type sequence
    mut_sequence: str  # mutant sequence
    position: int  # 0-indexed mutation position
    wt_residue: str  # original amino acid
    mut_residue: str  # mutated amino acid
    ddg: float  # ΔΔG in kcal/mol (STANDARD: positive = destabilizing)


def parse_mutation(mut_type: str) -> tuple[str, int, str]:
    """Parse mutation string like 'E1Q' -> (wt_aa, position_1indexed, mut_aa)."""
    wt_aa = mut_type[0]
    mut_aa = mut_type[-1]
    pos = int(mut_type[1:-1])
    return wt_aa, pos, mut_aa


def get_wt_sequence(mutant_seq: str, mut_type: str) -> str:
    """Reconstruct WT sequence from mutant sequence and mutation type."""
    wt_aa, pos, mut_aa = parse_mutation(mut_type)
    assert mutant_seq[pos - 1] == mut_aa, f"Expected {mut_aa} at pos {pos}, got {mutant_seq[pos-1]}"
    return mutant_seq[: pos - 1] + wt_aa + mutant_seq[pos:]


class MegaScaleDataset(Dataset):
    """PyTorch Dataset for MegaScale stability data."""

    # Standard amino acid vocabulary
    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

    def __init__(
        self,
        fold: int = 0,
        split: str = "train",
        cache_wt_sequences: bool = True,
    ):
        """
        Load MegaScale dataset for a specific fold and split.

        Args:
            fold: CV fold (0-4)
            split: One of 'train', 'val', 'test'
            cache_wt_sequences: Whether to cache WT sequences (recommended)
        """
        assert 0 <= fold <= 4, f"fold must be 0-4, got {fold}"
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"

        self.fold = fold
        self.split = split

        # Load from HuggingFace
        ds = load_dataset("RosettaCommons/MegaScale", "dataset3_single_cv")
        self.data = ds[f"{split}_{fold}"]

        # Cache WT sequences per protein (reconstruct from first mutation seen)
        self._wt_cache: dict[str, str] = {}
        if cache_wt_sequences:
            self._build_wt_cache()

    def _build_wt_cache(self) -> None:
        """Build cache of WT sequences for each protein."""
        for i in range(len(self.data)):
            wt_name = self.data[i]["WT_name"]
            if wt_name not in self._wt_cache:
                mut_seq = self.data[i]["aa_seq"]
                mut_type = self.data[i]["mut_type"]
                self._wt_cache[wt_name] = get_wt_sequence(mut_seq, mut_type)

    def get_wt_sequence(self, wt_name: str) -> str:
        """Get cached WT sequence for a protein."""
        if wt_name in self._wt_cache:
            return self._wt_cache[wt_name]
        # Fallback: find first mutation for this protein and reconstruct
        for i in range(len(self.data)):
            if self.data[i]["WT_name"] == wt_name:
                mut_seq = self.data[i]["aa_seq"]
                mut_type = self.data[i]["mut_type"]
                wt_seq = get_wt_sequence(mut_seq, mut_type)
                self._wt_cache[wt_name] = wt_seq
                return wt_seq
        raise ValueError(f"Protein {wt_name} not found in dataset")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MutationSample:
        row = self.data[idx]
        wt_aa, pos_1idx, mut_aa = parse_mutation(row["mut_type"])

        # Flip sign to convert from MegaScale convention to standard convention
        # MegaScale: positive = stabilizing, standard: positive = destabilizing
        ddg_standard = -row["ddG_ML"]

        return MutationSample(
            wt_name=row["WT_name"],
            wt_sequence=self.get_wt_sequence(row["WT_name"]),
            mut_sequence=row["aa_seq"],
            position=pos_1idx - 1,  # convert to 0-indexed
            wt_residue=wt_aa,
            mut_residue=mut_aa,
            ddg=ddg_standard,
        )

    @property
    def unique_proteins(self) -> list[str]:
        """Get list of unique protein names in this split."""
        return list(self._wt_cache.keys())

    @property
    def wt_sequences(self) -> dict[str, str]:
        """Get all WT sequences as {protein_name: sequence}."""
        return dict(self._wt_cache)

    def encode_residue(self, aa: str) -> int:
        """Encode amino acid to index (0-19)."""
        return self.AA_TO_IDX.get(aa, 20)  # 20 for unknown

    def get_protein_mutations(self, wt_name: str) -> list[int]:
        """Get all indices for mutations of a specific protein."""
        return [i for i in range(len(self.data)) if self.data[i]["WT_name"] == wt_name]


def collate_mutations(samples: list[MutationSample]) -> dict:
    """Collate mutation samples into a batch dict."""
    return {
        "wt_names": [s.wt_name for s in samples],
        "wt_sequences": [s.wt_sequence for s in samples],
        "positions": torch.tensor([s.position for s in samples], dtype=torch.long),
        "wt_residues": torch.tensor(
            [MegaScaleDataset.AA_TO_IDX.get(s.wt_residue, 20) for s in samples],
            dtype=torch.long,
        ),
        "mut_residues": torch.tensor(
            [MegaScaleDataset.AA_TO_IDX.get(s.mut_residue, 20) for s in samples],
            dtype=torch.long,
        ),
        "ddg": torch.tensor([s.ddg for s in samples], dtype=torch.float32),
    }


def get_dataloader(
    fold: int = 0,
    split: str = "train",
    batch_size: int = 128,
    shuffle: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for MegaScale data."""
    dataset = MegaScaleDataset(fold=fold, split=split)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_mutations,
    )
