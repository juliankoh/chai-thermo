"""Balanced sampling strategies for protein stability training.

Prevents large proteins (many mutations) from dominating gradients by
sampling uniformly across proteins, then sampling mutations within each.
"""

import random
from collections import defaultdict
from typing import Iterator

import torch
from torch.utils.data import Sampler

from src.data.dataset import MegaScaleDataset


class BalancedProteinSampler(Sampler[int]):
    """
    Sampler that balances across proteins.

    Each epoch:
    1. Shuffles proteins
    2. For each protein, samples up to `variants_per_protein` mutations
    3. Ensures all proteins contribute roughly equally to each epoch

    This prevents proteins with 1000+ mutations from dominating training
    over proteins with only 50 mutations.
    """

    def __init__(
        self,
        dataset: MegaScaleDataset,
        variants_per_protein: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        """
        Args:
            dataset: MegaScaleDataset instance
            variants_per_protein: Max mutations to sample per protein per epoch
            shuffle: Whether to shuffle proteins and mutations
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.variants_per_protein = variants_per_protein
        self.shuffle = shuffle

        self.rng = random.Random(seed)

        # Build protein -> indices mapping
        self.protein_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset.data[idx]
            self.protein_to_indices[sample["WT_name"]].append(idx)

        self.proteins = list(self.protein_to_indices.keys())
        self._epoch_indices: list[int] = []
        self._regenerate_indices()

    def _regenerate_indices(self) -> None:
        """Regenerate indices for a new epoch."""
        self._epoch_indices = []

        proteins = self.proteins.copy()
        if self.shuffle:
            self.rng.shuffle(proteins)

        for protein in proteins:
            indices = self.protein_to_indices[protein].copy()
            if self.shuffle:
                self.rng.shuffle(indices)

            # Take up to variants_per_protein
            selected = indices[: self.variants_per_protein]
            self._epoch_indices.extend(selected)

    def __iter__(self) -> Iterator[int]:
        self._regenerate_indices()
        return iter(self._epoch_indices)

    def __len__(self) -> int:
        # Approximate length (exact value set after _regenerate_indices)
        return len(self.proteins) * self.variants_per_protein

    @property
    def num_proteins(self) -> int:
        return len(self.proteins)


class FullDatasetSampler(Sampler[int]):
    """
    Standard sampler that goes through all data points.

    Use this for validation/test to ensure all mutations are evaluated.
    """

    def __init__(self, dataset: MegaScaleDataset, shuffle: bool = False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            indices = self.indices.copy()
            random.shuffle(indices)
            return iter(indices)
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.dataset)


class StratifiedProteinBatchSampler(Sampler[list[int]]):
    """
    Batch sampler that ensures each batch contains mutations from multiple proteins.

    Useful when you want diversity within each batch, not just across the epoch.
    """

    def __init__(
        self,
        dataset: MegaScaleDataset,
        batch_size: int = 128,
        proteins_per_batch: int = 16,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
    ):
        """
        Args:
            dataset: MegaScaleDataset instance
            batch_size: Total batch size
            proteins_per_batch: Number of proteins to sample per batch
            shuffle: Whether to shuffle
            seed: Random seed
            drop_last: Drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.proteins_per_batch = proteins_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.rng = random.Random(seed)

        # Build protein -> indices mapping
        self.protein_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset.data[idx]
            self.protein_to_indices[sample["WT_name"]].append(idx)

        self.proteins = list(self.protein_to_indices.keys())

        # Mutations per protein per batch
        self.mutations_per_protein = max(1, batch_size // proteins_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        # Copy and optionally shuffle proteins
        proteins = self.proteins.copy()
        if self.shuffle:
            self.rng.shuffle(proteins)

        # Copy indices and shuffle within each protein
        protein_indices = {
            p: self.protein_to_indices[p].copy() for p in proteins
        }
        if self.shuffle:
            for indices in protein_indices.values():
                self.rng.shuffle(indices)

        # Track position within each protein
        protein_positions = {p: 0 for p in proteins}

        # Generate batches
        batches = []
        protein_idx = 0

        while True:
            batch = []
            proteins_used = 0

            # Try to fill batch with proteins_per_batch proteins
            start_protein_idx = protein_idx
            while proteins_used < self.proteins_per_batch and len(batch) < self.batch_size:
                protein = proteins[protein_idx % len(proteins)]
                indices = protein_indices[protein]
                pos = protein_positions[protein]

                # Get mutations from this protein
                n_take = min(
                    self.mutations_per_protein,
                    len(indices) - pos,
                    self.batch_size - len(batch),
                )

                if n_take > 0:
                    batch.extend(indices[pos : pos + n_take])
                    protein_positions[protein] = pos + n_take

                proteins_used += 1
                protein_idx += 1

                # Check if we've cycled through all proteins
                if protein_idx - start_protein_idx >= len(proteins):
                    break

            if not batch:
                break

            if len(batch) >= self.batch_size or not self.drop_last:
                if len(batch) > 0:
                    batches.append(batch)

            # Check if all proteins are exhausted
            all_exhausted = all(
                protein_positions[p] >= len(protein_indices[p])
                for p in proteins
            )
            if all_exhausted:
                break

        return iter(batches)

    def __len__(self) -> int:
        # Approximate
        total_mutations = len(self.dataset)
        return (total_mutations + self.batch_size - 1) // self.batch_size
