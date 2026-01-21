"""Mutation feature encoding from Chai-1 embeddings.

Extracts structural context around a mutation site using single and pair
representations from the Chai-1 trunk.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MutationFeatures:
    """Encoded features for a single mutation."""

    local_single: Tensor  # [D_single] - mutated residue embedding
    global_single: Tensor  # [D_single] - protein-wide average
    pair_global: Tensor  # [D_pair] - average pairwise interactions
    pair_local_seq: Tensor  # [D_pair] - sequence neighbor interactions
    pair_structural: Tensor  # [D_pair] - long-range contact interactions
    mutation_feat: Tensor  # [41] - WT/MUT one-hots + relative position


def encode_mutation(
    single: Tensor,
    pair: Tensor,
    position: int,
    wt_residue: int,
    mut_residue: int,
    k_structural: int = 10,
    seq_window: int = 5,
) -> MutationFeatures:
    """
    Encode mutation site features from Chai-1 embeddings.

    Args:
        single: Per-residue embeddings [L, D_single]
        pair: Pairwise interaction embeddings [L, L, D_pair]
        position: Mutation site (0-indexed)
        wt_residue: Wild-type amino acid index (0-19)
        mut_residue: Mutant amino acid index (0-19)
        k_structural: Number of top structural neighbors to aggregate
        seq_window: Half-width of sequence window for local interactions

    Returns:
        MutationFeatures with all encoded tensors
    """
    L = single.shape[0]
    i = position
    device = single.device

    # === Single track features ===
    local_single = single[i]  # [D_single]
    global_single = single.mean(dim=0)  # [D_single]

    # === Pair track features ===
    pair_row = pair[i, :, :]  # [L, D_pair]

    # Global: average interaction with all residues (excluding self)
    mask_global = torch.ones(L, dtype=torch.bool, device=device)
    mask_global[i] = False
    if mask_global.sum() > 0:
        pair_global = pair_row[mask_global].mean(dim=0)  # [D_pair]
    else:
        pair_global = pair_row.mean(dim=0)  # fallback for L=1

    # Local (sequence): interactions with sequence neighbors (excluding self)
    start = max(0, i - seq_window)
    end = min(L, i + seq_window + 1)
    local_indices = [j for j in range(start, end) if j != i]

    if local_indices:
        pair_local_seq = pair[i, local_indices, :].mean(dim=0)  # [D_pair]
    else:
        pair_local_seq = torch.zeros_like(pair_row[0])  # [D_pair]

    # Structural: top-k strongest interactions OUTSIDE local window
    # This captures long-range contacts distinct from pair_local_seq
    pair_magnitudes = pair_row.norm(dim=-1)  # [L]

    # Mask out self AND local window
    structural_mask = torch.ones(L, dtype=torch.bool, device=device)
    structural_mask[i] = False  # exclude self
    for j in range(start, end):
        structural_mask[j] = False  # exclude local window

    n_candidates = structural_mask.sum().item()
    k_actual = min(k_structural, n_candidates)

    if k_actual > 0:
        # Set masked positions to -inf so they're not selected by topk
        masked_magnitudes = pair_magnitudes.clone()
        masked_magnitudes[~structural_mask] = float("-inf")
        topk_indices = masked_magnitudes.topk(k=k_actual).indices
        pair_structural = pair[i, topk_indices, :].mean(dim=0)  # [D_pair]
    else:
        # Fallback for very short sequences: use global
        pair_structural = pair_global.clone()

    # === Mutation identity ===
    wt_onehot = F.one_hot(torch.tensor(wt_residue, device=device), 20).float()
    mut_onehot = F.one_hot(torch.tensor(mut_residue, device=device), 20).float()
    rel_position = torch.tensor([i / L], device=device, dtype=torch.float32)
    mutation_feat = torch.cat([wt_onehot, mut_onehot, rel_position])  # [41]

    return MutationFeatures(
        local_single=local_single,
        global_single=global_single,
        pair_global=pair_global,
        pair_local_seq=pair_local_seq,
        pair_structural=pair_structural,
        mutation_feat=mutation_feat,
    )


class EmbeddingCache:
    """Lazy-loading cache for protein embeddings."""

    def __init__(self, embedding_dir: Path | str):
        self.embedding_dir = Path(embedding_dir)
        self._cache: dict[str, tuple[Tensor, Tensor]] = {}

    def get(self, protein_name: str) -> tuple[Tensor, Tensor]:
        """
        Get (single, pair) embeddings for a protein.

        Returns:
            Tuple of (single [L, D_single], pair [L, L, D_pair]) as float32
        """
        if protein_name not in self._cache:
            path = self.embedding_dir / f"{protein_name}.pt"
            if not path.exists():
                raise FileNotFoundError(f"No embedding file for {protein_name} at {path}")

            data = torch.load(path, map_location="cpu", weights_only=False)
            # Convert to float32 (embeddings may be stored as float16)
            single = data["single"].float()
            pair = data["pair"].float()
            self._cache[protein_name] = (single, pair)

        return self._cache[protein_name]

    def preload(self, protein_names: list[str]) -> None:
        """Preload embeddings for a list of proteins."""
        for name in protein_names:
            self.get(name)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    @property
    def loaded_proteins(self) -> list[str]:
        """List of currently cached proteins."""
        return list(self._cache.keys())


def encode_batch(
    cache: EmbeddingCache,
    protein_names: list[str],
    positions: list[int],
    wt_residues: list[int],
    mut_residues: list[int],
    k_structural: int = 10,
    seq_window: int = 5,
) -> dict[str, Tensor]:
    """
    Encode a batch of mutations.

    Args:
        cache: EmbeddingCache instance
        protein_names: List of protein identifiers
        positions: List of mutation positions (0-indexed)
        wt_residues: List of WT residue indices (0-19)
        mut_residues: List of mutant residue indices (0-19)
        k_structural: Number of structural neighbors
        seq_window: Sequence window half-width

    Returns:
        Dict with batched feature tensors, each [B, D]
    """
    batch_features = {
        "local_single": [],
        "global_single": [],
        "pair_global": [],
        "pair_local_seq": [],
        "pair_structural": [],
        "mutation_feat": [],
    }

    for prot, pos, wt, mut in zip(protein_names, positions, wt_residues, mut_residues):
        single, pair = cache.get(prot)
        features = encode_mutation(
            single=single,
            pair=pair,
            position=pos,
            wt_residue=wt,
            mut_residue=mut,
            k_structural=k_structural,
            seq_window=seq_window,
        )

        batch_features["local_single"].append(features.local_single)
        batch_features["global_single"].append(features.global_single)
        batch_features["pair_global"].append(features.pair_global)
        batch_features["pair_local_seq"].append(features.pair_local_seq)
        batch_features["pair_structural"].append(features.pair_structural)
        batch_features["mutation_feat"].append(features.mutation_feat)

    # Stack and ensure float32 dtype
    return {k: torch.stack(v, dim=0).float() for k, v in batch_features.items()}
