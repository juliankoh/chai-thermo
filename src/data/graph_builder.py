"""Graph construction utilities for MPNN-based stability prediction.

Builds PyTorch Geometric Data objects from Chai-1 embeddings, creating
local subgraphs around mutation sites.
"""

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Data


def build_mutation_graph(
    single: Tensor,
    pair: Tensor,
    position: int,
    wt_residue: int,
    mut_residue: int,
    k_neighbors: int = 30,
    ddg: Optional[float] = None,
) -> Data:
    """
    Build a PyG Data object for a single mutation.

    Creates a subgraph centered on the mutation site with its k-nearest neighbors.
    Neighbors are selected based on pair embedding magnitude (proxy for structural proximity).
    The mutated residue is always placed at node index 0.

    Args:
        single: Per-residue embeddings [L, D_single]
        pair: Pairwise interaction embeddings [L, L, D_pair]
        position: Mutation site (0-indexed)
        wt_residue: Wild-type amino acid index (0-19)
        mut_residue: Mutant amino acid index (0-19)
        k_neighbors: Number of neighbors to include (excluding mutation site)
        ddg: Optional target value

    Returns:
        PyG Data object with:
            - x: [K+1, D_single] node features (mutation site + k neighbors)
            - edge_index: [2, (K+1)*(K)] edges (fully connected)
            - edge_attr: [num_edges, D_pair] edge features
            - wt_idx, mut_idx: amino acid indices
            - rel_pos: relative position in sequence
            - y: target ddG (if provided)
    """
    L = single.shape[0]
    device = single.device

    # Select k-nearest neighbors based on pair embedding magnitude
    pair_row = pair[position]  # [L, D_pair]
    pair_magnitudes = pair_row.norm(dim=-1)  # [L]

    # Mask out the mutation site itself
    pair_magnitudes[position] = float("-inf")

    # Get top-k neighbors
    k_actual = min(k_neighbors, L - 1)
    neighbor_indices = pair_magnitudes.topk(k=k_actual).indices  # [k_actual]

    # Build node list: mutation site first, then neighbors
    # This ensures the mutated residue is always at index 0
    node_indices = torch.cat([
        torch.tensor([position], device=device),
        neighbor_indices,
    ])  # [K+1]

    num_nodes = len(node_indices)

    # Extract node features (single embeddings)
    x = single[node_indices]  # [K+1, D_single]

    # Build fully connected edge index (excluding self-loops) - fully vectorized
    # Create all pairs (i, j) where i != j using meshgrid
    arange = torch.arange(num_nodes, device=device)
    grid_i, grid_j = torch.meshgrid(arange, arange, indexing='ij')
    mask = grid_i != grid_j  # Exclude self-loops
    src = grid_i[mask]  # [E]
    dst = grid_j[mask]  # [E]
    edge_index = torch.stack([src, dst])  # [2, E]

    # Extract edge features (pair embeddings) - vectorized
    # Map from local indices to global indices
    global_src = node_indices[src]  # [E]
    global_dst = node_indices[dst]  # [E]
    edge_attr = pair[global_src, global_dst]  # [E, D_pair]

    # Mutation info
    wt_idx = torch.tensor(wt_residue, dtype=torch.long, device=device)
    mut_idx = torch.tensor(mut_residue, dtype=torch.long, device=device)
    rel_pos = torch.tensor(position / L, dtype=torch.float32, device=device)

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        wt_idx=wt_idx,
        mut_idx=mut_idx,
        rel_pos=rel_pos,
    )

    if ddg is not None:
        data.y = torch.tensor([ddg], dtype=torch.float32, device=device)

    return data


class GraphEmbeddingCache:
    """
    Cache for protein embeddings with graph construction.

    Extends the basic EmbeddingCache to also build graphs for mutations.
    """

    def __init__(self, embedding_dir: Path | str):
        self.embedding_dir = Path(embedding_dir)
        self._cache: dict[str, tuple[Tensor, Tensor]] = {}

    def get(self, protein_name: str) -> tuple[Tensor, Tensor]:
        """Get (single, pair) embeddings for a protein, converted to float32."""
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

    def build_graph(
        self,
        protein_name: str,
        position: int,
        wt_residue: int,
        mut_residue: int,
        k_neighbors: int = 30,
        ddg: Optional[float] = None,
    ) -> Data:
        """
        Build a mutation graph for a specific protein and position.

        Args:
            protein_name: Protein identifier
            position: Mutation position (0-indexed)
            wt_residue: WT amino acid index (0-19)
            mut_residue: Mutant amino acid index (0-19)
            k_neighbors: Number of neighbors in subgraph
            ddg: Optional target value

        Returns:
            PyG Data object
        """
        single, pair = self.get(protein_name)
        return build_mutation_graph(
            single=single,
            pair=pair,
            position=position,
            wt_residue=wt_residue,
            mut_residue=mut_residue,
            k_neighbors=k_neighbors,
            ddg=ddg,
        )


def collate_graphs(graphs: list[Data]) -> Data:
    """
    Collate a list of Data objects into a batched Data object.

    This is a simple wrapper that uses PyG's Batch.from_data_list.
    Use this with torch.utils.data.DataLoader's collate_fn parameter,
    or use torch_geometric.loader.DataLoader directly.
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(graphs)
