"""Evaluate model on the official ThermoMPNN test split.

Loads mega_splits.pkl from the ThermoMPNN repo and filters the MegaScale
dataset to only include proteins in their test set, enabling direct comparison.

Supports both MLP and MPNN model architectures.

Usage:
    # MLP model
    python scripts/evaluate_thermompnn_split.py --model-path outputs/<run>/fold_0/model.pt

    # MPNN model
    python scripts/evaluate_thermompnn_split.py --model-path outputs/<run>/fold_0/model.pt --model-type mpnn

    # Custom splits file
    python scripts/evaluate_thermompnn_split.py --model-path outputs/<run>/fold_0/model.pt --splits-file mega_splits.pkl
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pickle
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from src.data.dataset import parse_mutation, get_wt_sequence, collate_mutations, MutationSample


class FilteredMegaScaleDataset(Dataset):
    """MegaScale dataset filtered to specific protein names."""

    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

    def __init__(self, protein_names: set[str]):
        """
        Load MegaScale and filter to specific proteins.

        Args:
            protein_names: Set of protein names to include (with or without .pdb suffix)
        """
        # Normalize protein names (strip .pdb if present)
        self.target_proteins = {
            p.replace('.pdb', '') for p in protein_names
        }

        # Load full dataset from HuggingFace (all folds combined have same data, just split differently)
        # We load all splits and dedupe to get the full dataset
        print("Loading MegaScale dataset from HuggingFace...")
        ds = load_dataset("RosettaCommons/MegaScale", "dataset3_single_cv")

        # Combine all splits to access full data
        all_rows = []
        seen = set()

        for split_name in ds.keys():
            for row in ds[split_name]:
                # Create unique key for deduplication
                key = (row['WT_name'], row['mut_type'])
                if key not in seen:
                    seen.add(key)
                    # Filter to target proteins
                    wt_name = row['WT_name'].replace('.pdb', '')
                    if wt_name in self.target_proteins:
                        all_rows.append(row)

        self.data = all_rows

        # Build WT sequence cache
        self._wt_cache: dict[str, str] = {}
        self._build_wt_cache()

        print(f"Found {len(self.data)} mutations across {len(self.unique_proteins)} proteins in test set")

    def _build_wt_cache(self) -> None:
        """Build cache of WT sequences for each protein."""
        for row in self.data:
            wt_name = row["WT_name"]
            if wt_name not in self._wt_cache:
                mut_seq = row["aa_seq"]
                mut_type = row["mut_type"]
                self._wt_cache[wt_name] = get_wt_sequence(mut_seq, mut_type)

    def get_wt_sequence(self, wt_name: str) -> str:
        """Get cached WT sequence for a protein."""
        return self._wt_cache[wt_name]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MutationSample:
        row = self.data[idx]
        wt_aa, pos_1idx, mut_aa = parse_mutation(row["mut_type"])

        # Flip sign: MegaScale uses inverted convention
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
        """Get list of unique protein names in this dataset."""
        return list(self._wt_cache.keys())

    def encode_residue(self, aa: str) -> int:
        """Encode amino acid to index (0-19)."""
        return self.AA_TO_IDX.get(aa, 20)


def load_thermompnn_splits(splits_file: Path) -> dict:
    """Load the mega_splits.pkl file from ThermoMPNN."""
    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)
    return splits


def compute_metrics(
    predictions: dict[str, list[float]],
    targets: dict[str, list[float]],
    min_mutations: int = 10,
) -> dict:
    """Compute evaluation metrics from per-protein predictions."""
    all_preds = []
    all_targets = []
    spearmans = []
    pearsons = []
    weights = []

    for protein in predictions.keys():
        preds = predictions[protein]
        trues = targets[protein]

        all_preds.extend(preds)
        all_targets.extend(trues)

        if len(preds) >= min_mutations:
            rho, _ = spearmanr(preds, trues)
            r, _ = pearsonr(preds, trues)

            if not np.isnan(rho):
                spearmans.append(rho)
                weights.append(len(preds))
            if not np.isnan(r):
                pearsons.append(r)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    weights = np.array(weights)

    global_spearman, _ = spearmanr(all_preds, all_targets)
    global_pearson, _ = pearsonr(all_preds, all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))

    if len(spearmans) > 0 and weights.sum() > 0:
        weighted_spearman = np.average(spearmans, weights=weights)
        weighted_pearson = np.average(pearsons, weights=weights)
    else:
        weighted_spearman = 0.0
        weighted_pearson = 0.0

    return {
        "mean_spearman": float(np.mean(spearmans)) if spearmans else 0.0,
        "std_spearman": float(np.std(spearmans)) if spearmans else 0.0,
        "median_spearman": float(np.median(spearmans)) if spearmans else 0.0,
        "weighted_spearman": float(weighted_spearman),
        "mean_pearson": float(np.mean(pearsons)) if pearsons else 0.0,
        "std_pearson": float(np.std(pearsons)) if pearsons else 0.0,
        "median_pearson": float(np.median(pearsons)) if pearsons else 0.0,
        "weighted_pearson": float(weighted_pearson),
        "global_spearman": float(global_spearman) if not np.isnan(global_spearman) else 0.0,
        "global_pearson": float(global_pearson) if not np.isnan(global_pearson) else 0.0,
        "rmse": float(rmse),
        "mae": float(mae),
        "n_proteins": len(spearmans),
        "n_mutations": len(all_preds),
    }


def print_results(results: dict, title: str = "Results") -> None:
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"                    Spearman    Pearson")
    print(f"{'='*60}")
    print(f"Global (all muts)   {results['global_spearman']:>8.4f}    {results['global_pearson']:>8.4f}")
    print(f"Per-protein mean    {results['mean_spearman']:>8.4f}    {results['mean_pearson']:>8.4f}  (±{results['std_spearman']:.4f})")
    print(f"Per-protein median  {results['median_spearman']:>8.4f}    {results['median_pearson']:>8.4f}")
    print(f"Weighted mean       {results['weighted_spearman']:>8.4f}    {results['weighted_pearson']:>8.4f}  (by n_mutations)")
    print(f"{'='*60}")
    print(f"RMSE: {results['rmse']:.4f} kcal/mol | MAE: {results['mae']:.4f} kcal/mol")
    print(f"n_proteins: {results['n_proteins']} | n_mutations: {results['n_mutations']}")


# ============================================================================
# MLP Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_mlp(
    model: nn.Module,
    dataset: FilteredMegaScaleDataset,
    embedding_cache,
    encode_batch_fn,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    """Evaluate MLP model on filtered dataset."""
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_mutations,
        num_workers=0,
    )

    predictions: dict[str, list[float]] = defaultdict(list)
    targets: dict[str, list[float]] = defaultdict(list)

    for batch in tqdm(dataloader, desc="Evaluating MLP"):
        features = encode_batch_fn(
            cache=embedding_cache,
            protein_names=batch["wt_names"],
            positions=batch["positions"].tolist(),
            wt_residues=batch["wt_residues"].tolist(),
            mut_residues=batch["mut_residues"].tolist(),
        )

        features = {k: v.to(device) for k, v in features.items()}
        preds = model.forward_dict(features).squeeze(-1)

        for i, protein in enumerate(batch["wt_names"]):
            predictions[protein].append(preds[i].cpu().item())
            targets[protein].append(batch["ddg"][i].item())

    return compute_metrics(predictions, targets, min_mutations=10)


def load_mlp_model(model_path: Path, config: dict, device: torch.device) -> nn.Module:
    """Load MLP model from checkpoint."""
    from src.models.pair_aware_mlp import PairAwareMLP

    model = PairAwareMLP(
        d_single=config.get("d_single", 384),
        d_pair=config.get("d_pair", 256),
        hidden_dim=config.get("hidden_dim", 512),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


# ============================================================================
# MPNN Evaluation
# ============================================================================

class FilteredMPNNDataset(Dataset):
    """Dataset for MPNN evaluation on filtered proteins."""

    def __init__(
        self,
        filtered_dataset: FilteredMegaScaleDataset,
        graph_cache,
        k_neighbors: int = 30,
        device: str = "cuda",
    ):
        self.k_neighbors = k_neighbors
        self.device = torch.device(device)

        # Move all embeddings to GPU
        print(f"Moving embeddings to {device}...")
        self.embeddings = {
            name: (single.to(self.device), pair.to(self.device))
            for name, (single, pair) in graph_cache._cache.items()
        }

        # Edge index template on GPU
        num_nodes = k_neighbors + 1
        arange = torch.arange(num_nodes, device=self.device)
        grid_i, grid_j = torch.meshgrid(arange, arange, indexing='ij')
        mask = grid_i != grid_j
        self.edge_index_template = torch.stack([grid_i[mask], grid_j[mask]])

        # Precompute graphs
        print(f"Precomputing graphs for {len(filtered_dataset)} samples...")
        self.samples = []
        self.protein_names = []

        for i in tqdm(range(len(filtered_dataset)), desc="Precomputing"):
            sample = filtered_dataset[i]
            self.protein_names.append(sample.wt_name)

            single, pair = self.embeddings[sample.wt_name]
            L = single.shape[0]

            # topk on GPU
            pair_row = pair[sample.position]
            magnitudes = pair_row.norm(dim=-1)
            magnitudes[sample.position] = float("-inf")
            k_actual = min(k_neighbors, L - 1)
            neighbor_indices = magnitudes.topk(k=k_actual).indices

            node_indices = torch.cat([
                torch.tensor([sample.position], device=self.device),
                neighbor_indices,
            ])

            self.samples.append({
                'protein': sample.wt_name,
                'node_indices': node_indices,
                'wt_idx': filtered_dataset.encode_residue(sample.wt_residue),
                'mut_idx': filtered_dataset.encode_residue(sample.mut_residue),
                'rel_pos': sample.position / L,
                'ddg': sample.ddg,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from torch_geometric.data import Data

        s = self.samples[idx]
        single, pair = self.embeddings[s['protein']]
        node_idx = s['node_indices']

        x = single[node_idx]
        edge_attr = pair[node_idx[self.edge_index_template[0]],
                        node_idx[self.edge_index_template[1]]]

        return Data(
            x=x,
            edge_index=self.edge_index_template.clone(),
            edge_attr=edge_attr,
            wt_idx=torch.tensor(s['wt_idx'], device=self.device),
            mut_idx=torch.tensor(s['mut_idx'], device=self.device),
            rel_pos=torch.tensor(s['rel_pos'], device=self.device),
            y=torch.tensor([s['ddg']], device=self.device),
        )


@torch.no_grad()
def evaluate_mpnn(
    model: nn.Module,
    dataset: FilteredMPNNDataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Evaluate MPNN model on filtered dataset."""
    from torch_geometric.loader import DataLoader as PyGDataLoader

    model.eval()

    dataloader = PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Evaluating MPNN"):
        batch = batch.to(device)
        preds = model(batch).squeeze(-1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch.y.squeeze(-1).cpu().numpy())

    # Build per-protein predictions
    predictions: dict[str, list[float]] = defaultdict(list)
    targets: dict[str, list[float]] = defaultdict(list)

    for i, protein in enumerate(dataset.protein_names):
        predictions[protein].append(all_preds[i])
        targets[protein].append(all_targets[i])

    return compute_metrics(predictions, targets, min_mutations=10)


def load_mpnn_model(model_path: Path, config: dict, device: torch.device) -> nn.Module:
    """Load MPNN model from checkpoint."""
    from src.models.mpnn import ChaiMPNN, ChaiMPNNWithMutationInfo

    use_mutation_info = config.get("use_mutation_info", True)

    if use_mutation_info:
        model = ChaiMPNNWithMutationInfo(
            node_in_dim=config.get("node_in_dim", 384),
            edge_in_dim=config.get("edge_in_dim", 256),
            hidden_dim=config.get("hidden_dim", 128),
            edge_hidden_dim=config.get("edge_hidden_dim", 128),
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.1),
            use_global_pool=config.get("use_global_pool", True),
        )
    else:
        model = ChaiMPNN(
            node_in_dim=config.get("node_in_dim", 384),
            edge_in_dim=config.get("edge_in_dim", 256),
            hidden_dim=config.get("hidden_dim", 128),
            edge_hidden_dim=config.get("edge_hidden_dim", 128),
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.1),
            use_global_pool=config.get("use_global_pool", True),
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate on ThermoMPNN test split")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model.pt")
    parser.add_argument("--splits-file", type=str, default="mega_splits.pkl", help="Path to mega_splits.pkl")
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings/chai_trunk")
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "mlp", "mpnn"],
                       help="Model type (auto-detects from config if not specified)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load ThermoMPNN splits
    print(f"Loading splits from {args.splits_file}...")
    splits = load_thermompnn_splits(Path(args.splits_file))

    # Get test proteins
    test_proteins = set(splits['test'])
    print(f"ThermoMPNN test set: {len(test_proteins)} proteins")
    print(f"Example proteins: {list(test_proteins)[:5]}")

    # Create filtered dataset
    dataset = FilteredMegaScaleDataset(test_proteins)

    if len(dataset) == 0:
        print("ERROR: No matching proteins found in dataset!")
        return

    # Load config
    model_dir = Path(args.model_path).parent
    config_path = model_dir.parent / "config.json"

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Determine model type
    model_type = args.model_type
    if model_type == "auto":
        # Auto-detect from config
        config_model_type = config.get("model_type", "pair_aware_mlp")
        if "mpnn" in config_model_type.lower():
            model_type = "mpnn"
        else:
            model_type = "mlp"
        print(f"Auto-detected model type: {model_type}")

    print(f"Loading model from {args.model_path}...")

    if model_type == "mlp":
        # MLP evaluation
        from src.features.mutation_encoder import EmbeddingCache, encode_batch

        model = load_mlp_model(Path(args.model_path), config, device)

        print(f"Loading embeddings from {args.embedding_dir}...")
        embedding_cache = EmbeddingCache(args.embedding_dir)
        embedding_cache.preload(dataset.unique_proteins)

        print("\nEvaluating MLP on ThermoMPNN test split...")
        results = evaluate_mlp(
            model=model,
            dataset=dataset,
            embedding_cache=embedding_cache,
            encode_batch_fn=encode_batch,
            device=device,
            batch_size=args.batch_size,
        )

    else:
        # MPNN evaluation
        from src.data.graph_builder import GraphEmbeddingCache

        model = load_mpnn_model(Path(args.model_path), config, device)

        print(f"Loading embeddings from {args.embedding_dir}...")
        graph_cache = GraphEmbeddingCache(args.embedding_dir)
        graph_cache.preload(dataset.unique_proteins)

        # Create MPNN dataset
        k_neighbors = config.get("k_neighbors", 30)
        mpnn_dataset = FilteredMPNNDataset(
            dataset, graph_cache, k_neighbors=k_neighbors, device=args.device
        )

        print("\nEvaluating MPNN on ThermoMPNN test split...")
        results = evaluate_mpnn(
            model=model,
            dataset=mpnn_dataset,
            device=device,
            batch_size=args.batch_size,
        )

    # Print results
    print_results(results, "ThermoMPNN Test Split Results")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                "split": "thermompnn_test",
                "model_type": model_type,
                "n_test_proteins_in_split": len(test_proteins),
                **results
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
