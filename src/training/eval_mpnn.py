"""Per-protein evaluation for MPNN models.

Generates detailed breakdown of model performance by protein to identify weaknesses.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data.graph_builder import GraphEmbeddingCache
from src.models.mpnn import ChaiMPNN, ChaiMPNNWithMutationInfo
from src.training.train_mpnn import (
    MPNNTrainingConfig,
    MutationGraphDataset,
    ThermoMPNNSplitDataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(
    run_dir: Path, device: str = "cuda", checkpoint: str | None = None
) -> tuple[torch.nn.Module, MPNNTrainingConfig]:
    """Load model and config from a run directory."""
    config = MPNNTrainingConfig.load(run_dir / "config.json")

    if config.use_mutation_info:
        model = ChaiMPNNWithMutationInfo(
            node_in_dim=config.node_in_dim,
            edge_in_dim=config.edge_in_dim,
            hidden_dim=config.hidden_dim,
            edge_hidden_dim=config.edge_hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_global_pool=config.use_global_pool,
        )
    else:
        model = ChaiMPNN(
            node_in_dim=config.node_in_dim,
            edge_in_dim=config.edge_in_dim,
            hidden_dim=config.hidden_dim,
            edge_hidden_dim=config.edge_hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_global_pool=config.use_global_pool,
        )

    # Determine model path
    if checkpoint:
        model_path = run_dir / checkpoint
        if not model_path.exists():
            model_path = Path(checkpoint)  # Try as absolute path
    else:
        model_path = run_dir / "model.pt"
        if not model_path.exists():
            # Try fold_0 for CV runs
            model_path = run_dir / "fold_0" / "model.pt"

    # Load weights (handle both checkpoint dict and raw state_dict)
    # weights_only=False is fine for our own checkpoints
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded checkpoint from epoch {state.get('epoch', '?')}")
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    return model, config


def evaluate_per_protein(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataset: ThermoMPNNSplitDataset,
    graph_dataset: MutationGraphDataset,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate model and return per-protein metrics."""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            preds = model(batch).squeeze(-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch.y.squeeze(-1).cpu().numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Get protein names for each sample
    protein_names = [dataset.data[i]["WT_name"] for i in range(len(dataset))]

    # Compute per-protein metrics
    results = []
    unique_proteins = list(set(protein_names))

    for prot in unique_proteins:
        mask = np.array([p == prot for p in protein_names])
        n_muts = mask.sum()

        if n_muts < 2:
            continue

        p_preds = preds[mask]
        p_targets = targets[mask]

        # Metrics
        rmse = np.sqrt(np.mean((p_preds - p_targets) ** 2))
        mae = np.mean(np.abs(p_preds - p_targets))

        # Correlation (need variance)
        if np.std(p_preds) > 1e-6 and np.std(p_targets) > 1e-6:
            spearman = spearmanr(p_preds, p_targets)[0]
            pearson = pearsonr(p_preds, p_targets)[0]
        else:
            spearman = np.nan
            pearson = np.nan

        # Range of ddG values (helps understand if low correlation is due to narrow range)
        ddg_range = p_targets.max() - p_targets.min()
        ddg_std = np.std(p_targets)

        results.append({
            "protein": prot.replace(".pdb", ""),
            "n_mutations": n_muts,
            "spearman": spearman,
            "pearson": pearson,
            "rmse": rmse,
            "mae": mae,
            "ddg_range": ddg_range,
            "ddg_std": ddg_std,
            "mean_pred": p_preds.mean(),
            "mean_target": p_targets.mean(),
        })

    df = pd.DataFrame(results)
    return df


def main():
    parser = argparse.ArgumentParser(description="Per-protein evaluation of MPNN model")
    parser.add_argument("run_dir", type=str, help="Path to training run directory")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: run_dir/eval_{split}.csv)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="spearman",
        choices=["spearman", "pearson", "rmse", "mae", "n_mutations"],
        help="Column to sort by",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending (default: descending for correlations, ascending for errors)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Show top N worst/best proteins in summary",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename to load (e.g., checkpoint_epoch_20.pt)",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    # Load model and config
    model, config = load_model(run_dir, args.device, args.checkpoint)
    device = torch.device(args.device)

    # Load dataset
    if not config.thermompnn_splits:
        raise ValueError("Currently only supports thermompnn_splits evaluation")

    logger.info(f"Loading {args.split} split from {config.thermompnn_splits}")
    dataset = ThermoMPNNSplitDataset(
        config.thermompnn_splits,
        split=args.split,
        cv_fold=config.thermompnn_cv_fold,
        data_path=getattr(config, 'data_path', None),
    )

    # Load embeddings
    graph_cache = GraphEmbeddingCache(config.embedding_dir)
    graph_cache.preload(dataset.unique_proteins)

    # Create graph dataset
    graph_dataset = MutationGraphDataset(
        dataset, graph_cache, config.k_neighbors, device=args.device
    )

    dataloader = DataLoader(
        graph_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Evaluate
    df = evaluate_per_protein(model, dataloader, dataset, graph_dataset, device)

    # Sort
    ascending = args.ascending
    if not args.ascending and args.sort_by in ["rmse", "mae"]:
        ascending = False  # Higher error = worse, show worst first
    elif not args.ascending and args.sort_by in ["spearman", "pearson"]:
        ascending = True  # Lower correlation = worse, show worst first

    df_sorted = df.sort_values(args.sort_by, ascending=ascending)

    # Save
    output_path = args.output or run_dir / f"eval_{args.split}.csv"
    df_sorted.to_csv(output_path, index=False)
    logger.info(f"Saved per-protein results to {output_path}")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Per-Protein Evaluation Summary ({args.split} split)")
    print(f"{'=' * 70}")
    print(f"Total proteins: {len(df)}")
    print(f"Total mutations: {df['n_mutations'].sum()}")
    print(f"\nAggregate metrics:")
    print(f"  Mean Spearman: {df['spearman'].mean():.4f} ± {df['spearman'].std():.4f}")
    print(f"  Mean Pearson:  {df['pearson'].mean():.4f} ± {df['pearson'].std():.4f}")
    print(f"  Mean RMSE:     {df['rmse'].mean():.4f} ± {df['rmse'].std():.4f}")
    print(f"  Mean MAE:      {df['mae'].mean():.4f} ± {df['mae'].std():.4f}")

    # Worst performers
    print(f"\n{'=' * 70}")
    print(f"Bottom {args.top_n} proteins by {args.sort_by}:")
    print(f"{'=' * 70}")

    cols = ["protein", "n_mutations", "spearman", "pearson", "rmse", "mae", "ddg_range"]
    print(df_sorted.head(args.top_n)[cols].to_string(index=False))

    # Best performers
    print(f"\n{'=' * 70}")
    print(f"Top {args.top_n} proteins by {args.sort_by}:")
    print(f"{'=' * 70}")
    print(df_sorted.tail(args.top_n)[cols].to_string(index=False))

    # Identify patterns in poor performers
    print(f"\n{'=' * 70}")
    print("Analysis of poor performers (bottom 20%):")
    print(f"{'=' * 70}")

    n_bottom = max(1, len(df) // 5)
    bottom_df = df_sorted.head(n_bottom)
    top_df = df_sorted.tail(n_bottom)

    print(f"Poor performers ({n_bottom} proteins):")
    print(f"  Avg mutations: {bottom_df['n_mutations'].mean():.1f}")
    print(f"  Avg ddG range: {bottom_df['ddg_range'].mean():.2f}")
    print(f"  Avg ddG std:   {bottom_df['ddg_std'].mean():.2f}")

    print(f"\nGood performers ({n_bottom} proteins):")
    print(f"  Avg mutations: {top_df['n_mutations'].mean():.1f}")
    print(f"  Avg ddG range: {top_df['ddg_range'].mean():.2f}")
    print(f"  Avg ddG std:   {top_df['ddg_std'].mean():.2f}")


if __name__ == "__main__":
    main()
