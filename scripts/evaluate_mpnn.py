#!/usr/bin/env python3
"""Evaluate trained MPNN models.

Usage:
    # Evaluate a single fold
    python scripts/evaluate_mpnn.py --run-dir outputs/mpnn_20240120_123456 --fold 0

    # Evaluate all folds and aggregate
    python scripts/evaluate_mpnn.py --run-dir outputs/mpnn_20240120_123456 --all-folds

    # Evaluate on validation set instead of test
    python scripts/evaluate_mpnn.py --run-dir outputs/mpnn_20240120_123456 --fold 0 --split val

    # Show per-protein breakdown
    python scripts/evaluate_mpnn.py --run-dir outputs/mpnn_20240120_123456 --fold 0 --per-protein
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MegaScaleDataset
from src.data.graph_builder import GraphEmbeddingCache
from src.models.mpnn import ChaiMPNN, ChaiMPNNWithMutationInfo
from src.training.train_mpnn import (
    MPNNTrainingConfig,
    MutationGraphDataset,
    EvaluationResults,
    evaluate_mpnn,
)


def load_model(
    model_path: Path,
    config: MPNNTrainingConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained MPNN model from checkpoint."""
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

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_fold(
    model: torch.nn.Module,
    fold: int,
    config: MPNNTrainingConfig,
    device: torch.device,
    split: str = "test",
    batch_size: int = 128,
) -> tuple[EvaluationResults, dict]:
    """
    Evaluate model on a fold.

    Args:
        model: Trained MPNN model
        fold: Fold number (0-4)
        config: Training config (for embedding_dir, k_neighbors)
        device: Device to run on
        split: Which split to evaluate ('test', 'val', 'train')
        batch_size: Batch size for evaluation

    Returns:
        (EvaluationResults, per_protein_dict)
    """
    # Load dataset
    megascale = MegaScaleDataset(fold=fold, split=split)

    # Load embeddings
    graph_cache = GraphEmbeddingCache(config.embedding_dir)
    print(f"Preloading embeddings for {len(megascale.unique_proteins)} proteins...")
    graph_cache.preload(megascale.unique_proteins)

    # Create graph dataset
    graph_dataset = MutationGraphDataset(megascale, graph_cache, config.k_neighbors)

    # DataLoader
    loader = DataLoader(
        graph_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Single pass evaluation - collect predictions and compute metrics
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split}"):
            batch = batch.to(device)
            preds = model(batch).squeeze(-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch.y.squeeze(-1).cpu().numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Get protein names directly from underlying data (faster than iterating __getitem__)
    protein_names = [megascale.data[i]["WT_name"] for i in range(len(megascale))]

    # Compute per-protein correlations
    unique_proteins = list(set(protein_names))
    spearmans = []
    pearsons = []
    per_protein = defaultdict(lambda: {"preds": [], "targets": []})

    for i, prot in enumerate(protein_names):
        per_protein[prot]["preds"].append(preds[i])
        per_protein[prot]["targets"].append(targets[i])

    for prot in unique_proteins:
        p_preds = np.array(per_protein[prot]["preds"])
        p_targets = np.array(per_protein[prot]["targets"])
        if len(p_preds) >= 3 and np.std(p_preds) > 1e-6 and np.std(p_targets) > 1e-6:
            spearmans.append(spearmanr(p_preds, p_targets)[0])
            pearsons.append(pearsonr(p_preds, p_targets)[0])

    # Global metrics
    global_spearman = spearmanr(preds, targets)[0]
    global_pearson = pearsonr(preds, targets)[0]
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))

    results = EvaluationResults(
        mean_spearman=np.mean(spearmans) if spearmans else 0.0,
        std_spearman=np.std(spearmans) if spearmans else 0.0,
        median_spearman=np.median(spearmans) if spearmans else 0.0,
        mean_pearson=np.mean(pearsons) if pearsons else 0.0,
        std_pearson=np.std(pearsons) if pearsons else 0.0,
        median_pearson=np.median(pearsons) if pearsons else 0.0,
        global_spearman=global_spearman,
        global_pearson=global_pearson,
        rmse=rmse,
        mae=mae,
        n_proteins=len(unique_proteins),
        n_mutations=len(preds),
    )

    return results, dict(per_protein)


def print_per_protein_breakdown(per_protein: dict, top_k: int = 10):
    """Print per-protein results sorted by Spearman."""
    protein_spearmans = []
    for protein, data in per_protein.items():
        preds = data["preds"]
        targs = data["targets"]
        if len(preds) >= 10:
            rho, _ = spearmanr(preds, targs)
            if not np.isnan(rho):
                protein_spearmans.append((protein, rho, len(preds)))

    protein_spearmans.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_k} proteins (by Spearman):")
    print("-" * 50)
    for protein, rho, n in protein_spearmans[:top_k]:
        print(f"  {protein}: {rho:.4f} (n={n})")

    print(f"\nBottom {top_k} proteins (by Spearman):")
    print("-" * 50)
    for protein, rho, n in protein_spearmans[-top_k:]:
        print(f"  {protein}: {rho:.4f} (n={n})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MPNN models")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to training run directory (contains config.json and fold_X/)")
    parser.add_argument("--fold", type=int, default=0, help="Fold to evaluate")
    parser.add_argument("--all-folds", action="store_true", help="Evaluate all available folds")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--per-protein", action="store_true", help="Show per-protein breakdown")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Save results to JSON")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Load config
    config_path = args.run_dir / "config.json"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    config = MPNNTrainingConfig.load(config_path)
    print(f"Loaded config from {config_path}")
    print(f"Model type: {config.model_type}, use_mutation_info: {config.use_mutation_info}")

    # Determine folds to evaluate
    if args.all_folds:
        folds = []
        for i in range(5):
            if (args.run_dir / f"fold_{i}" / "model.pt").exists():
                folds.append(i)
        if not folds:
            print("Error: No fold directories found")
            sys.exit(1)
        print(f"Found {len(folds)} folds: {folds}")
    else:
        folds = [args.fold]

    all_results = []

    for fold in folds:
        model_path = args.run_dir / f"fold_{fold}" / "model.pt"
        if not model_path.exists():
            print(f"Warning: No model for fold {fold} at {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating Fold {fold} ({args.split} set)")
        print(f"{'='*60}")

        model = load_model(model_path, config, device)
        results, per_protein = evaluate_fold(
            model, fold, config, device,
            split=args.split, batch_size=args.batch_size,
        )
        all_results.append(results)

        print(results.summary())

        if args.per_protein:
            print_per_protein_breakdown(per_protein)

    # Aggregate across folds
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATION SUMMARY ({len(all_results)} folds)")
        print(f"{'='*70}")
        print(f"                         Mean ± Std      (per-fold values)")
        print(f"{'-'*70}")

        # Global metrics
        global_sp = [r.global_spearman for r in all_results]
        global_pe = [r.global_pearson for r in all_results]
        print(f"Global Spearman:         {np.mean(global_sp):.4f} ± {np.std(global_sp):.4f}   {[f'{x:.3f}' for x in global_sp]}")
        print(f"Global Pearson:          {np.mean(global_pe):.4f} ± {np.std(global_pe):.4f}   {[f'{x:.3f}' for x in global_pe]}")

        # Per-protein mean
        mean_sp = [r.mean_spearman for r in all_results]
        mean_pe = [r.mean_pearson for r in all_results]
        print(f"Per-protein Spearman:    {np.mean(mean_sp):.4f} ± {np.std(mean_sp):.4f}   {[f'{x:.3f}' for x in mean_sp]}")
        print(f"Per-protein Pearson:     {np.mean(mean_pe):.4f} ± {np.std(mean_pe):.4f}   {[f'{x:.3f}' for x in mean_pe]}")

        # RMSE/MAE
        rmses = [r.rmse for r in all_results]
        maes = [r.mae for r in all_results]
        print(f"{'-'*70}")
        print(f"RMSE (kcal/mol):         {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
        print(f"MAE (kcal/mol):          {np.mean(maes):.4f} ± {np.std(maes):.4f}")
        print(f"{'='*70}")

    # Save results
    if args.output and all_results:
        output_data = {
            "folds": [r.to_dict() for r in all_results],
            "summary": {
                "mean_spearman": float(np.mean([r.mean_spearman for r in all_results])),
                "std_spearman": float(np.std([r.mean_spearman for r in all_results])),
                "mean_pearson": float(np.mean([r.mean_pearson for r in all_results])),
                "mean_rmse": float(np.mean([r.rmse for r in all_results])),
            } if len(all_results) > 1 else all_results[0].to_dict()
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
