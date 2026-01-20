#!/usr/bin/env python3
"""Evaluate trained models and run ablations.

Usage:
    # Evaluate a single fold
    python scripts/evaluate.py --model outputs/fold_0/model.pt --fold 0

    # Evaluate all folds and aggregate
    python scripts/evaluate.py --model-dir outputs --all-folds

    # Run ablation (zero out pair features)
    python scripts/evaluate.py --model outputs/fold_0/model.pt --fold 0 --ablation no-pair

    # Show per-protein breakdown
    python scripts/evaluate.py --model outputs/fold_0/model.pt --fold 0 --per-protein
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MegaScaleDataset, collate_mutations
from src.features.mutation_encoder import EmbeddingCache, encode_batch
from src.models.pair_aware_mlp import PairAwareMLP
from src.training.sampler import FullDatasetSampler
from src.training.evaluate import compute_metrics, EvaluationResults


def load_model(model_path: Path, device: torch.device) -> PairAwareMLP:
    """Load a trained model."""
    model = PairAwareMLP(
        d_single=384,
        d_pair=256,
        hidden_dim=512,
        dropout=0.1,
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_fold(
    model: PairAwareMLP,
    fold: int,
    embedding_dir: Path,
    device: torch.device,
    split: str = "test",
    ablation: str | None = None,
    batch_size: int = 128,
) -> tuple[EvaluationResults, dict]:
    """
    Evaluate model on a fold.

    Args:
        model: Trained model
        fold: Fold number (0-4)
        embedding_dir: Path to embeddings
        device: Device to run on
        split: Which split to evaluate ('test', 'val', 'train')
        ablation: Optional ablation type ('no-pair', 'no-single', 'pair-global-only', etc.)
        batch_size: Batch size

    Returns:
        (EvaluationResults, per_protein_dict)
    """
    dataset = MegaScaleDataset(fold=fold, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=FullDatasetSampler(dataset, shuffle=False),
        collate_fn=collate_mutations,
        num_workers=0,
    )

    cache = EmbeddingCache(embedding_dir)
    cache.preload(dataset.unique_proteins)

    predictions = defaultdict(list)
    targets = defaultdict(list)
    wt_residues_collected = defaultdict(list)
    mut_residues_collected = defaultdict(list)

    for batch in tqdm(loader, desc=f"Evaluating fold {fold} {split}"):
        features = encode_batch(
            cache=cache,
            protein_names=batch["wt_names"],
            positions=batch["positions"].tolist(),
            wt_residues=batch["wt_residues"].tolist(),
            mut_residues=batch["mut_residues"].tolist(),
        )

        # Apply ablation if specified
        features = apply_ablation(features, ablation)

        # Move to device
        features = {k: v.to(device) for k, v in features.items()}

        # Forward pass
        preds = model.forward_dict(features).squeeze(-1)

        # Collect results
        for i, protein in enumerate(batch["wt_names"]):
            predictions[protein].append(preds[i].cpu().item())
            targets[protein].append(batch["ddg"][i].item())
            wt_residues_collected[protein].append(
                MegaScaleDataset.AA_VOCAB[batch["wt_residues"][i]]
            )
            mut_residues_collected[protein].append(
                MegaScaleDataset.AA_VOCAB[batch["mut_residues"][i]]
            )

    results = compute_metrics(predictions, targets)

    per_protein = {
        "predictions": dict(predictions),
        "targets": dict(targets),
        "wt_residues": dict(wt_residues_collected),
        "mut_residues": dict(mut_residues_collected),
    }

    return results, per_protein


def apply_ablation(features: dict, ablation: str | None) -> dict:
    """Apply ablation by zeroing out feature groups."""
    if ablation is None:
        return features

    features = {k: v.clone() for k, v in features.items()}

    if ablation == "no-pair":
        # Zero out all pair features
        features["pair_global"] = torch.zeros_like(features["pair_global"])
        features["pair_local_seq"] = torch.zeros_like(features["pair_local_seq"])
        features["pair_structural"] = torch.zeros_like(features["pair_structural"])

    elif ablation == "no-single":
        # Zero out single features
        features["local_single"] = torch.zeros_like(features["local_single"])
        features["global_single"] = torch.zeros_like(features["global_single"])

    elif ablation == "pair-global-only":
        # Keep only pair_global
        features["local_single"] = torch.zeros_like(features["local_single"])
        features["global_single"] = torch.zeros_like(features["global_single"])
        features["pair_local_seq"] = torch.zeros_like(features["pair_local_seq"])
        features["pair_structural"] = torch.zeros_like(features["pair_structural"])

    elif ablation == "pair-structural-only":
        # Keep only pair_structural
        features["local_single"] = torch.zeros_like(features["local_single"])
        features["global_single"] = torch.zeros_like(features["global_single"])
        features["pair_global"] = torch.zeros_like(features["pair_global"])
        features["pair_local_seq"] = torch.zeros_like(features["pair_local_seq"])

    elif ablation == "single-only":
        # Same as no-pair
        features["pair_global"] = torch.zeros_like(features["pair_global"])
        features["pair_local_seq"] = torch.zeros_like(features["pair_local_seq"])
        features["pair_structural"] = torch.zeros_like(features["pair_structural"])

    elif ablation == "no-mutation-feat":
        # Zero out mutation identity features
        features["mutation_feat"] = torch.zeros_like(features["mutation_feat"])

    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    return features


def print_per_protein_breakdown(per_protein: dict, top_k: int = 10):
    """Print per-protein results sorted by Spearman."""
    from scipy.stats import spearmanr

    protein_spearmans = []
    for protein in per_protein["predictions"].keys():
        preds = per_protein["predictions"][protein]
        targs = per_protein["targets"][protein]
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


def run_ablation_suite(
    model_dir: Path,
    embedding_dir: Path,
    device: torch.device,
    folds: list[int] = [0],
):
    """Run a suite of ablations across folds."""
    ablations = [
        None,  # Full model
        "no-pair",  # Single features only
        "no-single",  # Pair features only
        "pair-global-only",
        "pair-structural-only",
        "no-mutation-feat",
    ]

    results = {abl or "full": [] for abl in ablations}

    for fold in folds:
        model_path = model_dir / f"fold_{fold}" / "model.pt"
        if not model_path.exists():
            print(f"Warning: No model for fold {fold}")
            continue

        model = load_model(model_path, device)

        for ablation in ablations:
            abl_name = ablation or "full"
            print(f"\nFold {fold}, Ablation: {abl_name}")

            eval_results, _ = evaluate_fold(
                model, fold, embedding_dir, device, ablation=ablation
            )
            results[abl_name].append(eval_results.mean_spearman)
            print(f"  Spearman: {eval_results.mean_spearman:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Ablation Summary (Mean Spearman across folds)")
    print("=" * 60)
    for abl_name, spearmans in results.items():
        if spearmans:
            mean = np.mean(spearmans)
            std = np.std(spearmans) if len(spearmans) > 1 else 0
            print(f"  {abl_name:25s}: {mean:.4f} ± {std:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", type=Path, help="Path to model.pt file")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs"), help="Directory with fold_X subdirs")
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/embeddings/chai_trunk"))
    parser.add_argument("--fold", type=int, default=0, help="Fold to evaluate")
    parser.add_argument("--all-folds", action="store_true", help="Evaluate all 5 folds")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no-pair", "no-single", "pair-global-only",
                                "pair-structural-only", "single-only", "no-mutation-feat"])
    parser.add_argument("--ablation-suite", action="store_true", help="Run all ablations")
    parser.add_argument("--per-protein", action="store_true", help="Show per-protein breakdown")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Save results to JSON")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Run ablation suite
    if args.ablation_suite:
        folds = list(range(5)) if args.all_folds else [args.fold]
        run_ablation_suite(args.model_dir, args.embedding_dir, device, folds)
        return

    # Evaluate single model or all folds
    if args.all_folds:
        all_results = []
        for fold in range(5):
            model_path = args.model_dir / f"fold_{fold}" / "model.pt"
            if not model_path.exists():
                print(f"Warning: No model for fold {fold}")
                continue

            model = load_model(model_path, device)
            results, per_protein = evaluate_fold(
                model, fold, args.embedding_dir, device,
                split=args.split, ablation=args.ablation
            )
            all_results.append(results)

            print(f"\nFold {fold} ({args.split}):")
            print(results.summary())

            if args.per_protein:
                print_per_protein_breakdown(per_protein)

        # Aggregate across folds
        if all_results:
            print(f"\n{'='*70}")
            print(f"CROSS-VALIDATION SUMMARY ({len(all_results)} folds)")
            print(f"{'='*70}")
            print(f"                         Mean ± Std      (per-fold values)")
            print(f"{'-'*70}")

            # Global
            global_sp = [r.global_spearman for r in all_results]
            global_pe = [r.global_pearson for r in all_results]
            print(f"Global Spearman:         {np.mean(global_sp):.4f} ± {np.std(global_sp):.4f}   {[f'{x:.3f}' for x in global_sp]}")
            print(f"Global Pearson:          {np.mean(global_pe):.4f} ± {np.std(global_pe):.4f}   {[f'{x:.3f}' for x in global_pe]}")

            # Per-protein mean
            mean_sp = [r.mean_spearman for r in all_results]
            mean_pe = [r.mean_pearson for r in all_results]
            print(f"Per-protein Spearman:    {np.mean(mean_sp):.4f} ± {np.std(mean_sp):.4f}   {[f'{x:.3f}' for x in mean_sp]}")
            print(f"Per-protein Pearson:     {np.mean(mean_pe):.4f} ± {np.std(mean_pe):.4f}   {[f'{x:.3f}' for x in mean_pe]}")

            # Weighted mean
            weighted_sp = [r.weighted_spearman for r in all_results]
            weighted_pe = [r.weighted_pearson for r in all_results]
            print(f"Weighted Spearman:       {np.mean(weighted_sp):.4f} ± {np.std(weighted_sp):.4f}   {[f'{x:.3f}' for x in weighted_sp]}")
            print(f"Weighted Pearson:        {np.mean(weighted_pe):.4f} ± {np.std(weighted_pe):.4f}   {[f'{x:.3f}' for x in weighted_pe]}")

            # RMSE/MAE
            rmses = [r.rmse for r in all_results]
            maes = [r.mae for r in all_results]
            print(f"{'-'*70}")
            print(f"RMSE (kcal/mol):         {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
            print(f"MAE (kcal/mol):          {np.mean(maes):.4f} ± {np.std(maes):.4f}")
            print(f"{'='*70}")

    else:
        # Single fold
        model_path = args.model or (args.model_dir / f"fold_{args.fold}" / "model.pt")
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)

        model = load_model(model_path, device)
        results, per_protein = evaluate_fold(
            model, args.fold, args.embedding_dir, device,
            split=args.split, ablation=args.ablation
        )

        print(f"\nResults (Fold {args.fold}, {args.split}):")
        print(results.summary())

        if args.per_protein:
            print_per_protein_breakdown(per_protein)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results.to_dict(), f, indent=2)
            print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
