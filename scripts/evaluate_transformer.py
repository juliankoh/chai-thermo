#!/usr/bin/env python3
"""Evaluate trained Chai-Thermo-Transformer models.

Usage:
    # Evaluate a saved run
    uv run python scripts/evaluate_transformer.py --run-dir outputs/transformer_test_20240115_120000

    # Evaluate on validation set instead of test
    uv run python scripts/evaluate_transformer.py --run-dir outputs/my_run --split val

    # Show per-protein breakdown
    uv run python scripts/evaluate_transformer.py --run-dir outputs/my_run --per-protein

    # Evaluate with specific model file
    uv run python scripts/evaluate_transformer.py --model outputs/my_run/model.pt --config outputs/my_run/config.json

    # Ablation: disable pair bias (gates=0)
    uv run python scripts/evaluate_transformer.py --run-dir outputs/my_run --ablation no-bias

    # Ablation: force full pair bias (gates=1)
    uv run python scripts/evaluate_transformer.py --run-dir outputs/my_run --ablation full-bias
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MutationSample
from src.data.megascale_loader import AA_TO_IDX, load_thermompnn_mutations
from src.features.mutation_encoder import EmbeddingCache
from src.models.transformer import ChaiThermoTransformer
from src.training.evaluate import compute_metrics, EvaluationResults


def load_config(config_path: Path) -> dict:
    """Load training config from JSON."""
    with open(config_path) as f:
        return json.load(f)


def load_model(
    model_path: Path,
    config: dict,
    device: torch.device,
    ablation: str = "learned",
) -> ChaiThermoTransformer:
    """Load a trained transformer model.

    Args:
        ablation: One of "learned", "no-bias", "full-bias"
            - learned: use the learned gate values (default)
            - no-bias: force gates to 0 (disable pair bias entirely)
            - full-bias: force gates to 1 (always use full pair bias)
    """
    model = ChaiThermoTransformer(
        single_dim=config.get("single_dim", 384),
        pair_dim=config.get("pair_dim", 256),
        d_model=config.get("d_model", 256),
        n_heads=config.get("n_heads", 8),
        n_layers=config.get("n_layers", 4),
        d_ff=config.get("d_ff", 512),
        dropout=config.get("dropout", 0.1),
        site_hidden=config.get("site_hidden", 128),
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Apply ablation by overriding gate logits
    if ablation == "no-bias":
        # sigmoid(-100) ≈ 0 → disables pair bias entirely
        model.bias_gate_logits.data.fill_(-100.0)
    elif ablation == "full-bias":
        # sigmoid(+100) ≈ 1 → always use full pair bias
        model.bias_gate_logits.data.fill_(100.0)
    elif ablation != "learned":
        raise ValueError(f"Unknown ablation mode: {ablation}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    model: ChaiThermoTransformer,
    mutations: list[MutationSample],
    embedding_cache: EmbeddingCache,
    device: torch.device,
    min_mutations: int = 10,
) -> tuple[EvaluationResults, dict]:
    """
    Evaluate model on mutations.

    Returns:
        (EvaluationResults, per_protein_dict)
    """
    # Group mutations by protein
    protein_to_mutations: dict[str, list[MutationSample]] = defaultdict(list)
    for mut in mutations:
        protein_to_mutations[mut.wt_name].append(mut)

    proteins = list(protein_to_mutations.keys())

    # Preload embeddings
    embedding_cache.preload(proteins)

    predictions: dict[str, list[float]] = defaultdict(list)
    targets: dict[str, list[float]] = defaultdict(list)
    wt_residues_collected: dict[str, list[str]] = defaultdict(list)
    mut_residues_collected: dict[str, list[str]] = defaultdict(list)

    for protein_name in tqdm(proteins, desc="Evaluating"):
        muts = protein_to_mutations[protein_name]

        single, pair = embedding_cache.get(protein_name)
        single = single.to(device)
        pair = pair.to(device)

        positions = torch.tensor([m.position for m in muts], device=device)
        wt_indices = torch.tensor([AA_TO_IDX[m.wt_residue] for m in muts], device=device)
        mut_indices = torch.tensor([AA_TO_IDX[m.mut_residue] for m in muts], device=device)

        preds = model(single, pair, positions, wt_indices, mut_indices)

        predictions[protein_name].extend(preds.cpu().tolist())
        targets[protein_name].extend([m.ddg for m in muts])
        wt_residues_collected[protein_name].extend([m.wt_residue for m in muts])
        mut_residues_collected[protein_name].extend([m.mut_residue for m in muts])

    results = compute_metrics(predictions, targets, min_mutations=min_mutations)

    per_protein = {
        "predictions": dict(predictions),
        "targets": dict(targets),
        "wt_residues": dict(wt_residues_collected),
        "mut_residues": dict(mut_residues_collected),
    }

    return results, per_protein


def print_per_protein_breakdown(per_protein: dict, top_k: int = 10):
    """Print per-protein results sorted by Spearman."""
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


def print_model_info(model: ChaiThermoTransformer):
    """Print model info including learned scales and gates."""
    print("\nModel Info:")
    print("-" * 50)
    print(f"  Parameters: {model.num_parameters:,}")

    # Bias scales
    scales = model.get_bias_scale_values()
    print(f"  Bias scales (per head): {[f'{s:.3f}' for s in scales]}")

    # Gate values
    gates = model.get_gate_values()
    for layer, vals in gates.items():
        if isinstance(vals, list):
            print(f"  Gates {layer}: {[f'{v:.3f}' for v in vals]}")
        else:
            print(f"  Gates {layer}: {vals:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chai-Thermo-Transformer")
    parser.add_argument("--run-dir", type=Path, help="Path to run directory (contains model.pt and config.json)")
    parser.add_argument("--model", type=Path, help="Path to model.pt file (requires --config)")
    parser.add_argument("--config", type=Path, help="Path to config.json file")
    parser.add_argument("--embedding-dir", type=Path, default=None,
                        help="Path to embeddings (default: from config)")
    parser.add_argument("--splits-file", type=Path, default=None,
                        help="Path to splits file (default: from config)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate")
    parser.add_argument("--per-protein", action=argparse.BooleanOptionalAction, default=True,
                        help="Show per-protein breakdown (use --no-per-protein to disable)")
    parser.add_argument("--model-info", action="store_true", help="Show model info (gates, scales)")
    parser.add_argument("--ablation", choices=["learned", "no-bias", "full-bias"], default="learned",
                        help="Ablation mode: learned (default), no-bias (gates=0), full-bias (gates=1)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Save results to JSON")

    args = parser.parse_args()

    # Resolve paths
    if args.run_dir:
        model_path = args.run_dir / "model.pt"
        config_path = args.run_dir / "config.json"
    elif args.model and args.config:
        model_path = args.model
        config_path = args.config
    else:
        parser.error("Must provide either --run-dir or both --model and --config")

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    # Load config
    config = load_config(config_path)

    # Override paths if provided
    embedding_dir = args.embedding_dir or Path(config.get("embedding_dir", "data/embeddings/chai_trunk"))
    splits_file = args.splits_file or Path(config.get("splits_file", "data/mega_splits.pkl"))
    data_path = Path(config.get("data_path", "data/megascale.parquet"))
    cv_fold = config.get("cv_fold")

    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Embeddings: {embedding_dir}")
    print(f"Splits: {splits_file}")
    print(f"CV fold: {cv_fold}")
    print(f"Split: {args.split}")

    # Load model
    device = torch.device(args.device)
    model = load_model(model_path, config, device, ablation=args.ablation)
    print(f"Device: {device}")
    print(f"Ablation: {args.ablation}")

    if args.model_info:
        print_model_info(model)

    # Load mutations for the requested split
    mutations = load_thermompnn_mutations(
        splits_file=splits_file,
        split=args.split,
        cv_fold=cv_fold,
        data_path=data_path,
    )

    print(f"\nEvaluating on {len(mutations)} mutations...")

    # Create embedding cache
    cache = EmbeddingCache(embedding_dir, max_cached=64)

    # Evaluate
    results, per_protein = evaluate(model, mutations, cache, device)

    print(f"\nResults ({args.split}):")
    print(results.summary())

    if args.per_protein:
        print_per_protein_breakdown(per_protein)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
