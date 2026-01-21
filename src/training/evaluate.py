"""Evaluation metrics for protein stability prediction.

Primary metric: Per-protein Spearman correlation (standard for MegaScale).
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from torch import Tensor


@dataclass
class EvaluationResults:
    """Container for evaluation metrics."""

    # Per-protein Spearman (primary metric)
    mean_spearman: float
    std_spearman: float
    median_spearman: float
    weighted_spearman: float  # weighted by n_mutations per protein

    # Per-protein Pearson
    mean_pearson: float
    std_pearson: float
    median_pearson: float
    weighted_pearson: float

    # Global metrics (across all mutations)
    global_spearman: float
    global_pearson: float
    rmse: float
    mae: float

    # Metadata
    n_proteins: int
    n_mutations: int

    def to_dict(self) -> dict:
        return {
            "mean_spearman": self.mean_spearman,
            "std_spearman": self.std_spearman,
            "median_spearman": self.median_spearman,
            "weighted_spearman": self.weighted_spearman,
            "mean_pearson": self.mean_pearson,
            "std_pearson": self.std_pearson,
            "median_pearson": self.median_pearson,
            "weighted_pearson": self.weighted_pearson,
            "global_spearman": self.global_spearman,
            "global_pearson": self.global_pearson,
            "rmse": self.rmse,
            "mae": self.mae,
            "n_proteins": self.n_proteins,
            "n_mutations": self.n_mutations,
        }

    def summary(self) -> str:
        return (
            f"{'='*60}\n"
            f"                    Spearman    Pearson\n"
            f"{'='*60}\n"
            f"Global (all muts)   {self.global_spearman:>8.4f}    {self.global_pearson:>8.4f}\n"
            f"Per-protein mean    {self.mean_spearman:>8.4f}    {self.mean_pearson:>8.4f}  (±{self.std_spearman:.4f})\n"
            f"Per-protein median  {self.median_spearman:>8.4f}    {self.median_pearson:>8.4f}\n"
            f"Weighted mean       {self.weighted_spearman:>8.4f}    {self.weighted_pearson:>8.4f}  (by n_mutations)\n"
            f"{'='*60}\n"
            f"RMSE: {self.rmse:.4f} kcal/mol | MAE: {self.mae:.4f} kcal/mol\n"
            f"n_proteins: {self.n_proteins} | n_mutations: {self.n_mutations}"
        )


def compute_metrics(
    predictions: dict[str, list[float]],
    targets: dict[str, list[float]],
    min_mutations: int = 10,
) -> EvaluationResults:
    """
    Compute evaluation metrics from per-protein predictions.

    Args:
        predictions: {protein_name: [pred1, pred2, ...]}
        targets: {protein_name: [true1, true2, ...]}
        min_mutations: Minimum mutations per protein to include in per-protein metrics

    Returns:
        EvaluationResults with all metrics
    """
    all_preds = []
    all_targets = []
    spearmans = []
    pearsons = []
    weights = []  # n_mutations per protein for weighted average

    for protein in predictions.keys():
        preds = predictions[protein]
        trues = targets[protein]

        if len(preds) != len(trues):
            raise ValueError(f"Mismatch for {protein}: {len(preds)} preds vs {len(trues)} targets")

        all_preds.extend(preds)
        all_targets.extend(trues)

        # Per-protein correlations (need enough points)
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

    # Global metrics
    global_spearman, _ = spearmanr(all_preds, all_targets)
    global_pearson, _ = pearsonr(all_preds, all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))

    # Weighted mean (by n_mutations per protein)
    if len(spearmans) > 0 and weights.sum() > 0:
        weighted_spearman = np.average(spearmans, weights=weights)
        weighted_pearson = np.average(pearsons, weights=weights)
    else:
        weighted_spearman = 0.0
        weighted_pearson = 0.0

    return EvaluationResults(
        mean_spearman=np.mean(spearmans) if spearmans else 0.0,
        std_spearman=np.std(spearmans) if spearmans else 0.0,
        median_spearman=np.median(spearmans) if spearmans else 0.0,
        weighted_spearman=weighted_spearman,
        mean_pearson=np.mean(pearsons) if pearsons else 0.0,
        std_pearson=np.std(pearsons) if pearsons else 0.0,
        median_pearson=np.median(pearsons) if pearsons else 0.0,
        weighted_pearson=weighted_pearson,
        global_spearman=global_spearman if not np.isnan(global_spearman) else 0.0,
        global_pearson=global_pearson if not np.isnan(global_pearson) else 0.0,
        rmse=rmse,
        mae=mae,
        n_proteins=len(spearmans),
        n_mutations=len(all_preds),
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    embedding_cache,
    encode_batch_fn,
    device: torch.device,
    min_mutations: int = 10,
) -> EvaluationResults:
    """
    Evaluate model on a dataloader.

    Args:
        model: The model to evaluate
        dataloader: DataLoader yielding batches
        embedding_cache: EmbeddingCache instance
        encode_batch_fn: Function to encode batch features
        device: Device to run on
        min_mutations: Min mutations per protein for per-protein metrics

    Returns:
        EvaluationResults
    """
    model.eval()

    predictions: dict[str, list[float]] = defaultdict(list)
    targets: dict[str, list[float]] = defaultdict(list)

    for batch in dataloader:
        # Encode features
        features = encode_batch_fn(
            cache=embedding_cache,
            protein_names=batch["wt_names"],
            positions=batch["positions"].tolist(),
            wt_residues=batch["wt_residues"].tolist(),
            mut_residues=batch["mut_residues"].tolist(),
        )

        # Move to device
        features = {k: v.to(device) for k, v in features.items()}

        # Forward pass
        preds = model.forward_dict(features).squeeze(-1)

        # Collect per-protein results
        for i, protein in enumerate(batch["wt_names"]):
            predictions[protein].append(preds[i].cpu().item())
            targets[protein].append(batch["ddg"][i].item())

    return compute_metrics(predictions, targets, min_mutations=min_mutations)


def evaluate_precomputed(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    min_mutations: int = 10,
) -> EvaluationResults:
    """
    Evaluate model on precomputed features.

    Args:
        model: The model to evaluate
        dataloader: DataLoader yielding (features, targets, protein_names) batches
        device: Device to run on
        min_mutations: Min mutations per protein for per-protein metrics

    Returns:
        EvaluationResults
    """
    model.eval()

    predictions: dict[str, list[float]] = defaultdict(list)
    targets: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for features, batch_targets, protein_names in dataloader:
            # Move to device
            features = {k: v.to(device) for k, v in features.items()}

            # Forward pass
            preds = model.forward_dict(features).squeeze(-1)

            # Collect per-protein results
            for i, protein in enumerate(protein_names):
                predictions[protein].append(preds[i].cpu().item())
                targets[protein].append(batch_targets[i].item())

    return compute_metrics(predictions, targets, min_mutations=min_mutations)


def analyze_by_mutation_type(
    predictions: dict[str, list[float]],
    targets: dict[str, list[float]],
    wt_residues: dict[str, list[str]],
    mut_residues: dict[str, list[str]],
) -> dict[str, dict]:
    """
    Analyze performance by mutation type (e.g., to identify Cys artifacts).

    Returns metrics grouped by WT residue and by mutation type.
    """
    results = {
        "by_wt_residue": defaultdict(lambda: {"preds": [], "targets": []}),
        "by_mut_residue": defaultdict(lambda: {"preds": [], "targets": []}),
        "cys_involved": {"preds": [], "targets": []},
    }

    for protein in predictions.keys():
        for i, (pred, target, wt, mut) in enumerate(
            zip(
                predictions[protein],
                targets[protein],
                wt_residues[protein],
                mut_residues[protein],
            )
        ):
            results["by_wt_residue"][wt]["preds"].append(pred)
            results["by_wt_residue"][wt]["targets"].append(target)
            results["by_mut_residue"][mut]["preds"].append(pred)
            results["by_mut_residue"][mut]["targets"].append(target)

            if wt == "C" or mut == "C":
                results["cys_involved"]["preds"].append(pred)
                results["cys_involved"]["targets"].append(target)

    # Compute metrics for each group
    summary = {}

    for group_name, group_data in results.items():
        if group_name == "cys_involved":
            if group_data["preds"]:
                preds = np.array(group_data["preds"])
                targs = np.array(group_data["targets"])
                rho, _ = spearmanr(preds, targs)
                rmse = np.sqrt(np.mean((preds - targs) ** 2))
                summary["cys_involved"] = {
                    "n": len(preds),
                    "spearman": rho if not np.isnan(rho) else 0.0,
                    "rmse": rmse,
                }
        else:
            summary[group_name] = {}
            for residue, data in group_data.items():
                if len(data["preds"]) >= 10:
                    preds = np.array(data["preds"])
                    targs = np.array(data["targets"])
                    rho, _ = spearmanr(preds, targs)
                    rmse = np.sqrt(np.mean((preds - targs) ** 2))
                    summary[group_name][residue] = {
                        "n": len(preds),
                        "spearman": rho if not np.isnan(rho) else 0.0,
                        "rmse": rmse,
                    }

    return summary
