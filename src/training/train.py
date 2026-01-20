"""Training loop for protein stability prediction.

Supports single-fold training and full cross-validation.
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import MegaScaleDataset, collate_mutations
from src.features.mutation_encoder import EmbeddingCache, encode_batch
from src.models.pair_aware_mlp import PairAwareMLP
from src.training.sampler import BalancedProteinSampler, FullDatasetSampler
from src.training.evaluate import evaluate_model, EvaluationResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Run identification
    run_name: Optional[str] = None  # Optional name for this run
    model_type: str = "pair_aware_mlp"  # Architecture identifier

    # Data
    fold: int = 0
    embedding_dir: str = "data/embeddings/chai_trunk"

    # Model
    d_single: int = 384
    d_pair: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 100
    patience: int = 10
    gradient_clip: float = 1.0

    # Sampler
    variants_per_protein: int = 32

    # Loss
    loss_fn: str = "mse"  # "mse" or "huber"
    huber_delta: float = 2.0

    # Mutation encoding
    k_structural: int = 10
    seq_window: int = 5

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "outputs"

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        with open(path) as f:
            return cls(**json.load(f))


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    embedding_cache: EmbeddingCache,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Encode features
        features = encode_batch(
            cache=embedding_cache,
            protein_names=batch["wt_names"],
            positions=batch["positions"].tolist(),
            wt_residues=batch["wt_residues"].tolist(),
            mut_residues=batch["mut_residues"].tolist(),
            k_structural=config.k_structural,
            seq_window=config.seq_window,
        )

        # Move to device
        features = {k: v.to(device) for k, v in features.items()}
        targets = batch["ddg"].to(device)

        # Forward
        preds = model.forward_dict(features).squeeze(-1)

        # Loss
        if config.loss_fn == "huber":
            loss = F.huber_loss(preds, targets, delta=config.huber_delta)
        else:
            loss = F.mse_loss(preds, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train_fold(
    config: TrainingConfig,
    verbose: bool = True,
) -> tuple[nn.Module, EvaluationResults, dict]:
    """
    Train on a single fold.

    Returns:
        Tuple of (trained model, test results, training history)
    """
    device = torch.device(config.device)

    # Set seed
    torch.manual_seed(config.seed)

    # Load datasets
    if verbose:
        logger.info(f"Loading fold {config.fold}...")

    train_dataset = MegaScaleDataset(fold=config.fold, split="train")
    val_dataset = MegaScaleDataset(fold=config.fold, split="val")
    test_dataset = MegaScaleDataset(fold=config.fold, split="test")

    if verbose:
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Dataloaders
    train_sampler = BalancedProteinSampler(
        train_dataset,
        variants_per_protein=config.variants_per_protein,
        shuffle=True,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        collate_fn=collate_mutations,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=FullDatasetSampler(val_dataset, shuffle=False),
        collate_fn=collate_mutations,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=FullDatasetSampler(test_dataset, shuffle=False),
        collate_fn=collate_mutations,
        num_workers=0,
    )

    # Load embeddings
    embedding_cache = EmbeddingCache(config.embedding_dir)

    # Preload embeddings for all proteins in this fold
    all_proteins = set(train_dataset.unique_proteins) | set(val_dataset.unique_proteins) | set(test_dataset.unique_proteins)
    if verbose:
        logger.info(f"Preloading embeddings for {len(all_proteins)} proteins...")
    embedding_cache.preload(list(all_proteins))

    # Model
    model = PairAwareMLP(
        d_single=config.d_single,
        d_pair=config.d_pair,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop
    history = {
        "train_loss": [],
        "val_spearman": [],
        "val_rmse": [],
    }

    best_val_spearman = -1.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, embedding_cache, config, device
        )
        history["train_loss"].append(train_loss)

        # Validate
        val_results = evaluate_model(
            model, val_loader, embedding_cache, encode_batch, device
        )
        history["val_spearman"].append(val_results.mean_spearman)
        history["val_rmse"].append(val_results.rmse)

        scheduler.step()

        if verbose:
            logger.info(
                f"Epoch {epoch+1}/{config.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val Spearman: {val_results.mean_spearman:.4f} | "
                f"Val RMSE: {val_results.rmse:.4f}"
            )

        # Early stopping
        if val_results.mean_spearman > best_val_spearman:
            best_val_spearman = val_results.mean_spearman
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Final test evaluation
    test_results = evaluate_model(
        model, test_loader, embedding_cache, encode_batch, device
    )

    if verbose:
        logger.info(f"\nTest Results (Fold {config.fold}):")
        logger.info(test_results.summary())

    return model, test_results, history


def generate_run_dir(base_output_dir: str, config: TrainingConfig) -> Path:
    """Generate a unique run directory name with timestamp and model info."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.run_name:
        # Use provided run name with timestamp
        run_dir_name = f"{timestamp}_{config.run_name}"
    else:
        # Auto-generate name from model type and key hyperparams
        run_dir_name = f"{timestamp}_{config.model_type}"

    return Path(base_output_dir) / run_dir_name


def train_cv(
    base_config: TrainingConfig,
    folds: list[int] = [0, 1, 2, 3, 4],
    save_models: bool = True,
) -> dict:
    """
    Train on all CV folds and aggregate results.

    Returns:
        Dict with per-fold and aggregated results
    """
    # Create timestamped run directory
    run_dir = generate_run_dir(base_config.output_dir, base_config)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run directory: {run_dir}")

    # Save config once at run level (without fold-specific info)
    base_config.save(run_dir / "config.json")

    all_results = []
    all_histories = []

    for fold in folds:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Fold {fold}")
        logger.info(f"{'='*50}")

        config = TrainingConfig(**{**asdict(base_config), "fold": fold})

        model, test_results, history = train_fold(config, verbose=True)

        all_results.append(test_results)
        all_histories.append(history)

        # Save model and results per fold
        if save_models:
            fold_dir = run_dir / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)

            torch.save(model.state_dict(), fold_dir / "model.pt")

            with open(fold_dir / "results.json", "w") as f:
                json.dump(test_results.to_dict(), f, indent=2)

            # Save training history
            with open(fold_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    # Aggregate results
    spearmans = [r.mean_spearman for r in all_results]
    pearsons = [r.mean_pearson for r in all_results]
    rmses = [r.rmse for r in all_results]

    import numpy as np

    cv_summary = {
        "mean_spearman": float(np.mean(spearmans)),
        "std_spearman": float(np.std(spearmans)),
        "mean_pearson": float(np.mean(pearsons)),
        "std_pearson": float(np.std(pearsons)),
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "per_fold": [r.to_dict() for r in all_results],
    }

    logger.info(f"\n{'='*50}")
    logger.info("Cross-Validation Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Mean Spearman: {cv_summary['mean_spearman']:.4f} ± {cv_summary['std_spearman']:.4f}")
    logger.info(f"Mean Pearson:  {cv_summary['mean_pearson']:.4f} ± {cv_summary['std_pearson']:.4f}")
    logger.info(f"Mean RMSE:     {cv_summary['mean_rmse']:.4f} ± {cv_summary['std_rmse']:.4f}")

    # Save summary
    with open(run_dir / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    logger.info(f"\nResults saved to: {run_dir}")

    return cv_summary


def main():
    parser = argparse.ArgumentParser(description="Train stability predictor")
    parser.add_argument("--fold", type=int, default=None, help="Single fold to train (0-4), or None for all")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run (optional)")
    parser.add_argument("--model-type", type=str, default="pair_aware_mlp", help="Model architecture identifier")
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings/chai_trunk")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = TrainingConfig(
        run_name=args.run_name,
        model_type=args.model_type,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        patience=args.patience,
        loss_fn=args.loss,
        device=args.device,
        seed=args.seed,
    )

    if args.fold is not None:
        config.fold = args.fold
        train_fold(config, verbose=True)
    else:
        train_cv(config)


if __name__ == "__main__":
    main()
