"""Training loop for Chai-Thermo-Transformer.

Structure-aware transformer that uses Chai-1 single embeddings with
pair embeddings as attention biases. Uses ThermoMPNN splits.
"""

import argparse
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.dataset import MutationSample
from src.data.megascale_loader import AA_TO_IDX, load_thermompnn_mutations
from src.features.mutation_encoder import EmbeddingCache
from src.models.transformer import ChaiThermoTransformer
from src.training.common import (
    BaseTrainingConfig,
    EarlyStopping,
    TrainingHistory,
    log_epoch,
    run_training,
    save_checkpoint,
)
from src.training.evaluate import EvaluationResults, compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================


class TransformerDataset(Dataset):
    """Dataset that provides protein batches for the transformer."""

    def __init__(
        self,
        mutations: list[MutationSample],
        embedding_cache: EmbeddingCache,
    ):
        self.mutations = mutations
        self.cache = embedding_cache

        # Group mutations by protein for efficient batching
        self.protein_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, mut in enumerate(mutations):
            self.protein_to_indices[mut.wt_name].append(idx)

        self.proteins = list(self.protein_to_indices.keys())

    def __len__(self):
        return len(self.mutations)

    @property
    def n_proteins(self) -> int:
        return len(self.proteins)

    def get_protein_batch(self, protein_name: str) -> dict:
        """Get all mutations for a protein as a batch."""
        indices = self.protein_to_indices[protein_name]
        mutations = [self.mutations[i] for i in indices]

        single, pair = self.cache.get(protein_name)

        # Sanity checks
        assert single.ndim == 2, f"single should be [L, D], got {single.shape}"
        assert pair.ndim == 3, f"pair should be [L, L, D], got {pair.shape}"
        assert pair.size(0) == single.size(0) == pair.size(1), (
            f"Shape mismatch: single {single.shape}, pair {pair.shape}"
        )

        positions = torch.tensor([m.position for m in mutations])
        L = single.size(0)
        assert (positions >= 0).all() and (positions < L).all(), (
            f"Position out of range [0, {L}): {positions[positions >= L].tolist()}"
        )

        return {
            "single": single,
            "pair": pair,
            "positions": positions,
            "wt_indices": torch.tensor([AA_TO_IDX[m.wt_residue] for m in mutations]),
            "mut_indices": torch.tensor([AA_TO_IDX[m.mut_residue] for m in mutations]),
            "targets": torch.tensor([m.ddg for m in mutations], dtype=torch.float32),
            "protein_name": protein_name,
        }


class ProteinBatchSampler:
    """Yields protein names for batch processing."""

    def __init__(self, dataset: TransformerDataset, shuffle: bool = True):
        self.proteins = dataset.proteins
        self.shuffle = shuffle

    def __iter__(self):
        proteins = self.proteins.copy()
        if self.shuffle:
            random.shuffle(proteins)
        for protein in proteins:
            yield protein

    def __len__(self):
        return len(self.proteins)


# =============================================================================
# Loss Function
# =============================================================================


class DDGLoss(nn.Module):
    """
    Combined loss for ddG prediction.

    - Huber loss: robust to outliers in ddG values
    - Ranking loss: directly optimizes for Spearman correlation
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        rank_weight: float = 0.1,
        rank_margin: float = 0.1,
        n_rank_pairs: int = 256,
        tie_eps: float = 1e-6,
    ):
        super().__init__()
        # Use reduction="sum" for mutation-weighted training
        # (each mutation contributes equally, not each protein)
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="sum")
        self.rank_weight = rank_weight
        self.rank_margin = rank_margin
        self.n_rank_pairs = n_rank_pairs
        self.tie_eps = tie_eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: [M] predictions
            target: [M] targets
        Returns:
            loss: scalar
        """
        pred = pred.view(-1)
        target = target.view(-1)

        # Regression loss
        loss_reg = self.huber(pred, target)

        if self.rank_weight == 0 or pred.numel() < 2:
            return loss_reg

        # Ranking loss: sample pairs, penalize wrong orderings
        M = pred.numel()
        n_pairs = min(self.n_rank_pairs, M * (M - 1) // 2)
        if n_pairs == 0:
            return loss_reg

        # Sample i in [0, M), then j in [0, M-1] shifted to ensure j != i
        i = torch.randint(0, M, (n_pairs,), device=pred.device)
        j = torch.randint(0, M - 1, (n_pairs,), device=pred.device)
        j = j + (j >= i).long()

        target_diff = target[i] - target[j]
        pred_diff = pred[i] - pred[j]

        # Ignore ties - don't penalize when ground truth values are equal
        non_tie = target_diff.abs() > self.tie_eps
        if non_tie.sum() == 0:
            return loss_reg

        target_sign = target_diff[non_tie].sign()
        pred_diff = pred_diff[non_tie]

        # Margin ranking loss (sum for mutation-weighted training)
        loss_rank = F.relu(self.rank_margin - target_sign * pred_diff).sum()

        return loss_reg + self.rank_weight * loss_rank


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TransformerConfig(BaseTrainingConfig):
    """Training hyperparameters for Chai-Thermo-Transformer."""

    model_type: str = "transformer"

    # Model architecture
    single_dim: int = 384
    pair_dim: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    site_hidden: int = 128

    # Training (override defaults)
    learning_rate: float = 1e-4

    # Fine-tuning (stage 2)
    finetune: bool = True
    finetune_lr_factor: float = 0.1  # LR = learning_rate * factor
    finetune_epochs: int = 30
    finetune_patience: int = 10

    # Loss
    huber_delta: float = 1.0
    rank_weight: float = 0.1
    rank_margin: float = 0.1
    n_rank_pairs: int = 256

    # Data
    embedding_cache_size: int = 64


# =============================================================================
# Training Functions
# =============================================================================


def train_epoch(
    model: ChaiThermoTransformer,
    dataset: TransformerDataset,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingWarmRestarts,
    loss_fn: DDGLoss,
    scaler: GradScaler | None,
    config: TransformerConfig,
    epoch: int,
    device: str = "cuda",
) -> dict:
    """Train for one epoch with per-step LR scheduling and mixed precision."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    sampler = ProteinBatchSampler(dataset, shuffle=True)
    n_steps = len(sampler)
    pbar = tqdm(sampler, desc="Training", leave=False)

    use_amp = scaler is not None and device == "cuda"

    for step, protein_name in enumerate(pbar):
        batch = dataset.get_protein_batch(protein_name)

        # Move to device (embeddings may already be on device from cache)
        single = batch["single"].to(device)
        pair = batch["pair"].to(device)
        positions = batch["positions"].to(device)
        wt_indices = batch["wt_indices"].to(device)
        mut_indices = batch["mut_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with optional mixed precision
        if use_amp:
            with autocast(device_type="cuda"):
                predictions = model(single, pair, positions, wt_indices, mut_indices)
                loss = loss_fn(predictions, targets)
            scaler.scale(loss).backward()
            if config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(single, pair, positions, wt_indices, mut_indices)
            loss = loss_fn(predictions, targets)
            loss.backward()
            if config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

        # Per-step LR scheduling (fractional epoch)
        scheduler.step(epoch + (step + 1) / n_steps)

        # Loss is already summed (mutation-weighted), so just accumulate
        total_loss += loss.item()
        n_samples += len(predictions)

        pbar.set_postfix(loss=f"{total_loss / n_samples:.4f}")

    return {"loss": total_loss / n_samples}


@torch.inference_mode()
def validate(
    model: ChaiThermoTransformer,
    dataset: TransformerDataset,
    device: str = "cuda",
    min_mutations: int = 10,
) -> EvaluationResults:
    """Evaluate model on validation/test set."""
    model.eval()

    predictions: dict[str, list[float]] = defaultdict(list)
    targets: dict[str, list[float]] = defaultdict(list)

    sampler = ProteinBatchSampler(dataset, shuffle=False)

    for protein_name in tqdm(sampler, desc="Evaluating", leave=False):
        batch = dataset.get_protein_batch(protein_name)

        single = batch["single"].to(device)
        pair = batch["pair"].to(device)
        positions = batch["positions"].to(device)
        wt_indices = batch["wt_indices"].to(device)
        mut_indices = batch["mut_indices"].to(device)

        preds = model(single, pair, positions, wt_indices, mut_indices)

        predictions[protein_name].extend(preds.cpu().tolist())
        targets[protein_name].extend(batch["targets"].tolist())

    return compute_metrics(predictions, targets, min_mutations=min_mutations)


def train(
    config: TransformerConfig,
    verbose: bool = True,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 20,
) -> tuple[nn.Module, EvaluationResults, TrainingHistory]:
    """
    Train transformer model.

    Returns:
        Tuple of (trained model, test results, training history)
    """
    device = config.device

    # Set seeds (including CUDA for reproducibility)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create embedding cache
    embedding_cache = EmbeddingCache(
        config.embedding_dir,
        device=device,
        max_cached=config.embedding_cache_size,
    )

    # Load datasets using ThermoMPNN splits
    if verbose:
        logger.info(f"Using splits from {config.splits_file}")
        if config.cv_fold is not None:
            logger.info(f"Using CV fold {config.cv_fold}")
        else:
            logger.info("Using main train/val/test split")

    train_mutations = load_thermompnn_mutations(
        Path(config.splits_file), "train", config.cv_fold,
        Path(config.data_path)
    )
    val_mutations = load_thermompnn_mutations(
        Path(config.splits_file), "val", config.cv_fold,
        Path(config.data_path)
    )
    test_mutations = load_thermompnn_mutations(
        Path(config.splits_file), "test", config.cv_fold,
        Path(config.data_path)
    )

    # Create datasets
    train_dataset = TransformerDataset(train_mutations, embedding_cache)
    val_dataset = TransformerDataset(val_mutations, embedding_cache)
    test_dataset = TransformerDataset(test_mutations, embedding_cache)

    if verbose:
        logger.info(
            f"Train: {len(train_mutations)} mutations from {train_dataset.n_proteins} proteins"
        )
        logger.info(
            f"Val: {len(val_mutations)} mutations from {val_dataset.n_proteins} proteins"
        )
        logger.info(
            f"Test: {len(test_mutations)} mutations from {test_dataset.n_proteins} proteins"
        )

    # Preload validation proteins for faster eval
    embedding_cache.preload(val_dataset.proteins)

    # Create model
    model = ChaiThermoTransformer(
        single_dim=config.single_dim,
        pair_dim=config.pair_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        site_hidden=config.site_hidden,
    ).to(device)

    if verbose:
        logger.info(f"Model parameters: {model.num_parameters:,}")

    # Loss function
    loss_fn = DDGLoss(
        huber_delta=config.huber_delta,
        rank_weight=config.rank_weight,
        rank_margin=config.rank_margin,
        n_rank_pairs=config.n_rank_pairs,
    )

    # Optimizer with separate param groups (exclude gates/scales from weight decay)
    no_decay_keywords = ["bias", "LayerNorm", "gate", "scale"]
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_mult
    )

    # Mixed precision scaler (only for CUDA)
    scaler = GradScaler() if device == "cuda" else None

    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience, mode="max")

    # Training loop
    history = TrainingHistory()
    history.extra["bias_gates"] = []
    history.extra["bias_scales"] = []

    for epoch in range(config.epochs):
        # Train (scheduler stepping is done per-step inside train_epoch)
        train_metrics = train_epoch(
            model, train_dataset, optimizer, scheduler, loss_fn,
            scaler, config, epoch, device
        )
        history.append_train_loss(train_metrics["loss"])

        # Log gate and scale values
        history.extra["bias_gates"].append(model.get_gate_values())
        history.extra["bias_scales"].append(model.get_bias_scale_values())

        # Validate every 5 epochs (or first/last epoch)
        eval_interval = 5
        should_eval = (epoch + 1) % eval_interval == 0 or epoch == 0 or epoch == config.epochs - 1

        if should_eval:
            val_results = validate(model, val_dataset, device)
            history.append_val_metrics(val_results.mean_spearman, val_results.rmse)

            if verbose:
                log_epoch(epoch, config.epochs, train_metrics["loss"], val_results)

            # Early stopping check
            improved = early_stopping(val_results.mean_spearman, model)
            if early_stopping.should_stop:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            if verbose:
                log_epoch(epoch, config.epochs, train_metrics["loss"])

        # Periodic checkpoint saving
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_dir, epoch + 1, model, optimizer, scheduler,
                early_stopping.best_score or 0.0
            )

    # Load best model from stage 1
    early_stopping.load_best_model(model, device)
    stage1_spearman = early_stopping.best_score

    if verbose:
        logger.info(f"\nStage 1 complete. Best val Spearman: {stage1_spearman:.4f}")

    # =========================================================================
    # Stage 2: Fine-tuning at lower LR
    # =========================================================================
    if config.finetune and config.finetune_epochs > 0:
        finetune_lr = config.learning_rate * config.finetune_lr_factor
        if verbose:
            logger.info(f"\nStarting fine-tuning stage: LR={finetune_lr:.2e}, "
                       f"epochs={config.finetune_epochs}")

        # New optimizer with lower LR (reuse same param grouping)
        decay_params_ft = []
        no_decay_params_ft = []
        for name, param in model.named_parameters():
            if any(kw in name for kw in no_decay_keywords):
                no_decay_params_ft.append(param)
            else:
                decay_params_ft.append(param)

        optimizer_ft = AdamW(
            [
                {"params": decay_params_ft, "weight_decay": config.weight_decay},
                {"params": no_decay_params_ft, "weight_decay": 0.0},
            ],
            lr=finetune_lr,
        )

        # New scheduler for fine-tuning (shorter cycle)
        scheduler_ft = CosineAnnealingWarmRestarts(
            optimizer_ft, T_0=max(5, config.finetune_epochs // 3), T_mult=1
        )

        # Reset early stopping with fine-tune patience, starting from stage 1 best
        early_stopping_ft = EarlyStopping(patience=config.finetune_patience, mode="max")
        early_stopping_ft.best_score = stage1_spearman
        early_stopping_ft.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        for epoch in range(config.finetune_epochs):
            train_metrics = train_epoch(
                model, train_dataset, optimizer_ft, scheduler_ft, loss_fn,
                scaler, config, epoch, device
            )
            history.append_train_loss(train_metrics["loss"])

            # Validate every epoch during fine-tuning (it's shorter)
            val_results = validate(model, val_dataset, device)
            history.append_val_metrics(val_results.mean_spearman, val_results.rmse)

            if verbose:
                log_epoch(
                    epoch, config.finetune_epochs, train_metrics["loss"],
                    val_results, prefix="FT "
                )

            improved = early_stopping_ft(val_results.mean_spearman, model)
            if early_stopping_ft.should_stop:
                if verbose:
                    logger.info(f"Fine-tuning early stopping at epoch {epoch + 1}")
                break

        # Load best model from fine-tuning (or stage 1 if no improvement)
        early_stopping_ft.load_best_model(model, device)

        if verbose:
            improvement = (early_stopping_ft.best_score or 0) - stage1_spearman
            logger.info(f"Fine-tuning complete. Best val Spearman: "
                       f"{early_stopping_ft.best_score:.4f} "
                       f"({'+'if improvement >= 0 else ''}{improvement:.4f})")

    # Preload test proteins
    embedding_cache.preload(test_dataset.proteins)

    # Final test evaluation
    test_results = validate(model, test_dataset, device)

    if verbose:
        logger.info("\nTest Results:")
        logger.info(test_results.summary())

    return model, test_results, history


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Chai-Thermo-Transformer")

    # Run configuration
    parser.add_argument("--run-name", type=str, default=None, help="Name for this run")
    parser.add_argument("--splits-file", type=str, default="data/mega_splits.pkl",
                        help="Path to mega_splits.pkl")
    parser.add_argument("--cv-fold", type=int, default=None,
                        help="CV fold (0-4) within splits, or None for main split")
    parser.add_argument("--output-dir", type=str, default="outputs")

    # Data
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings/chai_trunk")
    parser.add_argument("--data-path", type=str, default="data/megascale.parquet")

    # Model architecture
    parser.add_argument("--d-model", type=int, default=256, help="Transformer hidden dim")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=512, help="FFN hidden dim")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--site-hidden", type=int, default=128, help="Site head hidden dim")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    # Fine-tuning (stage 2)
    parser.add_argument("--no-finetune", action="store_true",
                        help="Disable fine-tuning stage")
    parser.add_argument("--finetune-lr-factor", type=float, default=0.1,
                        help="LR multiplier for fine-tuning (e.g., 0.1 = lr/10)")
    parser.add_argument("--finetune-epochs", type=int, default=30,
                        help="Max epochs for fine-tuning stage")
    parser.add_argument("--finetune-patience", type=int, default=10,
                        help="Early stopping patience for fine-tuning")

    # Loss
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.1,
                        help="Weight for ranking loss (0 to disable)")
    parser.add_argument("--rank-margin", type=float, default=0.1)

    # Misc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-size", type=int, default=64,
                        help="Number of proteins to cache on GPU")

    args = parser.parse_args()

    config = TransformerConfig(
        run_name=args.run_name,
        splits_file=args.splits_file,
        cv_fold=args.cv_fold,
        embedding_dir=args.embedding_dir,
        data_path=args.data_path,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        site_hidden=args.site_hidden,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
        finetune=not args.no_finetune,
        finetune_lr_factor=args.finetune_lr_factor,
        finetune_epochs=args.finetune_epochs,
        finetune_patience=args.finetune_patience,
        huber_delta=args.huber_delta,
        rank_weight=args.rank_weight,
        rank_margin=args.rank_margin,
        embedding_cache_size=args.cache_size,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    run_training(config, train)


if __name__ == "__main__":
    main()
