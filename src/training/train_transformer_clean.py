"""Clean transformer training script v2 with antisymmetry + optional ranking loss.

Key features:
- Single training loop (no two-stage fine-tuning)
- Warmup + cosine decay schedule
- Huber loss + antisymmetry constraint
- Optional ranking loss for Spearman optimization
- Optional mutation-weighted loss scaling
- Higher default LR (3e-4) matching MPNN

Changes from v1:
- Added optional ranking loss (rank_weight, rank_margin, n_rank_pairs)
- Added loss_weighting option ("uniform" or "mutation")
- Proper config field support for all parameters
- Larger default capacity option (d_model=384, d_ff=1024)
"""

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from src.data.dataset import MutationSample
from src.data.megascale_loader import AA_TO_IDX, load_thermompnn_mutations
from src.features.mutation_encoder import EmbeddingCache
from src.models.transformer import ChaiThermoTransformer
from src.training.evaluate import EvaluationResults, compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TransformerCleanConfig:
    """Clean transformer training config v2."""

    # Run identification
    run_name: Optional[str] = None
    model_type: str = "transformer_clean_v2"

    # Data paths
    splits_file: str = "data/mega_splits.pkl"
    cv_fold: Optional[int] = None
    embedding_dir: str = "data/embeddings/chai_trunk"
    data_path: str = "data/megascale.parquet"

    # Model architecture
    single_dim: int = 384
    pair_dim: int = 256
    d_model: int = 384  # Bumped from 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024  # Bumped from 512
    dropout: float = 0.1
    site_hidden: int = 128

    # Training
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    epochs: int = 100
    patience: int = 20
    gradient_clip: float = 1.0
    warmup_steps: int = 1000  # Increased for larger model

    # Gradient accumulation
    accumulation_steps: int = 1

    # Loss - Huber
    huber_delta: float = 1.0

    # Antisymmetry
    antisymmetric: bool = True
    antisym_lambda: float = 1.0

    # Ranking loss (optional, for Spearman optimization)
    rank_weight: float = 0.0  # 0 = disabled, try 0.02 for gentle ranking
    rank_margin: float = 0.1
    n_rank_pairs: int = 512

    # Loss weighting: "uniform" (each protein equal) or "mutation" (weight by #muts)
    loss_weighting: str = "uniform"

    # Evaluation
    eval_interval: int = 5
    min_mutations_eval: int = 10

    # Misc
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    output_dir: str = "outputs"
    embedding_cache_size: int = 64

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TransformerCleanConfig":
        """Load config, ignoring unknown fields (e.g., old finetune_* fields)."""
        with open(path) as f:
            data = json.load(f)
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


# =============================================================================
# Dataset (unchanged from train_transformer.py)
# =============================================================================


class TransformerDataset:
    """Dataset that provides protein batches for the transformer."""

    def __init__(
        self,
        mutations: list[MutationSample],
        embedding_cache: EmbeddingCache,
    ):
        self.mutations = mutations
        self.cache = embedding_cache

        # Group mutations by protein
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

        positions = torch.tensor([m.position for m in mutations])
        L = single.size(0)

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

    def __init__(self, proteins: list[str], shuffle: bool = True):
        self.proteins = proteins
        self.shuffle = shuffle

    def __iter__(self):
        proteins = self.proteins.copy()
        if self.shuffle:
            random.shuffle(proteins)
        yield from proteins

    def __len__(self):
        return len(self.proteins)


# =============================================================================
# Scheduler: Warmup + Cosine Decay
# =============================================================================


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay to min_lr."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * (self.current_step / max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(1.0, progress)  # Clamp
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]


# =============================================================================
# Early Stopping
# =============================================================================


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.should_stop = False

    def __call__(self, metric: float, model: nn.Module) -> bool:
        score = metric if self.mode == "max" else -metric
        improved = self.best_score is None or score > self.best_score

        if improved:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def load_best_model(self, model: nn.Module, device: str):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            model.to(device)


# =============================================================================
# Training History
# =============================================================================


@dataclass
class TrainingHistory:
    """Container for training metrics."""

    train_loss: list[float] = field(default_factory=list)
    val_spearman: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Training Functions
# =============================================================================


def compute_ranking_loss(
    preds: Tensor,
    targets: Tensor,
    n_pairs: int,
    margin: float,
    tie_eps: float = 1e-6,
) -> Tensor:
    """
    Compute pairwise ranking loss to optimize Spearman correlation.

    Samples random pairs and penalizes incorrect orderings.
    """
    M = preds.numel()
    if M < 2 or n_pairs == 0:
        return torch.tensor(0.0, device=preds.device)

    n_pairs = min(n_pairs, M * (M - 1) // 2)

    # Sample pairs: i != j
    i = torch.randint(0, M, (n_pairs,), device=preds.device)
    j = torch.randint(0, M - 1, (n_pairs,), device=preds.device)
    j = j + (j >= i).long()

    target_diff = targets[i] - targets[j]
    pred_diff = preds[i] - preds[j]

    # Ignore ties
    non_tie = target_diff.abs() > tie_eps
    if non_tie.sum() == 0:
        return torch.tensor(0.0, device=preds.device)

    target_sign = target_diff[non_tie].sign()
    pred_diff = pred_diff[non_tie]

    # Margin ranking loss
    return F.relu(margin - target_sign * pred_diff).sum()


def compute_loss(
    model: ChaiThermoTransformer,
    single: Tensor,
    pair: Tensor,
    positions: Tensor,
    wt_indices: Tensor,
    mut_indices: Tensor,
    targets: Tensor,
    config: TransformerCleanConfig,
) -> tuple[Tensor, Tensor]:
    """
    Compute combined loss: Huber + antisymmetry + optional ranking.

    Returns:
        (total_loss, predictions) - predictions for logging/metrics
    """
    # Forward prediction: A → B
    preds_fwd = model(single, pair, positions, wt_indices, mut_indices)

    # Huber loss (forward)
    loss_huber = F.huber_loss(preds_fwd, targets, delta=config.huber_delta, reduction="sum")

    # Antisymmetry loss
    loss_antisym = torch.tensor(0.0, device=preds_fwd.device)
    if config.antisymmetric:
        # Reverse prediction: B → A
        preds_rev = model(single, pair, positions, mut_indices, wt_indices)

        # Reverse Huber: pred(B→A) should match -ddG
        loss_rev = F.huber_loss(preds_rev, -targets, delta=config.huber_delta, reduction="sum")

        # Consistency: pred(A→B) + pred(B→A) should = 0
        consistency = preds_fwd + preds_rev
        loss_consistency = (consistency ** 2).sum()

        loss_antisym = loss_rev + config.antisym_lambda * loss_consistency

    # Ranking loss (optional)
    loss_rank = torch.tensor(0.0, device=preds_fwd.device)
    if config.rank_weight > 0:
        loss_rank = config.rank_weight * compute_ranking_loss(
            preds_fwd, targets, config.n_rank_pairs, config.rank_margin
        )

    total_loss = loss_huber + loss_antisym + loss_rank
    return total_loss, preds_fwd


def train_epoch(
    model: ChaiThermoTransformer,
    dataset: TransformerDataset,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: Optional[GradScaler],
    config: TransformerCleanConfig,
    device: str,
    avg_muts_per_protein: float,
) -> dict:
    """Train for one epoch with gradient accumulation, antisymmetry, and optional ranking."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    sampler = ProteinBatchSampler(dataset.proteins, shuffle=True)
    pbar = tqdm(sampler, desc="Training", leave=False)

    use_amp = scaler is not None and device == "cuda"
    accum_steps = config.accumulation_steps
    use_mutation_weighting = config.loss_weighting == "mutation"

    optimizer.zero_grad(set_to_none=True)

    for step, protein_name in enumerate(pbar):
        batch = dataset.get_protein_batch(protein_name)

        single = batch["single"].to(device)
        pair = batch["pair"].to(device)
        positions = batch["positions"].to(device)
        wt_indices = batch["wt_indices"].to(device)
        mut_indices = batch["mut_indices"].to(device)
        targets = batch["targets"].to(device)

        n_muts = len(positions)

        # Mutation-weighted scaling: scale so each mutation contributes equally
        # regardless of how many mutations per protein
        if use_mutation_weighting:
            # Normalize: (n_muts / avg_muts) keeps expected gradient magnitude stable
            weight = n_muts / avg_muts_per_protein
        else:
            weight = 1.0

        # Forward + loss
        if use_amp:
            with autocast(device_type="cuda"):
                loss, _ = compute_loss(
                    model, single, pair, positions, wt_indices, mut_indices,
                    targets, config,
                )
                # Apply weighting and accumulation scaling
                loss = (loss * weight) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                if config.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        else:
            loss, _ = compute_loss(
                model, single, pair, positions, wt_indices, mut_indices,
                targets, config,
            )
            loss = (loss * weight) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                if config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        # Track loss (unscaled by accumulation, but keep weight for fair comparison)
        total_loss += loss.item() * accum_steps
        n_samples += n_muts

        pbar.set_postfix(
            loss=f"{total_loss / n_samples:.4f}",
            lr=f"{scheduler.get_lr():.2e}",
        )

    return {"loss": total_loss / n_samples}


@torch.inference_mode()
def validate(
    model: ChaiThermoTransformer,
    dataset: TransformerDataset,
    device: str,
    config: TransformerCleanConfig,
) -> EvaluationResults:
    """Evaluate model on validation/test set."""
    model.eval()

    predictions: dict[str, list[float]] = defaultdict(list)
    targets: dict[str, list[float]] = defaultdict(list)

    sampler = ProteinBatchSampler(dataset.proteins, shuffle=False)

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

    return compute_metrics(predictions, targets, min_mutations=config.min_mutations_eval)


def train(config: TransformerCleanConfig) -> tuple[nn.Module, EvaluationResults, TrainingHistory]:
    """Main training function."""
    device = config.device

    # Seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Embedding cache
    embedding_cache = EmbeddingCache(
        config.embedding_dir,
        device=device,
        max_cached=config.embedding_cache_size,
    )

    # Load data
    logger.info(f"Loading data from {config.splits_file}")
    train_mutations = load_thermompnn_mutations(
        Path(config.splits_file), "train", config.cv_fold, Path(config.data_path)
    )
    val_mutations = load_thermompnn_mutations(
        Path(config.splits_file), "val", config.cv_fold, Path(config.data_path)
    )
    test_mutations = load_thermompnn_mutations(
        Path(config.splits_file), "test", config.cv_fold, Path(config.data_path)
    )

    train_dataset = TransformerDataset(train_mutations, embedding_cache)
    val_dataset = TransformerDataset(val_mutations, embedding_cache)
    test_dataset = TransformerDataset(test_mutations, embedding_cache)

    logger.info(f"Train: {len(train_mutations)} mutations, {train_dataset.n_proteins} proteins")
    logger.info(f"Val: {len(val_mutations)} mutations, {val_dataset.n_proteins} proteins")
    logger.info(f"Test: {len(test_mutations)} mutations, {test_dataset.n_proteins} proteins")

    # Compute average mutations per protein for mutation-weighted training
    avg_muts_per_protein = len(train_mutations) / train_dataset.n_proteins
    logger.info(f"Avg mutations per protein: {avg_muts_per_protein:.1f}")

    # Preload val proteins
    embedding_cache.preload(val_dataset.proteins)

    # Model
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

    logger.info(f"Model parameters: {model.num_parameters:,}")

    # Optimizer with weight decay groups
    no_decay = ["bias", "LayerNorm", "gate", "scale"]
    decay_params = [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)]
    no_decay_params = [p for n, p in model.named_parameters() if any(k in n for k in no_decay)]

    optimizer = AdamW([
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.learning_rate)

    # Scheduler: warmup + cosine
    steps_per_epoch = len(train_dataset.proteins) // config.accumulation_steps
    total_steps = steps_per_epoch * config.epochs

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        max_lr=config.learning_rate,
        min_lr=config.min_lr,
    )

    # Mixed precision
    scaler = GradScaler() if device == "cuda" else None

    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience, mode="max")

    # History
    history = TrainingHistory()

    logger.info(f"\nTraining config:")
    logger.info(f"  Model: d_model={config.d_model}, d_ff={config.d_ff}, n_layers={config.n_layers}")
    logger.info(f"  LR: {config.learning_rate} → {config.min_lr} (cosine)")
    logger.info(f"  Warmup: {config.warmup_steps} steps")
    logger.info(f"  Antisymmetric: {config.antisymmetric} (lambda={config.antisym_lambda})")
    logger.info(f"  Ranking loss: weight={config.rank_weight}, margin={config.rank_margin}")
    logger.info(f"  Loss weighting: {config.loss_weighting}")
    logger.info(f"  Gradient accumulation: {config.accumulation_steps}")
    logger.info(f"  Total steps: {total_steps}")

    # Training loop
    for epoch in range(config.epochs):
        train_metrics = train_epoch(
            model, train_dataset, optimizer, scheduler, scaler, config, device,
            avg_muts_per_protein,
        )
        history.train_loss.append(train_metrics["loss"])
        history.learning_rates.append(scheduler.get_lr())

        # Validate periodically
        should_eval = (
            (epoch + 1) % config.eval_interval == 0
            or epoch == 0
            or epoch == config.epochs - 1
        )

        if should_eval:
            val_results = validate(model, val_dataset, device, config)
            history.val_spearman.append(val_results.mean_spearman)
            history.val_rmse.append(val_results.rmse)

            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Val Spearman: {val_results.mean_spearman:.4f} | "
                f"Val RMSE: {val_results.rmse:.4f} | "
                f"LR: {scheduler.get_lr():.2e}"
            )

            improved = early_stopping(val_results.mean_spearman, model)
            if improved:
                logger.info(f"  ↑ New best!")

            if early_stopping.should_stop:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"LR: {scheduler.get_lr():.2e}"
            )

    # Load best model
    early_stopping.load_best_model(model, device)
    logger.info(f"\nBest val Spearman: {early_stopping.best_score:.4f}")

    # Test evaluation
    embedding_cache.preload(test_dataset.proteins)
    test_results = validate(model, test_dataset, device, config)

    logger.info("\nTest Results:")
    logger.info(test_results.summary())

    return model, test_results, history


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Chai-Thermo-Transformer v2 (clean version with ranking loss)"
    )

    parser.add_argument(
        "--config", type=str, help="Path to JSON config file (overrides other args)"
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")

    # Data
    parser.add_argument("--splits-file", type=str, default="data/mega_splits.pkl")
    parser.add_argument("--cv-fold", type=int, default=None)
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings/chai_trunk")
    parser.add_argument("--data-path", type=str, default="data/megascale.parquet")

    # Model (defaults bumped for v2)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--accumulation-steps", type=int, default=1)

    # Antisymmetry
    parser.add_argument("--no-antisym", action="store_true", help="Disable antisymmetry")
    parser.add_argument("--antisym-lambda", type=float, default=1.0)

    # Ranking loss
    parser.add_argument("--rank-weight", type=float, default=0.0,
                        help="Weight for ranking loss (0=disabled, try 0.02)")
    parser.add_argument("--rank-margin", type=float, default=0.1)
    parser.add_argument("--n-rank-pairs", type=int, default=512)

    # Loss weighting
    parser.add_argument("--loss-weighting", type=str, default="uniform",
                        choices=["uniform", "mutation"],
                        help="'uniform': each protein equal, 'mutation': weight by #muts")

    args = parser.parse_args()

    # Load config from file or create from args
    if args.config:
        config = TransformerCleanConfig.load(Path(args.config))
        logger.info(f"Loaded config from {args.config}")
        # Override with CLI args if provided
        if args.device:
            config.device = args.device
        if args.run_name:
            config.run_name = args.run_name
    else:
        config = TransformerCleanConfig(
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
            learning_rate=args.lr,
            min_lr=args.min_lr,
            epochs=args.epochs,
            patience=args.patience,
            warmup_steps=args.warmup_steps,
            accumulation_steps=args.accumulation_steps,
            antisymmetric=not args.no_antisym,
            antisym_lambda=args.antisym_lambda,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            n_rank_pairs=args.n_rank_pairs,
            loss_weighting=args.loss_weighting,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            output_dir=args.output_dir,
        )

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or config.model_type
    run_dir = Path(config.output_dir) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run directory: {run_dir}")

    # Save config
    config.save(run_dir / "config.json")

    # Train
    model, test_results, history = train(config)

    # Save outputs
    torch.save(model.state_dict(), run_dir / "model.pt")

    with open(run_dir / "results.json", "w") as f:
        json.dump(test_results.to_dict(), f, indent=2)

    history.save(run_dir / "history.json")

    # Save model info
    eval_dict = {
        "test_results": test_results.to_dict(),
        "model_info": {
            "bias_gates": model.get_gate_values(),
            "bias_scales": model.get_bias_scale_values(),
            "num_parameters": model.num_parameters,
        },
    }
    with open(run_dir / "eval.json", "w") as f:
        json.dump(eval_dict, f, indent=2)

    logger.info(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
