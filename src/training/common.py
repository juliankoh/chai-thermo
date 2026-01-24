"""Shared training utilities for all model architectures.

Provides common functionality:
- Base training configuration
- Run directory generation
- Early stopping
- Checkpoint saving
- Training history
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from src.training.evaluate import EvaluationResults

logger = logging.getLogger(__name__)


# =============================================================================
# Base Configuration
# =============================================================================


@dataclass
class BaseTrainingConfig:
    """
    Base training configuration with fields common to all architectures.

    Subclass this for architecture-specific configs.
    """

    # Run identification
    run_name: Optional[str] = None
    model_type: str = "base"

    # Data paths (splits_file is required for training)
    splits_file: str = "data/mega_splits.pkl"  # ThermoMPNN splits
    cv_fold: Optional[int] = None  # Optional CV fold (0-4) within ThermoMPNN splits
    embedding_dir: str = "data/embeddings/chai_trunk"
    data_path: str = "data/megascale.parquet"

    # Common training hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 100
    patience: int = 15
    gradient_clip: float = 1.0

    # Scheduler (CosineAnnealingWarmRestarts)
    T_0: int = 10
    T_mult: int = 2

    # Misc
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    output_dir: str = "outputs"

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BaseTrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Run Directory Management
# =============================================================================


def generate_run_dir(base_output_dir: str, config: BaseTrainingConfig) -> Path:
    """
    Generate a unique run directory name with timestamp and model info.

    Format: {timestamp}_{run_name or model_type}
    Example: 20240115_143527_transformer
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.run_name:
        run_dir_name = f"{timestamp}_{config.run_name}"
    else:
        run_dir_name = f"{timestamp}_{config.model_type}"

    return Path(base_output_dir) / run_dir_name


# =============================================================================
# Checkpoint Management
# =============================================================================


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_val_metric: float,
    extra: Optional[dict] = None,
) -> Path:
    """
    Save a training checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer to save
        scheduler: LR scheduler to save
        best_val_metric: Best validation metric so far
        extra: Additional data to save

    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_metric": best_val_metric,
    }

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda",
) -> dict:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to

    Returns:
        Checkpoint dict with additional metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# =============================================================================
# Early Stopping
# =============================================================================


class EarlyStopping:
    """
    Early stopping handler.

    Tracks validation metric and signals when to stop.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        """
        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy, 'min' for metrics like loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_model_state: Optional[dict] = None
        self.should_stop = False

    def __call__(self, metric: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            metric: Current validation metric
            model: Current model (state saved if improved)

        Returns:
            True if improved, False otherwise
        """
        if self.mode == "max":
            score = metric
            improved = (
                self.best_score is None or score > self.best_score + self.min_delta
            )
        else:
            score = -metric
            improved = (
                self.best_score is None or score > self.best_score + self.min_delta
            )

        if improved:
            self.best_score = score
            self.best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def load_best_model(self, model: nn.Module, device: str = "cuda") -> None:
        """Load the best model state back into the model."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            model.to(device)


# =============================================================================
# Training History
# =============================================================================


@dataclass
class TrainingHistory:
    """Container for training metrics history."""

    train_loss: list[float] = field(default_factory=list)
    val_spearman: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def append_train_loss(self, loss: float) -> None:
        self.train_loss.append(float(loss))

    def append_val_metrics(self, spearman: float, rmse: float) -> None:
        self.val_spearman.append(float(spearman))
        self.val_rmse.append(float(rmse))

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_spearman": self.val_spearman,
            "val_rmse": self.val_rmse,
            **self.extra,
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Logging Utilities
# =============================================================================


def log_config(config: BaseTrainingConfig, fields: Optional[list[str]] = None) -> None:
    """
    Log configuration hyperparameters.

    Args:
        config: Training configuration
        fields: Optional list of field names to log. If None, logs all fields.
    """
    logger.info(f"\n{'=' * 50}")
    logger.info("Hyperparameters")
    logger.info(f"{'=' * 50}")

    config_dict = asdict(config)

    if fields is None:
        fields = list(config_dict.keys())

    for field_name in fields:
        if field_name in config_dict:
            logger.info(f"  {field_name}: {config_dict[field_name]}")

    logger.info(f"{'=' * 50}\n")


def log_epoch(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_results: Optional[EvaluationResults] = None,
    prefix: str = "",
) -> None:
    """Log epoch progress."""
    if val_results is not None:
        logger.info(
            f"{prefix}Epoch {epoch + 1}/{total_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Spearman: {val_results.mean_spearman:.4f} | "
            f"Val RMSE: {val_results.rmse:.4f}"
        )
    else:
        logger.info(f"{prefix}Epoch {epoch + 1}/{total_epochs} | Loss: {train_loss:.4f}")


# =============================================================================
# Training Runner
# =============================================================================


def run_training(
    config: BaseTrainingConfig,
    train_fn: Callable[
        [BaseTrainingConfig, bool, Optional[Path], int],
        tuple[nn.Module, EvaluationResults, TrainingHistory],
    ],
    checkpoint_interval: int = 20,
) -> tuple[Path, nn.Module, EvaluationResults, TrainingHistory]:
    """
    Run training with the given config and training function.

    Args:
        config: Training configuration
        train_fn: Function that performs training, with signature:
            (config, verbose, checkpoint_dir, checkpoint_interval) -> (model, results, history)
        checkpoint_interval: Save checkpoint every N epochs

    Returns:
        Tuple of (run_dir, model, results, history)
    """
    # Create run directory
    run_dir = generate_run_dir(config.output_dir, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # Save config
    config.save(run_dir / "config.json")

    # Train
    model, test_results, history = train_fn(
        config,
        verbose=True,
        checkpoint_dir=run_dir,
        checkpoint_interval=checkpoint_interval,
    )

    # Save model and results
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Build comprehensive evaluation dict
    eval_dict = {
        "test_results": test_results.to_dict(),
    }

    # Add model info if available (gates, scales for transformer)
    if hasattr(model, "get_gate_values"):
        eval_dict["model_info"] = {
            "bias_gates": model.get_gate_values(),
            "bias_scales": model.get_bias_scale_values() if hasattr(model, "get_bias_scale_values") else None,
            "num_parameters": model.num_parameters if hasattr(model, "num_parameters") else None,
        }

    with open(run_dir / "eval.json", "w") as f:
        json.dump(eval_dict, f, indent=2)

    # Also save legacy results.json for backwards compatibility
    with open(run_dir / "results.json", "w") as f:
        json.dump(test_results.to_dict(), f, indent=2)

    history_dict = history.to_dict() if hasattr(history, 'to_dict') else history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    logger.info(f"\nResults saved to: {run_dir}")
    logger.info(f"  - model.pt, config.json, eval.json, history.json")

    return run_dir, model, test_results, history
