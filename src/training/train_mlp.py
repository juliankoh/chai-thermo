"""Training loop for protein stability prediction with PairAwareMLP.

Uses ThermoMPNN splits for training/validation/test.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.dataset import collate_mutations
from src.data.megascale_loader import (
    ThermoMPNNSplitDatasetHF,
    ThermoMPNNSplitDatasetParquet,
)
from src.features.mutation_encoder import EmbeddingCache, encode_batch
from src.models.pair_aware_mlp import PairAwareMLP
from src.training.common import (
    BaseTrainingConfig,
    EarlyStopping,
    TrainingHistory,
    log_epoch,
    run_training,
    save_checkpoint,
)
from src.training.evaluate import EvaluationResults, evaluate_model, evaluate_precomputed
from src.training.sampler import BalancedProteinSampler, FullDatasetSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Precomputed Features Dataset
# =============================================================================


class PrecomputedDataset(Dataset):
    """
    Efficient dataset that loads pre-encoded tensors directly from RAM.

    Use scripts/precompute.py to generate the .pt files.
    """

    def __init__(self, pt_file_path: str | Path):
        self.path = Path(pt_file_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Precomputed file not found: {self.path}")

        logger.info(f"Loading precomputed data from {self.path}...")
        data = torch.load(self.path, map_location="cpu", weights_only=False)
        self.features = data["features"]
        self.targets = data["targets"]
        self.protein_names = data.get("protein_names", None)
        self.length = self.targets.shape[0]
        logger.info(f"  Loaded {self.length} samples")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        feats = {k: v[idx] for k, v in self.features.items()}
        protein_name = self.protein_names[idx] if self.protein_names else None
        return feats, self.targets[idx], protein_name


def collate_precomputed(batch: list) -> tuple[dict, torch.Tensor, list]:
    """Collate function for PrecomputedDataset."""
    features_list, targets_list, protein_names = zip(*batch)

    keys = features_list[0].keys()
    batched_features = {k: torch.stack([f[k] for f in features_list]) for k in keys}
    batched_targets = torch.stack(targets_list)

    return batched_features, batched_targets, list(protein_names)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingConfig(BaseTrainingConfig):
    """Training hyperparameters for PairAwareMLP."""

    model_type: str = "pair_aware_mlp"

    # Data
    precomputed_dir: Optional[str] = None  # Path to precomputed features

    # Model
    d_single: int = 384
    d_pair: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1

    # Training (override defaults)
    batch_size: int = 128
    learning_rate: float = 3e-4
    patience: int = 10  # MLP converges faster

    # Sampler
    variants_per_protein: int = 32

    # Loss
    loss_fn: str = "mse"  # "mse" or "huber"
    huber_delta: float = 2.0

    # Mutation encoding
    k_structural: int = 10
    seq_window: int = 5


# =============================================================================
# Training Functions
# =============================================================================


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
    embedding_cache: Optional[EmbeddingCache] = None,
    precomputed: bool = False,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        if precomputed:
            features, targets, _ = batch
        else:
            features = encode_batch(
                cache=embedding_cache,
                protein_names=batch["wt_names"],
                positions=batch["positions"].tolist(),
                wt_residues=batch["wt_residues"].tolist(),
                mut_residues=batch["mut_residues"].tolist(),
                k_structural=config.k_structural,
                seq_window=config.seq_window,
            )
            targets = batch["ddg"]

        features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
        targets = targets.to(device, non_blocking=True)

        preds = model.forward_dict(features).squeeze(-1)

        if config.loss_fn == "huber":
            loss = F.huber_loss(preds, targets, delta=config.huber_delta)
        else:
            loss = F.mse_loss(preds, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train(
    config: TrainingConfig,
    verbose: bool = True,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 20,
) -> tuple[nn.Module, EvaluationResults, TrainingHistory]:
    """
    Train MLP model.

    Returns:
        Tuple of (trained model, test results, training history)
    """
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    # Check if precomputed features are available
    use_precomputed = False
    if config.precomputed_dir:
        precomputed_path = Path(config.precomputed_dir)
        train_pt = precomputed_path / "train_features.pt"
        if train_pt.exists():
            use_precomputed = True
            if verbose:
                logger.info(f"FAST MODE: Using precomputed features from {config.precomputed_dir}")

    if use_precomputed:
        # === FAST PATH: Precomputed features ===
        train_dataset = PrecomputedDataset(precomputed_path / "train_features.pt")
        val_dataset = PrecomputedDataset(precomputed_path / "val_features.pt")
        test_dataset = PrecomputedDataset(precomputed_path / "test_features.pt")

        if verbose:
            logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_precomputed,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_precomputed,
            num_workers=2,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_precomputed,
            num_workers=2,
            pin_memory=True,
        )
        embedding_cache = None

    else:
        # === SLOW PATH: On-the-fly encoding ===
        if verbose:
            logger.info("Using on-the-fly feature encoding")
            logger.info(f"Using splits from {config.splits_file}")
            if config.cv_fold is not None:
                logger.info(f"Using CV fold {config.cv_fold}")
            else:
                logger.info("Using main train/val/test split")

        # Prefer local parquet if available to avoid network fetch
        from pathlib import Path
        if config.data_path and Path(config.data_path).exists():
            if verbose:
                logger.info(f"Using local parquet at {config.data_path}")
            train_dataset = ThermoMPNNSplitDatasetParquet(
                config.splits_file, split="train", cv_fold=config.cv_fold, data_path=config.data_path
            )
            val_dataset = ThermoMPNNSplitDatasetParquet(
                config.splits_file, split="val", cv_fold=config.cv_fold, data_path=config.data_path
            )
            test_dataset = ThermoMPNNSplitDatasetParquet(
                config.splits_file, split="test", cv_fold=config.cv_fold, data_path=config.data_path
            )
        else:
            if verbose and config.data_path:
                logger.info(
                    f"Local parquet not found at {config.data_path}; falling back to HuggingFace dataset"
                )
            train_dataset = ThermoMPNNSplitDatasetHF(
                config.splits_file, split="train", cv_fold=config.cv_fold
            )
            val_dataset = ThermoMPNNSplitDatasetHF(
                config.splits_file, split="val", cv_fold=config.cv_fold
            )
            test_dataset = ThermoMPNNSplitDatasetHF(
                config.splits_file, split="test", cv_fold=config.cv_fold
            )

        if verbose:
            logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

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

        embedding_cache = EmbeddingCache(config.embedding_dir)
        all_proteins = (
            set(train_dataset.unique_proteins)
            | set(val_dataset.unique_proteins)
            | set(test_dataset.unique_proteins)
        )
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
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_mult)

    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience, mode="max")

    # Training loop
    history = TrainingHistory()

    for epoch in range(config.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            config,
            device,
            embedding_cache=embedding_cache,
            precomputed=use_precomputed,
        )
        history.append_train_loss(train_loss)

        # Validate
        if use_precomputed:
            val_results = evaluate_precomputed(model, val_loader, device)
        else:
            val_results = evaluate_model(
                model, val_loader, embedding_cache, encode_batch, device
            )
        history.append_val_metrics(val_results.mean_spearman, val_results.rmse)

        scheduler.step()

        if verbose:
            log_epoch(epoch, config.epochs, train_loss, val_results)

        # Early stopping check
        improved = early_stopping(val_results.mean_spearman, model)
        if early_stopping.should_stop:
            if verbose:
                logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        # Periodic checkpoint saving
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_dir, epoch + 1, model, optimizer, scheduler,
                early_stopping.best_score or 0.0
            )

    # Load best model
    early_stopping.load_best_model(model, config.device)

    # Final test evaluation
    if use_precomputed:
        test_results = evaluate_precomputed(model, test_loader, device)
    else:
        test_results = evaluate_model(
            model, test_loader, embedding_cache, encode_batch, device
        )

    if verbose:
        logger.info("\nTest Results:")
        logger.info(test_results.summary())

    return model, test_results, history


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train MLP stability predictor")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run")
    parser.add_argument("--splits-file", type=str, default="data/mega_splits.pkl",
                        help="Path to mega_splits.pkl")
    parser.add_argument("--cv-fold", type=int, default=None,
                        help="CV fold (0-4) within splits, or None for main split")
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings/chai_trunk")
    parser.add_argument("--data-path", type=str, default="data/megascale.parquet",
                        help="Path to local MegaScale parquet file (uses local if exists)")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--precomputed-dir", type=str, default=None,
                        help="Path to precomputed features (use scripts/precompute.py to generate)")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])

    # Misc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = TrainingConfig(
        run_name=args.run_name,
        splits_file=args.splits_file,
        cv_fold=args.cv_fold,
        embedding_dir=args.embedding_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        precomputed_dir=args.precomputed_dir,
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

    run_training(config, train)


if __name__ == "__main__":
    main()
