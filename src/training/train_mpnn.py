"""Training loop for MPNN-based protein stability prediction.

Uses PyTorch Geometric for graph batching and message passing.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data.dataset import MegaScaleDataset
from src.data.graph_builder import GraphEmbeddingCache
from src.models.mpnn import ChaiMPNN, ChaiMPNNWithMutationInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MPNNTrainingConfig:
    """Training hyperparameters for MPNN."""

    # Run identification
    run_name: Optional[str] = None
    model_type: str = "chai_mpnn"

    # Data
    fold: int = 0
    embedding_dir: str = "data/embeddings/chai_trunk"

    # Model architecture
    node_in_dim: int = 384  # Chai single embedding dim
    edge_in_dim: int = 256  # Chai pair embedding dim
    hidden_dim: int = 128
    edge_hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    use_global_pool: bool = True
    use_mutation_info: bool = True  # Use ChaiMPNNWithMutationInfo

    # Graph construction
    k_neighbors: int = 30

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 100
    patience: int = 15
    gradient_clip: float = 1.0

    # Loss
    loss_fn: str = "mse"  # "mse" or "huber"
    huber_delta: float = 2.0

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "outputs"
    num_workers: int = 0

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MPNNTrainingConfig":
        with open(path) as f:
            return cls(**json.load(f))


class MutationGraphDataset(Dataset):
    """
    Dataset that returns PyG Data objects for each mutation.

    Wraps MegaScaleDataset and builds graphs on-the-fly using cached embeddings.
    """

    def __init__(
        self,
        megascale_dataset: MegaScaleDataset,
        graph_cache: GraphEmbeddingCache,
        k_neighbors: int = 30,
    ):
        self.megascale = megascale_dataset
        self.graph_cache = graph_cache
        self.k_neighbors = k_neighbors

    def __len__(self) -> int:
        return len(self.megascale)

    def __getitem__(self, idx: int) -> Data:
        sample = self.megascale[idx]

        # Get amino acid indices
        wt_idx = self.megascale.encode_residue(sample.wt_residue)
        mut_idx = self.megascale.encode_residue(sample.mut_residue)

        # Build graph
        graph = self.graph_cache.build_graph(
            protein_name=sample.wt_name,
            position=sample.position,
            wt_residue=wt_idx,
            mut_residue=mut_idx,
            k_neighbors=self.k_neighbors,
            ddg=sample.ddg,
        )

        return graph


@dataclass
class EvaluationResults:
    """Evaluation metrics."""

    mean_spearman: float
    std_spearman: float
    median_spearman: float
    mean_pearson: float
    std_pearson: float
    median_pearson: float
    global_spearman: float
    global_pearson: float
    rmse: float
    mae: float
    n_proteins: int
    n_mutations: int

    def summary(self) -> str:
        return (
            f"Mean Spearman: {self.mean_spearman:.4f} ± {self.std_spearman:.4f}\n"
            f"Mean Pearson:  {self.mean_pearson:.4f} ± {self.std_pearson:.4f}\n"
            f"Global Spearman: {self.global_spearman:.4f}\n"
            f"Global Pearson:  {self.global_pearson:.4f}\n"
            f"RMSE: {self.rmse:.4f}, MAE: {self.mae:.4f}\n"
            f"Proteins: {self.n_proteins}, Mutations: {self.n_mutations}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert numpy types to Python types for JSON serialization
        for k, v in d.items():
            if hasattr(v, 'item'):
                d[k] = v.item()
        return d


def evaluate_mpnn(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    megascale_dataset: MegaScaleDataset,
) -> EvaluationResults:
    """Evaluate MPNN model on a dataset."""
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    model.eval()
    all_preds = []
    all_targets = []
    all_proteins = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            preds = model(batch).squeeze(-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch.y.squeeze(-1).cpu().numpy())

    # We need protein assignments for per-protein metrics
    # Since PyG batching loses this info, we'll reconstruct it
    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Get protein names for each sample
    protein_names = [megascale_dataset[i].wt_name for i in range(len(megascale_dataset))]

    # Per-protein correlations
    unique_proteins = list(set(protein_names))
    spearmans = []
    pearsons = []

    for prot in unique_proteins:
        mask = np.array([p == prot for p in protein_names])
        if mask.sum() < 3:
            continue
        p_preds = preds[mask]
        p_targets = targets[mask]
        if np.std(p_preds) > 1e-6 and np.std(p_targets) > 1e-6:
            spearmans.append(spearmanr(p_preds, p_targets)[0])
            pearsons.append(pearsonr(p_preds, p_targets)[0])

    # Global metrics
    global_spearman = spearmanr(preds, targets)[0]
    global_pearson = pearsonr(preds, targets)[0]
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))

    return EvaluationResults(
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: MPNNTrainingConfig,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        preds = model(batch).squeeze(-1)
        targets = batch.y.squeeze(-1)

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

        # Update progress bar with running loss
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / n_batches


def train_fold(
    config: MPNNTrainingConfig,
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

    train_megascale = MegaScaleDataset(fold=config.fold, split="train")
    val_megascale = MegaScaleDataset(fold=config.fold, split="val")
    test_megascale = MegaScaleDataset(fold=config.fold, split="test")

    if verbose:
        logger.info(
            f"Train: {len(train_megascale)}, Val: {len(val_megascale)}, Test: {len(test_megascale)}"
        )

    # Load embeddings
    graph_cache = GraphEmbeddingCache(config.embedding_dir)

    all_proteins = (
        set(train_megascale.unique_proteins)
        | set(val_megascale.unique_proteins)
        | set(test_megascale.unique_proteins)
    )
    if verbose:
        logger.info(f"Preloading embeddings for {len(all_proteins)} proteins...")
    graph_cache.preload(list(all_proteins))

    # Create graph datasets
    train_dataset = MutationGraphDataset(train_megascale, graph_cache, config.k_neighbors)
    val_dataset = MutationGraphDataset(val_megascale, graph_cache, config.k_neighbors)
    test_dataset = MutationGraphDataset(test_megascale, graph_cache, config.k_neighbors)

    # DataLoaders (using PyG's DataLoader)
    use_persistent = config.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )

    # Model
    if config.use_mutation_info:
        model = ChaiMPNNWithMutationInfo(
            node_in_dim=config.node_in_dim,
            edge_in_dim=config.edge_in_dim,
            hidden_dim=config.hidden_dim,
            edge_hidden_dim=config.edge_hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_global_pool=config.use_global_pool,
        ).to(device)
    else:
        model = ChaiMPNN(
            node_in_dim=config.node_in_dim,
            edge_in_dim=config.edge_in_dim,
            hidden_dim=config.hidden_dim,
            edge_hidden_dim=config.edge_hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_global_pool=config.use_global_pool,
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
        train_loss = train_epoch(model, train_loader, optimizer, config, device)
        history["train_loss"].append(train_loss)

        # Validate
        val_results = evaluate_mpnn(model, val_loader, device, val_megascale)
        history["val_spearman"].append(val_results.mean_spearman)
        history["val_rmse"].append(val_results.rmse)

        scheduler.step()

        if verbose:
            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} | "
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
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Final test evaluation
    test_results = evaluate_mpnn(model, test_loader, device, test_megascale)

    if verbose:
        logger.info(f"\nTest Results (Fold {config.fold}):")
        logger.info(test_results.summary())

    return model, test_results, history


def generate_run_dir(base_output_dir: str, config: MPNNTrainingConfig) -> Path:
    """Generate a unique run directory name with timestamp and model info."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.run_name:
        run_dir_name = f"{timestamp}_{config.run_name}"
    else:
        run_dir_name = f"{timestamp}_{config.model_type}"

    return Path(base_output_dir) / run_dir_name


def train_cv(
    base_config: MPNNTrainingConfig,
    folds: list[int] = [0, 1, 2, 3, 4],
    save_models: bool = True,
) -> dict:
    """
    Train on all CV folds and aggregate results.

    Returns:
        Dict with per-fold and aggregated results
    """
    import numpy as np

    # Create timestamped run directory
    run_dir = generate_run_dir(base_config.output_dir, base_config)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run directory: {run_dir}")

    # Log hyperparameters
    logger.info(f"\n{'=' * 50}")
    logger.info("Hyperparameters")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Model: {base_config.model_type}")
    logger.info(f"  hidden_dim: {base_config.hidden_dim}")
    logger.info(f"  edge_hidden_dim: {base_config.edge_hidden_dim}")
    logger.info(f"  num_layers: {base_config.num_layers}")
    logger.info(f"  k_neighbors: {base_config.k_neighbors}")
    logger.info(f"  dropout: {base_config.dropout}")
    logger.info(f"  use_global_pool: {base_config.use_global_pool}")
    logger.info(f"  use_mutation_info: {base_config.use_mutation_info}")
    logger.info(f"  batch_size: {base_config.batch_size}")
    logger.info(f"  learning_rate: {base_config.learning_rate}")
    logger.info(f"  weight_decay: {base_config.weight_decay}")
    logger.info(f"  epochs: {base_config.epochs}")
    logger.info(f"  patience: {base_config.patience}")
    logger.info(f"  loss_fn: {base_config.loss_fn}")
    logger.info(f"  device: {base_config.device}")
    logger.info(f"  seed: {base_config.seed}")
    logger.info(f"  num_workers: {base_config.num_workers}")
    logger.info(f"{'=' * 50}\n")

    # Save config once at run level
    base_config.save(run_dir / "config.json")

    all_results = []
    all_histories = []

    for fold in folds:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Training Fold {fold}")
        logger.info(f"{'=' * 50}")

        config = MPNNTrainingConfig(**{**asdict(base_config), "fold": fold})

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

            with open(fold_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    # Aggregate results
    spearmans = [r.mean_spearman for r in all_results]
    pearsons = [r.mean_pearson for r in all_results]
    rmses = [r.rmse for r in all_results]

    cv_summary = {
        "mean_spearman": float(np.mean(spearmans)),
        "std_spearman": float(np.std(spearmans)),
        "mean_pearson": float(np.mean(pearsons)),
        "std_pearson": float(np.std(pearsons)),
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "per_fold": [r.to_dict() for r in all_results],
    }

    logger.info(f"\n{'=' * 50}")
    logger.info("Cross-Validation Summary")
    logger.info(f"{'=' * 50}")
    logger.info(
        f"Mean Spearman: {cv_summary['mean_spearman']:.4f} ± {cv_summary['std_spearman']:.4f}"
    )
    logger.info(
        f"Mean Pearson:  {cv_summary['mean_pearson']:.4f} ± {cv_summary['std_pearson']:.4f}"
    )
    logger.info(f"Mean RMSE:     {cv_summary['mean_rmse']:.4f} ± {cv_summary['std_rmse']:.4f}")

    # Save summary
    with open(run_dir / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    logger.info(f"\nResults saved to: {run_dir}")

    return cv_summary


def main():
    parser = argparse.ArgumentParser(description="Train MPNN stability predictor")
    parser.add_argument(
        "--fold", type=int, default=None, help="Single fold to train (0-4), or None for all"
    )
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run")
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings/chai_trunk")
    parser.add_argument("--output-dir", type=str, default="outputs")

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--edge-hidden-dim", type=int, default=128, help="Edge hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of MPNN layers")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--k-neighbors", type=int, default=30, help="Number of neighbors in subgraph"
    )
    parser.add_argument("--no-global-pool", action="store_true", help="Disable global pooling")
    parser.add_argument(
        "--no-mutation-info", action="store_true", help="Don't use mutation identity info"
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of DataLoader workers")

    args = parser.parse_args()

    config = MPNNTrainingConfig(
        run_name=args.run_name,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        k_neighbors=args.k_neighbors,
        use_global_pool=not args.no_global_pool,
        use_mutation_info=not args.no_mutation_info,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        loss_fn=args.loss,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    if args.fold is not None:
        config.fold = args.fold
        train_fold(config, verbose=True)
    else:
        train_cv(config)


if __name__ == "__main__":
    main()
