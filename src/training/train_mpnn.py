"""Training loop for MPNN-based protein stability prediction.

Uses PyTorch Geometric for graph batching and message passing.
Uses ThermoMPNN splits for training/validation/test.
"""

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
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

from src.data.graph_builder import GraphEmbeddingCache
from src.data.megascale_loader import ThermoMPNNSplitDatasetParquet
from src.models.mpnn import ChaiMPNN, ChaiMPNNWithMutationInfo
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
# Configuration
# =============================================================================


@dataclass
class MPNNTrainingConfig(BaseTrainingConfig):
    """Training hyperparameters for MPNN."""

    model_type: str = "chai_mpnn"

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

    # Training (override defaults)
    batch_size: int = 64
    learning_rate: float = 3e-4

    # Loss
    loss_fn: str = "mse"  # "mse" or "huber"
    huber_delta: float = 2.0
    antisymmetric: bool = False  # Use antisymmetric loss
    antisymmetric_lambda: float = 1.0  # Weight for consistency term


# =============================================================================
# Graph Dataset
# =============================================================================


def build_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    """Build fully connected edge index for a graph with num_nodes nodes."""
    arange = torch.arange(num_nodes, device=device)
    grid_i, grid_j = torch.meshgrid(arange, arange, indexing='ij')
    mask = grid_i != grid_j
    return torch.stack([grid_i[mask], grid_j[mask]])


class MutationGraphDataset(Dataset):
    """
    Dataset that returns PyG Data objects for each mutation.

    Keeps embeddings on GPU for fast indexing.
    """

    def __init__(
        self,
        megascale_dataset,
        graph_cache: GraphEmbeddingCache,
        k_neighbors: int = 30,
        device: str = "cuda",
    ):
        self.k_neighbors = k_neighbors
        self.device = torch.device(device)

        # Move embeddings to GPU
        logger.info(f"Moving embeddings to {device}...")
        self.embeddings = {
            name: (single.to(self.device), pair.to(self.device))
            for name, (single, pair) in graph_cache._cache.items()
        }

        # Cache edge index templates for different graph sizes
        self._edge_index_cache: dict[int, torch.Tensor] = {}

        # Precompute neighbor indices on GPU
        logger.info(f"Precomputing graphs for {len(megascale_dataset)} samples...")
        self.samples = []

        for i in tqdm(range(len(megascale_dataset)), desc="Precomputing"):
            sample = megascale_dataset[i]
            single, pair = self.embeddings[sample.wt_name]
            L = single.shape[0]

            pair_row = pair[sample.position]
            magnitudes = pair_row.norm(dim=-1)
            magnitudes[sample.position] = float("-inf")
            k_actual = min(k_neighbors, L - 1)
            neighbor_indices = magnitudes.topk(k=k_actual).indices

            node_indices = torch.cat([
                torch.tensor([sample.position], device=self.device),
                neighbor_indices,
            ])

            # Store actual number of nodes (no padding)
            num_nodes = len(node_indices)

            self.samples.append({
                'protein': sample.wt_name,
                'node_indices': node_indices,
                'num_nodes': num_nodes,
                'wt_idx': megascale_dataset.encode_residue(sample.wt_residue),
                'mut_idx': megascale_dataset.encode_residue(sample.mut_residue),
                'rel_pos': sample.position / L,
                'ddg': sample.ddg,
            })

    def _get_edge_index(self, num_nodes: int) -> torch.Tensor:
        """Get or create edge index for a graph with num_nodes nodes."""
        if num_nodes not in self._edge_index_cache:
            self._edge_index_cache[num_nodes] = build_edge_index(num_nodes, self.device)
        return self._edge_index_cache[num_nodes]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        s = self.samples[idx]
        single, pair = self.embeddings[s['protein']]
        node_idx = s['node_indices']
        num_nodes = s['num_nodes']

        # Get edge index for this graph size
        edge_index = self._get_edge_index(num_nodes)

        x = single[node_idx]
        edge_attr = pair[node_idx[edge_index[0]], node_idx[edge_index[1]]]

        return Data(
            x=x,
            edge_index=edge_index.clone(),
            edge_attr=edge_attr,
            wt_idx=torch.tensor(s['wt_idx'], device=self.device),
            mut_idx=torch.tensor(s['mut_idx'], device=self.device),
            rel_pos=torch.tensor(s['rel_pos'], device=self.device),
            y=torch.tensor([s['ddg']], device=self.device),
        )


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_mpnn(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    graph_dataset: MutationGraphDataset,
    min_mutations: int = 10,
) -> EvaluationResults:
    """
    Evaluate MPNN model on a dataset.

    Args:
        model: The MPNN model to evaluate
        dataloader: DataLoader with shuffle=False to preserve order
        device: Device to run on
        graph_dataset: MutationGraphDataset (has samples with protein info)
        min_mutations: Minimum mutations per protein for per-protein metrics

    Returns:
        EvaluationResults with all metrics
    """
    model.eval()

    # Collect predictions in dataset order (dataloader must have shuffle=False)
    all_preds: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            preds = model(batch).squeeze(-1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch.y.squeeze(-1).cpu().tolist())

    # Group by protein using dataset's sample info
    predictions: dict[str, list[float]] = defaultdict(list)
    targets: dict[str, list[float]] = defaultdict(list)

    for i, (pred, target) in enumerate(zip(all_preds, all_targets)):
        protein = graph_dataset.samples[i]["protein"]
        predictions[protein].append(pred)
        targets[protein].append(target)

    return compute_metrics(predictions, targets, min_mutations=min_mutations)


# =============================================================================
# Loss Computation
# =============================================================================


def compute_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    config: MPNNTrainingConfig,
) -> torch.Tensor:
    """Compute loss (MSE or Huber)."""
    if config.loss_fn == "huber":
        return F.huber_loss(preds, targets, delta=config.huber_delta)
    return F.mse_loss(preds, targets)


# =============================================================================
# Training Functions
# =============================================================================


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
        targets = batch.y.squeeze(-1)

        # Forward prediction (A -> B)
        preds_fwd = model(batch).squeeze(-1)

        if config.antisymmetric:
            # Antisymmetric loss: train on both A->B and B->A
            wt_idx_orig = batch.wt_idx.clone()
            mut_idx_orig = batch.mut_idx.clone()

            batch.wt_idx = mut_idx_orig
            batch.mut_idx = wt_idx_orig

            # Reverse prediction (B -> A)
            preds_rev = model(batch).squeeze(-1)

            # Restore original indices
            batch.wt_idx = wt_idx_orig
            batch.mut_idx = mut_idx_orig

            # Forward loss: pred(A->B) should match ddG
            loss_fwd = compute_loss(preds_fwd, targets, config)

            # Reverse loss: pred(B->A) should match -ddG
            loss_rev = compute_loss(preds_rev, -targets, config)

            # Consistency loss: pred(A->B) + pred(B->A) should equal 0
            loss_consistency = F.mse_loss(preds_fwd + preds_rev, torch.zeros_like(preds_fwd))

            loss = loss_fwd + loss_rev + config.antisymmetric_lambda * loss_consistency
        else:
            # Standard loss
            loss = compute_loss(preds_fwd, targets, config)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / n_batches


def train(
    config: MPNNTrainingConfig,
    verbose: bool = True,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 20,
) -> tuple[nn.Module, EvaluationResults, TrainingHistory]:
    """
    Train MPNN model.

    Returns:
        Tuple of (trained model, test results, training history)
    """
    device = torch.device(config.device)

    # Set seed
    torch.manual_seed(config.seed)

    # Load datasets using ThermoMPNN splits
    if verbose:
        logger.info(f"Using splits from {config.splits_file}")
        if config.cv_fold is not None:
            logger.info(f"Using CV fold {config.cv_fold}")
        else:
            logger.info("Using main train/val/test split")

    train_megascale = ThermoMPNNSplitDatasetParquet(
        config.splits_file, split="train", cv_fold=config.cv_fold,
        data_path=config.data_path,
    )
    val_megascale = ThermoMPNNSplitDatasetParquet(
        config.splits_file, split="val", cv_fold=config.cv_fold,
        data_path=config.data_path,
    )
    test_megascale = ThermoMPNNSplitDatasetParquet(
        config.splits_file, split="test", cv_fold=config.cv_fold,
        data_path=config.data_path,
    )

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

    # Create graph datasets on GPU
    train_dataset = MutationGraphDataset(train_megascale, graph_cache, config.k_neighbors, device=config.device)
    val_dataset = MutationGraphDataset(val_megascale, graph_cache, config.k_neighbors, device=config.device)
    test_dataset = MutationGraphDataset(test_megascale, graph_cache, config.k_neighbors, device=config.device)

    # DataLoaders - no workers needed since data is on GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
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
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_mult)

    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience, mode="max")

    # Training loop
    history = TrainingHistory()

    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, config, device)
        history.append_train_loss(train_loss)

        scheduler.step()

        # Validate every 5 epochs (or first epoch)
        eval_interval = 5
        should_eval = (epoch + 1) % eval_interval == 0 or epoch == 0

        if should_eval:
            val_results = evaluate_mpnn(model, val_loader, device, val_dataset)
            history.append_val_metrics(val_results.mean_spearman, val_results.rmse)

            if verbose:
                log_epoch(epoch, config.epochs, train_loss, val_results)

            # Early stopping check
            improved = early_stopping(val_results.mean_spearman, model)
            if early_stopping.should_stop:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            if verbose:
                log_epoch(epoch, config.epochs, train_loss)

        # Periodic checkpoint saving
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_dir, epoch + 1, model, optimizer, scheduler,
                early_stopping.best_score or 0.0
            )

    # Load best model
    early_stopping.load_best_model(model, config.device)

    # Final test evaluation
    test_results = evaluate_mpnn(model, test_loader, device, test_dataset)

    if verbose:
        logger.info("\nTest Results:")
        logger.info(test_results.summary())

    return model, test_results, history


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train MPNN stability predictor")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run")
    parser.add_argument("--splits-file", type=str, default="data/mega_splits.pkl",
                        help="Path to mega_splits.pkl")
    parser.add_argument("--cv-fold", type=int, default=None,
                        help="CV fold (0-4) within splits, or None for main split")
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings/chai_trunk")
    parser.add_argument("--data-path", type=str, default="data/megascale.parquet",
                        help="Path to local MegaScale parquet file")
    parser.add_argument("--output-dir", type=str, default="outputs")

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--edge-hidden-dim", type=int, default=128, help="Edge hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of MPNN layers")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--k-neighbors", type=int, default=30, help="Number of neighbors in subgraph")
    parser.add_argument("--no-global-pool", action="store_true", help="Disable global pooling")
    parser.add_argument("--no-mutation-info", action="store_true", help="Don't use mutation identity info")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--antisymmetric", action="store_true",
                        help="Use antisymmetric loss: train on A->B and B->A with consistency constraint")
    parser.add_argument("--antisymmetric-lambda", type=float, default=1.0,
                        help="Weight for antisymmetric consistency term (default: 1.0)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = MPNNTrainingConfig(
        run_name=args.run_name,
        splits_file=args.splits_file,
        cv_fold=args.cv_fold,
        embedding_dir=args.embedding_dir,
        data_path=args.data_path,
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
        antisymmetric=args.antisymmetric,
        antisymmetric_lambda=args.antisymmetric_lambda,
        device=args.device,
        seed=args.seed,
    )

    run_training(config, train)


if __name__ == "__main__":
    main()
