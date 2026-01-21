"""
Precompute mutation features for fast training.

Encodes all mutations once and saves them to disk, eliminating
the CPU encoding bottleneck during training.

Usage:
    uv run python scripts/precompute.py --splits-file mega_splits.pkl
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import collate_mutations
from src.features.mutation_encoder import EmbeddingCache, encode_batch
from src.training.train import ThermoMPNNSplitDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def precompute_split(
    dataset,
    embedding_cache: EmbeddingCache,
    split_name: str,
    output_dir: Path,
    batch_size: int = 256,
    k_structural: int = 10,
    seq_window: int = 5,
):
    """Encodes a dataset and saves tensors to disk."""
    logger.info(f"Processing {split_name} ({len(dataset)} samples)...")

    # num_workers=0 because embedding_cache isn't picklable
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_mutations,
        num_workers=0,
        shuffle=False,
    )

    all_features = []
    all_targets = []
    all_protein_names = []

    for batch in tqdm(loader, desc=f"Encoding {split_name}"):
        features = encode_batch(
            cache=embedding_cache,
            protein_names=batch["wt_names"],
            positions=batch["positions"].tolist(),
            wt_residues=batch["wt_residues"].tolist(),
            mut_residues=batch["mut_residues"].tolist(),
            k_structural=k_structural,
            seq_window=seq_window,
        )

        # Clone to ensure we own the memory
        features = {k: v.clone().detach() for k, v in features.items()}
        targets = batch["ddg"].clone().detach()
        protein_names = batch["wt_names"]

        all_features.append(features)
        all_targets.append(targets)
        all_protein_names.extend(protein_names)

    if not all_features:
        logger.warning(f"No data found for {split_name}!")
        return

    # Concatenate into single tensors
    logger.info("Concatenating tensors...")
    keys = all_features[0].keys()
    merged_features = {}

    for key in keys:
        merged_features[key] = torch.cat([b[key] for b in all_features], dim=0)

    merged_targets = torch.cat(all_targets, dim=0)

    # Save
    output_path = output_dir / f"{split_name}_features.pt"
    torch.save(
        {
            "features": merged_features,
            "targets": merged_targets,
            "protein_names": all_protein_names,
        },
        output_path,
    )

    logger.info(f"Saved {split_name} to {output_path}")
    logger.info(f"  Samples: {merged_targets.shape[0]}")
    for key, val in merged_features.items():
        logger.info(f"  {key}: {val.shape}")


def main():
    parser = argparse.ArgumentParser(description="Precompute mutation features")
    parser.add_argument(
        "--splits-file",
        type=str,
        required=True,
        help="Path to mega_splits.pkl",
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="data/embeddings/chai_trunk",
        help="Directory containing .pt embedding files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/precomputed",
        help="Output directory for precomputed features",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--k-structural",
        type=int,
        default=10,
        help="Number of structural neighbors (must match training config)",
    )
    parser.add_argument(
        "--seq-window",
        type=int,
        default=5,
        help="Sequence window half-width (must match training config)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embedding cache once
    logger.info(f"Loading embeddings from {args.embedding_dir}")
    embedding_cache = EmbeddingCache(args.embedding_dir)

    # Process all splits
    for split in ["train", "val", "test"]:
        ds = ThermoMPNNSplitDataset(args.splits_file, split=split)
        logger.info(f"Preloading embeddings for {len(ds.unique_proteins)} proteins...")
        embedding_cache.preload(ds.unique_proteins)

        precompute_split(
            ds,
            embedding_cache,
            split,
            output_dir,
            batch_size=args.batch_size,
            k_structural=args.k_structural,
            seq_window=args.seq_window,
        )

    logger.info("Precomputation complete!")


if __name__ == "__main__":
    main()
