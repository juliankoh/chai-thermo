#!/usr/bin/env python
"""Extract Chai-1 trunk embeddings for all WT proteins.

Usage:
    uv run python scripts/02_extract_embeddings.py

Requirements:
    - Linux with CUDA GPU (A100 80GB recommended)
    - ~10-15 min for 298 proteins on A100

This extracts single [L, D_single] and pair [L, L, D_pair] representations
from Chai-1's trunk, skipping the diffusion (structure prediction) step.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    import torch

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Chai-1 requires a GPU.")
        print("Run this script on a Linux machine with CUDA.")
        sys.exit(1)

    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from src.embeddings.chai_extractor import extract_all_proteins

    # Load sequences
    sequences_path = Path("data/wt_sequences.json")
    if not sequences_path.exists():
        print(f"ERROR: {sequences_path} not found.")
        print("Run the data loading step first to generate this file.")
        sys.exit(1)

    with open(sequences_path) as f:
        sequences = json.load(f)

    print(f"Loaded {len(sequences)} WT sequences")

    # Output directory
    output_dir = Path("data/embeddings/chai_trunk")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract embeddings
    extract_all_proteins(
        sequences=sequences,
        output_dir=output_dir,
        device="cuda:0",
        num_trunk_recycles=3,
        use_esm_embeddings=True,
        skip_existing=True,
    )

    # Summary
    saved_files = list(output_dir.glob("*.pt"))
    print(f"\nDone! Saved {len(saved_files)} embedding files to {output_dir}")


if __name__ == "__main__":
    main()
