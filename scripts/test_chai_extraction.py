#!/usr/bin/env python
"""Quick test of Chai-1 embedding extraction on a single protein.

Usage:
    uv run python scripts/test_chai_extraction.py

This tests the extraction pipeline on one small protein to verify
everything works before running the full extraction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        print("This script requires a Linux machine with CUDA GPU.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from src.embeddings.chai_extractor import ChaiTrunkExtractor, ChaiEmbeddings

    # Test with a small protein (47 aa)
    test_sequence = "EPELVFKVRVRTKDGRELEIEVSAEDLEKLLEALPDIEEVEIEEVEP"
    test_name = "test_protein"

    print(f"\nExtracting embeddings for test protein ({len(test_sequence)} aa)...")
    print(f"Sequence: {test_sequence[:30]}...")

    # Create extractor (loads models once, keeps in VRAM)
    extractor = ChaiTrunkExtractor(device="cuda:0")

    embeddings = extractor.extract(
        sequence=test_sequence,
        protein_name=test_name,
        num_trunk_recycles=3,
        use_esm_embeddings=True,
    )

    print(f"\n=== Results ===")
    print(f"Protein: {embeddings.protein_name}")
    print(f"Sequence length: {len(embeddings.sequence)}")
    print(f"Single embedding shape: {embeddings.single.shape}")
    print(f"Pair embedding shape: {embeddings.pair.shape}")
    print(f"Single dtype: {embeddings.single.dtype}")
    print(f"Pair dtype: {embeddings.pair.dtype}")

    # Sanity checks
    L = len(test_sequence)
    assert embeddings.single.shape[0] == L, f"Expected L={L}, got {embeddings.single.shape[0]}"
    assert embeddings.pair.shape[0] == L, f"Pair dim 0 mismatch"
    assert embeddings.pair.shape[1] == L, f"Pair dim 1 mismatch"

    D_single = embeddings.single.shape[1]
    D_pair = embeddings.pair.shape[2]
    print(f"\nDimensions: D_single={D_single}, D_pair={D_pair}")

    # Save test embedding
    test_output = Path("data/embeddings/chai_trunk/test_protein.pt")
    test_output.parent.mkdir(parents=True, exist_ok=True)
    embeddings.save(test_output)
    print(f"\nSaved to {test_output}")

    # Test loading
    loaded = ChaiEmbeddings.load(test_output)
    assert torch.allclose(embeddings.single, loaded.single)
    assert torch.allclose(embeddings.pair, loaded.pair)
    print("Load/save test passed!")

    print("\n=== SUCCESS ===")
    print("Chai-1 embedding extraction is working correctly.")


if __name__ == "__main__":
    main()
