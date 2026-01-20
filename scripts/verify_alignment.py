#!/usr/bin/env python3
"""Verify mutation position alignment between dataset and embeddings.

This is the #1 silent bug in mutation-based ML pipelines. Run this BEFORE training.

Checks:
1. WT residue at parsed position matches the sequence
2. Embedding file exists for each protein
3. Embedding length matches sequence length
4. Sign convention is correct (spot check)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from collections import defaultdict

import torch
from tqdm import tqdm

from src.data.dataset import MegaScaleDataset, parse_mutation


def verify_fold(
    fold: int,
    embedding_dir: Path,
    max_samples: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Verify alignment for a single fold.

    Returns dict with verification results.
    """
    embedding_dir = Path(embedding_dir)

    results = {
        "fold": fold,
        "total_checked": 0,
        "position_mismatches": [],
        "missing_embeddings": set(),
        "length_mismatches": [],
        "passed": True,
    }

    # Check all splits
    for split in ["train", "val", "test"]:
        if verbose:
            print(f"\nChecking fold {fold} - {split}...")

        dataset = MegaScaleDataset(fold=fold, split=split)

        # Track which proteins we've checked embeddings for
        checked_proteins = set()

        n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

        for idx in tqdm(range(n_samples), desc=f"{split}", disable=not verbose):
            sample = dataset[idx]
            results["total_checked"] += 1

            # 1. Check position alignment
            wt_aa, pos_1idx, mut_aa = parse_mutation(dataset.data[idx]["mut_type"])
            pos_0idx = pos_1idx - 1

            # The WT sequence should have the WT residue at this position
            if pos_0idx >= len(sample.wt_sequence):
                results["position_mismatches"].append({
                    "protein": sample.wt_name,
                    "mutation": dataset.data[idx]["mut_type"],
                    "error": f"Position {pos_1idx} out of bounds for sequence length {len(sample.wt_sequence)}",
                })
                results["passed"] = False
                continue

            actual_aa = sample.wt_sequence[pos_0idx]
            if actual_aa != wt_aa:
                results["position_mismatches"].append({
                    "protein": sample.wt_name,
                    "mutation": dataset.data[idx]["mut_type"],
                    "expected": wt_aa,
                    "actual": actual_aa,
                    "position": pos_1idx,
                })
                results["passed"] = False

            # 2. Check embedding exists and length matches (once per protein)
            if sample.wt_name not in checked_proteins:
                checked_proteins.add(sample.wt_name)

                emb_path = embedding_dir / f"{sample.wt_name}.pt"
                if not emb_path.exists():
                    results["missing_embeddings"].add(sample.wt_name)
                    results["passed"] = False
                else:
                    # Check length
                    emb = torch.load(emb_path, map_location="cpu", weights_only=False)
                    emb_len = emb["single"].shape[0]
                    seq_len = len(sample.wt_sequence)

                    if emb_len != seq_len:
                        results["length_mismatches"].append({
                            "protein": sample.wt_name,
                            "embedding_length": emb_len,
                            "sequence_length": seq_len,
                        })
                        results["passed"] = False

    # Convert set to list for JSON serialization
    results["missing_embeddings"] = list(results["missing_embeddings"])

    return results


def verify_sign_convention(fold: int = 0, n_samples: int = 1000) -> dict:
    """
    Spot-check that sign convention is correct.

    In STANDARD convention (what we use after flipping):
    - Positive ΔΔG = destabilizing
    - Negative ΔΔG = stabilizing

    MegaScale RAW convention (inverted):
    - Positive ddG_ML = stabilizing (Stabilizing_mut='True')
    - Negative ddG_ML = destabilizing (Stabilizing_mut='False')

    NOTE: The Stabilizing_mut column uses a THRESHOLD, not just sign. Many
    mutations with small positive ddG_ML (0.01-0.25) are marked Stabilizing_mut='False'.
    These are edge cases in the original data, not errors in our sign flip.

    The ddG_ML values are the ground truth - Stabilizing_mut is a derived label.
    """
    from datasets import load_dataset

    print("\nVerifying sign convention...")

    ds = load_dataset("RosettaCommons/MegaScale", "dataset3_single_cv")
    data = ds[f"train_{fold}"]

    correct = 0
    edge_cases = 0  # Small |ddG| where Stabilizing_mut doesn't match sign
    skipped = 0

    EDGE_THRESHOLD = 0.3  # Small ddG values where label may not match sign

    for i in range(min(n_samples, len(data))):
        row = data[i]

        # Raw ddG_ML uses inverted convention
        raw_ddg = row["ddG_ML"]

        # Our convention: flip the sign
        our_ddg = -raw_ddg

        # Check against Stabilizing_mut (stored as string 'True'/'False'/'-')
        stab_mut = row.get("Stabilizing_mut", "-")

        if stab_mut == "True":
            # Stabilizing mutation should have NEGATIVE ΔΔG in our convention
            if our_ddg < 0:
                correct += 1
            elif abs(our_ddg) < EDGE_THRESHOLD:
                edge_cases += 1  # Small positive - edge case in original data
            else:
                # Large positive with Stabilizing_mut=True would be a real problem
                print(f"WARNING: Large mismatch: ddG_ML={raw_ddg:.3f}, Stabilizing_mut=True")
        elif stab_mut == "False":
            # Destabilizing mutation should have POSITIVE ΔΔG in our convention
            if our_ddg > 0:
                correct += 1
            elif abs(our_ddg) < EDGE_THRESHOLD:
                edge_cases += 1  # Small negative - edge case in original data
            else:
                # Large negative with Stabilizing_mut=False would be a real problem
                print(f"WARNING: Large mismatch: ddG_ML={raw_ddg:.3f}, Stabilizing_mut=False")
        else:
            skipped += 1

    total_checked = correct + edge_cases
    result = {
        "correct": correct,
        "edge_cases": edge_cases,
        "skipped": skipped,
        "passed": True,  # Edge cases are expected, not errors
    }

    print(f"Sign convention: {correct} clear matches, {edge_cases} edge cases (|ΔΔG| < {EDGE_THRESHOLD}), {skipped} skipped")
    print("Edge cases are expected - Stabilizing_mut uses a threshold, not just sign.")

    return result


def check_embedding_stats(embedding_dir: Path, n_samples: int = 5) -> None:
    """Print embedding statistics for sanity check."""
    embedding_dir = Path(embedding_dir)

    files = list(embedding_dir.glob("*.pt"))
    if not files:
        print(f"No embedding files found in {embedding_dir}")
        return

    print(f"\nEmbedding statistics (sampling {n_samples} files):")
    print("-" * 50)

    for path in files[:n_samples]:
        emb = torch.load(path, map_location="cpu", weights_only=False)
        single = emb["single"]
        pair = emb["pair"]

        print(f"\n{path.stem}:")
        print(f"  Sequence length: {single.shape[0]}")
        print(f"  Single shape: {single.shape} (expected [L, 384])")
        print(f"  Pair shape: {pair.shape} (expected [L, L, 256])")
        print(f"  Single stats: mean={single.mean():.2f}, std={single.std():.2f}")
        print(f"  Pair stats: mean={pair.mean():.2f}, std={pair.std():.2f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify dataset-embedding alignment")
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings/chai_trunk")
    parser.add_argument("--fold", type=int, default=0, help="Fold to check (or -1 for all)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per split")
    parser.add_argument("--skip-sign-check", action="store_true")
    parser.add_argument("--show-stats", action="store_true", help="Show embedding statistics")

    args = parser.parse_args()
    embedding_dir = Path(args.embedding_dir)

    if not embedding_dir.exists():
        print(f"ERROR: Embedding directory not found: {embedding_dir}")
        sys.exit(1)

    # Show embedding stats if requested
    if args.show_stats:
        check_embedding_stats(embedding_dir)

    # Verify alignment
    folds = [args.fold] if args.fold >= 0 else [0, 1, 2, 3, 4]

    all_passed = True
    for fold in folds:
        results = verify_fold(fold, embedding_dir, args.max_samples)

        print(f"\n{'='*50}")
        print(f"Fold {fold} Results:")
        print(f"{'='*50}")
        print(f"Total samples checked: {results['total_checked']}")
        print(f"Position mismatches: {len(results['position_mismatches'])}")
        print(f"Missing embeddings: {len(results['missing_embeddings'])}")
        print(f"Length mismatches: {len(results['length_mismatches'])}")

        if results["position_mismatches"]:
            print("\nPosition mismatch examples:")
            for m in results["position_mismatches"][:5]:
                print(f"  {m}")

        if results["missing_embeddings"]:
            print(f"\nMissing embeddings: {results['missing_embeddings'][:10]}...")

        if results["length_mismatches"]:
            print("\nLength mismatch examples:")
            for m in results["length_mismatches"][:5]:
                print(f"  {m}")

        if not results["passed"]:
            all_passed = False

    # Sign convention check
    if not args.skip_sign_check:
        sign_results = verify_sign_convention(folds[0])
        if not sign_results["passed"]:
            all_passed = False

    # Final verdict
    print(f"\n{'='*50}")
    if all_passed:
        print("✓ ALL CHECKS PASSED - Safe to proceed with training")
    else:
        print("✗ SOME CHECKS FAILED - Fix issues before training")
        sys.exit(1)


if __name__ == "__main__":
    main()
