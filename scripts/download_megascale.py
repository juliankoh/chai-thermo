"""Download MegaScale dataset from HuggingFace and save locally.

Run once to cache the data:
    uv run python scripts/download_megascale.py

This creates data/megascale.parquet which is used by training/eval scripts.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download MegaScale dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="data/megascale.parquet",
        help="Output path for the parquet file",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Dataset already exists at {output_path}")
        df = pd.read_parquet(output_path)
        logger.info(f"  {len(df)} rows, {df['WT_name'].nunique()} proteins")
        return

    logger.info("Downloading MegaScale dataset from HuggingFace...")
    ds = load_dataset("RosettaCommons/MegaScale", "dataset3_single_cv")
    logger.info(f"Dataset loaded. Splits: {list(ds.keys())}")

    # Collect all unique mutations across all splits
    all_rows = []
    seen = set()

    for split_name in ds.keys():
        logger.info(f"Processing split: {split_name}")
        for row in ds[split_name]:
            key = (row["WT_name"], row["mut_type"])
            if key not in seen:
                seen.add(key)
                all_rows.append({
                    "WT_name": row["WT_name"],
                    "mut_type": row["mut_type"],
                    "aa_seq": row["aa_seq"],
                    "ddG_ML": row["ddG_ML"],
                })

    df = pd.DataFrame(all_rows)
    logger.info(f"Collected {len(df)} unique mutations from {df['WT_name'].nunique()} proteins")

    # Save as parquet
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    # Print some stats
    logger.info("\nDataset statistics:")
    logger.info(f"  Total mutations: {len(df)}")
    logger.info(f"  Unique proteins: {df['WT_name'].nunique()}")
    logger.info(f"  ddG range: [{df['ddG_ML'].min():.2f}, {df['ddG_ML'].max():.2f}]")


if __name__ == "__main__":
    main()
