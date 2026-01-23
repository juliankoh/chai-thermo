"""Analyze discrepancy between evaluated mutants and official ThermoMPNN benchmark."""

import pickle
from pathlib import Path
import pandas as pd

# Paths
SPLITS_FILE = Path("mega_splits.pkl")
DATA_FILE = Path("data/megascale.parquet")
EMBEDDINGS_DIR = Path("data/embeddings/chai_trunk")


def main():
    print("=" * 70)
    print("Analyzing Mutant Discrepancy")
    print("=" * 70)

    # 1. Load splits file
    print("\n1. Loading ThermoMPNN splits...")
    with open(SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)

    print(f"   Available keys: {list(splits.keys())}")

    # 2. Count proteins and extract all protein names per split
    print("\n2. Protein counts per split:")
    all_proteins = set()
    test_proteins = set()

    for key in ["train", "val", "test"]:
        if key in splits:
            proteins = set()
            for p in splits[key]:
                if hasattr(p, "item"):
                    p = p.item()
                proteins.add(p.replace(".pdb", ""))
            print(f"   {key}: {len(proteins)} proteins")
            all_proteins.update(proteins)
            if key == "test":
                test_proteins = proteins

    print(f"   Total unique proteins: {len(all_proteins)}")

    # 3. Load megascale data
    print("\n3. Loading MegaScale data...")
    df = pd.read_parquet(DATA_FILE)
    df["protein_name"] = df["WT_name"].str.replace(".pdb", "", regex=False)
    print(f"   Total rows in parquet: {len(df)}")
    print(f"   Unique proteins in parquet: {df['protein_name'].nunique()}")

    # 4. Check which split proteins exist in megascale
    print("\n4. Matching splits to megascale data...")
    megascale_proteins = set(df["protein_name"].unique())

    missing_from_megascale = all_proteins - megascale_proteins
    extra_in_megascale = megascale_proteins - all_proteins

    print(f"   Proteins in splits but NOT in megascale: {len(missing_from_megascale)}")
    if missing_from_megascale:
        print(f"      Examples: {list(missing_from_megascale)[:5]}")

    print(f"   Proteins in megascale but NOT in splits: {len(extra_in_megascale)}")

    # 5. Count mutants per split
    print("\n5. Mutant counts when filtering megascale by split proteins:")
    total_mutants = 0
    for key in ["train", "val", "test"]:
        if key in splits:
            proteins = set()
            for p in splits[key]:
                if hasattr(p, "item"):
                    p = p.item()
                proteins.add(p.replace(".pdb", ""))

            mutants = df[df["protein_name"].isin(proteins)]
            count = len(mutants)
            total_mutants += count
            print(f"   {key}: {count} mutants")

    print(f"   Total: {total_mutants} mutants")

    # 6. Check embeddings
    print("\n6. Checking embeddings...")
    embedding_files = list(EMBEDDINGS_DIR.glob("*.pt"))
    embedding_proteins = {f.stem.replace(".pdb", "") for f in embedding_files}
    print(f"   Total embedding files: {len(embedding_files)}")

    # Which test proteins are missing embeddings?
    test_missing_embeddings = test_proteins - embedding_proteins
    all_missing_embeddings = all_proteins - embedding_proteins

    print(f"   Test proteins missing embeddings: {len(test_missing_embeddings)}")
    if test_missing_embeddings:
        print(f"      Missing: {sorted(test_missing_embeddings)}")

    print(f"   All split proteins missing embeddings: {len(all_missing_embeddings)}")
    if all_missing_embeddings:
        print(f"      Missing: {sorted(all_missing_embeddings)[:10]}...")

    # 7. Count mutants that would be lost due to missing embeddings
    print("\n7. Mutant impact from missing embeddings:")
    for key in ["train", "val", "test"]:
        if key in splits:
            proteins = set()
            for p in splits[key]:
                if hasattr(p, "item"):
                    p = p.item()
                proteins.add(p.replace(".pdb", ""))

            # Proteins with embeddings
            proteins_with_emb = proteins & embedding_proteins
            proteins_without_emb = proteins - embedding_proteins

            mutants_with_emb = df[df["protein_name"].isin(proteins_with_emb)]
            mutants_without_emb = df[df["protein_name"].isin(proteins_without_emb)]

            print(f"   {key}:")
            print(f"      Proteins with embeddings: {len(proteins_with_emb)}")
            print(f"      Proteins without embeddings: {len(proteins_without_emb)}")
            print(f"      Mutants with embeddings: {len(mutants_with_emb)}")
            print(f"      Mutants without embeddings (LOST): {len(mutants_without_emb)}")

    # 8. Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Official count
    official_test = 28312
    your_count = 28172
    difference = official_test - your_count

    print(f"   Official test mutants: {official_test}")
    print(f"   Your evaluated mutants: {your_count}")
    print(f"   Difference: {difference}")

    # Calculate what we found
    test_mutants_in_data = len(df[df["protein_name"].isin(test_proteins)])
    test_mutants_with_emb = len(
        df[df["protein_name"].isin(test_proteins & embedding_proteins)]
    )

    print(f"\n   Test mutants in your megascale.parquet: {test_mutants_in_data}")
    print(f"   Test mutants with embeddings available: {test_mutants_with_emb}")

    # List the specific proteins missing embeddings and their mutant counts
    if test_missing_embeddings:
        print(f"\n   Missing proteins and their mutant counts:")
        for prot in sorted(test_missing_embeddings):
            count = len(df[df["protein_name"] == prot])
            print(f"      {prot}: {count} mutants")

        total_missing = sum(
            len(df[df["protein_name"] == prot]) for prot in test_missing_embeddings
        )
        print(f"\n   Total mutants lost from missing embeddings: {total_missing}")


def compare_with_huggingface():
    """Compare local parquet with HuggingFace dataset."""
    print("\n" + "=" * 70)
    print("Comparing with HuggingFace MegaScale dataset")
    print("=" * 70)

    from datasets import load_dataset

    # Load HF dataset
    print("\nLoading HuggingFace dataset...")
    ds = load_dataset("RosettaCommons/MegaScale", "dataset3_single_cv")

    # Count total rows across all splits
    hf_total = sum(len(ds[split]) for split in ds.keys())
    print(f"HuggingFace total rows: {hf_total}")
    for split in ds.keys():
        print(f"   {split}: {len(ds[split])}")

    # Load local parquet
    df = pd.read_parquet(DATA_FILE)
    print(f"\nLocal parquet total rows: {len(df)}")

    # Load splits
    with open(SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)

    # Get test proteins
    test_proteins = set()
    for p in splits["test"]:
        if hasattr(p, "item"):
            p = p.item()
        test_proteins.add(p.replace(".pdb", ""))

    # Count test mutants in HF dataset
    print("\nCounting test mutants in HuggingFace dataset...")
    hf_test_mutants = []
    seen = set()

    for hf_split in ds.keys():
        for row in ds[hf_split]:
            key = (row["WT_name"], row["mut_type"])
            if key not in seen:
                seen.add(key)
                wt_name = row["WT_name"].replace(".pdb", "")
                if wt_name in test_proteins:
                    hf_test_mutants.append(row)

    print(f"   HuggingFace test mutants (deduplicated): {len(hf_test_mutants)}")

    # Count in local
    df["protein_name"] = df["WT_name"].str.replace(".pdb", "", regex=False)
    local_test = df[df["protein_name"].isin(test_proteins)]
    print(f"   Local parquet test mutants: {len(local_test)}")

    # Find the difference
    print("\nFinding missing mutants...")

    # Create sets of (protein, mutation)
    local_mutations = set(zip(local_test["WT_name"], local_test["mut_type"]))
    hf_mutations = {(r["WT_name"], r["mut_type"]) for r in hf_test_mutants}

    missing = hf_mutations - local_mutations
    extra = local_mutations - hf_mutations

    print(f"   Mutants in HF but NOT in local: {len(missing)}")
    print(f"   Mutants in local but NOT in HF: {len(extra)}")

    if missing:
        # Group by protein
        missing_by_protein = {}
        for wt, mut in missing:
            prot = wt.replace(".pdb", "")
            if prot not in missing_by_protein:
                missing_by_protein[prot] = []
            missing_by_protein[prot].append(mut)

        print("\n   Missing mutants by protein:")
        for prot, muts in sorted(missing_by_protein.items()):
            print(f"      {prot}: {len(muts)} mutants")
            if len(muts) <= 5:
                for m in muts:
                    print(f"         - {m}")

    # Also count WITH duplicates (maybe ThermoMPNN counts all rows, not deduped)
    print("\n" + "=" * 70)
    print("Counting WITHOUT deduplication (raw row counts)")
    print("=" * 70)

    hf_test_with_dupes = 0
    for hf_split in ds.keys():
        for row in ds[hf_split]:
            wt_name = row["WT_name"].replace(".pdb", "")
            if wt_name in test_proteins:
                hf_test_with_dupes += 1

    print(f"   HuggingFace test mutants (with dupes): {hf_test_with_dupes}")

    # Check what configurations are available
    print("\n" + "=" * 70)
    print("Checking other dataset configurations")
    print("=" * 70)

    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names("RosettaCommons/MegaScale")
        print(f"   Available configs: {configs}")

        # Load default config and compare
        ds_default = load_dataset("RosettaCommons/MegaScale", "default")
        print(f"\n   Default config splits: {list(ds_default.keys())}")
        for split in ds_default.keys():
            print(f"      {split}: {len(ds_default[split])}")

        # Count test mutants in default
        default_test_count = 0
        for hf_split in ds_default.keys():
            for row in ds_default[hf_split]:
                wt_name = row["WT_name"].replace(".pdb", "")
                if wt_name in test_proteins:
                    default_test_count += 1
        print(f"\n   Default config test mutants (proteins in test split): {default_test_count}")

    except Exception as e:
        print(f"   Error loading configs: {e}")

    # Check dataset3_single specifically
    print("\n" + "=" * 70)
    print("Checking dataset3_single config (non-CV version)")
    print("=" * 70)

    try:
        ds_single = load_dataset("RosettaCommons/MegaScale", "dataset3_single")
        print(f"   Splits: {list(ds_single.keys())}")
        for split in ds_single.keys():
            print(f"      {split}: {len(ds_single[split])}")

        # Count test mutants
        single_test_dedup = set()
        single_test_all = 0
        for hf_split in ds_single.keys():
            for row in ds_single[hf_split]:
                wt_name = row["WT_name"].replace(".pdb", "")
                if wt_name in test_proteins:
                    single_test_all += 1
                    single_test_dedup.add((row["WT_name"], row["mut_type"]))

        print(f"\n   dataset3_single test mutants (deduped): {len(single_test_dedup)}")
        print(f"   dataset3_single test mutants (with dupes): {single_test_all}")

    except Exception as e:
        print(f"   Error: {e}")

    # Check dataset3 (maybe has different processing)
    print("\n" + "=" * 70)
    print("Checking dataset3 config")
    print("=" * 70)

    try:
        ds3 = load_dataset("RosettaCommons/MegaScale", "dataset3")
        print(f"   Splits: {list(ds3.keys())}")
        for split in ds3.keys():
            print(f"      {split}: {len(ds3[split])}")

        # This might include multi-mutations
        columns = list(ds3[list(ds3.keys())[0]].features.keys())
        print(f"\n   Columns: {columns}")

        # Count test mutants
        d3_test_dedup = set()
        d3_test_all = 0
        for hf_split in ds3.keys():
            for row in ds3[hf_split]:
                wt_name = row["WT_name"].replace(".pdb", "")
                if wt_name in test_proteins:
                    d3_test_all += 1
                    # Use mut_type if exists, otherwise use aa_seq as key
                    if "mut_type" in row:
                        d3_test_dedup.add((row["WT_name"], row["mut_type"]))
                    else:
                        d3_test_dedup.add((row["WT_name"], row.get("aa_seq", str(row))))

        print(f"\n   dataset3 test mutants (deduped): {len(d3_test_dedup)}")
        print(f"   dataset3 test mutants (with dupes): {d3_test_all}")

    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    main()
    compare_with_huggingface()
