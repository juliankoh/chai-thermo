#!/usr/bin/env python3
"""Benchmark dataloader components to find bottlenecks."""

import time
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MegaScaleDataset
from src.data.graph_builder import GraphEmbeddingCache
from src.training.train_mpnn import MutationGraphDataset


def benchmark_megascale_access(megascale: MegaScaleDataset, n_samples: int = 1000):
    """Benchmark raw HuggingFace dataset access."""
    # Warmup
    _ = megascale[0]

    t0 = time.time()
    for i in range(n_samples):
        sample = megascale[i]
    elapsed = time.time() - t0
    print(f"MegaScale access: {elapsed*1000:.1f}ms for {n_samples} samples ({elapsed/n_samples*1000:.2f}ms/sample)")
    return elapsed


def benchmark_embedding_access(graph_cache: GraphEmbeddingCache, protein_names: list[str], n_samples: int = 1000):
    """Benchmark embedding cache access (dict lookup, should be O(1))."""
    # Warmup
    _ = graph_cache.get(protein_names[0])

    t0 = time.time()
    for i in range(n_samples):
        prot = protein_names[i % len(protein_names)]
        single, pair = graph_cache.get(prot)
    elapsed = time.time() - t0
    print(f"Embedding access: {elapsed*1000:.1f}ms for {n_samples} samples ({elapsed/n_samples*1000:.2f}ms/sample)")
    return elapsed


def benchmark_graph_build(
    graph_cache: GraphEmbeddingCache,
    megascale: MegaScaleDataset,
    n_samples: int = 1000,
):
    """Benchmark graph building (assumes embeddings are cached)."""
    # Pre-fetch samples to isolate graph build time
    samples = []
    for i in range(n_samples):
        sample = megascale[i]
        wt_idx = megascale.encode_residue(sample.wt_residue)
        mut_idx = megascale.encode_residue(sample.mut_residue)
        samples.append((sample.wt_name, sample.position, wt_idx, mut_idx, sample.ddg))

    # Warmup
    wt_name, position, wt_idx, mut_idx, ddg = samples[0]
    _ = graph_cache.build_graph(
        protein_name=wt_name, position=position, wt_residue=wt_idx,
        mut_residue=mut_idx, k_neighbors=30, ddg=ddg,
    )

    t0 = time.time()
    for wt_name, position, wt_idx, mut_idx, ddg in samples:
        graph = graph_cache.build_graph(
            protein_name=wt_name,
            position=position,
            wt_residue=wt_idx,
            mut_residue=mut_idx,
            k_neighbors=30,
            ddg=ddg,
        )
    elapsed = time.time() - t0
    print(f"Graph build only: {elapsed*1000:.1f}ms for {n_samples} samples ({elapsed/n_samples*1000:.2f}ms/sample)")
    return elapsed


def benchmark_full_getitem(dataset: MutationGraphDataset, n_samples: int = 1000):
    """Benchmark full __getitem__ (sample access + graph build)."""
    # Warmup
    _ = dataset[0]

    t0 = time.time()
    for i in range(n_samples):
        graph = dataset[i]
    elapsed = time.time() - t0
    print(f"Full __getitem__: {elapsed*1000:.1f}ms for {n_samples} samples ({elapsed/n_samples*1000:.2f}ms/sample)")
    return elapsed


def benchmark_dataloader_only(dataset: MutationGraphDataset, batch_size: int = 1024, n_batches: int = 10):
    """Benchmark DataLoader batching/collation only, no GPU."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Warmup
    _ = next(iter(loader))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        # Don't move to GPU - just measure collation
    elapsed = time.time() - t0
    print(f"DataLoader only (no GPU): {elapsed:.2f}s for {n_batches} batches ({elapsed/n_batches:.2f}s/batch)")
    return elapsed


def benchmark_forward_pass(dataset: MutationGraphDataset, batch_size: int = 1024, n_batches: int = 10):
    """Benchmark DataLoader + model forward pass."""
    from src.models.mpnn import ChaiMPNNWithMutationInfo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChaiMPNNWithMutationInfo(
        node_in_dim=384,
        edge_in_dim=256,
        hidden_dim=128,
        edge_hidden_dim=128,
        num_layers=3,
        dropout=0.1,
    ).to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Warmup
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        _ = model(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = batch.to(device)
            _ = model(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"DataLoader + Forward: {elapsed:.2f}s for {n_batches} batches ({elapsed/n_batches:.2f}s/batch)")
    return elapsed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark dataloader components")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings/chai_trunk")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    print(f"Loading MegaScaleDataset (fold={args.fold}, split={args.split})...")
    megascale = MegaScaleDataset(fold=args.fold, split=args.split)
    print(f"Dataset size: {len(megascale)}")

    print(f"\nLoading embeddings from {args.embedding_dir}...")
    graph_cache = GraphEmbeddingCache(args.embedding_dir)
    graph_cache.preload(megascale.unique_proteins)
    print(f"Loaded embeddings for {len(megascale.unique_proteins)} proteins")

    print(f"\n{'='*60}")
    print("COMPONENT BENCHMARKS")
    print(f"{'='*60}\n")

    # Individual components
    t_megascale = benchmark_megascale_access(megascale, args.n_samples)
    t_embedding = benchmark_embedding_access(graph_cache, megascale.unique_proteins, args.n_samples)
    t_graph = benchmark_graph_build(graph_cache, megascale, args.n_samples)

    print(f"\n{'='*60}")
    print("DATASET BENCHMARKS")
    print(f"{'='*60}\n")

    # Create MutationGraphDataset (this will cache samples)
    print("Creating MutationGraphDataset...")
    dataset = MutationGraphDataset(megascale, graph_cache, k_neighbors=30)

    t_getitem = benchmark_full_getitem(dataset, args.n_samples)

    print(f"\n{'='*60}")
    print("DATALOADER BENCHMARKS")
    print(f"{'='*60}\n")

    t_dataloader = benchmark_dataloader_only(dataset, args.batch_size, args.n_batches)
    t_forward = benchmark_forward_pass(dataset, args.batch_size, args.n_batches)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Per-sample breakdown (ms):")
    print(f"  MegaScale access:  {t_megascale/args.n_samples*1000:.2f}ms")
    print(f"  Embedding lookup:  {t_embedding/args.n_samples*1000:.2f}ms")
    print(f"  Graph build:       {t_graph/args.n_samples*1000:.2f}ms")
    print(f"  Full __getitem__:  {t_getitem/args.n_samples*1000:.2f}ms")
    print(f"\nWith batch_size={args.batch_size}:")
    print(f"  Expected batch time from __getitem__: {t_getitem/args.n_samples*args.batch_size:.2f}s")
    print(f"  Actual DataLoader (no GPU):           {t_dataloader/args.n_batches:.2f}s/batch")
    print(f"  Actual DataLoader + Forward:          {t_forward/args.n_batches:.2f}s/batch")
    print(f"\nBottleneck analysis:")
    gpu_time = t_forward - t_dataloader
    print(f"  DataLoader overhead: {t_dataloader/args.n_batches:.2f}s/batch")
    print(f"  GPU forward pass:    {gpu_time/args.n_batches:.2f}s/batch")


if __name__ == "__main__":
    main()
