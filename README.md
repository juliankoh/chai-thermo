# Chai-1 Stability Predictor

**Protein stability (ΔΔG) prediction using Chai-1's pair representations.**

## Results

**5-Fold Cross-Validation on MegaScale `dataset3_single_cv`:**

| Metric | Result |
|--------|--------|
| **Mean Spearman** | **0.716 ± 0.006** |
| Mean Pearson | 0.755 ± 0.009 |
| Mean RMSE | 0.68 kcal/mol |

This significantly exceeds initial targets (0.55 Spearman) and validates the hypothesis that Chai-1's explicit pair representations capture stability-relevant structural information.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Pre-extracted Chai-1 embeddings in `data/embeddings/chai_trunk/`

### Installation

```bash
uv sync  # or pip install -r requirements.txt
```

### Download Data

```bash
# Download MegaScale dataset from HuggingFace (one-time, ~50MB)
uv run python scripts/download_megascale.py
```

This creates `data/megascale.parquet` which is used by training/eval scripts.

### Training

**MLP Model (original):**
```bash
# Train on a single fold
uv run python -m src.training.train --fold 0 --embedding-dir data/embeddings/chai_trunk

# Train all 5 folds (full cross-validation)
uv run python -m src.training.train --embedding-dir data/embeddings/chai_trunk
```

**MPNN Model:**
```bash
# Train with ThermoMPNN's official splits
uv run python -m src.training.train_mpnn \
    --run-name mpnn_thermompnn_splits \
    --thermompnn-splits mega_splits.pkl \
    --batch-size 1024 \
    --epochs 50

# With antisymmetric loss (recommended for better generalization)
uv run python -m src.training.train_mpnn \
    --run-name mpnn_antisym \
    --thermompnn-splits mega_splits.pkl \
    --antisymmetric \
    --batch-size 1024 \
    --epochs 50

# Train with HuggingFace CV splits
uv run python -m src.training.train_mpnn --fold 0

# Full 5-fold CV
uv run python -m src.training.train_mpnn
```

### Evaluation

**MPNN per-protein evaluation:**
```bash
# Evaluate trained model with per-protein breakdown
uv run python -m src.training.eval_mpnn outputs/YYYYMMDD_HHMMSS_run_name/

# Evaluate on validation split, sorted by RMSE
uv run python -m src.training.eval_mpnn outputs/run_dir/ --split val --sort-by rmse

# Show top 50 worst/best proteins
uv run python -m src.training.eval_mpnn outputs/run_dir/ --top-n 50
```

**MLP evaluation:**
```bash
uv run python scripts/evaluate.py --all-folds
uv run python scripts/evaluate.py --all-folds --ablation-suite
```

### Verification (run before training)

```bash
uv run python scripts/verify_alignment.py --embedding-dir data/embeddings/chai_trunk
```

---

## Architecture

### Key Insight

Chai-1's trunk produces **pair representations** `[L, L, 256]` that explicitly encode pairwise residue interactions. For stability prediction, these capture:
- Salt bridges
- Hydrophobic packing
- Steric clashes
- Hydrogen bond networks

ESM-2 infers these from sequence patterns. Chai-1 computes them explicitly.

### Feature Encoding

For each mutation at position `i`, we extract:

| Feature | Dim | Description |
|---------|-----|-------------|
| `local_single` | 384 | Embedding of mutated residue |
| `global_single` | 384 | Protein-average embedding |
| `pair_global` | 256 | Average pairwise interactions (excl. self) |
| `pair_local_seq` | 256 | Interactions with sequence neighbors ±5 |
| `pair_structural` | 256 | Top-10 strongest long-range contacts |
| `mutation_feat` | 41 | WT/MUT one-hots + relative position |

**Total: 1577 dimensions**

### Models

**MLP Model:**
```
Features → [LayerNorm per group] → Concat → MLP(1577→512→512→256→1) → ΔΔG
```

- **~1.2M parameters**
- Separate LayerNorm for each feature group (raw embeddings have std ~400-600)
- GELU activations, dropout 0.1
- Trained with AdamW, cosine annealing, early stopping on validation Spearman

**MPNN Model (ChaiMPNN):**
```
Local subgraph (k=30 neighbors) → Message Passing → Mutation site pooling → MLP → ΔΔG
```

- Graph neural network operating on Chai-1 embeddings
- Node features: single embeddings (384-dim)
- Edge features: pair embeddings (256-dim)
- 3 message passing layers with edge-conditioned updates
- Mutation identity encoding (WT/MUT amino acid + relative position)
- Trained with AdamW, cosine annealing, early stopping

---

## Data

### Source

**MegaScale** dataset from HuggingFace (`RosettaCommons/MegaScale`), config `dataset3_single_cv`:
- ~271k unique single mutations
- ~298 proteins (32-72 aa)
- 5-fold CV splits (by protein, no leakage)
- Standard benchmark from ThermoMPNN

### Splits

Two split options are supported:
- **HuggingFace CV splits**: Default 5-fold cross-validation from the dataset
- **ThermoMPNN splits**: Official train/val/test splits from `mega_splits.pkl` for direct comparison with ThermoMPNN paper results

### Sign Convention

MegaScale raw data uses **inverted** convention. Our code automatically flips to standard:
- **Positive ΔΔG = destabilizing**
- **Negative ΔΔG = stabilizing**

---

## Project Structure

```
chai-thermo/
├── src/
│   ├── data/
│   │   └── dataset.py          # MegaScaleDataset with sign flip
│   ├── embeddings/
│   │   └── chai_extractor.py   # Chai-1 trunk embedding extraction
│   ├── features/
│   │   └── mutation_encoder.py # encode_mutation() + EmbeddingCache
│   ├── models/
│   │   ├── pair_aware_mlp.py   # PairAwareMLP with LayerNorms
│   │   └── mpnn.py             # ChaiMPNN graph neural network
│   └── training/
│       ├── train.py            # MLP training loop + CV
│       ├── train_mpnn.py       # MPNN training (supports ThermoMPNN splits)
│       ├── eval_mpnn.py        # Per-protein MPNN evaluation
│       ├── sampler.py          # BalancedProteinSampler
│       └── evaluate.py         # Metrics computation
├── scripts/
│   ├── download_megascale.py   # Download dataset from HuggingFace
│   ├── verify_alignment.py     # Pre-training sanity checks
│   └── evaluate.py             # Evaluation + ablations
├── data/
│   ├── megascale.parquet       # Local copy of MegaScale dataset
│   └── embeddings/
│       └── chai_trunk/         # Cached embeddings ({protein}.pt)
└── outputs/
    ├── YYYYMMDD_HHMMSS_run_name/   # MPNN run directory
    │   ├── config.json
    │   ├── model.pt                # Final best model
    │   ├── results.json            # Test metrics
    │   ├── history.json            # Training history
    │   ├── eval_test.csv           # Per-protein evaluation
    │   └── checkpoint_epoch_*.pt   # Periodic checkpoints
    ├── fold_0/                     # MLP fold directory
    │   ├── model.pt
    │   ├── config.json
    │   └── results.json
    └── cv_summary.json
```

---

## Embedding Extraction

Embeddings must be pre-extracted using `src/embeddings/chai_extractor.py` (requires GPU):

```bash
# On a machine with A100/H100
python -m src.embeddings.chai_extractor \
    --sequences data/wt_sequences.json \
    --output-dir data/embeddings/chai_trunk
```

Each protein produces a `.pt` file containing:
- `single`: `[L, 384]` per-residue embeddings
- `pair`: `[L, L, 256]` pairwise interaction embeddings

Total storage: ~426 MB for 298 proteins.

---

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Batch size | 128 |
| Max epochs | 100 |
| Early stopping | patience=10 on val Spearman |
| Scheduler | CosineAnnealingWarmRestarts |
| Gradient clip | 1.0 |

### Balanced Sampling

Each epoch samples uniformly across proteins (32 mutations per protein), preventing large proteins from dominating gradients.

### Antisymmetric Loss (MPNN)

Optional training augmentation that enforces physical reversibility:

```
Loss = MSE(pred(A→B), ΔΔG) + MSE(pred(B→A), -ΔΔG) + λ·MSE(pred(A→B) + pred(B→A), 0)
```

**Why it helps:**
- Forces the model to learn that mutation effects are reversible
- Acts as a regularizer, preventing memorization of "mutating to X is always bad"
- Forces learning the context of the fit, not just amino acid identity
- Particularly effective on de novo designs where signal-to-noise is lower

Enable with `--antisymmetric` flag. Adjust consistency weight with `--antisymmetric-lambda` (default: 1.0).

---

## Ablation Studies

To understand feature contributions:

```bash
uv run python scripts/evaluate.py --all-folds --ablation-suite
```

Available ablations:
- `no-pair` — Single features only (ESM-style baseline)
- `no-single` — Pair features only
- `pair-global-only` — Only average pairwise interactions
- `pair-structural-only` — Only top-k structural neighbors
- `no-mutation-feat` — Remove WT/MUT identity

---

## Known Limitations

1. **Domain size**: Trained on 32-72 aa proteins; may not generalize to larger domains
2. **Dynamic range**: MegaScale spans -3 to +5 kcal/mol; extreme values are extrapolations
3. **Cysteine artifacts**: Surface Cys mutations may show assay artifacts (disulfide-related)
4. **De novo designs**: ~148/298 proteins are de novo designs with no evolutionary history

---

## Citation

If you use this work, please cite:

- **Chai-1**: [Chai Discovery](https://chaidiscovery.com)
- **MegaScale**: Tsuboyama et al., "Mega-scale experimental analysis of protein folding stability in biology and design" (2023)
- **ThermoMPNN**: [Reference for benchmark splits]
