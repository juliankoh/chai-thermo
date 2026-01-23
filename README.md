# Chai-1 Stability Predictor

**Protein stability (ddG) prediction using Chai-1's pair representations.**

Uses ThermoMPNN's official data splits for training and evaluation.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Pre-extracted Chai-1 embeddings in `data/embeddings/chai_trunk/`
- ThermoMPNN splits file (`data/mega_splits.pkl`)

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

**MLP Model:**
```bash
uv run python -m src.training.train_mlp --run-name mlp_baseline
```

**MPNN Model:**
```bash
# Standard training
uv run python -m src.training.train_mpnn --run-name mpnn_baseline

# With antisymmetric loss (recommended for better generalization)
uv run python -m src.training.train_mpnn \
    --run-name mpnn_antisym \
    --antisymmetric \
    --batch-size 1024
```

**Transformer Model:**
```bash
uv run python -m src.training.train_transformer --run-name transformer_baseline
```

All training scripts support:
- `--splits-file PATH` - Path to mega_splits.pkl (default: `data/mega_splits.pkl`)
- `--cv-fold N` - Use CV fold 0-4 within splits, or omit for main split
- `--run-name NAME` - Name for this training run

### Evaluation

**MPNN per-protein evaluation:**
```bash
# Evaluate trained model with per-protein breakdown
uv run python -m src.training.eval_mpnn outputs/YYYYMMDD_HHMMSS_run_name/

# Evaluate on validation split, sorted by RMSE
uv run python -m src.training.eval_mpnn outputs/run_dir/ --split val --sort-by rmse
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
| `pair_local_seq` | 256 | Interactions with sequence neighbors +/-5 |
| `pair_structural` | 256 | Top-10 strongest long-range contacts |
| `mutation_feat` | 41 | WT/MUT one-hots + relative position |

**Total: 1577 dimensions**

### Models

**MLP Model:**
```
Features -> [LayerNorm per group] -> Concat -> MLP(1577->512->512->256->1) -> ddG
```

- **~1.2M parameters**
- Separate LayerNorm for each feature group (raw embeddings have std ~400-600)
- GELU activations, dropout 0.1
- Trained with AdamW, cosine annealing, early stopping on validation Spearman

**MPNN Model (ChaiMPNN):**
```
Local subgraph (k=30 neighbors) -> Message Passing -> Mutation site pooling -> MLP -> ddG
```

- Graph neural network operating on Chai-1 embeddings
- Node features: single embeddings (384-dim)
- Edge features: pair embeddings (256-dim)
- 3 message passing layers with edge-conditioned updates
- Mutation identity encoding (WT/MUT amino acid + relative position)
- Trained with AdamW, cosine annealing, early stopping

**Transformer Model (ChaiThermoTransformer):**
```
Single embeddings -> Transformer (pair embeddings as attention bias) -> Site pooling -> MLP -> ddG
```

- Structure-aware transformer using Chai-1 representations
- Single embeddings as token features (384-dim projected to d_model)
- Pair embeddings as learned attention biases (256-dim)
- 4 transformer layers with 8 attention heads
- Combined Huber + ranking loss for robust training
- Processes all mutations for a protein in one forward pass

### Shared Training Infrastructure

All models use common utilities from `src/training/common.py`:

- **`EmbeddingCache`**: Unified cache for protein embeddings with optional LRU eviction (prevents OOM on GPU)
- **`TrainingHistory`**: Standardized training metrics tracking
- **`EarlyStopping`**: Patience-based stopping with best model checkpointing
- **`compute_metrics()`**: Per-protein Spearman/Pearson correlation computation
- **`run_training()`**: Standard training loop with checkpointing and result saving

---

## Data

### Source

**MegaScale** dataset from HuggingFace (`RosettaCommons/MegaScale`):
- ~271k unique single mutations
- ~298 proteins (32-72 aa)
- Standard benchmark from ThermoMPNN

### Splits

Uses **ThermoMPNN's official splits** from `mega_splits.pkl`:
- Main train/val/test split for final evaluation
- Optional CV folds (0-4) for hyperparameter tuning

### Sign Convention

MegaScale raw data uses **inverted** convention. Our code automatically flips to standard:
- **Positive ddG = destabilizing**
- **Negative ddG = stabilizing**

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
│   │   └── mutation_encoder.py # encode_mutation() + EmbeddingCache (unified, LRU-capable)
│   ├── models/
│   │   ├── pair_aware_mlp.py   # PairAwareMLP with LayerNorms
│   │   ├── mpnn.py             # ChaiMPNN graph neural network
│   │   └── transformer.py      # ChaiThermoTransformer
│   └── training/
│       ├── common.py           # Shared: BaseTrainingConfig, EarlyStopping, TrainingHistory
│       ├── evaluate.py         # Shared: compute_metrics(), EvaluationResults
│       ├── train_mlp.py        # MLP training
│       ├── train_mpnn.py       # MPNN training
│       ├── train_transformer.py# Transformer training
│       ├── eval_mpnn.py        # Per-protein MPNN evaluation
│       └── sampler.py          # BalancedProteinSampler
├── configs/
│   └── transformer.yaml        # Example config file
├── scripts/
│   ├── download_megascale.py   # Download dataset from HuggingFace
│   └── verify_alignment.py     # Pre-training sanity checks
├── data/
│   ├── megascale.parquet       # Local copy of MegaScale dataset
│   ├── mega_splits.pkl         # ThermoMPNN train/val/test splits
│   └── embeddings/
│       └── chai_trunk/         # Cached embeddings ({protein}.pt)
└── outputs/
    └── YYYYMMDD_HHMMSS_run_name/   # Run directory (all models)
        ├── config.json
        ├── model.pt                # Final best model
        ├── results.json            # Test metrics
        ├── history.json            # Training history
        └── checkpoint_epoch_*.pt   # Periodic checkpoints
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

**Loading embeddings in code:**
```python
from src.features.mutation_encoder import EmbeddingCache

# CPU cache (unlimited)
cache = EmbeddingCache("data/embeddings/chai_trunk")

# GPU cache with LRU eviction (prevents OOM)
cache = EmbeddingCache("data/embeddings/chai_trunk", device="cuda", max_cached=64)

single, pair = cache.get("protein_name")
```

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
Loss = MSE(pred(A->B), ddG) + MSE(pred(B->A), -ddG) + lambda*MSE(pred(A->B) + pred(B->A), 0)
```

**Why it helps:**
- Forces the model to learn that mutation effects are reversible
- Acts as a regularizer, preventing memorization of "mutating to X is always bad"
- Forces learning the context of the fit, not just amino acid identity
- Particularly effective on de novo designs where signal-to-noise is lower

Enable with `--antisymmetric` flag. Adjust consistency weight with `--antisymmetric-lambda` (default: 1.0).

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
