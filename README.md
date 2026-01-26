# Chai-1 Stability Predictor

**Protein stability (ddG) prediction using Chai-1's pair representations.**

Uses ThermoMPNN's official data splits for training and evaluation.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Pre-extracted Chai-1 embeddings in `data/embeddings/chai_trunk/`
- ThermoMPNN splits file (`data/mega_splits.pkl`)

### Installation

```bash
uv sync  # recommended

# or with pip (no requirements.txt provided)
pip install .        # install as a package
# pip install -e .   # editable install for development
```

Notes:
- If you don’t have `uv`: pipx install uv (or `pip install uv`).
- `torch-geometric` may require CUDA-specific wheels. If installation fails, follow: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### Download Data

```bash
# Download MegaScale dataset from HuggingFace (one-time, ~50MB)
uv run python scripts/download_megascale.py
```

This creates `data/megascale.parquet` which is used by training/eval scripts.

### ThermoMPNN Splits (required)

- Download the official ThermoMPNN splits file `mega_splits.pkl` from the ThermoMPNN repository and place it at `data/mega_splits.pkl`.
- The same file also contains optional CV folds (`cv_train_{0-4}`/`cv_val_{0-4}`/`cv_test_{0-4}`).

Example layout:
```
data/
  megascale.parquet
  mega_splits.pkl
```

### Training

This repo supports three training modes:

1) MLP — fast baseline using pre-encoded features
2) MPNN — graph neural network over local structural subgraphs
3) Transformer — structure-aware transformer using pair biases

**MLP Model:**
```bash
uv run python -m src.training.train_mlp --run-name mlp_baseline
```

Key options:
- `--precomputed-dir PATH` to train from precomputed features (see Precomputation below)
- `--splits-file PATH` (default `data/mega_splits.pkl`)
- `--cv-fold N` (0–4) for CV splits, or omit for main split
- `--run-name NAME`

**MPNN Model:**
```bash
# Standard training
uv run python -m src.training.train_mpnn --run-name mpnn_baseline

# With antisymmetric loss (recommended for better generalization)
uv run python -m src.training.train_mpnn \
    --run-name mpnn_antisym \
    --antisymmetric \
    --batch-size 128
```

Key options:
- `--data-path PATH` to use local parquet (faster repeated access)
- `--splits-file PATH` (default `data/mega_splits.pkl`)
- `--cv-fold N` (0–4), `--run-name NAME`

**Transformer Model:**
```bash
uv run python -m src.training.train_transformer --run-name transformer_baseline
```

Key options:
- `--data-path PATH` to use local parquet (faster)
- `--splits-file PATH` (default `data/mega_splits.pkl`)
- `--cv-fold N` (0–4), `--run-name NAME`

### Evaluation

**MPNN per-protein evaluation:**
```bash
# Evaluate trained model with per-protein breakdown
uv run python -m src.training.eval_mpnn outputs/YYYYMMDD_HHMMSS_run_name/

# Evaluate on validation split, sorted by RMSE
uv run python -m src.training.eval_mpnn outputs/run_dir/ --split val --sort-by rmse
```

**Transformer evaluation:**
```bash
uv run python scripts/evaluate_transformer.py \
  --config configs/transformer.yaml \
  --splits-file data/mega_splits.pkl \
  --model-path outputs/run_dir/model.pt
```

### Verification (run before training)

```bash
uv run python scripts/verify_alignment.py --embedding-dir data/embeddings/chai_trunk
```

Useful flags:
- `--fold -1` to check all folds
- `--show-stats` to print per-embedding shape/stats

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

### Creating `data/wt_sequences.json`

If you don’t already have WT sequences, you can extract them from the MegaScale data restricted to ThermoMPNN splits:

```bash
uv run python - <<'PY'
from pathlib import Path
import json
from src.data.megascale_loader import ThermoMPNNSplitDatasetHF

splits = 'data/mega_splits.pkl'
ds = ThermoMPNNSplitDatasetHF(splits_file=splits, split='train')
# Normalize names and materialize dict
seqs = {k.replace('.pdb',''): v for k,v in ds.wt_sequences.items()}
Path('data').mkdir(exist_ok=True)
with open('data/wt_sequences.json','w') as f:
    json.dump(seqs, f)
print(f'wrote {len(seqs)} sequences to data/wt_sequences.json')
PY
```

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

For the MLP training path, each epoch samples uniformly across proteins (32 mutations per protein), preventing large proteins from dominating gradients.

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

## Top Results

Best runs per model class on ThermoMPNN splits:

- MLP (PairAwareMLP, ~5M):
  - Spearman: 0.7218, RMSE: 0.7175
  - Params: ~5,017,217
  - Run: `outputs/20260126_205313_mlp_5m`

- MPNN (ChaiMPNN, antisymmetric):
  - Spearman: 0.7683, RMSE: 0.6982
  - Params: ~5,271,105
  - Run: `outputs/20260122_054138_mpnn_antisym_scaled`

- Transformer (ChaiThermoTransformer v2 + ranking):
  - Spearman: 0.7456, RMSE: 0.7004
  - Params: ~5,797,956
  - Run: `outputs/20260124_084950_transformer_v2_rank`

Notes:
- Primary metric is mean per-protein Spearman (higher is better).
- RMSE reported on the same splits for comparability.

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
- **ThermoMPNN**: Add the official ThermoMPNN citation and GitHub link for benchmark splits
