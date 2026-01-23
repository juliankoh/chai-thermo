# Chai-Thermo-Transformer: Structure-Aware Transformer for ΔΔG Prediction

## Overview

A transformer-based model that uses Chai-1's single embeddings as the sequence representation and Chai-1's pair embeddings as attention biases. This replaces the current MPNN approach with attention-based structure integration.

```
Chai single [L, 384]  ──→  project + norm  ──→  Transformer layers  ──→  site head  ──→  ΔΔG
                                                      ↑
Chai pair [L, L, 256] ──→  project (zero-init)  ──→  gated attention bias
```

## Motivation

- Current MPNN achieves Spearman ~0.716 using pair embeddings as edge features
- Injecting structure into attention (rather than message passing) can be more effective
- Chai-1 already provides both single and pair representations - we just need to combine them differently
- Attention-based fusion is closer to how Chai-1/AlphaFold2 work internally

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                   ChaiThermoTransformer Model                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Chai single [L, 384]           Chai pair [L, L, 256]           │
│         │                              │                        │
│         ▼                              ▼                        │
│  Linear(384 → d_model)         PairBiasProjection               │
│  + LayerNorm                   (zero-init output)               │
│         │                              │                        │
│         ▼                              ▼                        │
│    h [L, d_model]              bias [n_heads, L, L]             │
│         │                              │                        │
│         └──────────────┬───────────────┘                        │
│                        ▼                                        │
│         ┌──────────────────────────────┐                        │
│         │  StructureAwareTransformer   │                        │
│         │  (N layers)                  │                        │
│         │                              │                        │
│         │  For each layer:             │                        │
│         │    - LayerNorm               │                        │
│         │    - MultiHeadAttention      │                        │
│         │      + gate[layer] * bias    │  ← per-layer gate      │
│         │    - Residual                │    (init 0)            │
│         │    - LayerNorm               │                        │
│         │    - FFN                     │                        │
│         │    - Residual                │                        │
│         └──────────────────────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│              h_out [L, d_model]                                 │
│                        │                                        │
│                        ▼                                        │
│         ┌──────────────────────────────┐                        │
│         │       SiteHead (20-way)      │                        │
│         │                              │                        │
│         │  h[pos] + h.mean() + rel_pos │                        │
│         │  → MLP → scores[20]          │                        │
│         │  → score[mut] - score[wt]    │  ← exact antisymmetry  │
│         │  → ΔΔG                       │                        │
│         └──────────────────────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Gated pair bias (init 0)**: Model starts as vanilla transformer, learns to use structure
2. **Zero-init bias projection**: Initial bias is ~0, preventing attention collapse
3. **20-way site head**: Predicts score per AA, takes difference → exact antisymmetry for free
4. **Vectorized readout**: All mutations for a protein in one forward pass
5. **LayerNorm after projection**: Stabilizes training, makes LR less brittle

---

## Component Details

### 1. Input Projections

```python
class InputProjection(nn.Module):
    """Project and normalize single embeddings."""

    def __init__(self, single_dim: int = 384, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(single_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, single: Tensor) -> Tensor:
        return self.norm(self.proj(single))
```

### 2. PairBiasProjection (Zero-Init Output)

```python
class PairBiasProjection(nn.Module):
    """Project pair embeddings to attention biases. Output layer is zero-initialized."""

    def __init__(self, pair_dim: int = 256, n_heads: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim // 4),
            nn.GELU(),
            nn.Linear(pair_dim // 4, n_heads),
        )
        # Zero-init the last layer so initial bias is ~0
        self._zero_init_output()

    def _zero_init_output(self):
        last_linear = self.net[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, pair: Tensor) -> Tensor:
        # pair: [L, L, pair_dim]
        # output: [n_heads, L, L]
        return self.net(pair).permute(2, 0, 1)
```

### 3. StructureAwareAttention (with Gated Bias)

```python
class StructureAwareAttention(nn.Module):
    """Multi-head attention with gated additive pair bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,              # [L, d_model]
        pair_bias: Tensor,      # [n_heads, L, L]
        gate: Tensor,           # [n_heads, 1, 1] - per-layer gate
    ) -> Tensor:
        L = x.size(0)
        qkv = self.qkv_proj(x).reshape(L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # each: [L, n_heads, head_dim]

        # Attention with gated pair bias
        # [n_heads, L, L]
        attn_logits = torch.einsum('ihd,jhd->hij', q, k) * self.scale
        attn_logits = attn_logits + gate * pair_bias  # gate starts at 0

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [L, n_heads, head_dim] -> [L, d_model]
        out = torch.einsum('hij,jhd->ihd', attn_weights, v)
        out = out.reshape(L, -1)

        return self.out_proj(out)
```

### 4. TransformerLayer

```python
class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with structure-aware attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = StructureAwareAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, pair_bias: Tensor, gate: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), pair_bias, gate)
        x = x + self.ffn(self.norm2(x))
        return x
```

### 5. SiteHead (20-way with Exact Antisymmetry)

This is the key architectural change: predict a score for each of the 20 amino acids at the mutation site, then compute ΔΔG as the difference.

```python
class SiteHead(nn.Module):
    """
    20-way site head that predicts AA preference scores.

    ΔΔG(wt→mut) = score[mut] - score[wt]

    This gives:
    - Exact antisymmetry: ΔΔG(A→B) = -ΔΔG(B→A) by construction
    - Parameter sharing across all substitutions at a site
    - Better ranking (consistent AA preference scale)
    """

    def __init__(self, d_model: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # Input: h_local [d_model] + h_global [d_model] + rel_pos [1]
        input_dim = d_model * 2 + 1

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 20),  # 20 AA scores
        )

    def forward(
        self,
        h: Tensor,              # [L, d_model]
        positions: Tensor,      # [M] mutation positions
        wt_indices: Tensor,     # [M] WT residue indices (0-19)
        mut_indices: Tensor,    # [M] mutant residue indices (0-19)
    ) -> Tensor:
        """
        Vectorized forward for M mutations on one protein.

        Returns: [M] ΔΔG predictions
        """
        M = positions.size(0)
        L = h.size(0)

        # Local features at mutation sites: [M, d_model]
        h_local = h[positions]

        # Global features (same for all mutations): [M, d_model]
        h_global = h.mean(dim=0, keepdim=True).expand(M, -1)

        # Relative positions: [M, 1]
        rel_pos = (positions.float() / L).unsqueeze(-1)

        # Concatenate features: [M, 2*d_model + 1]
        features = torch.cat([h_local, h_global, rel_pos], dim=-1)

        # Predict 20 AA scores per site: [M, 20]
        scores = self.mlp(features)

        # ΔΔG = score(mut) - score(wt)
        # This gives exact antisymmetry: ΔΔG(A→B) = -ΔΔG(B→A)
        wt_scores = scores.gather(1, wt_indices.unsqueeze(-1)).squeeze(-1)   # [M]
        mut_scores = scores.gather(1, mut_indices.unsqueeze(-1)).squeeze(-1) # [M]

        return mut_scores - wt_scores  # [M]
```

### 6. Full Model

```python
class ChaiThermoTransformer(nn.Module):
    """
    Structure-aware transformer for ΔΔG prediction.

    Key features:
    - Chai pair embeddings injected as gated attention biases
    - Per-layer bias gates (initialized to 0)
    - Zero-initialized bias projection
    - 20-way site head with exact antisymmetry
    - Vectorized readout for efficient protein-batch processing
    """

    def __init__(
        self,
        single_dim: int = 384,
        pair_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        site_hidden: int = 128,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Input projections
        self.single_proj = nn.Linear(single_dim, d_model)
        self.single_norm = nn.LayerNorm(d_model)
        self.pair_bias_proj = PairBiasProjection(pair_dim, n_heads)

        # Per-layer bias gates (initialized to 0)
        # Shape: [n_layers, n_heads, 1, 1]
        self.bias_gates = nn.Parameter(torch.zeros(n_layers, n_heads, 1, 1))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # 20-way site head
        self.site_head = SiteHead(d_model, site_hidden, dropout)

    def forward(
        self,
        single: Tensor,         # [L, 384]
        pair: Tensor,           # [L, L, 256]
        positions: Tensor,      # [M] mutation positions
        wt_indices: Tensor,     # [M] WT residue indices (0-19)
        mut_indices: Tensor,    # [M] mutant residue indices (0-19)
    ) -> Tensor:
        """
        Forward pass for all mutations on one protein.

        Args:
            single: Chai single embeddings [L, 384]
            pair: Chai pair embeddings [L, L, 256]
            positions: Mutation positions [M]
            wt_indices: WT residue indices [M]
            mut_indices: Mutant residue indices [M]

        Returns:
            predictions: ΔΔG predictions [M]
        """
        # Project and normalize single embeddings
        h = self.single_norm(self.single_proj(single))  # [L, d_model]

        # Project pair to attention biases
        pair_bias = self.pair_bias_proj(pair)  # [n_heads, L, L]

        # Transformer layers with gated pair bias
        for layer_idx, layer in enumerate(self.layers):
            gate = self.bias_gates[layer_idx]  # [n_heads, 1, 1]
            h = layer(h, pair_bias, gate)

        h = self.final_norm(h)  # [L, d_model]

        # Predict ΔΔG via 20-way site head
        return self.site_head(h, positions, wt_indices, mut_indices)  # [M]
```

---

## File Structure

```
src/
├── models/
│   ├── mpnn.py              (existing - keep for comparison)
│   ├── pair_aware_mlp.py    (existing - keep for comparison)
│   └── transformer.py       (NEW)
│       ├── PairBiasProjection
│       ├── StructureAwareAttention
│       ├── TransformerLayer
│       ├── SiteHead
│       └── ChaiThermoTransformer
└── training/
    ├── train_mpnn.py        (existing)
    └── train_transformer.py (NEW)
        ├── TransformerDataset
        ├── EmbeddingCache (LRU)
        ├── ProteinBatchSampler
        ├── train_epoch
        ├── validate
        ├── train_fold
        └── main
```

---

## Data Pipeline

### EmbeddingCache (LRU, Memory-Safe)

Don't eagerly load all embeddings to GPU. Use an LRU cache instead.

```python
from functools import lru_cache
from typing import Dict, Tuple
import torch

class EmbeddingCache:
    """
    LRU cache for protein embeddings.

    Keeps recently-used proteins on GPU, evicts old ones.
    Prevents OOM when running multiple folds or large datasets.
    """

    def __init__(
        self,
        embedding_dir: Path,
        device: str = "cuda",
        max_cached: int = 64,
    ):
        self.embedding_dir = embedding_dir
        self.device = device
        self.max_cached = max_cached
        self._cache: Dict[str, Dict[str, Tensor]] = {}
        self._access_order: list = []

    def get(self, protein_name: str) -> Dict[str, Tensor]:
        """Get embeddings for a protein, loading if needed."""
        if protein_name not in self._cache:
            self._load(protein_name)

        # Update access order
        if protein_name in self._access_order:
            self._access_order.remove(protein_name)
        self._access_order.append(protein_name)

        return self._cache[protein_name]

    def _load(self, protein_name: str):
        """Load embeddings from disk to GPU."""
        # Evict if at capacity
        while len(self._cache) >= self.max_cached:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        # Load from disk
        path = self.embedding_dir / f"{protein_name}.pt"
        data = torch.load(path, map_location=self.device)
        self._cache[protein_name] = {
            "single": data["single"],  # [L, 384]
            "pair": data["pair"],      # [L, L, 256]
        }

    def preload(self, protein_names: list[str]):
        """Preload a set of proteins (e.g., for validation)."""
        for name in protein_names[:self.max_cached]:
            if name not in self._cache:
                self._load(name)


class TransformerDataset(Dataset):
    """Dataset that uses LRU cache for memory-safe embedding access."""

    def __init__(
        self,
        mutations: list[MutationSample],
        embedding_cache: EmbeddingCache,
    ):
        self.mutations = mutations
        self.cache = embedding_cache

        # Group mutations by protein for efficient batching
        self.protein_to_indices: Dict[str, list[int]] = defaultdict(list)
        for idx, mut in enumerate(mutations):
            self.protein_to_indices[mut.wt_name].append(idx)

        self.proteins = list(self.protein_to_indices.keys())

    def __len__(self):
        return len(self.mutations)

    def get_protein_batch(self, protein_name: str) -> Dict:
        """Get all mutations for a protein as a batch."""
        indices = self.protein_to_indices[protein_name]
        mutations = [self.mutations[i] for i in indices]

        emb = self.cache.get(protein_name)

        return {
            "single": emb["single"],
            "pair": emb["pair"],
            "positions": torch.tensor([m.position for m in mutations]),
            "wt_indices": torch.tensor([AA_TO_IDX[m.wt_residue] for m in mutations]),
            "mut_indices": torch.tensor([AA_TO_IDX[m.mut_residue] for m in mutations]),
            "targets": torch.tensor([m.ddg for m in mutations]),
            "protein_name": protein_name,
        }
```

### ProteinBatchSampler

```python
class ProteinBatchSampler:
    """Yields protein names for batch processing."""

    def __init__(self, dataset: TransformerDataset, shuffle: bool = True):
        self.proteins = dataset.proteins
        self.shuffle = shuffle

    def __iter__(self):
        proteins = self.proteins.copy()
        if self.shuffle:
            random.shuffle(proteins)
        for protein in proteins:
            yield protein

    def __len__(self):
        return len(self.proteins)
```

---

## Training Loop

### Loss Function (Huber + Optional Ranking)

```python
class DDGLoss(nn.Module):
    """
    Combined loss for ΔΔG prediction.

    - Huber loss: robust to outliers in ΔΔG values
    - Ranking loss: directly optimizes for Spearman correlation
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        rank_weight: float = 0.1,
        rank_margin: float = 0.1,
        n_rank_pairs: int = 256,
    ):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.rank_weight = rank_weight
        self.rank_margin = rank_margin
        self.n_rank_pairs = n_rank_pairs

    def forward(
        self,
        pred: Tensor,    # [M]
        target: Tensor,  # [M]
    ) -> Tensor:
        # Regression loss
        loss_reg = self.huber(pred, target)

        if self.rank_weight == 0 or pred.size(0) < 2:
            return loss_reg

        # Ranking loss: sample pairs, penalize wrong orderings
        M = pred.size(0)
        n_pairs = min(self.n_rank_pairs, M * (M - 1) // 2)

        # Sample random pairs
        i = torch.randint(0, M, (n_pairs,), device=pred.device)
        j = torch.randint(0, M, (n_pairs,), device=pred.device)

        # Ensure i != j
        mask = i != j
        i, j = i[mask], j[mask]

        if i.size(0) == 0:
            return loss_reg

        # Target ordering: sign of (target[i] - target[j])
        target_diff = target[i] - target[j]
        pred_diff = pred[i] - pred[j]

        # Margin ranking loss
        # If target[i] > target[j], we want pred[i] > pred[j] - margin
        loss_rank = F.relu(
            self.rank_margin - target_diff.sign() * pred_diff
        ).mean()

        return loss_reg + self.rank_weight * loss_rank
```

### Training Epoch

```python
def train_epoch(
    model: ChaiThermoTransformer,
    dataset: TransformerDataset,
    optimizer: torch.optim.Optimizer,
    loss_fn: DDGLoss,
    config: dict,
    device: str = "cuda",
) -> dict:
    model.train()
    total_loss = 0.0
    n_samples = 0

    sampler = ProteinBatchSampler(dataset, shuffle=True)

    for protein_name in tqdm(sampler, desc="Training"):
        batch = dataset.get_protein_batch(protein_name)

        # Move to device
        single = batch["single"].to(device)
        pair = batch["pair"].to(device)
        positions = batch["positions"].to(device)
        wt_indices = batch["wt_indices"].to(device)
        mut_indices = batch["mut_indices"].to(device)
        targets = batch["targets"].to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(single, pair, positions, wt_indices, mut_indices)

        # Loss
        loss = loss_fn(predictions, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()

        total_loss += loss.item() * len(predictions)
        n_samples += len(predictions)

    return {"loss": total_loss / n_samples}
```

### Validation

```python
@torch.no_grad()
def validate(
    model: ChaiThermoTransformer,
    dataset: TransformerDataset,
    device: str = "cuda",
) -> dict:
    model.eval()

    all_preds = []
    all_targets = []
    all_proteins = []

    sampler = ProteinBatchSampler(dataset, shuffle=False)

    for protein_name in sampler:
        batch = dataset.get_protein_batch(protein_name)

        single = batch["single"].to(device)
        pair = batch["pair"].to(device)
        positions = batch["positions"].to(device)
        wt_indices = batch["wt_indices"].to(device)
        mut_indices = batch["mut_indices"].to(device)

        predictions = model(single, pair, positions, wt_indices, mut_indices)

        all_preds.extend(predictions.cpu().tolist())
        all_targets.extend(batch["targets"].tolist())
        all_proteins.extend([protein_name] * len(predictions))

    # Compute per-protein metrics
    return per_protein_metrics(all_preds, all_targets, all_proteins)
```

---

## Hyperparameters

### Model

| Parameter | Value | Notes |
|-----------|-------|-------|
| `single_dim` | 384 | Chai single embedding dim |
| `pair_dim` | 256 | Chai pair embedding dim |
| `d_model` | 256 | Transformer hidden dim |
| `n_heads` | 8 | Attention heads (32 dim per head) |
| `n_layers` | 4 | Transformer layers |
| `d_ff` | 512 | FFN hidden dim (2x d_model) |
| `dropout` | 0.1 | Dropout rate |
| `site_hidden` | 128 | Site head MLP hidden dim |

### Training

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | 1e-4 | Lower than MLP due to transformer |
| `weight_decay` | 0.01 | AdamW regularization |
| `epochs` | 100 | Max epochs |
| `early_stopping_patience` | 15 | Stop if no improvement |
| `gradient_clip` | 1.0 | Gradient clipping |
| `huber_delta` | 1.0 | Huber loss threshold |
| `rank_weight` | 0.1 | Weight for ranking loss (0 to disable) |
| `rank_margin` | 0.1 | Margin for ranking loss |
| `scheduler` | CosineAnnealingWarmRestarts | Learning rate schedule |
| `T_0` | 10 | Scheduler restart period |
| `T_mult` | 2 | Scheduler period multiplier |

### Config File

```yaml
# configs/transformer.yaml

model:
  single_dim: 384
  pair_dim: 256
  d_model: 256
  n_heads: 8
  n_layers: 4
  d_ff: 512
  dropout: 0.1
  site_hidden: 128

training:
  learning_rate: 1.0e-4
  weight_decay: 0.01
  epochs: 100
  early_stopping_patience: 15
  gradient_clip: 1.0

  loss:
    huber_delta: 1.0
    rank_weight: 0.1      # set to 0 to disable ranking loss
    rank_margin: 0.1
    n_rank_pairs: 256

  scheduler:
    name: CosineAnnealingWarmRestarts
    T_0: 10
    T_mult: 2

data:
  embedding_cache_size: 64  # proteins to keep on GPU

evaluation:
  metrics: [spearman, pearson, rmse, mae]
  primary_metric: spearman
```

---

## Parameter Count

| Component | Parameters |
|-----------|------------|
| single_proj (384 → 256) | 98,560 |
| single_norm | 512 |
| pair_bias_proj (256 → 64 → 8) | 16,904 |
| bias_gates (4 × 8) | 32 |
| 4× TransformerLayer | 1,581,056 |
| final_norm | 512 |
| site_head MLP (513 → 128 → 64 → 20) | 74,708 |
| **Total** | **~1.77M** |

Comparable to current MPNN (~1.5M parameters).

---

## Training Script CLI

```bash
# Single fold
python -m src.training.train_transformer \
    --fold 0 \
    --output-dir outputs/transformer_fold0

# 5-fold CV
python -m src.training.train_transformer \
    --cv \
    --output-dir outputs/transformer_cv

# With custom config
python -m src.training.train_transformer \
    --cv \
    --config configs/transformer.yaml \
    --output-dir outputs/transformer_experiment

# ThermoMPNN splits (for comparison)
python -m src.training.train_transformer \
    --splits thermompnn \
    --output-dir outputs/transformer_thermompnn

# Disable ranking loss
python -m src.training.train_transformer \
    --cv \
    --rank-weight 0 \
    --output-dir outputs/transformer_no_rank
```

---

## Ablations

| Experiment | Description | Hypothesis |
|------------|-------------|------------|
| `no_pair_bias` | Set all bias_gates to 0 (frozen) | Baseline: how much does structure help? |
| `no_gate` | Remove gating, use bias directly | May cause instability early |
| `no_zero_init` | Don't zero-init bias projection | May cause attention collapse |
| `n_layers` | 2, 4, 6, 8 layers | 4 is likely sweet spot |
| `d_model` | 128, 256, 384 | Match single_dim (384) might help |
| `rank_weight` | 0, 0.1, 0.3, 0.5 | How much ranking loss helps |
| `per_layer_proj` | Separate PairBiasProjection per layer | More capacity |
| `pos_encoding` | Add learnable position embeddings | May help generalization |

---

## Implementation Checklist

### Must-Have (before first run)

- [ ] `PairBiasProjection` with zero-init output
- [ ] Per-layer `bias_gates` initialized to 0
- [ ] `LayerNorm` after `single_proj`
- [ ] Vectorized `SiteHead` with 20-way scores
- [ ] `ΔΔG = score[mut] - score[wt]` (exact antisymmetry)
- [ ] `EmbeddingCache` with LRU eviction
- [ ] `HuberLoss` instead of MSE
- [ ] Assert `d_model % n_heads == 0`

### Nice-to-Have (can add later)

- [ ] Ranking loss term
- [ ] Mixed precision (bf16/amp)
- [ ] `F.scaled_dot_product_attention` with additive mask
- [ ] Per-layer PairBiasProjection
- [ ] Learnable position embeddings
- [ ] Attention visualization hooks

---

## Expected Outcomes

| Model | Spearman (5-fold CV) | Notes |
|-------|---------------------|-------|
| Current MPNN | ~0.716 | Baseline |
| ChaiThermoTransformer | 0.72-0.76 | Hypothesis: attention-based fusion + site head |
| + ranking loss | +0.01-0.02 | Directly optimizes correlation |

**Note:** Compare on same splits (HF CV or ThermoMPNN) to avoid apples-to-oranges.

---

## Open Questions

1. **Diagonal masking**: Should we zero out diagonal of pair_bias (self-attention bias)?

2. **Position encoding**: Chai singles likely encode position already. Ablate to confirm.

3. **Multi-mutation support**: Current design is single-mutation only. Site head naturally extends to multi-mutants by summing score differences.

4. **Attention patterns**: Add hooks to visualize which residue pairs the model attends to. Could reveal learned structural motifs.
