# Chai-1 Stability Predictor (v4) — Performance Edition

## Goal

**Build the best possible protein stability predictor using Chai-1's full representational capacity.**

We are exploiting everything Chai-1 offers — single representations, pair representations — to maximize prediction accuracy on ΔΔG for short protein domains.

**Scope clarification:** We're building "the best ΔΔG predictor for short (40–72 aa) domains measured by cDNA display proteolysis," not a universal protein stability oracle. MegaScale is one of the best training corpora for this specific target.

**Why Chai-1 should win:**
Stability is determined by pairwise interactions: salt bridges, hydrophobic packing, steric clashes, hydrogen bond networks. ESM-2 infers these from sequence patterns. Chai-1 computes them explicitly in its pair track. The pair track is the key unlock.

---

## Data

### Source: MegaScale Dataset (RosettaCommons/MegaScale on HuggingFace)

The dataset has multiple configs. Here's what they mean:

| Config | Rows | Description |
|--------|------|-------------|
| `dataset1` | 1,841,285 | All quality groups (G0–G11) |
| `dataset2` | 776,298 | High-quality (G0 + G1), includes doubles |
| `dataset3` | 607,839 | G0 only, WT ΔG < 4.75 kcal/mol (reliable ΔΔG) |
| `dataset3_single` | 1,836,063 | Singles from dataset3, ThermoMPNN splits |
| **`dataset3_single_cv`** | ~271,000 | Singles from dataset3, 5-fold CV splits |

### Which Config to Use

**Use `dataset3_single_cv`** — This is the clean, deduplicated set with explicit CV folds.

**Why not `dataset3_single`?**
The 1.5M+ rows contain duplicates — same `(WT_name, mut_type)` appearing multiple times with identical labels. Training on raw rows silently re-weights some mutations 6–10×, which:
- Distorts the loss
- Inflates apparent data size without adding information
- Makes balanced sampling less effective

**Real data size:**
- **~271k unique single mutants** across ~298 proteins
- **~160k training examples per CV fold**
- This is the true supervision signal

### Fold Structure (dataset3_single_cv)

Each of the 5 folds contains roughly:
- Train: ~160–165k mutations
- Val: ~55k mutations
- Test: ~51k mutations

Labels are proper floats (`dG_ML`, `ddG_ML`), not strings.

### Label Semantics

**⚠️ CRITICAL: MegaScale uses INVERTED sign convention in raw data!**

```
RAW MegaScale data (ddG_ML column):
  ΔΔG = ΔG(wildtype) − ΔG(mutant)   ← INVERTED
  - Positive ΔΔG = STABILIZING (lower mutant energy)
  - Negative ΔΔG = DESTABILIZING (higher mutant energy)

Evidence: Stabilizing_mut=True entries have POSITIVE ddG_ML values.

STANDARD convention (what we use after loading):
  ΔΔG = ΔG(mutant) − ΔG(wildtype)   ← STANDARD
  - Positive ΔΔG = destabilizing mutation
  - Negative ΔΔG = stabilizing mutation

Units: kcal/mol
```

**Our `MegaScaleDataset` class automatically flips the sign to standard convention.**

### Dataset Limitations (Know Your Bounds)

#### 1. Limited Dynamic Range

MegaScale spans roughly **−3 to +5 kcal/mol ΔΔG**. Models trained on it struggle on outliers outside this range. This is fine for in-dataset performance, but don't expect generalization to extreme stability changes.

#### 2. Surface Cysteine Artifact

Surface cysteine mutations can appear highly stabilizing due to **disulfide-related artifacts** in the cDNA display proteolysis assay. This is an assay quirk, not real physics. Options:
- Keep them for leaderboard performance (you're learning the assay)
- Filter them for a "cleaner" model (loses some data)
- Flag predictions involving Cys mutations as lower confidence

#### 3. De Novo Designed Proteins

**148 of ~479 domains are de novo designs** with no natural evolutionary history. This affects MSA strategy (see below).

### Data Sanity Checklist

```
□ Load dataset3_single_cv, verify fold structure
□ Check label columns are float (dG_ML, ddG_ML)
□ Verify no duplicate (WT_name, mut_type, aa_seq) within a fold
□ Confirm sign convention on known destabilizing mutations
□ Plot ΔΔG distribution (should be roughly −3 to +5 kcal/mol)
□ Count Cys mutations, decide whether to flag/filter
□ Check for any NaN/missing values
```

### Optional: Deduplication / Aggregation

If you later want to use `dataset3_single` (more rows), you need to deduplicate:

```python
# Group by unique mutation identity
group_cols = ['WT_name', 'mut_type', 'aa_seq']

# Aggregate: take mean of ddG_ML, or use CI columns as weights
df_unique = df.groupby(group_cols).agg({
    'ddG_ML': 'mean',
    'dG_ML': 'mean',
    # Optionally use CI columns for uncertainty weighting
}).reset_index()
```

---

## Splits: Use Provided CV Folds

**Do not create your own splits.** The `dataset3_single_cv` folds are already:
- Split by protein (no leakage)
- Curated by ThermoMPNN authors
- Standard benchmark for comparison

Using the provided folds lets you compare directly to published baselines.

### If You Want Homology-Aware Validation

The provided splits may still have homologous proteins across folds. If you want stricter generalization:
1. Cluster all WT sequences at 30% identity
2. Check which clusters span multiple folds
3. Report results both ways (standard folds + homology-strict)

---

## Embedding Extraction: Single + Pair Tracks

### The Key Insight

Chai-1's trunk produces two outputs:

| Track | Shape | What it encodes |
|-------|-------|-----------------|
| **Single** | `[L, D_single]` | Per-residue features (what each residue "is") |
| **Pair** | `[L, L, D_pair]` | Pairwise interactions (who talks to whom) |

The pair track explicitly encodes spatial proximity, interaction strength, and contact patterns. **For stability prediction, the pair track matters more than the single track.**

### MSA Expectations (Adjusted)

**Don't expect MSAs to help much.**

~148 of ~479 domains are de novo designed proteins with no natural homologs. For these:
- MSA search returns zero or near-zero hits
- The MSA track will be empty
- No evolutionary signal to exploit

For the ~331 natural domains, MSAs might help, but:
- Computing MSAs is expensive
- The lift is uncertain
- The pair track works regardless of evolutionary history

**Recommendation:** Start with single-sequence mode (no MSAs). The pair track is your main bet. Only add MSAs later if you plateau and want to squeeze out extra performance on natural proteins.

### Extraction Code (Conceptual)

```python
def extract_chai_features(sequence: str) -> dict:
    """
    Extract both single and pair representations from Chai-1 trunk.
    Run in single-sequence mode (no MSA).
    """
    trunk_output = chai_model.forward_trunk(
        sequence=sequence,
        use_msa=False,  # Start without MSAs
    )
    
    return {
        "single": trunk_output.single,  # [L, D_single]
        "pair": trunk_output.pair,      # [L, L, D_pair]
        "length": len(sequence),
        "sequence": sequence,
    }
```

### Caching Strategy

Given ~298 proteins with ~70 aa average length:

**Option A: Cache raw representations**
```python
{
    "single": Tensor[L, D_single],  # float16
    "pair": Tensor[L, L, D_pair],   # float16
}
```
Storage: ~1.4 GB total

**Option B: Cache pre-aggregated features (recommended)**
```python
{
    "single": Tensor[L, D_single],           # [L, D_single]
    "single_mean": Tensor[D_single],         # precomputed
    "pair_row_means": Tensor[L, D_pair],     # pair[i,:].mean for each i
    "pair_global_mean": Tensor[D_pair],      # overall mean
}
```
Storage: ~200 MB total, faster training

---

## Mutation Representation: Structure-Aware Features

For a mutation at position `i`, extract features that capture the structural environment:

```python
def encode_mutation(
    single: Tensor,           # [L, D_single]
    pair: Tensor,             # [L, L, D_pair]
    position: int,            # mutation site i
    wt_residue: int,          # 0-19
    mut_residue: int,         # 0-19
    k_structural: int = 10,   # number of structural neighbors
    seq_window: int = 5,      # sequence window size
) -> Tensor:
    L = single.shape[0]
    
    # === Single track features ===
    local_single = single[position]                    # [D_single]
    global_single = single.mean(dim=0)                 # [D_single]
    
    # === Pair track features ===
    pair_row = pair[position, :, :]                    # [L, D_pair]
    
    # Global: average interaction with all residues
    pair_global = pair_row.mean(dim=0)                 # [D_pair]
    
    # Local (sequence): interactions with sequence neighbors
    start = max(0, position - seq_window)
    end = min(L, position + seq_window + 1)
    pair_local_seq = pair[position, start:end, :].mean(dim=0)  # [D_pair]
    
    # Structural: top-k strongest interactions (spatial neighbors)
    pair_magnitudes = pair_row.norm(dim=-1)            # [L]
    topk_indices = pair_magnitudes.topk(k=k_structural).indices
    pair_structural = pair[position, topk_indices, :].mean(dim=0)  # [D_pair]
    
    # === Mutation identity ===
    wt_onehot = F.one_hot(torch.tensor(wt_residue), 20).float()
    mut_onehot = F.one_hot(torch.tensor(mut_residue), 20).float()
    rel_position = torch.tensor([position / L])
    
    # === Concatenate ===
    return torch.cat([
        local_single,       # [D_single]
        global_single,      # [D_single]
        pair_global,        # [D_pair]
        pair_local_seq,     # [D_pair]
        pair_structural,    # [D_pair]
        wt_onehot,          # [20]
        mut_onehot,         # [20]
        rel_position,       # [1]
    ])
    # Total: 2*D_single + 3*D_pair + 41
```

### Feature Summary

| Feature | Dimension | What it captures |
|---------|-----------|------------------|
| `local_single` | D_single | Properties of mutated residue |
| `global_single` | D_single | Overall protein context |
| `pair_global` | D_pair | Average interaction pattern |
| `pair_local_seq` | D_pair | Interactions with sequence neighbors |
| `pair_structural` | D_pair | Interactions with spatial neighbors |
| `wt_onehot` | 20 | Original residue |
| `mut_onehot` | 20 | New residue |
| `rel_position` | 1 | Position in sequence |

---

## Model Architecture

### Primary: Pair-Aware MLP

```python
class PairAwareMLP(nn.Module):
    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = 2 * d_single + 3 * d_pair + 41
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)
```

### Alternative: Attention Head (If MLP Plateaus)

```python
class PairAttentionHead(nn.Module):
    """
    Let mutation site attend over all residues, weighted by pair interactions.
    """
    def __init__(self, d_single: int, d_pair: int, hidden_dim: int = 256):
        super().__init__()
        self.pair_to_attn = nn.Linear(d_pair, 1)
        self.single_proj = nn.Linear(d_single, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 41, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, single, pair, position, mutation_feat):
        B, L, _ = single.shape
        
        # Attention from pair features
        pair_row = pair[torch.arange(B), position]  # [B, L, D_pair]
        attn_logits = self.pair_to_attn(pair_row).squeeze(-1)  # [B, L]
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, L]
        
        # Weighted sum of single representations
        values = self.single_proj(single)  # [B, L, hidden]
        context = (attn_weights.unsqueeze(-1) * values).sum(dim=1)  # [B, hidden]
        
        return self.head(torch.cat([context, mutation_feat], dim=-1))
```

---

## Training

### Sampling: Per-Protein Balanced

```python
class BalancedProteinSampler:
    """
    Each batch samples proteins uniformly, then k variants per protein.
    Prevents large scans from dominating gradients.
    """
    def __init__(self, dataset, variants_per_protein: int = 8):
        self.protein_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset):
            self.protein_to_indices[item['WT_name']].append(idx)
        self.proteins = list(self.protein_to_indices.keys())
```

### Loss

```python
loss = F.mse_loss(predicted_ddg, true_ddg)
# Or Huber for outlier robustness:
loss = F.huber_loss(predicted_ddg, true_ddg, delta=2.0)
```

### Hyperparameters

```yaml
optimizer: AdamW
learning_rate: 3e-4
weight_decay: 0.01
batch_size: 128
epochs: 100
early_stopping: patience=10 on validation Spearman
scheduler: CosineAnnealingWarmRestarts
gradient_clip: 1.0
```

### Cross-Validation Strategy

Train on all 5 folds independently, report mean ± std of test metrics. This is standard for MegaScale benchmarking.

---

## Evaluation

### Primary Metric: Per-Protein Spearman

```python
def evaluate(model, test_loader) -> dict:
    protein_results = defaultdict(lambda: {"pred": [], "true": []})
    
    for batch in test_loader:
        preds = model(batch["features"])
        for i, pid in enumerate(batch["WT_name"]):
            protein_results[pid]["pred"].append(preds[i].item())
            protein_results[pid]["true"].append(batch["ddG_ML"][i].item())
    
    spearmans = []
    for pid, results in protein_results.items():
        if len(results["pred"]) >= 30:
            rho, _ = spearmanr(results["pred"], results["true"])
            if not np.isnan(rho):
                spearmans.append(rho)
    
    return {
        "mean_spearman": np.mean(spearmans),
        "std_spearman": np.std(spearmans),
        "median_spearman": np.median(spearmans),
        "n_proteins": len(spearmans),
    }
```

### Targets

| Metric | Target | Stretch |
|--------|--------|---------|
| Mean Spearman | 0.55 | 0.65+ |
| Median Spearman | 0.58 | 0.68+ |
| RMSE (kcal/mol) | < 1.0 | < 0.8 |

### Baseline Comparison

ThermoMPNN (trained on this exact dataset) is the natural comparison. Check their reported numbers on `dataset3_single_cv` folds.

---

## Implementation Plan

### Phase 1: Data Pipeline (Days 1-2)

```
□ Load dataset3_single_cv from HuggingFace
□ Verify fold structure and label types
□ Run sanity checklist
□ Implement Dataset class
□ Count Cys mutations, decide on handling
```

### Phase 2: Chai-1 Feature Extraction (Days 3-5)

```
□ Clone chai-lab repo
□ Find trunk output point (before diffusion)
□ Patch to return single + pair representations
□ Run on 5 proteins, verify shapes
□ Extract features for all ~298 WT proteins
□ Cache to disk (option A or B)
```

### Phase 3: Pair-Aware MLP (Days 6-8)

```
□ Implement encode_mutation()
□ Implement PairAwareMLP
□ Implement balanced sampler
□ Train on fold 0
□ Evaluate on fold 0 test
□ Target: Spearman > 0.50
```

### Phase 4: Full CV + Iteration (Days 9-12)

```
□ Train on all 5 folds
□ Report mean ± std across folds
□ Ablations:
  □ pair_global only
  □ pair_local_seq only  
  □ pair_structural only
  □ All three combined
□ Hyperparameter sweep if needed
□ Try attention head if MLP plateaus
```

### Phase 5: Polish (Days 13-14)

```
□ Error analysis: which proteins/mutations fail?
□ Check if Cys mutations are outliers
□ Compare to ThermoMPNN baseline
□ Document final architecture
□ Clean up codebase
```

### Phase 6: Extensions (Week 3+)

```
□ Test on external benchmarks (ProTherm, S669, FireProtDB)
□ Multi-mutation prediction (dataset has ~210k doubles)
□ LoRA fine-tuning (if cluster access available)
```

---

## Technical Risks & Mitigations

### Risk 1: Pair track extraction is hard

**Mitigation:** Pair track is fundamental to structure prediction; it must exist. Dig into trunk module. Worst case: predict structure, compute pairwise distances/contacts from coordinates.

### Risk 2: Pair features don't help on short domains

Short domains (40–72 aa) have limited long-range contacts. Pair track may be less valuable than on larger proteins.

**Mitigation:** This is a genuine risk. If pair features don't help, the negative result is informative. Ablations will reveal this quickly.

### Risk 3: Overfitting to ~298 proteins

**Mitigation:** 
- CV folds provide honest generalization estimate
- Strong regularization (dropout, weight decay)
- Monitor train/val gap
- The ~160k mutations per fold is enough supervision for an MLP head

### Risk 4: Assay artifacts distort learning

Surface Cys mutations, dynamic range limits.

**Mitigation:**
- Flag Cys predictions as lower confidence
- Don't expect generalization outside −3 to +5 kcal/mol range
- This is a predictor for *this assay*, not universal stability

---

## File Structure

```
chai1-stability/
├── data/
│   ├── processed/
│   │   └── megascale_cv/
│   │       ├── fold_0/
│   │       │   ├── train.parquet
│   │       │   ├── val.parquet
│   │       │   └── test.parquet
│   │       └── ...
│   └── embeddings/
│       └── chai_full/
│           └── {protein_id}.pt
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── embeddings/
│   │   └── chai_extractor.py
│   ├── features/
│   │   └── mutation_encoder.py
│   ├── models/
│   │   ├── pair_aware_mlp.py
│   │   └── pair_attention_head.py
│   ├── training/
│   │   ├── train.py
│   │   ├── sampler.py
│   │   └── evaluate.py
│   └── utils/
│       └── metrics.py
├── configs/
│   └── default.yaml
├── scripts/
│   ├── 01_load_dataset.py
│   ├── 02_extract_chai_features.py
│   ├── 03_train_fold.py
│   └── 04_evaluate_cv.py
├── notebooks/
│   ├── data_exploration.ipynb
│   └── results.ipynb
└── README.md
```

---

## Success Criteria

| Outcome | Mean Spearman (5-fold CV) | Interpretation |
|---------|---------------------------|----------------|
| **Minimum** | 0.50 | Pair features add value |
| **Good** | 0.55-0.60 | Competitive with ThermoMPNN |
| **Great** | 0.60-0.65 | Clear win, worth extending |
| **Excellent** | 0.65+ | SOTA on this benchmark |

---

## Key Changes from v3

| Aspect | v3 | v4 |
|--------|----|----|
| Dataset | Generic MegaScale | `dataset3_single_cv` (deduplicated, CV folds) |
| Data size | "~1M rows" | ~271k unique mutants, ~160k/fold |
| MSA | "Enable for best performance" | Skip (148/479 domains are de novo) |
| Splits | Create homology clusters | Use provided ThermoMPNN folds |
| Artifacts | Not mentioned | Cys mutations flagged, range limits noted |
| Baseline | None | ThermoMPNN (trained on same data) |