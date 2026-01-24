"""
Chai-Thermo-Transformer: Structure-Aware Transformer for ΔΔG Prediction.

Uses Chai-1 single embeddings as sequence representation and pair embeddings
as attention biases. Key features:
- Gated pair bias with learnable per-layer gates
- Small random init on bias projection (enables gradient flow)
- tanh + learnable scale to prevent attention collapse
- 20-way site head with exact antisymmetry: ΔΔG = score[mut] - score[wt]
- Vectorized readout for efficient protein-batch processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PairBiasProjection(nn.Module):
    """
    Project pair embeddings to attention biases.

    Uses tanh + learnable scale for stable training:
    - tanh clamps raw output to [-1, 1] preventing softmax saturation
    - learnable bias_scale starts small (0.1) so model begins with
      small but non-zero bias, learning to use structure if it helps

    Output layer uses small random init (std=0.02) to ensure gradient flow.
    """

    def __init__(self, pair_dim: int = 256, n_heads: int = 8, init_scale: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.net = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim // 4),
            nn.GELU(),
            nn.Linear(pair_dim // 4, n_heads),
        )
        # Learnable scale per head, initialized small
        # Shape: [n_heads, 1, 1] for broadcasting with [n_heads, L, L]
        self.bias_scale = nn.Parameter(torch.full((n_heads, 1, 1), init_scale))
        self._init_output()

    def _init_output(self):
        """Small random init so gradients can flow from the start."""
        last_linear = self.net[-1]
        nn.init.normal_(last_linear.weight, std=0.02)
        nn.init.zeros_(last_linear.bias)  # bias can stay zero

    def forward(self, pair: Tensor) -> Tensor:
        """
        Args:
            pair: [L, L, pair_dim]
        Returns:
            bias: [n_heads, L, L]
        """
        raw = self.net(pair)           # [L, L, n_heads]
        raw = raw.permute(2, 0, 1)     # [n_heads, L, L]
        bias = torch.tanh(raw)         # clamp to [-1, 1]
        bias = self.bias_scale * bias  # learnable scale per head
        return bias


class StructureAwareAttention(nn.Module):
    """
    Multi-head attention with gated additive pair bias.

    The pair bias is gated per-layer, with gates initialized to 0.
    This means the model starts as a vanilla transformer and learns
    to incorporate structural information gradually.
    """

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
        x: Tensor,
        pair_bias: Tensor,
        gate: Tensor,
    ) -> Tensor:
        """
        Args:
            x: [L, d_model]
            pair_bias: [n_heads, L, L]
            gate: [n_heads, 1, 1] per-layer gate (starts at 0)
        Returns:
            out: [L, d_model]
        """
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


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with structure-aware attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
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


class SiteHead(nn.Module):
    """
    20-way site head that predicts AA preference scores.

    ΔΔG(wt→mut) = score[mut] - score[wt]

    This gives:
    - Exact antisymmetry: ΔΔG(A→B) = -ΔΔG(B→A) by construction
    - Parameter sharing across all substitutions at a site
    - Better ranking (consistent AA preference scale)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
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
        h: Tensor,
        positions: Tensor,
        wt_indices: Tensor,
        mut_indices: Tensor,
    ) -> Tensor:
        """
        Vectorized forward for M mutations on one protein.

        Args:
            h: [L, d_model] - transformer output
            positions: [M] - mutation positions
            wt_indices: [M] - WT residue indices (0-19)
            mut_indices: [M] - mutant residue indices (0-19)

        Returns:
            predictions: [M] - ΔΔG predictions
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
        wt_scores = scores.gather(1, wt_indices.unsqueeze(-1)).squeeze(-1)
        mut_scores = scores.gather(1, mut_indices.unsqueeze(-1)).squeeze(-1)

        return mut_scores - wt_scores


class ChaiThermoTransformer(nn.Module):
    """
    Structure-aware transformer for ΔΔG prediction.

    Key features:
    - Chai pair embeddings injected as gated attention biases
    - Per-layer bias gates (initialized ~0.12 via sigmoid(-2))
    - tanh + learnable scale on bias projection (prevents softmax saturation)
    - Small random init on bias projection (enables gradient flow)
    - 20-way site head with exact antisymmetry
    - Vectorized readout for efficient protein-batch processing

    Args:
        single_dim: Chai single embedding dimension (default: 384)
        pair_dim: Chai pair embedding dimension (default: 256)
        d_model: Transformer hidden dimension (default: 256)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 4)
        d_ff: FFN hidden dimension (default: 512)
        dropout: Dropout rate (default: 0.1)
        site_hidden: Site head MLP hidden dimension (default: 128)
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

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Input projections
        self.single_proj = nn.Linear(single_dim, d_model)
        self.single_norm = nn.LayerNorm(d_model)
        self.pair_bias_proj = PairBiasProjection(pair_dim, n_heads)

        # Per-layer bias gate logits (sigmoid applied in forward)
        # Initialized to -2 → sigmoid(-2) ≈ 0.12, so model starts mostly vanilla
        # and learns to incorporate structure bias if helpful
        # Shape: [n_layers, n_heads, 1, 1]
        self.bias_gate_logits = nn.Parameter(torch.full((n_layers, n_heads, 1, 1), -2.0))

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
        single: Tensor,
        pair: Tensor,
        positions: Tensor,
        wt_indices: Tensor,
        mut_indices: Tensor,
    ) -> Tensor:
        """
        Forward pass for all mutations on one protein.

        Args:
            single: Chai single embeddings [L, 384]
            pair: Chai pair embeddings [L, L, 256]
            positions: Mutation positions [M]
            wt_indices: WT residue indices (0-19) [M]
            mut_indices: Mutant residue indices (0-19) [M]

        Returns:
            predictions: ΔΔG predictions [M]
        """
        # Project and normalize single embeddings
        h = self.single_norm(self.single_proj(single))  # [L, d_model]

        # Project pair to attention biases
        pair_bias = self.pair_bias_proj(pair)  # [n_heads, L, L]

        # Transformer layers with gated pair bias
        for layer_idx, layer in enumerate(self.layers):
            gate = torch.sigmoid(self.bias_gate_logits[layer_idx])  # [n_heads, 1, 1]
            h = layer(h, pair_bias, gate)

        h = self.final_norm(h)  # [L, d_model]

        # Predict ΔΔG via 20-way site head
        return self.site_head(h, positions, wt_indices, mut_indices)

    def get_gate_values(self) -> dict:
        """Return current gate values (after sigmoid) for logging/debugging."""
        gates = torch.sigmoid(self.bias_gate_logits)
        return {
            f"layer_{i}": gates[i].detach().squeeze().tolist()
            for i in range(self.n_layers)
        }

    def get_bias_scale_values(self) -> list:
        """Return current bias scale values for logging/debugging."""
        return self.pair_bias_proj.bias_scale.detach().squeeze().tolist()

    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
