"""Pair-Aware MLP for protein stability prediction.

Uses Chai-1 single and pair representations with per-group LayerNorm
to predict ΔΔG from mutation features.
"""

import torch
import torch.nn as nn
from torch import Tensor


class PairAwareMLP(nn.Module):
    """
    MLP that predicts ΔΔG from Chai-1 mutation features.

    Architecture:
    - Separate LayerNorm for each feature group (handles raw embedding scale ~400-600)
    - 3-layer MLP with GELU activations and dropout
    - Single scalar output (ΔΔG in kcal/mol)
    """

    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_single: Dimension of single-track embeddings (Chai-1: 384)
            d_pair: Dimension of pair-track embeddings (Chai-1: 256)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.d_single = d_single
        self.d_pair = d_pair

        # Input dimension: 2*d_single + 3*d_pair + 41 = 1577
        input_dim = 2 * d_single + 3 * d_pair + 41

        # LayerNorm for each feature group (raw embeddings have std ~400-600)
        self.norm_local_single = nn.LayerNorm(d_single)
        self.norm_global_single = nn.LayerNorm(d_single)
        self.norm_pair_global = nn.LayerNorm(d_pair)
        self.norm_pair_local_seq = nn.LayerNorm(d_pair)
        self.norm_pair_structural = nn.LayerNorm(d_pair)

        # MLP backbone
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

    def forward(
        self,
        local_single: Tensor,
        global_single: Tensor,
        pair_global: Tensor,
        pair_local_seq: Tensor,
        pair_structural: Tensor,
        mutation_feat: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            local_single: [B, D_single] - embedding at mutation site
            global_single: [B, D_single] - protein-average embedding
            pair_global: [B, D_pair] - average pairwise interactions
            pair_local_seq: [B, D_pair] - local sequence neighbor interactions
            pair_structural: [B, D_pair] - long-range structural interactions
            mutation_feat: [B, 41] - WT/MUT one-hots + relative position

        Returns:
            [B, 1] predicted ΔΔG values
        """
        # Normalize each feature group
        local_single = self.norm_local_single(local_single)
        global_single = self.norm_global_single(global_single)
        pair_global = self.norm_pair_global(pair_global)
        pair_local_seq = self.norm_pair_local_seq(pair_local_seq)
        pair_structural = self.norm_pair_structural(pair_structural)

        # Concatenate all features
        features = torch.cat(
            [
                local_single,
                global_single,
                pair_global,
                pair_local_seq,
                pair_structural,
                mutation_feat,
            ],
            dim=-1,
        )

        return self.net(features)

    def forward_dict(self, features: dict[str, Tensor]) -> Tensor:
        """Forward pass from a dict of features (convenience method)."""
        return self.forward(
            local_single=features["local_single"],
            global_single=features["global_single"],
            pair_global=features["pair_global"],
            pair_local_seq=features["pair_local_seq"],
            pair_structural=features["pair_structural"],
            mutation_feat=features["mutation_feat"],
        )


class PairAwareMLPWithUncertainty(PairAwareMLP):
    """
    Variant that predicts both mean and log-variance for uncertainty estimation.

    Useful for identifying low-confidence predictions (e.g., Cys mutations).
    """

    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__(d_single, d_pair, hidden_dim, dropout)

        # Replace final layer with two heads
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
        )

        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        local_single: Tensor,
        global_single: Tensor,
        pair_global: Tensor,
        pair_local_seq: Tensor,
        pair_structural: Tensor,
        mutation_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            Tuple of (mean [B, 1], log_variance [B, 1])
        """
        # Normalize each feature group
        local_single = self.norm_local_single(local_single)
        global_single = self.norm_global_single(global_single)
        pair_global = self.norm_pair_global(pair_global)
        pair_local_seq = self.norm_pair_local_seq(pair_local_seq)
        pair_structural = self.norm_pair_structural(pair_structural)

        features = torch.cat(
            [
                local_single,
                global_single,
                pair_global,
                pair_local_seq,
                pair_structural,
                mutation_feat,
            ],
            dim=-1,
        )

        hidden = self.net(features)
        mean = self.mean_head(hidden)
        logvar = self.logvar_head(hidden)

        return mean, logvar


def gaussian_nll_loss(
    pred_mean: Tensor,
    pred_logvar: Tensor,
    target: Tensor,
    min_var: float = 1e-6,
) -> Tensor:
    """
    Gaussian negative log-likelihood loss for uncertainty-aware training.

    Args:
        pred_mean: Predicted mean [B, 1]
        pred_logvar: Predicted log-variance [B, 1]
        target: True values [B, 1] or [B]
        min_var: Minimum variance for numerical stability

    Returns:
        Scalar loss
    """
    if target.dim() == 1:
        target = target.unsqueeze(-1)

    var = torch.exp(pred_logvar).clamp(min=min_var)
    loss = 0.5 * (torch.log(var) + (target - pred_mean) ** 2 / var)
    return loss.mean()
