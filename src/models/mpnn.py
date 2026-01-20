"""Message Passing Neural Network for protein stability prediction.

Uses Chai-1 single embeddings as node features and pair embeddings as edge features.
The model constructs a local subgraph around each mutation site and uses message
passing to aggregate structural context.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch


class MPNNLayer(MessagePassing):
    """Single MPNN layer with edge-conditioned message passing."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add")

        # Message network: combines source node, target node, and edge features
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update network: combines old node state with aggregated messages
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]

        Returns:
            Updated node features [N, hidden_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """Compute messages from source nodes j to target nodes i."""
        # x_i: target node features, x_j: source node features
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_net(inputs)

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        """Update node features with aggregated messages."""
        inputs = torch.cat([x, aggr_out], dim=-1)
        # Residual connection
        return x + self.dropout(self.update_net(inputs))


class ChaiMPNN(nn.Module):
    """
    MPNN for stability prediction using Chai-1 embeddings.

    Architecture:
    - Encoder: Projects node (single) and edge (pair) features to hidden dim
    - Processor: Stack of MPNN layers for message passing
    - Decoder: Predicts ΔΔG from the mutated residue's final representation

    The input is a PyG Data object where:
    - x: Node features (single embeddings of k-nearest neighbors)
    - edge_index: Fully connected edges within the neighborhood
    - edge_attr: Edge features (pair embeddings between nodes)
    - The mutated residue is always at node index 0
    """

    def __init__(
        self,
        node_in_dim: int = 384,
        edge_in_dim: int = 256,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_global_pool: bool = True,
    ):
        """
        Args:
            node_in_dim: Input node feature dimension (Chai single: 384)
            edge_in_dim: Input edge feature dimension (Chai pair: 256)
            hidden_dim: Hidden dimension for node features
            edge_hidden_dim: Hidden dimension for edge features
            num_layers: Number of MPNN layers
            dropout: Dropout probability
            use_global_pool: Whether to include global mean pooling in readout
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_global_pool = use_global_pool

        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.GELU(),
        )

        # MPNN layers
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Readout head
        readout_dim = hidden_dim * 2 if use_global_pool else hidden_dim
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: Data | Batch) -> Tensor:
        """
        Forward pass.

        Args:
            data: PyG Data or Batch object with:
                - x: [N, node_in_dim] node features
                - edge_index: [2, E] edge connectivity
                - edge_attr: [E, edge_in_dim] edge features
                - ptr: [B+1] batch pointers (for batched data)

        Returns:
            [B, 1] predicted ΔΔG values
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encode
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Message passing
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Readout: extract mutated node (always index 0 in each subgraph)
        if hasattr(data, "ptr"):
            # Batched data: ptr gives start indices of each graph
            mutated_indices = data.ptr[:-1]
        else:
            # Single graph
            mutated_indices = torch.tensor([0], device=x.device)

        mutated_features = x[mutated_indices]  # [B, hidden_dim]

        if self.use_global_pool:
            # Add global mean pooling over each graph
            if hasattr(data, "batch"):
                # Scatter mean over batch
                from torch_geometric.nn import global_mean_pool
                global_features = global_mean_pool(x, data.batch)  # [B, hidden_dim]
            else:
                global_features = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]

            readout = torch.cat([mutated_features, global_features], dim=-1)
        else:
            readout = mutated_features

        return self.head(readout)


class ChaiMPNNWithMutationInfo(ChaiMPNN):
    """
    MPNN variant that also incorporates mutation identity information.

    Adds WT/MUT amino acid embeddings and relative position to the readout.
    """

    def __init__(
        self,
        node_in_dim: int = 384,
        edge_in_dim: int = 256,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_global_pool: bool = True,
        num_amino_acids: int = 20,
    ):
        super().__init__(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            edge_hidden_dim=edge_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_global_pool=use_global_pool,
        )

        # Amino acid embeddings for mutation identity
        self.aa_embedding = nn.Embedding(num_amino_acids, 16)

        # Update readout head to include mutation info
        readout_dim = hidden_dim * 2 if use_global_pool else hidden_dim
        readout_dim += 32 + 1  # +32 for WT/MUT embeddings, +1 for relative position

        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: Data | Batch) -> Tensor:
        """
        Forward pass.

        Args:
            data: PyG Data or Batch with additional fields:
                - wt_idx: [B] wild-type amino acid indices
                - mut_idx: [B] mutant amino acid indices
                - rel_pos: [B] relative position in sequence

        Returns:
            [B, 1] predicted ΔΔG values
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encode
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Message passing
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Readout: extract mutated node
        if hasattr(data, "ptr"):
            mutated_indices = data.ptr[:-1]
        else:
            mutated_indices = torch.tensor([0], device=x.device)

        mutated_features = x[mutated_indices]

        if self.use_global_pool:
            from torch_geometric.nn import global_mean_pool
            if hasattr(data, "batch"):
                global_features = global_mean_pool(x, data.batch)
            else:
                global_features = x.mean(dim=0, keepdim=True)
            graph_features = torch.cat([mutated_features, global_features], dim=-1)
        else:
            graph_features = mutated_features

        # Add mutation identity info
        wt_emb = self.aa_embedding(data.wt_idx)  # [B, 16]
        mut_emb = self.aa_embedding(data.mut_idx)  # [B, 16]
        rel_pos = data.rel_pos.unsqueeze(-1)  # [B, 1]

        readout = torch.cat([graph_features, wt_emb, mut_emb, rel_pos], dim=-1)

        return self.head(readout)
