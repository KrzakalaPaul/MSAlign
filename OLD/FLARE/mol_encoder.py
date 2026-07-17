import torch
import torch.nn as nn
from dgllife.model import GCN, GAT
import dgl

class MolEncoder(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 gnn_type = "gcn",
                 gnn_dropout = 0.0,
                 mlp_dropout = 0.0,
                 gnn_channels = [64,128,256],
                 attn_heads = [12,12,12],
                 gnn_hidden_dim = 512,
                 pool: str | None = None,
                 ):
        super().__init__()

        dropout = [gnn_dropout for _ in range(len(gnn_channels))]
        batchnorm = [True for _ in range(len(gnn_channels))]
        gnn_map = {
            "gcn": GCN(in_dim, gnn_channels, batchnorm = batchnorm, dropout = dropout),
            "gat": GAT(in_dim, gnn_channels, attn_heads)
        }
        self.GNN = gnn_map[gnn_type]
        self.pool = pool
        
        if pool is not None:
            self.out_mlp = nn.Sequential(
                nn.Linear(gnn_channels[-1], gnn_hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(gnn_hidden_dim * 2, out_dim),
            )
        else:
            self.out_linear = nn.Linear(gnn_channels[-1], out_dim)
    
    def _dgl_to_node_embeddings(self, g) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract node embeddings and masks from a batched DGL graph.
        Returns:
            node_embeddings: (B, N_max, d)
            node_masks:      (B, N_max) — True = valid node
        """
        graphs = dgl.unbatch(g)                           # list of B individual graphs
        embeddings = [gi.ndata['f'] for gi in graphs]    # list of (n_i, d)

        B     = len(embeddings)
        N_max = max(e.shape[0] for e in embeddings)
        d     = embeddings[0].shape[1]
        device = embeddings[0].device

        node_embeddings = torch.zeros(B, N_max, d, device=device)
        node_masks      = torch.zeros(B, N_max, dtype=torch.bool, device=device)

        for i, e in enumerate(embeddings):
            n_i = e.shape[0]
            node_embeddings[i, :n_i] = e
            node_masks[i, :n_i]      = True

        return node_embeddings, node_masks

    def forward(self, g) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # GNN forward pass
        f = self.GNN(g, g.ndata['h'])
        g.ndata['f'] = f

        # Convert DGL graph to node embeddings and masks
        node_embeddings, node_masks = self._dgl_to_node_embeddings(g)

        # Apply pooling using node_embeddings and node_masks
        if self.pool == "max":
            # Set masked-out nodes to -inf so they don't win the max
            masked = node_embeddings.masked_fill(~node_masks.unsqueeze(-1), float('-inf'))
            out = masked.max(dim=1).values
            out = self.out_mlp(out)
        elif self.pool == "mean":
            # Zero out masked nodes and average over valid ones
            masked = node_embeddings * node_masks.unsqueeze(-1)
            out = masked.sum(dim=1) / node_masks.sum(dim=1, keepdim=True).clamp(min=1)
            out = self.out_mlp(out)
        elif self.pool == "sum":
            masked = node_embeddings * node_masks.unsqueeze(-1)
            out = masked.sum(dim=1)
            out = self.out_mlp(out)
        elif self.pool is None:
            out = {'node_embeddings': self.out_linear(node_embeddings), 'node_masks': node_masks}
        else:
            raise ValueError(f"Unsupported pooling method: {self.pool}")
        
        return out


            