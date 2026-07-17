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
                 pool = "max"
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
    
        self.fc1_graph = nn.Linear(gnn_channels[len(gnn_channels) - 1], gnn_hidden_dim * 2)
        self.fc2_graph = nn.Linear(gnn_hidden_dim * 2, out_dim)

        self.dropout = nn.Dropout(mlp_dropout)
        self.relu = nn.ReLU()

    def forward(self, g) -> torch.Tensor:
        g1 = g
        f1 = g.ndata['h']
        f = self.GNN(g1, f1)
        g.ndata['f'] = f
        # Pooling f 
        if self.pool == "max":
            h = dgl.max_nodes(g, 'f')
        elif self.pool == "mean":
            h = dgl.mean_nodes(g, 'f')  
        else:
            raise ValueError(f"Invalid pooling method: {self.pool}")
        h1 = self.relu(self.fc1_graph(h))
        h1 = self.dropout(h1)
        h1 = self.fc2_graph(h1)
        h1 = self.dropout(h1)
        return h1