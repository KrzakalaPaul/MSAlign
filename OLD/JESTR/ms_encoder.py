import torch.nn as nn
import torch

class MSEncoder(nn.Module):
    def __init__(self, 
                 max_mz, 
                 bin_width,
                 mlp_dropout,
                 hidden_dim,
                 out_dim):
        super(MSEncoder, self).__init__()
        bin_size = int(max_mz / bin_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.mz_fc1 = nn.Linear(bin_size, hidden_dim)
        self.mz_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mz_fc3 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, mzi_b):
                
       h1 = self.mz_fc1(mzi_b)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       h1 = self.mz_fc2(h1)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       mz_vec = self.mz_fc3(h1)
       mz_vec = self.dropout(mz_vec)
       
       return mz_vec
