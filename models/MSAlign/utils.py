from torch.optim.lr_scheduler import LambdaLR
import torch
import numpy as np
import torch.nn as nn

def optimizer_with_scheduler(parameters, lr, weight_decay, n_warmup_steps, n_max_steps):
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    warmup_steps = n_warmup_steps
    total_steps = n_max_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine annealing after warmup
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = {
        "scheduler": LambdaLR(optimizer, lr_lambda),
        "interval": "step",     # update per step
        "frequency": 1
    }
    return [optimizer], [scheduler]


class AlignmentMLP(nn.Module):
    def __init__(
        self,
        d_in,
        d_hidden,
        d_shared,
        n_hidden_layers=1,
        dropout=0.0,
        layernorm=False,
        residual=False,
        orthogonal_init=False
    ):
        super().__init__()
            
        if n_hidden_layers == 0:
            self.hidden = nn.Identity()
            self.output_layer = nn.Linear(d_in, d_shared)
            self.residual = False
            self.layernorm = False
        else:
            
            self.residual = residual
            self.layernorm = layernorm
            layers = []
            for i in range(n_hidden_layers):
                in_dim = d_in if i == 0 else d_hidden

                layers.append(nn.Linear(in_dim, d_hidden))

                if layernorm:
                    layers.append(nn.LayerNorm(d_hidden))

                layers.append(nn.GELU())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            self.hidden = nn.Sequential(*layers)
            self.output_layer = nn.Linear(d_hidden, d_shared)

            if residual and d_in != d_shared:
                self.residual_proj = nn.Linear(d_in, d_shared)
            else:
                self.residual_proj = None

        if orthogonal_init:
            self._orthogonal_init()

    def _orthogonal_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.hidden(x)
        out = self.output_layer(h)

        if self.residual:
            if self.residual_proj is not None:
                res = self.residual_proj(x)
            else:
                res = x
            out = out + res

        return out
    
