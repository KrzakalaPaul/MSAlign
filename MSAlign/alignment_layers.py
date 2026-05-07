import torch.nn as nn

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
    
