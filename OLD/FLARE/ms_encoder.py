import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors")

class MSEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        d_hidden: int,
        d_out: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
        pool: str | None = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_hidden,
            dropout=dropout,
            batch_first=True,  # expects (N, L, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, d_out)
        self.pool = pool


    def forward(self, ms: dict) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ms: dict with
                'tokens': (N, L, d_in)
                'mask':   (N, L) boolean, True = ignore (pytorch convention)
        Returns:
            pool=None:  (node_embeddings, node_masks) → (N, L, d_out), (N, L)
            pool='mean': (N, d_out) mean-pooled representation
            pool='max':  (N, d_out) max-pooled representation
        """
        x = self.input_proj(ms['tokens'])                         # (N, L, d_model)
        x = self.transformer(x, src_key_padding_mask=ms['mask'])  # (N, L, d_model)
        x = self.output_proj(x)                                   # (N, L, d_out)

        # Switch mask to True = keep convention
        keep_mask = ~ms['mask']                                   # (N, L)

        if self.pool == "mean":
            h = (x * keep_mask.unsqueeze(-1)).sum(dim=1) / keep_mask.sum(dim=1, keepdim=True).clamp(min=1)
            return h
        elif self.pool == "max":
            h = x.masked_fill(~keep_mask.unsqueeze(-1), float('-inf')).max(dim=1).values
            return h
        elif self.pool is None:
            out = {'peak_embeddings': x, 'peak_masks': keep_mask}
            return out
        else:
            raise ValueError(f"Unsupported pooling method: {self.pool}")