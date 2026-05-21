from torch.optim.lr_scheduler import LambdaLR
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

def batch_infonce(X, Y, temperature=0.07):
    """
    Compute the InfoNCE loss for a batch of paired samples (X, Y).
    Expect X and Y to be L2-normalized embeddings of shape (B, D).
    
    Args:
        X: Tensor of shape (B, D) - embeddings from modality 1 (e.g., spectra)
        Y: Tensor of shape (B, D) - embeddings from modality 2 (e.g., molecules)
        temperature: Scaling factor for the logits
    """

    similarity_matrix = torch.matmul(X, Y.T) / temperature  # (B, B)    
    labels = torch.arange(X.size(0)).to(X.device)  # (B,)
    loss_x_to_y = nn.CrossEntropyLoss()(similarity_matrix, labels)
    loss_y_to_x = nn.CrossEntropyLoss()(similarity_matrix.T, labels)
    loss = (loss_x_to_y + loss_y_to_x) / 2
    acc = (similarity_matrix.argmax(dim=1) == labels).float().mean().item()
    return loss, acc

def candidate_infonce(ms, candidates, candidates_mask, temperature=0.07):
    # Compute query / candidate similarity + scale by temperature
    ms_candidates_sim = torch.einsum('id,ikd->ik', ms, candidates)  # (B, K)
    ms_candidates_sim = ms_candidates_sim.masked_fill(~candidates_mask, -1e9)  # Mask out NaN candidates
    ms_candidates_sim = ms_candidates_sim / temperature
    log_probs = F.log_softmax(ms_candidates_sim, dim=-1)  # (B, K)
    loss = -log_probs[:, 0].mean()
    acc = log_probs.argmax(dim=-1).eq(0).float().mean().item()  # Assuming the first candidate is the positive one
    return loss, acc

def candidate_retrieval_accuracy(ms, candidates, candidates_mask):
    ms_candidates_sim = torch.einsum('id,ikd->ik', ms, candidates)  # (B, K)
    ms_candidates_sim = ms_candidates_sim.masked_fill(~candidates_mask, -1e9)

    gt_sim = ms_candidates_sim[:, :1]        # (B, 1) - ground truth similarity
    other_sim = ms_candidates_sim[:, 1:]     # (B, K-1)
    correct_rank = (other_sim > gt_sim).sum(dim=-1)  # (B,) number of candidates ranked above gt

    K = ms_candidates_sim.size(1)
    log = {}
    for k in [1, 5, 20]:
        if k <= K:
            log[f'R@{k}'] = (correct_rank < k).float().mean().item()
    return log