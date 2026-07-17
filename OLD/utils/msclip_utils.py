from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def optimizer_with_scheduler(self):
    optimizer = torch.optim.AdamW([
        {"params": self.parameters(), "lr": self.hparams["base_lr"]},
    ], weight_decay=self.hparams["weight_decay"])
    
    warmup_steps = self.hparams.n_warmup_steps
    total_steps = self.hparams.n_max_steps

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

def optimizer_without_scheduler(self):
    # Group 1: self.ms_encoder, base_lr_ms_backbone
    # Group 2: self.fc_ms and self.fc_mol, base_lr
    '''
    # Old version with only one learning rate
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["base_lr"], weight_decay=self.hparams["weight_decay"])
    '''
    optimizer = torch.optim.AdamW([
        {"params": self.parameters(), "lr": self.hparams["base_lr"]}
    ], weight_decay=self.hparams["weight_decay"])
    return optimizer

def retrieval_accuracy(ms_emb, candidates_emb, candidates_similarity=None):
        '''
        ms_emb: (B, D)
        candidates_emb: (B, K, D)
        candidates_similarity: (B, K) or None, if not None, it contains the similarity of each candidate to the true molecule
        If provided returns similarity of the top-1 candidate,
        '''
    
        ms_emb = F.normalize(ms_emb, dim=-1)
        candidates_emb = F.normalize(candidates_emb, dim=-1)
        
        n_valid_candidates = torch.sum(~torch.isnan(candidates_emb).any(dim=-1), dim=1)  # (B,)

        logits = torch.einsum("bd,bkd->bk", ms_emb, candidates_emb)  # / tau
        logits = torch.nan_to_num(logits, nan=-1e9)

        labels = torch.zeros(ms_emb.size(0), dtype=torch.long, device=ms_emb.device)

        # Top-k recalls
        top1 = (logits.argmax(dim=1) == labels).float().mean()
        top5 = (
            (logits.topk(5, dim=1).indices == labels.unsqueeze(1))
            .any(dim=1)
            .float()
            .mean()
        )
        top20 = (
            (logits.topk(20, dim=1).indices == labels.unsqueeze(1))
            .any(dim=1)
            .float()
            .mean()
        )

        # Compute ranks
        sorted_indices = logits.argsort(dim=1, descending=True)
        ranks = (sorted_indices == labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
        reciprocal_ranks = 1.0 / ranks.float()
        mrr = reciprocal_ranks.mean()
        
        # If candidates_similarity is provided, compute the similarity of the top-1 candidate
        if candidates_similarity is not None:
            top1_indices = logits.argmax(dim=1)  # (B,)
            top1_similarity = candidates_similarity[torch.arange(candidates_similarity.size(0)), top1_indices]  # (B,)
            log = {
                "R@1": top1.item(),
                "R@5": top5.item(),
                "R@20": top20.item(),
                "MRR": mrr.item(),
                "Avg. Valid Candidates": n_valid_candidates.float().mean().item(),
                "Tanimoto@1": top1_similarity.mean().item()
            }
        else:
            log = {
                "R@1": top1.item(),
                "R@5": top5.item(),
                "R@20": top20.item(),
                "MRR": mrr.item(),
                "Avg. Valid Candidates": n_valid_candidates.float().mean().item()
            }

        return log
    
def top_k_from_logits(logits):
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    # Top-k recalls
    top1 = (logits.argmax(dim=1) == labels).float().mean().item()
    if logits.size(1) >= 5:
        top5 = (
            (logits.topk(5, dim=1).indices == labels.unsqueeze(1))
            .any(dim=1)
            .float()
            .mean().item()
        )
    else:
        top5 = 1
    if logits.size(1) >= 20:
        top20 = (
            (logits.topk(20, dim=1).indices == labels.unsqueeze(1))
            .any(dim=1)
            .float()
            .mean().item()
        )
    else:        
        top20 = 1
    return top1, top5, top20

    
def shuffle_candidates(candidates_emb):
    '''
    shuffle the negative candidates for each sample in the batch, but keep the positive candidate in place (first position)
    candidates_emb: (B, K, D)
    '''
    target = candidates_emb[:, 0, :].clone()  # (B, D')
    shuffled_candidates_emb = candidates_emb[:,1:, :].clone().reshape(-1, candidates_emb.size(-1))  # (B*(K-1), D')
    shuffled_candidates_emb = shuffled_candidates_emb[torch.randperm(shuffled_candidates_emb.size(0))]  # Shuffle
    shuffled_candidates_emb = shuffled_candidates_emb.reshape(candidates_emb.size(0), candidates_emb.size(1)-1, candidates_emb.size(2))  # (B, K-1, D')
    candidates_emb_shuffled = torch.cat([target.unsqueeze(1), shuffled_candidates_emb], dim=1)  # (B, K, D')
    return candidates_emb_shuffled 
    
def infonce_loss(X, Y, temperature=0.07):
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

def infonce_loss_from_logits(logits):
    """
    Compute the InfoNCE loss for a batch of paired samples (X, Y).
    Expect X and Y to be L2-normalized embeddings of shape (B, D).
    
    Args:
        X: Tensor of shape (B, D) - embeddings from modality 1 (e.g., spectra)
        Y: Tensor of shape (B, D) - embeddings from modality 2 (e.g., molecules)
        temperature: Scaling factor for the logits
    """

    similarity_matrix = logits   
    labels = torch.arange(logits.size(0)).to(logits.device)  # (B,)
    loss_x_to_y = nn.CrossEntropyLoss()(similarity_matrix, labels)
    loss_y_to_x = nn.CrossEntropyLoss()(similarity_matrix.T, labels)
    loss = (loss_x_to_y + loss_y_to_x) / 2
    acc = (similarity_matrix.argmax(dim=1) == labels).float().mean().item()
    return loss, acc

def siglip_loss(pred, target, eps, tol):
    '''
    sigmoid loss centered at tol, with a slope of 1/eps at the inflection point
    '''
    pred = (pred - tol) / eps
    return - F.logsigmoid(pred * target)
