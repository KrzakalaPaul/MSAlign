from lightning.pytorch import LightningModule
import torch
from .alignment_layers import get_fc_ms, get_fc_mol
from .utils import optimizer_with_scheduler
import torch.nn.functional as F
import torch.nn as nn


class MSAlign(LightningModule):
    def __init__(self, config):
        super(MSAlign, self).__init__()
        self.fc_ms = get_fc_ms(config) 
        self.fc_mol = get_fc_mol(config) 
        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(0.07)), requires_grad=True)
        self.save_hyperparameters(config)

    def encode_ms(self, ms):
        ms = F.normalize(self.fc_ms(ms), p=2, dim=-1)               # (B, D')
        return ms
    
    def encode_mol(self, mol):
        mol = F.normalize(self.fc_mol(mol), p=2, dim=-1)
        return mol
        
    def forward(self, ms, mol):
        ms = self.encode_ms(ms)
        mol = self.encode_mol(mol)
        return ms, mol
    
    def configure_optimizers(self):
        return optimizer_with_scheduler(self)

   
    def compute_losses(self, ms, candidates, candidates_mask):

        # Compute query / candidate similarity + scale by temperature
        ms_candidates_sim = torch.einsum('id,ikd->ik', ms, candidates)  # (B, K)
        ms_candidates_sim = ms_candidates_sim.masked_fill(~candidates_mask, -1e9)  # Mask out NaN candidates
        temperature = torch.exp(self.log_epsilon)
        ms_candidates_sim = ms_candidates_sim / temperature
        log_probs = F.log_softmax(ms_candidates_sim, dim=-1)  # (B, K)
        loss = -log_probs[:, 0].mean()
        log = self.retrieval_metrics(ms, candidates, candidates_mask)
        log['loss'] = loss.detach().item()
        return loss, log
    
    def retrieval_accuracy(self, ms, candidates, candidates_mask):
        ms_candidates_sim = torch.einsum('id,ikd->ik', ms, candidates)  # (B, K)
        ms_candidates_sim = ms_candidates_sim.masked_fill(~candidates_mask, -1e9)

        K = ms_candidates_sim.size(1)
        ranks = ms_candidates_sim.argsort(dim=-1, descending=True)  # (B, K)
        correct_rank = (ranks == 0).nonzero(as_tuple=True)[1]       # (B,) position of gt for each row

        log = {}
        for k in [1, 5, 20]:
            if k <= K:
                log[f'R@{k}'] = (correct_rank < k).float().mean().item()

        return log
    
    def training_step(self, batch):
        ms, candidates, candidates_mask = batch  # ms: (B, D), candidates: (B, K, D), candidates_mask: (B, K)
        
        # Encode all
        ms = self.encode_ms(ms)
        candidates = self.encode_mol(candidates)
        loss, log = self.compute_losses(ms, candidates, candidates_mask)
        
        self.log_dict(
            {f'{k} (train)': v for k, v in log.items()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=ms.size(0)
        )

        return loss
    
    def validation_step(self, batch):
        ms, candidates, candidates_mask = batch  # ms: (B, D), candidates: (B, K, D), candidates_mask: (B, K)
        
        # Project to shared space
        ms = self.encode_ms(ms)
        candidates = self.encode_mol(candidates)
        
        # Do it once with the correct "hard" candidates
        log = self.retrieval_accuracy(ms, candidates, candidates_mask)
        log = {f'{k} (val)': v for k, v in log.items()}
        
        self.log_dict(
            log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=ms.size(0)
        )
        
        return log['R@1 (val)']  # Return hard candidate top1 accuracy for checkpointing
    
    def test_step(self, batch):
        ms, candidates, candidates_mask = batch  # ms: (B, D), candidates: (B, K, D), candidates_mask: (B, K)

        # Project to shared space
        ms = self.encode_ms(ms)
        candidates = self.encode_mol(candidates)
        
        # Do it once with the correct "hard" candidates
        log = self.retrieval_accuracy(ms, candidates, candidates_mask)
        log = {f'{k} (test)': v for k, v in log.items()}
        
        self.log_dict(
            log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=ms.size(0)
        )
        
        return log['R@1 (test)']  # Return hard candidate top1 accuracy for checkpointing



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    similarity_matrix = torch.linspace(-1, 1, steps=1000)
    tol = 0.2
    reg_loss = F.relu(similarity_matrix.abs() + tol - 1) 
    plt.plot(similarity_matrix.numpy(), reg_loss.numpy())
    plt.xlabel("Similarity")
    plt.ylabel("Regularization Loss")
    plt.title("Intra-cluster Regularization Loss vs Similarity")
    plt.grid()
    plt.show()