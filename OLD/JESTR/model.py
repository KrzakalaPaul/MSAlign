from lightning.pytorch import LightningModule
import torch
from utils.msclip_utils import optimizer_with_scheduler, retrieval_accuracy, infonce_loss
from .mol_encoder import MolEncoder
from .ms_encoder import MSEncoder
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from JESTR.params import *

class JESTR_model(LightningModule):
    def __init__(self, config):
        super(JESTR_model, self).__init__()
        self.mol_encoder = MolEncoder(NODE_FEATURES,
                                      SHARED_DIM,
                                      gnn_type = GNN_TYPE,
                                      gnn_dropout = GNN_DROPOUT,
                                      mlp_dropout = MLP_DROPOUT,
                                      gnn_channels = GNN_CHANNELS,
                                      attn_heads = ATTN_HEADS,
                                      gnn_hidden_dim = GNN_HIDDEN_DIM,
                                      pool = POOLING)
        
        self.ms_encoder = MSEncoder(max_mz=MAX_MZ,
                                    bin_width=BIN_WIDTH,
                                    mlp_dropout=MLP_DROPOUT,
                                    hidden_dim=MS_ENCODER_HIDDEN_DIM,
                                    out_dim=SHARED_DIM)
        
        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(0.07)), requires_grad=False)
        self.mode = config['mode']
        assert self.mode in ['pretrain', 'finetune'], "Mode must be either 'pretrain' or 'finetune'"
        self.save_hyperparameters(config)
        

    def forward(self, ms, mol):
        ms = self.ms_encoder(ms)
        mol = self.mol_encoder(mol)
        return ms, mol
    
    def configure_optimizers(self):
        return optimizer_with_scheduler(self)
    
    def robust_stack(self, tensors, pad_value=0):
        max_size = max(t.size(0) for t in tensors)
        padded_tensors = []
        for t in tensors:
            if t.size(0) < max_size:
                padding = torch.full((max_size - t.size(0), t.size(1)), pad_value, device=t.device)
                padded_tensors.append(torch.cat([t, padding], dim=0))
            else:
                padded_tensors.append(t)
        return torch.stack(padded_tensors, dim=0)
    
    def weak_loss(self, ms, mol):
        ms = self.ms_encoder(ms)
        mol = self.mol_encoder(mol)
        ms = F.normalize(ms, p=2, dim=1)
        mol = F.normalize(mol, p=2, dim=1)
        loss, acc = infonce_loss(ms, mol, temperature=self.log_epsilon.exp())
        return loss, acc
        
    def strong_loss(self, ms, candidates):
        ms = self.ms_encoder(ms)
        ms = F.normalize(ms, p=2, dim=1) 
        candidates_emb = []
        for c in candidates:
            candidates_emb.append(F.normalize(self.mol_encoder(c), p=2, dim=1))  
        candidates_emb = self.robust_stack(candidates_emb, pad_value=float('nan'))  # (B, K, D)
        
        # First recompute the weak loss
        mol = candidates_emb[:, 0, :]  # Assuming the first candidate is the positive one
        loss_weak, acc_weak = infonce_loss(ms, mol, temperature=self.log_epsilon.exp())
        
        # Then compute the strong loss with all candidates
        candidates_mask = candidates_emb.isnan().any(axis=-1)
        candidates_emb = candidates_emb.masked_fill(candidates_mask.unsqueeze(-1), 0.0)  # Replace NaNs with zeros for masked positions
        ms_candidates_sim = torch.einsum('id,ikd->ik', ms, candidates_emb)  # (B, K)
        ms_candidates_sim = ms_candidates_sim.masked_fill(candidates_mask, -1e9)  # Mask out NaN candidates
        temperature = torch.exp(self.log_epsilon)
        ms_candidates_sim = ms_candidates_sim / temperature

        # Shared log-probs over candidates
        log_probs = F.log_softmax(ms_candidates_sim, dim=-1)  # (B, K)
        acc = log_probs.argmax(dim=-1).eq(0).float().mean()  # Assuming the first candidate is the positive one

        # InfoNCE loss
        loss_strong = -log_probs[:, 0].mean()
        
        loss = 0.1*loss_weak + loss_strong
        
        return loss, acc, acc_weak
    
    
    def training_step(self, batch):
        if self.mode == 'pretrain':
            ms, mol = batch  
            loss, acc = self.weak_loss(ms, mol)
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            return loss
        elif self.mode == 'finetune':
            ms, candidates = batch  
            loss, acc, acc_weak = self.strong_loss(ms, candidates)
            self.log('train_loss (finetune)', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            self.log('train_acc (finetune)', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            self.log('train_acc_weak (finetune)', acc_weak, on_step=True, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            return loss
    
    def validation_step(self, batch):
        if self.mode == 'pretrain':
            ms, mol = batch  
            loss, acc = self.weak_loss(ms, mol)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            return loss
        elif self.mode == 'finetune':
            ms, candidates = batch  
            loss, acc, acc_weak = self.strong_loss(ms, candidates)
            self.log('val_loss (finetune)', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            self.log('val_acc (finetune)', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            self.log('val_acc_weak (finetune)', acc_weak, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            return loss

    
    def test_step(self, batch):
        ms, candidates = batch 
        ms = F.normalize(self.ms_encoder(ms), p=2, dim=1) 
        candidates_emb = []
        for c in candidates:
            candidates_emb.append(F.normalize(self.mol_encoder(c), p=2, dim=1))  
        candidates_emb = self.robust_stack(candidates_emb, pad_value=float('nan'))  # (B, K, D)
        
        # Do it once with the correct "hard" candidates
        log = retrieval_accuracy(ms, candidates_emb)
        log = {f'Hard Candidates {k} (test)': v for k, v in log.items()}
        
        self.log_dict(
            log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=ms.size(0)
        )

        return log['Hard Candidates R@1 (test)']  # Return hard candidate top1 accuracy for checkpointing

