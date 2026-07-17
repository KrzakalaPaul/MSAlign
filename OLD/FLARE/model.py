from lightning.pytorch import LightningModule
import torch
from utils.msclip_utils import optimizer_with_scheduler, retrieval_accuracy, infonce_loss, infonce_loss_from_logits 
from .mol_encoder import MolEncoder
from .ms_encoder import MSEncoder
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from .params import *

class FLARE_model(LightningModule):
    def __init__(self, config):
        super(FLARE_model, self).__init__()
        
        self.mol_encoder = MolEncoder(NODE_FEATURES,
                                      SHARED_DIM,
                                      gnn_type = GNN_TYPE,
                                      gnn_dropout = GNN_DROPOUT,
                                      mlp_dropout = MLP_DROPOUT,
                                      gnn_channels = GNN_CHANNELS,
                                      attn_heads = ATTN_HEADS,
                                      gnn_hidden_dim = GNN_HIDDEN_DIM,
                                      pool = None if MAX_SIM else POOLING,
                                      )
        
        self.ms_encoder = MSEncoder(d_in = PEAKS_FEATURES,
                                    d_model=TRANSFORMER_D_MODEL,
                                    d_hidden=TRANSFORMER_D_HIDDEN,
                                    d_out=SHARED_DIM,
                                    n_layers=TRANSFORMER_N_LAYERS,
                                    n_heads=TRANSFORMER_N_HEADS,
                                    dropout=TRANSFORMER_DROPOUT,
                                    pool = None if MAX_SIM else POOLING,
                                    )
        
        self.lambda_strong = LAMBDA_STRONG
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
    
    def robust_stack(self, tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stack a ragged list of tensors into a padded batch.

        Args:
            tensors: list of B tensors, each of shape (c_i, d)
        Returns:
            stacked:  (B, C_max, d) padded tensors
            pad_mask: (B, C_max) boolean, True = padded (invalid)
        """
        B, C_max, d = len(tensors), max(t.shape[0] for t in tensors), tensors[0].shape[1]
        device = tensors[0].device

        stacked  = torch.zeros(B, C_max, d, device=device)
        pad_mask = torch.ones(B, C_max, dtype=torch.bool, device=device)  # True = padded

        for i, t in enumerate(tensors):
            stacked[i, :t.shape[0]]  = t
            pad_mask[i, :t.shape[0]] = False

        return stacked, pad_mask
    
    def weak_infonce_loss(self, ms, mol):
        
        if not MAX_SIM:
            ms = F.normalize(ms, p=2, dim=1) 
            mol = F.normalize(mol, p=2, dim=1)
            logits = torch.matmul(ms, mol.t()) / self.log_epsilon.exp()
        else:
            ms_tokens, ms_masks = ms['peak_embeddings'], ms['peak_masks']
            mol_nodes, mol_masks = mol['node_embeddings'], mol['node_masks']
            ms_tokens  = F.normalize(ms_tokens,  p=2, dim=-1)
            mol_nodes  = F.normalize(mol_nodes,  p=2, dim=-1)

            # All pairwise peak-node cosine similarities
            logits = torch.einsum('spd,mnd->smpn', ms_tokens, mol_nodes)  # (Spec, Mol, Peak, Node)

            # Mask padded nodes → won't win the max
            logits = logits.masked_fill(~mol_masks.unsqueeze(0).unsqueeze(2), float('-inf'))  # (Spec, Mol, Peak, Node)

            # Max over nodes: best node match for each peak
            logits = logits.max(dim=-1).values  # (Spec, Mol, Peak)

            # Mask padded peaks → contribute 0 to the mean
            logits = logits.masked_fill(~ms_masks.unsqueeze(1), 0.0)  # (Spec, Mol, Peak)

            # Mean over valid peaks only
            peak_counts = ms_masks.sum(dim=-1, keepdim=True).clamp(min=1)  # (Spec, 1)
            logits = logits.sum(dim=-1) / peak_counts                       # (Spec, Mol)

            logits = logits / self.log_epsilon.exp()
            
        similarity_matrix = logits   
        labels = torch.arange(logits.size(0)).to(logits.device)  # (B,)
        loss_x_to_y = nn.CrossEntropyLoss()(similarity_matrix, labels)
        loss_y_to_x = nn.CrossEntropyLoss()(similarity_matrix.T, labels)
        loss = (loss_x_to_y + loss_y_to_x) / 2
        acc = (similarity_matrix.argmax(dim=1) == labels).float().mean().item()
        return loss, acc

    def strong_loss(self, ms, candidates):

        batchsize = len(candidates)
        loss = 0.0
        top1 = 0
        top5 = 0
        top20 = 0

        for i in range(batchsize):
            cands_i = candidates[i] 
            if not MAX_SIM:
                ms_i    = ms[i]                        # (P, d) or (d,)
                ms_i    = F.normalize(ms_i,    p=2, dim=-1)  # (d,)
                cands_i = F.normalize(cands_i, p=2, dim=-1)  # (C, d)
                logits_i = torch.einsum('d,cd->c', ms_i, cands_i) / self.log_epsilon.exp()  # (C,)

            else:
                ms_tokens, ms_mask   = ms['peak_embeddings'][i], ms['peak_masks'][i]           # (P, d), (P,)
                mol_nodes, mol_masks = cands_i['node_embeddings'], cands_i['node_masks']      # (C, N, d), (C, N)
                ms_tokens = F.normalize(ms_tokens, p=2, dim=-1)
                mol_nodes = F.normalize(mol_nodes, p=2, dim=-1)

                # All pairwise peak-node cosine similarities
                logits_i = torch.einsum('pd,cnd->cpn', ms_tokens, mol_nodes)             # (C, P, N)
                # Mask padded nodes → won't win the max
                logits_i = logits_i.masked_fill(~mol_masks.unsqueeze(1), float('-inf'))  # (C, P, N)
                # Max over nodes: best node match for each peak
                logits_i = logits_i.max(dim=-1).values                                   # (C, P)
                # Mask padded peaks → contribute 0 to the mean
                logits_i = logits_i.masked_fill(~ms_mask.unsqueeze(0), 0.0)              # (C, P)
                # Mean over valid peaks only
                logits_i = logits_i.sum(dim=-1) / ms_mask.sum().clamp(min=1)             # (C,)
                logits_i = logits_i / self.log_epsilon.exp()

            # First candidate is always the positive
            loss += -logits_i.log_softmax(dim=0)[0]
            top1 += logits_i.argmax().eq(0).float().item()
            top5 += logits_i.topk(5, dim=0)[1].eq(0).any().float().item() if len(logits_i) >= 5 else 1
            top20 += logits_i.topk(20, dim=0)[1].eq(0).any().float().item() if len(logits_i) >= 20 else 1

        loss = loss / batchsize
        log = {'R@1': top1 / batchsize,
               'R@5': top5 / batchsize,
               'R@20': top20 / batchsize}
        return loss, log
    
    def training_step(self, batch):
        batch_size = batch[0]['tokens'].size(0)
        if self.mode == 'pretrain':
            ms, mol = batch  
            ms = self.ms_encoder(ms)
            mol = self.mol_encoder(mol)
            loss, acc = self.weak_infonce_loss(ms, mol)
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        elif self.mode == 'finetune':
            ms, mol, candidates = batch
            ms = self.ms_encoder(ms)
            mol = self.mol_encoder(mol)
            candidates = [self.mol_encoder(cands) for cands in candidates]
            loss_weak, acc_weak = self.weak_infonce_loss(ms, mol)
            loss_strong, log = self.strong_loss(ms, candidates)
            acc_strong = log['R@1']
            loss = loss_weak + self.lambda_strong * loss_strong
            self.log('train_loss (finetune)', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('train_acc (finetune)', acc_strong, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('train_acc_weak (finetune)', acc_weak, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch):
        batch_size = batch[0]['tokens'].size(0)
        if self.mode == 'pretrain':
            ms, mol = batch  
            ms = self.ms_encoder(ms)
            mol = self.mol_encoder(mol)
            loss, acc = self.weak_infonce_loss(ms, mol)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        elif self.mode == 'finetune':
            ms, mol, candidates = batch
            ms = self.ms_encoder(ms)
            mol = self.mol_encoder(mol)
            candidates = [self.mol_encoder(cands) for cands in candidates]
            loss_weak, acc_weak = self.weak_infonce_loss(ms, mol)
            loss_strong, log = self.strong_loss(ms, candidates)
            acc_strong = log['R@1']
            loss = loss_weak + self.lambda_strong * loss_strong
            self.log('val_loss (finetune)', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('val_acc (finetune)', acc_strong, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('val_acc_weak (finetune)', acc_weak, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    
    def test_step(self, batch):
        ms, _, candidates = batch
        batch_size = ms['tokens'].size(0)
        ms = self.ms_encoder(ms)
        candidates = [self.mol_encoder(cands) for cands in candidates]
        _, log = self.strong_loss(ms, candidates)
        log = {f'Hard Candidates {k} (test)': v for k, v in log.items()}
        self.log_dict(
            log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size
        )
        return log['Hard Candidates R@1 (test)']  # Return hard candidate top1 accuracy for checkpointing

