from lightning.pytorch import LightningModule
import torch
from .utils import MSEncoder, MolEncoder
from transforms.spectra_transforms import Subformula_Transform
from transforms.molecules_transforms import MoleculeToGraph
import torch.nn.functional as F
import torch.nn as nn
from models.MSAlign.utils import optimizer_with_scheduler
from .utils import candidate_infonce_from_logits, batch_infonce_from_logits, candidate_retrieval_accuracy_from_logits

class FLARE(LightningModule):
    def __init__(self, config):
        super(FLARE, self).__init__()
        
        self.use_max_sim = config['use_max_sim']
        
        mol_transform = MoleculeToGraph()
        self.mol_encoder = MolEncoder(mol_transform.get_dim(),
                                      config['shared_dim'],
                                      gnn_type = config['gnn_type'],
                                      gnn_dropout = config['gnn_dropout'],
                                      mlp_dropout = config['mlp_dropout'],
                                      gnn_channels = config['gnn_channels'],
                                      attn_heads = config['attn_heads'],
                                      gnn_hidden_dim = config['gnn_hidden_dim'],
                                      pool = None if self.use_max_sim else config['pooling_mol'])
        
        ms_transform = Subformula_Transform()
        self.ms_encoder = MSEncoder(d_in = ms_transform.get_dim(),
                                    d_model=config['transformer_d_model'],
                                    d_hidden=config['transformer_d_hidden'],
                                    d_out=config['shared_dim'],
                                    n_layers=config['transformer_n_layers'],
                                    n_heads=config['transformer_n_heads'],
                                    dropout=config['transformer_dropout'],
                                    pool=None if self.use_max_sim else config['pooling_ms']
                                    )
        
        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(0.07)), requires_grad=False)
        
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.n_warmup_steps = config['n_warmup_steps']
        self.n_max_steps = config['n_max_steps']
        
        self.save_hyperparameters(config)
        
    def forward(self, ms, mol):
        ms = self.ms_encoder(ms, pooling=None if self.use_max_sim else self.pooling)
        mol = self.mol_encoder(mol)
        return ms, mol
    
    def configure_optimizers(self):
        return optimizer_with_scheduler(
            parameters=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_warmup_steps=self.n_warmup_steps,
            n_max_steps=self.n_max_steps
        )
        
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
    
    def loss(self, ms, mol):
        
        if not self.use_max_sim:
            ms = F.normalize(self.ms_encoder(ms), p=2, dim=1)
            mol = F.normalize(self.mol_encoder(mol), p=2, dim=1)
            logits = torch.matmul(ms, mol.t()) / self.log_epsilon.exp()
        else:
            ms_tokens, peak_masks = self.ms_encoder(ms)
            mol_nodes, node_masks = self.mol_encoder(mol)
            ms_tokens  = F.normalize(ms_tokens,  p=2, dim=-1)
            mol_nodes  = F.normalize(mol_nodes,  p=2, dim=-1)

            # All pairwise peak-node cosine similarities
            # ms_tokens = (spec, peak, dim)
            # mol_nodes = (mol, node, dim)
            logits = torch.einsum('spd,mnd->smpn', ms_tokens, mol_nodes)  # (Spec, Mol, Peak, Node)

            # Mask padded nodes → won't win the max
            logits = logits.masked_fill(~node_masks.unsqueeze(0).unsqueeze(2), float('-inf'))  # (Spec, Mol, Peak, Node)

            # Max over nodes: best node match for each peak
            logits = logits.max(dim=-1).values  # (Spec, Mol, Peak)

            # Mask padded peaks → contribute 0 to the mean
            logits = logits.masked_fill(~peak_masks.unsqueeze(1), 0.0)  # (Spec, Mol, Peak)

            # Mean over valid peaks only
            peak_counts = peak_masks.sum(dim=-1, keepdim=True).clamp(min=1)  # (Spec, 1)
            logits = logits.sum(dim=-1) / peak_counts                       # (Spec, Mol)

            logits = logits / self.log_epsilon.exp()
            
        return batch_infonce_from_logits(logits)

    def retrieval_accuracy(self, ms, candidates):

        batchsize = len(candidates)
        n_max_candidates = max(cands.batch_size for cands in candidates) # cands is a dgl batch
        # Invalid padded candidate positions must never outrank real candidates,
        # whose cosine similarities may legitimately be negative.
        logits = torch.full(
            (batchsize, n_max_candidates),
            fill_value=float('-inf'),
            device=ms['tokens'].device,
        )
        if not self.use_max_sim:
            ms = F.normalize(self.ms_encoder(ms), p=2, dim=1)
            for i in range(batchsize):
                cands_i = candidates[i]
                cands_i = F.normalize(self.mol_encoder(cands_i), p=2, dim=-1)
                logits_i = torch.einsum('d,cd->c', ms[i], cands_i) / self.log_epsilon.exp()
                logits[i, :cands_i.size(0)] = logits_i
        else:
            ms_tokens, peak_masks = self.ms_encoder(ms)
            ms_tokens = F.normalize(ms_tokens, p=2, dim=-1)
            for i in range(batchsize):
                cand_i = candidates[i]
                mol_nodes_i, node_masks_i = self.mol_encoder(cand_i)
                mol_nodes_i = F.normalize(mol_nodes_i, p=2, dim=-1)
                ms_tokens_i = ms_tokens[i]  # (P, d)
                peak_masks_i = peak_masks[i]  # (P,)
                # All pairwise peak-node cosine similarities
                logits_i = torch.einsum('pd,cnd->cpn', ms_tokens_i, mol_nodes_i)             # (C, P, N)
                # Mask padded nodes → won't win the max
                logits_i = logits_i.masked_fill(~node_masks_i.unsqueeze(1), float('-inf'))  # (C, P, N)
                # Max over nodes: best node match for each peak
                logits_i = logits_i.max(dim=-1).values                                   # (C, P)
                # Mask padded peaks → contribute 0 to the mean
                logits_i = logits_i.masked_fill(~peak_masks_i.unsqueeze(0), 0.0)              # (C, P)
                # Mean over valid peaks only
                logits_i = logits_i.sum(dim=-1) / peak_masks_i.sum().clamp(min=1)             # (C,)
                logits_i = logits_i / self.log_epsilon.exp()
                logits[i, :cand_i.batch_size] = logits_i
        log = candidate_retrieval_accuracy_from_logits(logits)
        print(log)
        return log
          
    
    def training_step(self, batch):
        batch_size = batch[0]['tokens'].size(0)
        ms, mol = batch  
        loss, acc = self.loss(ms, mol)
        self.log('loss (train)', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('R@1 - batch (train)', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch):
        batch_size = batch[0]['tokens'].size(0)
        ms, mol = batch  
        loss, acc = self.loss(ms, mol)
        self.log('loss (val)', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('R@1 - batch (val)', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    
    def test_step(self, batch):
        ms, candidates = batch
        batch_size = ms['tokens'].size(0)
        log = self.retrieval_accuracy(ms, candidates)
        log = {f'{k} (test)': v for k, v in log.items()}
        self.log_dict(
            log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size
        )
        return log['R@1 (test)']  # Return hard candidate top1 accuracy for checkpointing
