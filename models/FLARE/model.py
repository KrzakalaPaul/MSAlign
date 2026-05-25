from lightning.pytorch import LightningModule
import torch
from models.JESTR.utils import MolEncoder, MSEncoder
import torch.nn.functional as F
import torch.nn as nn
from models.MSAlign.utils import optimizer_with_scheduler, batch_infonce, candidate_infonce, candidate_retrieval_accuracy

class FLARE(LightningModule):
    def __init__(self, config):
        super(FLARE, self).__init__()
        self.mol_encoder = MolEncoder(78,
                                      config['shared_dim'],
                                      gnn_type = config['gnn_type'],
                                      gnn_dropout = config['gnn_dropout'],
                                      mlp_dropout = config['mlp_dropout'],
                                      gnn_channels = config['gnn_channels'],
                                      attn_heads = config['attn_heads'],
                                      gnn_hidden_dim = config['gnn_hidden_dim'],
                                      pool = config['pooling'])
        
        self.ms_encoder = MSEncoder(max_mz=config['max_mz'],
                                    bin_width=config['bin_width'],
                                    mlp_dropout=config['mlp_dropout'],
                                    hidden_dim=config['ms_encoder_hidden_dim'],
                                    out_dim=config['shared_dim'])
        
        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(0.07)), requires_grad=False)
        
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.n_warmup_steps = config['n_warmup_steps']
        self.n_max_steps = config['n_max_steps']
        
        self.use_max_sim = config['use_max_sim']
        
        self.save_hyperparameters(config)
        
    def forward(self, ms, mol):
        ms = self.ms_encoder(ms)
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