from models.MSAlign.utils import AlignmentMLP
from models.MSAlign.model import MSAlign
import torch.nn.functional as F
import torch.nn as nn
from transforms.molecules_transforms import MorganFingerprintTransform
from transforms.spectra_transforms import BIN_Transform
import torch

class EmbCos(MSAlign):
    def __init__(self, config):

        mol_transform = MorganFingerprintTransform(fp_size=config['fingerprint_size'])
        d_mol = mol_transform.get_dim()
        spectra_transform = BIN_Transform(max_mz=config['max_mz'], bin_width=config['bin_width'])
        d_ms = spectra_transform.get_dim()
        
        self.fc_ms = AlignmentMLP(d_in=d_ms,
                                  d_hidden=config['d_hidden'],
                                  d_shared=config['d_shared'],
                                  n_hidden_layers=config['n_hidden_layers'],
                                  dropout=config['dropout'],
                                  layernorm=config['layernorm'],
                                  residual=False,
                                  orthogonal_init=False) 
        
        self.fc_mol = AlignmentMLP(d_in=d_mol,
                                   d_hidden=config['d_hidden'],
                                   d_shared=config['d_shared'],
                                   n_hidden_layers=config['n_hidden_layers'],
                                   dropout=config['dropout'],
                                   layernorm=config['layernorm'],
                                   residual=False,
                                   orthogonal_init=False)
        
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.n_warmup_steps = config['n_warmup_steps']
        self.n_max_steps = config['n_max_steps']

        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(0.07)), requires_grad=True)
        
        self.save_hyperparameters(config)
