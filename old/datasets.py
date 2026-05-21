# This is an attempt to write a unified dataset class for all models
# Abandonned because all the "if" make the code hard to read. Instead we write separate dataset classes for each model (with a lot of copy-pasting but at least it's clear and straightforward).

##### Core Idea ######
# We provide 2 types of datasets:
#   - PairsDataset: get item return a pair (ms, mol)
#   - CandidatesDataset: get item return a triplet (ms, candidates, candidates_mask) the first candidate is always the true molecule

# For each type we propose different preprocessing (correspondig to E_mol and E_ms in the paper):

# For spectra, the options are:
#   - 'dreams': in that case ms is a vector of dim d = 1024 (dreams encoding)
#   - 'none': in that case ms is a vector of dim d = 2 x N_max_peaks (m/z and intensity of the N_max_peaks most intense peaks, padded with zeros and flattened)
#   - 'bins': in that case ms is a vector of dim d = N_bins (binned spectrum, with bins of size 0.1 Da, padded with zeros)
#   - 'subformulas': in that case ms is a dictionnary {"tokens":...,"masks":...} where each token represent a peak that has been attributed to a subformula masking is used to distinguish real peaks from padding. 

# For molecules, the options are:
#   - 'chemberta': in that case mol is a vector of dim d = 768 (chemberta encoding)
#   - 'morgan_2048': in that case mol is a vector of dim d = 2048  (morgan fingerprint)
#   - 'morgan_4096': in that case mol is a vector of dim d = 4096  (morgan fingerprint)
#   - 'graph': in that case mol is a DGL object


from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import h5py
import dgl
import json
from ..transforms.molecules_transforms import MorganFingerprintTransform, MoleculeToGraph
from ..transforms.spectra_transforms import BIN_Transform, Subformula_Transform

def collate_tokens(batch: list[torch.Tensor], padding_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        batch: list of N tensors of shape (L_i, d)
        padding_value: value to pad with (default 0.0)
    Returns:
        tokens: (N, L_max, d) padded token tensor
        mask:   (N, L_max) boolean mask, True where valid (not padding)
    """
    max_len = max(tokens.shape[0] for tokens in batch)
    N = len(batch)
    d = batch[0].shape[1]

    padded = torch.full((N, max_len, d), fill_value=padding_value, dtype=torch.float)
    mask = torch.ones(N, max_len, dtype=torch.bool)

    for i, tokens in enumerate(batch):
        L = tokens.shape[0]
        padded[i, :L] = tokens
        mask[i, :L] = False

    return {"tokens": padded, "mask": mask}

def load_molecules(dataset_name, mol_preprocessing):
    if mol_preprocessing == "chemberta_13M":
        molecules = np.load(f'data/{dataset_name}/mol_embeddings/chemberta_13M.npy') # (n_unique_smiles, 768)
        transform = None
    elif mol_preprocessing == "chemberta_100M":
        molecules = np.load(f'data/{dataset_name}/mol_embeddings/chemberta_100M.npy') # (n_unique_smiles, 768)
        transform = None
    elif mol_preprocessing in ["morgan_2048", "morgan_4096"]:
        molecules = pd.read_csv(f'data/{dataset_name}/unique_smiles.csv')['smiles'].values # (n_unique_smiles,)
        transform = MorganFingerprintTransform()
    elif mol_preprocessing == "graph":
        molecules = pd.read_csv(f'data/{dataset_name}/unique_smiles.csv')['smiles'].values # (n_unique_smiles,)
        transform = MoleculeToGraph()
    else:
        raise ValueError(f"Unknown molecule preprocessing: {mol_preprocessing}")
    return molecules, transform

def load_spectra(dataset_name, spectra_preprocessing):
    if spectra_preprocessing == "dreams":
        spectra = np.load(f'data/{dataset_name}/spectra_embeddings/dreams.npy') # (n_samples, 1024)
        transform = None
    elif spectra_preprocessing in ["none"]:
        spectra = np.load(f'data/{dataset_name}/spectra.npy')  # (n_samples, n_peaks, 2)
        transform = None
    elif spectra_preprocessing == "bins": 
        spectra = np.load(f'data/{dataset_name}/spectra.npy')  # (n_samples, n_peaks, 2)
        transform = BIN_Transform()
    elif spectra_preprocessing == "subformulas":
        with open(f"data/{dataset_name}/annotated_peaks.json", "r") as f:
            spectra = json.load(f)  # spectra['mz'][i] = list of float, spectra['intensities'][i] list of float, spectra['subformulas'][i] list of string
        transform = Subformula_Transform()
    else:
        raise ValueError(f"Unknown spectra preprocessing: {spectra_preprocessing}")
    return spectra, transform

class PairsDataset(Dataset):
    
    def __init__(self, 
                 dataset_name,
                 split_method,
                 fold,
                 mol_preprocessing="chemberta_13M",
                 spectra_preprocessing="dreams",
                 ):

        super().__init__()
        
        # Load split indices
        split = pd.read_csv(f'data/{dataset_name}/splits/{split_method}.csv')['fold']
        self.split_indices = [i for i, fold_val in enumerate(split) if fold_val == fold]  # list of indices for this fold
        
        # Load metadata
        self.metadata = pd.read_csv(f'data/{dataset_name}/metadata.csv') # unique_smiles_idx, collision_energy, adduct, precursor_mz
        
        # Load spectra
        self.spectra_preprocessing = spectra_preprocessing
        self.spectra, self.subformula_transform = load_spectra(dataset_name, spectra_preprocessing)
        
        # Load molecules
        self.mol_preprocessing = mol_preprocessing  
        self.molecules, self.mol_transform = load_molecules(dataset_name, mol_preprocessing)
         
    def __len__(self):
        return len(self.split_indices)
    
    def __getitem__(self, idx):
        idx = self.split_indices[idx]
        ms = self.spectra[idx]
        smiles_idx = self.metadata['unique_smiles_idx'].values[idx]
        
        # Spectra preprocessing
        if self.spectra_preprocessing == "dreams":
            ms = torch.from_numpy(ms).float()  # (1024,)
        elif self.spectra_preprocessing == "none":
            ms = torch.from_numpy(ms).float().flatten()  # (n_peaks*2,)
        elif self.spectra_preprocessing == "subformulas":
            mz, intensities, subformulas = self.spectra['mz'][idx], self.spectra['intensities'][idx], self.spectra['subformulas'][idx]
            ms = torch.from_numpy(self.subformula_transform(mz, intensities, subformulas)).float()
        elif self.spectra_preprocessing == "bins":
            ms = torch.from_numpy(self.bin_transform(ms)).float()  # (n_bins,)  
        else:
            raise ValueError(f"Unknown spectra preprocessing: {self.spectra_preprocessing}")
        
        # Molecule preprocessing
        if self.mol_preprocessing in ['chemberta_13M', 'chemberta_100M']:
            mol = torch.from_numpy(self.molecules[smiles_idx]).float()  # (768,)
        elif self.mol_preprocessing in ["morgan_2048", "morgan_4096"]:
            smiles = self.molecules[smiles_idx]
            mol = torch.from_numpy(self.mol_transform(smiles)).float()  # (2048,) or (4096,)
        elif self.mol_preprocessing == "graph":
            smiles = self.molecules[smiles_idx]
            mol = self.mol_transform(smiles)
            
        return ms, mol
    
    def get_spectra_dim(self):
        if self.spectra_preprocessing == "dreams":
            return 1024
        elif self.spectra_preprocessing == "none":
            return self.spectra.shape[1] * 2
        elif self.spectra_preprocessing == "subformulas":
            return self.subformula_transform.get_dim()
        elif self.spectra_preprocessing == "bins":
            return self.bin_transform.get_dim()
        else:
            raise ValueError(f"Unknown spectra preprocessing: {self.spectra_preprocessing}")
        
    def get_mol_dim(self):
        if self.mol_preprocessing in ['chemberta_13M', 'chemberta_100M']:
            return 768
        elif self.mol_preprocessing in ["morgan_2048", "morgan_4096"]:
            return self.mol_transform.get_dim()
        elif self.mol_preprocessing == "graph":
            return self.mol_transform.get_dim()
        else:
            raise ValueError(f"Unknown molecule preprocessing: {self.mol_preprocessing}")
    
    def collate_fn(self, batch):
        ms_batch, mol_batch = zip(*batch)
        
        # Collate spectra
        if self.spectra_preprocessing in ["dreams", "bins", "none"]:
            ms_batch = torch.stack(ms_batch)  # (batch_size, d_ms)
        elif self.spectra_preprocessing == "subformulas":
            ms_batch = collate_tokens(ms_batch)  # {"tokens": (batch_size, L_max, d_token), "mask": (batch_size, L_max)}
        else:
            raise ValueError(f"Unknown spectra preprocessing: {self.spectra_preprocessing}")
        
        # Collate molecules
        if self.mol_preprocessing in ['chemberta_13M', 'chemberta_100M', "morgan_2048", "morgan_4096"]:
            mol_batch = torch.stack(mol_batch)  # (batch_size, d_mol)
        elif self.mol_preprocessing == "graph":
            mol_batch = dgl.batch(mol_batch)
            
        return ms_batch, mol_batch
        
