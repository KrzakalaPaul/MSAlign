from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import h5py
import lightning.pytorch as pl
import os
import json


def keep_only_k_candidates(candidates_embedding, candidates_mask, k):
    '''
    Select a random subset of k candidates.
    Select non-masked candidates first, then randomly select from the masked candidates if needed.
    '''
    # Get the indices of the non-masked candidates
    valid_indices = torch.where(candidates_mask)[0]
    
    # Randomly select from the valid indices if there are more than k
    if len(valid_indices) > k:
        selected_indices = torch.randperm(len(valid_indices))[:k]
        valid_indices = valid_indices[selected_indices]
    
    # Select the corresponding embeddings and update the mask
    selected_embeddings = candidates_embedding[valid_indices]
    selected_mask = candidates_mask[valid_indices]
    
    # Pad with zeros if there are less than k valid candidates
    if len(valid_indices) < k:
        padding_size = k - len(valid_indices)
        selected_embeddings = torch.cat([selected_embeddings, torch.zeros(padding_size, candidates_embedding.shape[1])], dim=0)
        selected_mask = torch.cat([selected_mask, torch.zeros(padding_size, dtype=torch.bool)], dim=0)
    
    return selected_embeddings, selected_mask


class CandidateDataset(Dataset):
    def __init__(self,
                 labelled_dataset_name,
                 candidate_map_name,
                 split_method,
                 fold,
                 k_candidates,
                 encoder_mol="chemberta_13M",
                 encoder_spectra="dreams",
                 ):
        
        self.k_candidates = k_candidates
        
        # Load All
        self.candidates_emb_path = f'data/{labelled_dataset_name}/candidate_embeddings/{candidate_map_name}/{encoder_mol}.h5'
        metadata = pd.read_csv(f'data/{labelled_dataset_name}/metadata.csv')
        split = pd.read_csv(f'data/{labelled_dataset_name}/splits/{split_method}.csv')['fold']
        split_mask = (split == fold).values
        spectra_emb = np.load(f'data/{labelled_dataset_name}/spectra_embeddings/{encoder_spectra}.npy')
        spectra_to_smiles = metadata['unique_smiles_idx'].values
        
        # Keep only fold data
        self.spectra_emb = spectra_emb[split_mask]
        self.spectra_to_smiles = spectra_to_smiles[split_mask]
        
        self._candidates_h5 = None

    def get_candidate_h5(self):
        """Open the HDF5 file lazily (once per worker)."""
        if self._candidates_h5 is None:
            self._candidates_h5 = h5py.File(self.candidates_emb_path, 'r')
        return self._candidates_h5

    def __len__(self):
        return len(self.spectra_emb)

    def __getitem__(self, idx):
        
        spectra_embedding = torch.from_numpy(self.spectra_emb[idx])
        smiles_idx = self.spectra_to_smiles[idx]

        h5 = self.get_candidate_h5()
        candidates_embedding = torch.from_numpy(h5['candidates_embeddings'][smiles_idx])  # (n_candidates, dim)
        candidates_mask = torch.from_numpy(h5['candidate_mask'][smiles_idx])              # (n_candidates,)

        if self.k_candidates is not None:
            candidates_embedding, candidates_mask = keep_only_k_candidates(
                candidates_embedding, candidates_mask, self.k_candidates
            )

        return spectra_embedding, candidates_embedding, candidates_mask