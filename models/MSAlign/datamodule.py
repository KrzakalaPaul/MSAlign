from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import h5py
import lightning.pytorch as pl


def keep_only_k_candidates(candidates_embedding, candidates_mask, k):
    '''
    Select a random subset of k candidates.
    Index 0 (ground truth) is always kept at position 0.
    '''
    valid_indices = torch.where(candidates_mask[1:])[0] + 1  # valid non-gt indices
    perm = torch.randperm(len(valid_indices))[:k - 1]
    selected_indices = torch.cat([torch.tensor([0]), valid_indices[perm]])  # gt first, then k-1 others

    selected_embeddings = candidates_embedding[selected_indices]
    selected_mask = candidates_mask[selected_indices]

    # Pad if not enough valid candidates
    if len(selected_indices) < k:
        padding_size = k - len(selected_indices)
        selected_embeddings = torch.cat([selected_embeddings, torch.zeros(padding_size, candidates_embedding.shape[1])], dim=0)
        selected_mask = torch.cat([selected_mask, torch.zeros(padding_size, dtype=torch.bool)], dim=0)

    return selected_embeddings, selected_mask

def collate_candidates(candidates_embeddings, candidates_masks):
    max_candidates = max(emb.shape[0] for emb in candidates_embeddings)
    embedding_dim = candidates_embeddings[0].shape[1]

    padded_embeddings = torch.zeros(len(candidates_embeddings), max_candidates, embedding_dim)
    padded_masks = torch.zeros(len(candidates_masks), max_candidates, dtype=torch.bool)

    for i, (emb, mask) in enumerate(zip(candidates_embeddings, candidates_masks)):
        n_candidates = emb.shape[0]
        padded_embeddings[i, :n_candidates] = emb
        padded_masks[i, :n_candidates] = mask

    return padded_embeddings, padded_masks

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
        self.candidates_emb_path = f'data/{labelled_dataset_name}/candidates/{candidate_map_name}/{encoder_mol}.h5'
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
    
    def collate_fn(self, batch):
        spectra_embeddings, candidates_embeddings, candidates_masks = zip(*batch)
        spectra_embeddings = torch.stack(spectra_embeddings)
        if self.k_candidates is None:
            # If k_candidates is None, the number of candidates can vary across samples, so we need to collate them with padding
            candidates_embeddings, candidates_masks = collate_candidates(candidates_embeddings, candidates_masks)
        else:
            candidates_embeddings, candidates_masks = torch.stack(candidates_embeddings), torch.stack(candidates_masks)

        return spectra_embeddings, candidates_embeddings, candidates_masks

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
    
    def get_pair(self, idx):
        # For loading only positive pairs (spectrum, ground truth candidate) for evaluation
        spectra_embedding = torch.from_numpy(self.spectra_emb[idx])
        smiles_idx = self.spectra_to_smiles[idx]
        h5 = self.get_candidate_h5()
        candidates_embedding = torch.from_numpy(h5['candidates_embeddings'][smiles_idx])  # (n_candidates, dim)
        mol_embedding = candidates_embedding[0]  # ground truth candidate is always at index 0
        return spectra_embedding, mol_embedding
    
class MSAlign_Datamodule(pl.LightningDataModule):

    def __init__(self, 
                 labelled_dataset_name,
                 candidate_map_name,
                 split_method,
                 k_candidates,
                 encoder_mol="chemberta_13M",
                 encoder_spectra="dreams",
                 batch_size=128, 
                 batch_size_test=16, # ALL candidates are loaded during validation/test, so we need to reduce the batch size to fit in memory
                 n_workers=8, 
                 prefetch_factor=2
                 ):
        super().__init__()
        
        ### Load datasets
        self.train_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                              candidate_map_name=candidate_map_name,
                                              split_method=split_method,
                                              fold='train',
                                              k_candidates=k_candidates,
                                              encoder_mol=encoder_mol,
                                              encoder_spectra=encoder_spectra)
        
        self.val_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                              candidate_map_name=candidate_map_name,
                                              split_method=split_method,
                                              fold='val',
                                              k_candidates=None,
                                              encoder_mol=encoder_mol,
                                              encoder_spectra=encoder_spectra)
        
        self.test_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                              candidate_map_name=candidate_map_name,
                                              split_method=split_method,
                                              fold='test',
                                              k_candidates=None,
                                              encoder_mol=encoder_mol,
                                              encoder_spectra=encoder_spectra)
        
        ### Save parameters
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.n_workers = n_workers
        self.prefetch_factor = prefetch_factor
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, persistent_workers=True)
    
    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size_test, shuffle=True, collate_fn=self.val_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, generator=generator)
    
    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=True, collate_fn=self.test_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, generator=generator)
    