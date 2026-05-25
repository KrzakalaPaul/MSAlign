from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import lightning.pytorch as pl
import json
from models.JESTR.datamodule import keep_only_k_candidates
from transforms.spectra_transforms import Subformula_Transform
from transforms.molecules_transforms import MoleculeToGraph

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

    return padded, mask

def load_fold_data(labelled_dataset_name, split_method, fold):
    '''
    In FLARE each epoch = one pass on the list of unique smiles.
    (this way, spectra with the same smiles will be seen in different epochs which prevents false negatives in the in-batch contrastive loss)
    
    This functions loads:
    - the list of spectra for the fold (with their mz, intensities and subformulas)
    - a mapping whose keys are the unique smiles in the fold, and the values are lists of indices of spectra with that smiles in the fold spectra list
    '''
    metadata = pd.read_csv(f'data/{labelled_dataset_name}/metadata.csv')
    split = pd.read_csv(f'data/{labelled_dataset_name}/splits/{split_method}.csv')['fold']
    split_mask = (split == fold).values
    with open(f"data/{labelled_dataset_name}/annotated_peaks.json", "r") as f:
        spectra = json.load(f)  # spectra['mz'][i] = list of float, spectra['intensities'][i] list of float, spectra['subformulas'][i] list of string
    spectra_to_smiles = metadata['unique_smiles_idx'].values
    unique_smiles = pd.read_csv(f'data/{labelled_dataset_name}/unique_smiles.csv')['smiles'].values

    print("Preparing safe iteration for FLARE...")
    spectra_fold = []    
    for i in range(len(spectra_to_smiles)):
        if split_mask[i]:
            spectra_fold.append({'mz': spectra['mz'][i], 'intensities': spectra['intensities'][i], 'subformulas': spectra['subformulas'][i]})
    spectra_to_smiles_fold = spectra_to_smiles[split_mask]
    map_smiles_to_spectra_fold = {}
    for idx, smiles_idx in enumerate(spectra_to_smiles_fold):
        smiles = unique_smiles[smiles_idx]
        if smiles not in map_smiles_to_spectra_fold:
            map_smiles_to_spectra_fold[smiles] = []
        map_smiles_to_spectra_fold[smiles].append(idx)
        
    return map_smiles_to_spectra_fold, spectra_fold

class PairDataset(Dataset):
    '''
    Used for pretraining JESTR, no negative loading, just return pairs of (spectrum, positive candidate)
    '''
    def __init__(self,
                 labelled_dataset_name,
                 split_method,
                 fold,
                 ):
        
        # Load fold data
        self.map_smiles_to_spectra_fold, self.spectra_fold = load_fold_data(labelled_dataset_name, split_method, fold)
        self.unique_smiles_fold = sorted(self.map_smiles_to_spectra_fold.keys()) # use sorted to ensure deterministic order of unique smiles for reproducibility
        
        # Load transforms
        self.mol_transform = MoleculeToGraph()
        self.spectra_transform = Subformula_Transform()
        
    def __len__(self):
        return len(self.unique_smiles_fold)
    
    def __getitem__(self, idx):
        smiles = self.unique_smiles_fold[idx]
        mol = self.mol_transform(smiles)
        spectra_indices = self.map_smiles_to_spectra_fold[smiles]
        idx = np.random.choice(spectra_indices) # shuffle the spectra indices for this smiles to ensure different spectra are seen in different epochs
        spectra = self.spectra_fold[idx] # randomly choose one spectrum for this smiles
        if spectra['mz'] is None or spectra['intensities'] is None or spectra['subformulas'] is None:
            tokens = torch.zeros((1, self.spectra_transform.get_dim()))
        else:
            tokens = self.spectra_transform(mz_list=spectra['mz'], intensities_list=spectra['intensities'], formulas_list=spectra['subformulas'])
        tokens = torch.from_numpy(tokens).to(torch.float32)
        return tokens, mol
    
    def collate_fn(self, batch):
        tokens, mol = zip(*batch)
        tokens, mask = collate_tokens(tokens)
        ms = {'tokens': tokens, 'mask': mask}
        mol = dgl.batch(mol)
        return ms, mol
    
    
class CandidateDataset(Dataset):
    '''
    Used for loading candidate molecules for each spectrum in the fold (evaluation and finetuning JESTR)
    '''
    def __init__(self,
                 labelled_dataset_name,
                 candidate_map_name,
                 split_method,
                 fold,
                 k_candidates,
                 ):
        
        self.k_candidates = k_candidates
        
        # Load fold data
        self.map_smiles_to_spectra_fold, self.spectra_fold = load_fold_data(labelled_dataset_name, split_method, fold)
        self.unique_smiles_fold = sorted(self.map_smiles_to_spectra_fold.keys()) # use sorted to ensure deterministic order of unique smiles for reproducibility
        candidate_map_path = f'data/{labelled_dataset_name}/candidates/{candidate_map_name}/map.json'
        with open(candidate_map_path, 'r') as f:
            self.candidate_map = json.load(f)
        # Load transforms
        self.mol_transform = MoleculeToGraph()
        self.spectra_transform = Subformula_Transform()
        
    def __len__(self):
        return len(self.unique_smiles_fold)
    
    def __getitem__(self, idx):
        smiles = self.unique_smiles_fold[idx]
        spectra_indices = self.map_smiles_to_spectra_fold[smiles]
        idx = np.random.choice(spectra_indices) # shuffle the spectra indices for this smiles to ensure different spectra are seen in different epochs
        
        spectra = self.spectra_fold[idx] # randomly choose one spectrum for this smiles
        if spectra['mz'] is None or spectra['intensities'] is None or spectra['subformulas'] is None:
            tokens = torch.zeros((1, self.spectra_transform.get_dim()))
        else:
            tokens = self.spectra_transform(mz_list=spectra['mz'], intensities_list=spectra['intensities'], formulas_list=spectra['subformulas'])
        tokens = torch.from_numpy(tokens).to(torch.float32)
        
        candidates = self.candidate_map[smiles]
        candidates_graphs = []
        for cand in candidates:
            try:
                graph = self.mol_transform(cand)
                assert graph.ndata['h'].shape[1] == 78, f"Expected node feature dimension to be 78, but got {graph.ndata['h'].shape[1]}"
                candidates_graphs.append(graph)
            except Exception as e:
                pass # if the candidate cannot be processed, skip it 

        candidates_graphs = keep_only_k_candidates(candidates_graphs, self.k_candidates)
            
        return tokens, dgl.batch(candidates_graphs)
    
    def collate_fn(self, batch):
        tokens, candidates_graphs = zip(*batch)
        tokens, mask = collate_tokens(tokens)
        ms = {'tokens': tokens, 'mask': mask}
        return ms, candidates_graphs # candidates_graphs is a list of batched graphs
    

    
class FLARE_Datamodule(pl.LightningDataModule):

    def __init__(self, 
                 labelled_dataset_name,
                 candidate_map_name,
                 split_method,
                 batch_size=128, 
                 batch_size_test=16, # ALL candidates are loaded during validation/test, so we need to reduce the batch size to fit in memory
                 n_workers=8, 
                 prefetch_factor=2
                 ):
        super().__init__()
        
        self.train_dataset = PairDataset(labelled_dataset_name=labelled_dataset_name,
                                        split_method=split_method,
                                        fold='train')
        
        self.val_dataset = PairDataset(labelled_dataset_name=labelled_dataset_name,
                                        split_method=split_method,
                                        fold='val')

        self.test_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                              candidate_map_name=candidate_map_name,
                                              split_method=split_method,
                                              fold='test',
                                              k_candidates=None)
        
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
        batch_size= self.batch_size if self.mode == 'pretrain' else self.batch_size_test
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.val_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, generator=generator)
    
    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=True, collate_fn=self.test_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, generator=generator)