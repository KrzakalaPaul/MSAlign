from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import lightning.pytorch as pl
import json
from transforms.molecules_transforms import MoleculeToGraph
from transforms.spectra_transforms import BIN_Transform
from models.MSAlign.datamodule import keep_only_k_candidates, collate_candidates
import dgl

def load_fold_data(labelled_dataset_name, split_method, fold):
    '''
    In JESTR each epoch = one pass on the list of unique smiles.
    (this way, spectra with the same smiles will be seen in different epochs which prevents false negatives in the in-batch contrastive loss)
    
    This functions loads:
    - the list of spectra for the fold
    - a mapping whose keys are the unique smiles in the fold, and the values are lists of indices of spectra with that smiles in the fold spectra list
    '''
    metadata = pd.read_csv(f'data/{labelled_dataset_name}/metadata.csv')
    split = pd.read_csv(f'data/{labelled_dataset_name}/splits/{split_method}.csv')['fold']
    split_mask = (split == fold).values
    spectra = np.load(f'data/{labelled_dataset_name}/spectra.npy')
    spectra_to_smiles = metadata['unique_smiles_idx'].values
    unique_smiles = pd.read_csv(f'data/{labelled_dataset_name}/unique_smiles.csv')['smiles'].values

    print("Preparing safe iteration for JESTR...")
    spectra_fold = spectra[split_mask]
    spectra_to_smiles_fold = spectra_to_smiles[split_mask]
    map_smiles_to_spectra_fold = {}
    for idx, smiles_idx in enumerate(spectra_to_smiles_fold):
        smiles = unique_smiles[smiles_idx]
        if smiles not in map_smiles_to_spectra_fold:
            map_smiles_to_spectra_fold[smiles] = []
        map_smiles_to_spectra_fold[smiles].append(idx)
        
    return map_smiles_to_spectra_fold, spectra_fold

def keep_only_k_candidates(candidates_graphs, k):
    n_valid_candidates = len(candidates_graphs) - 1 # exclude the ground truth which is added manually
    if k is not None and n_valid_candidates > k:
        indices = np.random.choice(range(1, n_valid_candidates + 1), size=k, replace=False)
        candidates_graphs = [candidates_graphs[0]] + [candidates_graphs[i] for i in indices]
    return candidates_graphs

class PairDataset(Dataset):
    '''
    Used for pretraining JESTR, no negative loading, just return pairs of (spectrum, positive candidate)
    '''
    def __init__(self,
                 labelled_dataset_name,
                 split_method,
                 fold,
                 bin_width=1.0,
                 max_mz=1005
                 ):
        
        # Load fold data
        self.map_smiles_to_spectra_fold, self.spectra_fold = load_fold_data(labelled_dataset_name, split_method, fold)
        self.unique_smiles_fold = sorted(self.map_smiles_to_spectra_fold.keys()) # use sorted to ensure deterministic order of unique smiles for reproducibility
        
        # Load transforms
        self.mol_transform = MoleculeToGraph()
        self.spectra_transform = BIN_Transform(max_mz=max_mz, bin_width=bin_width)
        
    def __len__(self):
        return len(self.unique_smiles_fold)
    
    def __getitem__(self, idx):
        smiles = self.unique_smiles_fold[idx]
        mol = self.mol_transform(smiles)
        spectra_indices = self.map_smiles_to_spectra_fold[smiles]
        spectra = np.random.choice(self.spectra_fold[spectra_indices]) # randomly choose one spectrum for this smiles
        ms = torch.from_numpy(self.spectra_transform(spectra)).to(torch.float32)
        return ms, mol
    
    def collate_fn(self, batch):
        ms, mol = zip(*batch)
        ms = torch.stack(ms)
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
                 bin_width=1.0,
                 max_mz=1005
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
        self.spectra_transform = BIN_Transform(max_mz=max_mz, bin_width=bin_width)
        
    def __len__(self):
        return len(self.unique_smiles_fold)
    
    def __getitem__(self, idx):
        smiles = self.unique_smiles_fold[idx]
        spectra_indices = self.map_smiles_to_spectra_fold[smiles]
        spectra = np.random.choice(self.spectra_fold[spectra_indices]) # randomly choose one spectrum for this smiles
        ms = torch.from_numpy(self.spectra_transform(spectra)).to(torch.float32)
        
        candidates = self.candidate_map[smiles]
        candidates_graphs = []
        for cand in candidates:
            try:
                graph = self.mol_preprocessor(cand)
                assert graph.ndata['h'].shape[1] == 78, f"Expected node feature dimension to be 78, but got {graph.ndata['h'].shape[1]}"
                candidates_graphs.append(graph)
            except Exception as e:
                pass # if the candidate cannot be processed, skip it 
        candidates_graphs = keep_only_k_candidates(candidates_graphs, self.k_candidates)
            
        return ms, dgl.batch(candidates_graphs)
    
    def collate_fn(self, batch):
        ms, candidates_graphs = zip(*batch)
        ms = torch.stack(ms)
        return ms, candidates_graphs # candidates_graphs is a list of batched graphs
    

    
class JESTER_Datamodule(pl.LightningDataModule):

    def __init__(self, 
                 labelled_dataset_name,
                 candidate_map_name,
                 split_method,
                 k_candidates,
                 mode,
                 bin_width=0.1,
                 max_mz=1005,
                 batch_size=128, 
                 batch_size_test=16, # ALL candidates are loaded during validation/test, so we need to reduce the batch size to fit in memory
                 n_workers=8, 
                 prefetch_factor=2
                 ):
        super().__init__()
        
        ### Load datasets
        if mode == 'pretrain':
            self.train_dataset = PairDataset(labelled_dataset_name=labelled_dataset_name,
                                            split_method=split_method,
                                            fold='train',
                                            bin_width=bin_width,
                                            max_mz=max_mz)
            
            self.val_dataset = PairDataset(labelled_dataset_name=labelled_dataset_name,
                                            split_method=split_method,
                                            fold='val',
                                            bin_width=bin_width,
                                            max_mz=max_mz)
        elif mode == 'finetune':
            self.train_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                                candidate_map_name=candidate_map_name,
                                                split_method=split_method,
                                                fold='train',
                                                k_candidates=k_candidates,
                                                bin_width=bin_width,
                                                max_mz=max_mz)
            
            self.val_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                                candidate_map_name=candidate_map_name,
                                                split_method=split_method,
                                                fold='val',
                                                k_candidates=None,
                                                bin_width=bin_width,
                                                max_mz=max_mz)
        else:
            raise ValueError(f"Invalid mode {mode}. Expected 'pretrain' or 'finetune'.")
        
        self.mode = mode
        
        self.test_dataset = CandidateDataset(labelled_dataset_name=labelled_dataset_name,
                                              candidate_map_name=candidate_map_name,
                                              split_method=split_method,
                                              fold='test',
                                              k_candidates=None,
                                              bin_width=bin_width,
                                              max_mz=max_mz)
        
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