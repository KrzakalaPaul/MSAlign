from torch.utils.data import Dataset, default_collate
import torch
import pandas as pd
import numpy as np
import lightning.pytorch as pl
from JESTR.transforms import MSPreprocessor, MOLPreprocessor
import dgl
import json
from torch.utils.data import default_collate
from JESTR.params import MAX_MZ, BIN_WIDTH
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class JESTR_Dataset(Dataset):
    
    def __init__(self, 
                 labelled_dataset_name,
                 split_method,
                 candidate_dataset_name,
                 fold,
                 ):
        
        #### Load Labelled Dataset ####
        # load molecules
        self.smiles = pd.read_csv(f'data/{labelled_dataset_name}/split_{split_method}/{fold}/smiles.csv')['smiles'].to_list()
        self.mol_preprocessor = MOLPreprocessor(atom_feature="full", bond_feature="full")
        # load spectra
        self.spectra = np.load(f'data/{labelled_dataset_name}/split_{split_method}/{fold}/spectra.npy')
        self.ms_binarizer = MSPreprocessor(max_mz=MAX_MZ, bin_width=BIN_WIDTH)
        # load spectra_metadata 
        self.spectra_metadata = pd.read_csv(f'data/{labelled_dataset_name}/split_{split_method}/{fold}/spectra_meta.csv')
        # load map from spectra to smiles idx
        self.spectra_to_smiles = np.load(f'data/{labelled_dataset_name}/split_{split_method}/{fold}/spectra_to_smiles.npy')
        # load map from mol to spectra idx start and end (possibly multiple spectra per molecule)
        self.smiles_to_ms_start = pd.read_csv(f'data/{labelled_dataset_name}/split_{split_method}/{fold}/smiles.csv')['spectra_idx_start'].to_numpy()
        self.smiles_to_ms_end = pd.read_csv(f'data/{labelled_dataset_name}/split_{split_method}/{fold}/smiles.csv')['spectra_idx_end'].to_numpy()
        
        ### Save metadata ###
        self.n_supervised = len(self.smiles)
        self.labelled_dataset_name = labelled_dataset_name
        self.candidate_dataset_name = candidate_dataset_name
        
    
    def __len__(self):
        return self.n_supervised
    
    def __getitem__(self, mol_idx):
        '''
        Returns:
            ms: the spectrum embedding of the idx-th spectrum
            ms_meta: metadata for the spectrum embedding
            mols: the molecule embedding of the molecule corresponding to the idx-th spectrum
            candidate_mols: the candidate molecule embeddings for the idx-th spectrum
        '''
        # Select a random spectrum for the molecule
        ms_start = self.smiles_to_ms_start[mol_idx]
        ms_end = self.smiles_to_ms_end[mol_idx]
        idx = np.random.randint(ms_start, ms_end) if ms_end > ms_start else ms_start
    
        # Load the spectra
        spectrum = self.spectra[idx]
        ms = self.ms_binarizer(spectrum)
            
        # Load the molecule
        smiles = self.smiles[mol_idx]
        graph = self.mol_preprocessor(smiles)
        
        return ms, graph

    def collate_fn(self, batch):
        '''
        Collate the data in the batch and sample extra unlabelled molecules (clustered by mass)
        '''
        ms, graph = zip(*batch)
        graph = dgl.batch(graph)
        ms = default_collate(ms)
        return ms, graph

    
class JESTR_Candidate_Dataset(Dataset):
    
    def __init__(self, 
                 fold,
                 labelled_dataset_name,
                 split_method,
                 candidate_dataset_name,
                 n_max_candidates=None
                 ):
        
        self.spectra = np.load(f'data/{labelled_dataset_name}/split_{split_method}/{fold}/spectra.npy')
        self.ms_binarizer = MSPreprocessor(max_mz=MAX_MZ, bin_width=BIN_WIDTH)
        self.spectra_metadata = pd.read_csv(f'data/{labelled_dataset_name}/split_{split_method}/{fold}/spectra_meta.csv')
        self.spectra_to_smiles = np.load(f'data/{labelled_dataset_name}/split_{split_method}/{fold}/spectra_to_smiles.npy')
        self.smiles = pd.read_csv(f'data/{labelled_dataset_name}/split_{split_method}/{fold}/smiles.csv')['smiles'].to_list()
        self.mol_preprocessor = MOLPreprocessor(atom_feature="full", bond_feature="full")
        candidate_path = f'data/{labelled_dataset_name}/split_{split_method}/{fold}/candidates/{candidate_dataset_name}_map.json'
        with open(candidate_path) as f:
            self.candidate_map = json.load(f)
        self.n_max_candidates = n_max_candidates
        
    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        ms = self.spectra[idx]
        ms = self.ms_binarizer(ms)
        # Load the molecule
        mol_idx = self.spectra_to_smiles[idx]
        smiles = self.smiles[mol_idx]
        candidates = self.candidate_map[smiles]
        candidates_graphs = []
        for cand in candidates:
            try:
                graph = self.mol_preprocessor(cand)
            except:     
                try:
                    graph = self.mol_preprocessor(self.basic_cleaning(cand))
                except:
                    graph = None
            if graph is not None:
                candidates_graphs.append(graph)
                
        n_valid_candidates = len(candidates_graphs) - 1 # exclude the ground truth which is added manually
        if self.n_max_candidates is not None and n_valid_candidates > self.n_max_candidates:
            indices = np.random.choice(range(1, n_valid_candidates + 1), size=self.n_max_candidates, replace=False)
            candidates_graphs = [candidates_graphs[0]] + [candidates_graphs[i] for i in indices]
        
        target_dim = 78
        candidates_graphs = [g for g in candidates_graphs if g.ndata['h'].shape[1] == target_dim]
        
        return ms, candidates_graphs
    
    def collate_fn(self, batch):
        ms, candidates_graphs = zip(*batch)
        candidates_graphs = [dgl.batch(cands) for cands in candidates_graphs]
        ms = default_collate(ms)
        return ms, candidates_graphs
   
class JESTR_Datamodule(pl.LightningDataModule):

    def __init__(self, 
                 labelled_dataset_name,
                 split_method,
                 candidate_dataset_train_name,
                 candidate_dataset_eval_name,
                 batch_size=32, 
                 batchsize_all_candidates=16, 
                 n_workers=8, 
                 prefetch_factor=2
                 ):
        super().__init__()
        
        ### Load datasets
        self.train_dataset = JESTR_Dataset(labelled_dataset_name=labelled_dataset_name,
                                          split_method=split_method,
                                          candidate_dataset_name=candidate_dataset_train_name,
                                          fold='train',
                                          )
        
        self.val_dataset = JESTR_Dataset(labelled_dataset_name=labelled_dataset_name,
                                        split_method=split_method,
                                        candidate_dataset_name=candidate_dataset_eval_name,
                                        fold='val',
                                        )
        
        self.test_dataset = JESTR_Candidate_Dataset(labelled_dataset_name=labelled_dataset_name,
                                            split_method=split_method,
                                            candidate_dataset_name=candidate_dataset_eval_name,
                                            fold='test')
        
        ### Save parameters
        self.batch_size = batch_size
        self.batchsize_all_candidates = batchsize_all_candidates
        self.n_workers = n_workers
        self.prefetch_factor = prefetch_factor
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, persistent_workers=True)
    
    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.val_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, generator=generator)
    
    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batchsize_all_candidates, shuffle=True, collate_fn=self.test_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, generator=generator)
  
class JESTR_Finetune_Datamodule(pl.LightningDataModule):

    def __init__(self, 
                 labelled_dataset_name,
                 split_method,
                 candidate_dataset_train_name,
                 candidate_dataset_eval_name,
                 n_max_candidates,
                 batchsize_all_candidates=16,
                 batch_size=32, 
                 n_workers=8, 
                 prefetch_factor=2
                 ):
        super().__init__()
        
        ### Load datasets
        self.train_dataset = JESTR_Candidate_Dataset(labelled_dataset_name=labelled_dataset_name,
                                            split_method=split_method,
                                            candidate_dataset_name=candidate_dataset_train_name,
                                            n_max_candidates=n_max_candidates,
                                            fold='train')
        
        self.val_dataset = JESTR_Candidate_Dataset(labelled_dataset_name=labelled_dataset_name,
                                            split_method=split_method,
                                            candidate_dataset_name=candidate_dataset_train_name,
                                            fold='val')
        
        self.test_dataset = JESTR_Candidate_Dataset(labelled_dataset_name=labelled_dataset_name,
                                            split_method=split_method,
                                            candidate_dataset_name=candidate_dataset_eval_name,
                                            fold='test')
        
        ### Save parameters
        self.batch_size = batch_size
        self.batchsize_all_candidates = batchsize_all_candidates
        self.n_workers = n_workers
        self.prefetch_factor = prefetch_factor
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, persistent_workers=True)
    
    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batchsize_all_candidates, shuffle=True, collate_fn=self.val_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, generator=generator)
    
    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batchsize_all_candidates, shuffle=True, collate_fn=self.test_dataset.collate_fn, num_workers=self.n_workers, prefetch_factor=self.prefetch_factor, generator=generator)
  