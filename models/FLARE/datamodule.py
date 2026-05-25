from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import lightning.pytorch as pl
import json
from models.JESTR.datamodule import PairDataset, CandidateDataset

    
class FLARE_Datamodule(pl.LightningDataModule):

    def __init__(self, 
                 labelled_dataset_name,
                 candidate_map_name,
                 split_method,
                 bin_width=0.1,
                 max_mz=1005,
                 batch_size=128, 
                 batch_size_test=16, # ALL candidates are loaded during validation/test, so we need to reduce the batch size to fit in memory
                 n_workers=8, 
                 prefetch_factor=2
                 ):
        super().__init__()
        
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