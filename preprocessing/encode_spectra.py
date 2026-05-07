from dreams.api import PreTrainedModel
from dreams.utils.dformats import DataFormatA
from dreams.utils.data import SpectrumPreprocessor
from torch.utils.data import DataLoader, Dataset
from preprocessing.definitions import N_HIGHEST_PEAKS
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import os 

class DreamsSpectraEncoder(nn.Module):
    def __init__(self):
        super(DreamsSpectraEncoder, self).__init__()
        self.model = PreTrainedModel.from_name('DreaMS_embedding').model
        
    def forward(self, spectra):
        embeddings = self.model(spectra)
        return embeddings
    
class SpectraDataset(Dataset):
    def __init__(self, spectra, meta):
        self.spectra = spectra
        self.meta = meta
        self.preprocessor = SpectrumPreprocessor(DataFormatA(), n_highest_peaks=N_HIGHEST_PEAKS)

    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        prec_mz = self.meta.iloc[idx]["precursor_mz"]
        ms = torch.tensor(self.preprocessor(spectrum, prec_mz), dtype=torch.float32)
        return ms

def get_dreams_embeddings(dataset_name, batch_size=32, n_workers=4, overwrite=False):
    if os.path.exists(f"data/{dataset_name}/spectra_embeddings/dreams.npy") and not overwrite:
        print(f"Encoded spectra already exist for {dataset_name}. Skipping encoding.")
        return
    device = 'cuda'
    spectra = np.load(f"data/{dataset_name}/spectra.npy")  # [N, n_peaks, 2]
    meta = pd.read_csv(f"data/{dataset_name}/metadata.csv")
    dataset = SpectraDataset(spectra, meta)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, drop_last=False)
    all_embeddings = []
    encoder = DreamsSpectraEncoder().to(device)
    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding spectra for {dataset_name}"):
            batch = batch.to(device)
            embeddings = encoder(batch)
            all_embeddings.append(embeddings.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    os.makedirs(f"data/{dataset_name}/spectra_embeddings", exist_ok=True)
    np.save(f"data/{dataset_name}/spectra_embeddings/dreams.npy", all_embeddings)