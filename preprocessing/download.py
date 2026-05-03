import os
import json
import shutil
import pandas as pd
from huggingface_hub import hf_hub_download
from pyteomics import mgf
import numpy as np
from preprocessing.definitions import N_HIGHEST_PEAKS
import urllib.request

def read_mgf_and_extract_data(mgf_path, metadata_keys=['precursor_mz']):
    smiles_list, all_spectra = [], []
    meta_data = {key: [] for key in metadata_keys}

    with mgf.read(mgf_path, use_index=False) as reader:
        for spectrum in reader:
            
            smiles_list.append(spectrum['params']['smiles'])
            for key in metadata_keys:
                meta_data[key].append(spectrum['params'].get(key, None))

            mz, intensities = spectrum['m/z array'], spectrum['intensity array']
            order = np.argsort(intensities)[::-1]
            mz, intensities = mz[order], intensities[order]

            s = np.zeros((N_HIGHEST_PEAKS, 2), dtype=np.float32)
            n_peaks = min(len(mz), N_HIGHEST_PEAKS)
            s[:n_peaks, 0] = mz[:n_peaks]
            s[:n_peaks, 1] = intensities[:n_peaks]
            all_spectra.append(s)

    assert len(smiles_list) == len(all_spectra), "Mismatch in lengths of extracted data"
    all_spectra = np.stack(all_spectra)
    return smiles_list, meta_data, all_spectra

def download_massspecgym(overwrite=False):
    
    os.makedirs("data/massspecgym", exist_ok=True)


    print("Downloading MassSpecGym raw mgf file...")
    mgf_path = hf_hub_download(
    repo_id="roman-bushuiev/MassSpecGym",
    filename="data/auxiliary/MassSpecGym.mgf",
    repo_type="dataset",
    )
    print(f"Done. Saved mgf file to {mgf_path}")
    
    
    if not os.path.exists("data/massspecgym/spectra.npy") or not os.path.exists("data/massspecgym/metadata.csv") or overwrite:
        print("Extracting spectra data from mgf file...")
        smiles_list, meta_data, all_spectra = read_mgf_and_extract_data(mgf_path, metadata_keys=['fold', 'collision_energy', 'adduct', 'precursor_mz'])
        np.save("data/massspecgym/spectra.npy", all_spectra)
        metadata_df = pd.DataFrame(meta_data)
        metadata_df['smiles'] = smiles_list
        metadata_df.to_csv("data/massspecgym/metadata.csv", index=False)
        print(f"Done. Extracted {len(smiles_list)} spectra from MGF file.")
    else:
        print("MassSpecGym data already extracted. Skipping extraction.")
    
def download_spectraverse(overwrite=False):

    os.makedirs("data/spectraverse/raw", exist_ok=True)
    mgf_path = "data/spectraverse/raw/spectra.mgf"    
    
    # Download files (skip if already present)
    if not os.path.exists(mgf_path):
        print("Downloading Spectraverse MGF file...")
        urllib.request.urlretrieve(
            "https://zenodo.org/records/17870921/files/spectraverse-1.0.1.mgf?download=1",
             mgf_path
        )
        print(f"Done. Saved mgf file to {mgf_path}")
    else:
        print("Spectraverse mgf file already exists. Skipping download.")
    
    
    if not os.path.exists("data/spectraverse/spectra.npy") or not os.path.exists("data/spectraverse/metadata.csv") or overwrite:
        
        print("Extracting spectra data from mgf file...")
        smiles_list, meta_data, all_spectra = read_mgf_and_extract_data(mgf_path, metadata_keys=['collision_energy', 'adduct', 'precursor_mz'])
        np.save("data/spectraverse/spectra.npy", all_spectra)
        metadata_df = pd.DataFrame(meta_data)
        metadata_df['smiles'] = smiles_list
        metadata_df.to_csv("data/spectraverse/metadata.csv", index=False)
        print(f"Done. Extracted {len(smiles_list)} spectra from MGF file.")
        
    else:
        print("Spectraverse data already extracted. Skipping extraction.")
