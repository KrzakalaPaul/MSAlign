import os
import pandas as pd
from huggingface_hub import hf_hub_download
from pyteomics import mgf
import numpy as np
from preprocessing.definitions import N_HIGHEST_PEAKS
import urllib.request
import json
from preprocessing.utils.rdkit_mp import normalize_candidate_dict, preprocess_smiles_list

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

def download_massspecgym_candidates(overwrite=False, n_threads=16, chunk_size=4096, by="mass"):

    if os.path.exists(f"data/massspecgym/candidates/{by}_candidates_original/map.json") and not overwrite:
        print(f"MassSpecGym {by} candidates already downloaded and normalized. Skipping.")
        return
    
    candidates_path = hf_hub_download(
    repo_id="roman-bushuiev/MassSpecGym",
    filename=f"data/molecules/MassSpecGym_retrieval_candidates_{by}.json",
    repo_type="dataset"
    )
    with open(candidates_path, "r") as f:
        candidate_map = json.load(f)

    # Normalize candidate maps 
    candidate_map = normalize_candidate_dict(candidate_map,
                                            n_threads=n_threads,
                                            chunk_size=chunk_size,
                                            canonical=True,
                                            isomeric=False,
                                            sanitize=True,
                                            None_For_invalid=False,
                                            description=f"Normalizing original massspecgym candidate map (by {by})")

    # Save normalized candidate maps
    os.makedirs(f"data/massspecgym/candidates/{by}_candidates_original", exist_ok=True)
    with open( f"data/massspecgym/candidates/{by}_candidates_original/map.json", 'w') as f:
        json.dump(candidate_map, f)
        
def download_large_unlabelled_smiles(overwrite=False):

    os.makedirs("data/large_unlabelled_smiles", exist_ok=True)
    smiles_path = "data/large_unlabelled_smiles/smiles.txt"
    
    if not os.path.exists(smiles_path) or overwrite:
        print("Downloading large unlabelled SMILES dataset...")
        urllib.request.urlretrieve(
            "https://zenodo.org/records/17870921/files/unlabelled_smiles.txt?download=1",
             smiles_path
        )
        print(f"Done. Saved unlabelled SMILES to {smiles_path}")
    else:
        print("Large unlabelled SMILES file already exists. Skipping download.")
        
def download_candidate_pool(subset='1M', overwrite=False, n_threads=16, chunk_size=4096):
    
    if os.path.exists(f"data/candidate_pools/{subset}.csv") and not overwrite:
        print(f"Candidate pool {subset} already exists. Skipping download and preprocessing.")
        return
    
    if subset == '1M':
        filename="data/molecules/candidate_pools/MassSpecGym_retrieval_molecules_1M.tsv"
    elif subset == '4M':
        filename="data/molecules/candidate_pools/MassSpecGym_retrieval_molecules_4M.tsv"
    elif subset == '118M':
        filename="data/molecules/candidate_pools/MassSpecGym_retrieval_molecules_pubchem_118M.tsv"

    path = hf_hub_download(
    repo_id="roman-bushuiev/MassSpecGym",
    filename=filename,
    repo_type="dataset",
    )

    df_unlabeled = pd.read_csv(path, sep='\t')
    smiles_unlabeled = sorted(set(df_unlabeled.iloc[:, 0].tolist()))

    smiles, formulas, masses = preprocess_smiles_list(smiles_unlabeled,
                                                        n_threads=n_threads,
                                                        chunk_size=chunk_size,
                                                        canonical=True,
                                                        isomeric=False,
                                                        sanitize=True,
                                                        description=f"Preprocessing candidate pool: {subset}")
        
    # Order by len(smiles) to speed up encoding (shorter SMILES are faster to encode on average)
    smiles, formulas, masses = zip(*sorted(zip(smiles, formulas, masses), key=lambda x: len(x[0])))

    # Save unlabeled data
    os.makedirs(f"data/candidate_pools/", exist_ok=True)
    df_path = f"data/candidate_pools/{subset}.csv" # ['smiles', 'mass']
    df = pd.DataFrame({'smiles': smiles, 'mass': masses, 'formula': formulas})
    df.to_csv(df_path, index=False)