import numpy as np
from ot.sliced import sliced_wasserstein_distance

def downsample(mol, ms, split, n_samples):
    """
    Downsample each fold independently so that every fold contains at most
    ``n_samples`` points.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    keep_indices = []
    for fold in np.unique(split):
        fold_indices = np.where(split == fold)[0]
        if len(fold_indices) > n_samples//2:
            fold_indices = np.random.choice(fold_indices, size=n_samples//2, replace=False)
        keep_indices.append(fold_indices)

    keep = np.sort(np.concatenate(keep_indices)) if keep_indices else np.array([], dtype=int)
    return mol[keep], ms[keep], split[keep]
    
def compute_shift_with_sliced_ot(mol, ms, split, n_projections=100):
    """
    Compute the Sliced Wasserstein distance between train and test samples for a given split.
    """
    # sliced ot scales very well but if there too much samples I recommend to downsample the data to avoid memory issues
    mol, ms, split = downsample(mol, ms, split, n_samples=50000)
    
    # Normalize molecule and spectrum embeddings separately
    mol_normalized = mol / np.linalg.norm(mol, axis=1, keepdims=True)
    ms_normalized = ms / np.linalg.norm(ms, axis=1, keepdims=True)

    # Concatenate normalized embeddings + divide by sqrt(2) to ensure that the concatenated vector has unit norm
    pair_embedding = np.concatenate([mol_normalized, ms_normalized], axis=1) / np.sqrt(2)  

    # Split the data into train and test based on the provided split
    train_samples = pair_embedding[split == "train"]
    test_samples = pair_embedding[split == "test"]

    # Compute the Sliced Wasserstein distance + std over the projections
    value, log = sliced_wasserstein_distance(train_samples, test_samples, n_projections=n_projections, log=True)
    std = np.std(log['projected_emds'])  # Use ddof=1 for sample standard deviation
    
    return value.item(), std.item()


######################################## DEMO with synthetic data ########################################

from time import time

def generate_data(n_data=10000, d_mol=768, d_ms=1024, test_fraction=0.25):
    rng = np.random.default_rng(seed=42)
    n_test = int(n_data * test_fraction)
    n_train = n_data - n_test

    joint = np.concatenate([
        rng.normal(-5.0, 1.0, size=(n_train, d_mol + d_ms)),
        rng.normal(+5.0, 1.0, size=(n_test, d_mol + d_ms)),
    ])
    ms, mol = joint[:, :d_mol], joint[:, d_mol:]

    split_shift = np.array(["train"] * n_train + ["test"] * n_test)
    split_random = rng.permutation(split_shift)
    return mol, ms, split_shift, split_random


def demo_synthetic_data():
    n_data = 10000
    ms, mol, split_shift, split_random = generate_data(n_data=n_data)

    start_time = time()

    swd_split_shift, _ = compute_shift_with_sliced_ot(mol, ms, split_shift, n_projections=100)
    swd_split_random, _ = compute_shift_with_sliced_ot(mol, ms, split_random, n_projections=100)

    print(f'Ratio: {swd_split_shift / swd_split_random:.1f}')

    end_time = time()
    
    print(f"Time taken: {end_time - start_time} seconds")


######################################## DEMO with real data ########################################

import pandas as pd
import numpy as np

def demo_real_data():

    labelled_dataset_name='massspecgym'
    encoder_spectra="dreams"
    encoder_mol="chemberta_13M"

    spectra_emb=np.load(f'data/{labelled_dataset_name}/spectra_embeddings/{encoder_spectra}.npy')
    mol_emb=np.load(f'data/{labelled_dataset_name}/mol_embeddings/{encoder_mol}.npy')
    spectra_to_smiles = pd.read_csv(f'data/{labelled_dataset_name}/metadata.csv')['unique_smiles_idx'].values

    ms = spectra_emb
    mol = mol_emb[spectra_to_smiles]

    split_mces= pd.read_csv(f'data/{labelled_dataset_name}/splits/as_provided.csv')['fold'].values
    split_formula = pd.read_csv(f'data/{labelled_dataset_name}/splits/formula.csv')['fold'].values
    split_random = pd.read_csv(f'data/{labelled_dataset_name}/splits/random.csv')['fold'].values

    start_time = time()

    shift_mces, _ = compute_shift_with_sliced_ot(mol, ms, split_mces)
    shift_formula, _ = compute_shift_with_sliced_ot(mol, ms, split_formula)
    shift_random, _ = compute_shift_with_sliced_ot(mol, ms, split_random)

    print("Shift (as_provided / random):", f"{shift_mces / shift_random:.1f}")
    print("Shift (formula / random):", f"{shift_formula / shift_random:.1f}")

    end_time = time()
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    #demo_synthetic_data()
    demo_real_data()
