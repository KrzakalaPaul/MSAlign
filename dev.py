import pandas as pd
import numpy as np

labelled_dataset_name='massspecgym'
encoder_spectra="dreams"
encoder_mol="chemberta_13M"

spectra_emb=np.load(f'data/{labelled_dataset_name}/spectra_embeddings/{encoder_spectra}.npy')
mol_emb=np.load(f'data/{labelled_dataset_name}/mol_embeddings/{encoder_mol}.npy')
spectra_to_smiles = pd.read_csv(f'data/{labelled_dataset_name}/metadata.csv')['unique_smiles_idx'].values

ms = spectra_emb
mol = mol_emb[spectra_to_smiles]

split_as_provided = pd.read_csv(f'data/{labelled_dataset_name}/splits/as_provided.csv')['fold'].values
split_formula = pd.read_csv(f'data/{labelled_dataset_name}/splits/formula.csv')['fold'].values
split_random = pd.read_csv(f'data/{labelled_dataset_name}/splits/random.csv')['fold'].values

from ot_shift import compute_shift_sswd
from time import time

start_time = time()
n_samples = None
n_projections_slice = 50

shift_as_provided = compute_shift_sswd(mol, ms, split_as_provided, normalize=True, n_projections_slice=n_projections_slice, n_samples=n_samples)
print(f"Shift between train and test distributions (as_provided): {shift_as_provided}")

shift_formula = compute_shift_sswd(mol, ms, split_formula, normalize=True, n_projections_slice=n_projections_slice, n_samples=n_samples)
print(f"Shift between train and test distributions (formula): {shift_formula}")

shift_random = compute_shift_sswd(mol, ms, split_random, normalize=True, n_projections_slice=n_projections_slice, n_samples=n_samples)
print(f"Shift between train and test distributions (random): {shift_random}")

end_time = time()
print(f"Time taken: {end_time - start_time} seconds")