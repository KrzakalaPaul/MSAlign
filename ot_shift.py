import numpy as np
from ot.sliced import sliced_wasserstein_distance, sliced_wasserstein_sphere, linear_sliced_wasserstein_sphere

def compute_shift_sswd(mol, ms, split, normalize=True, n_projections_slice=100, n_samples=None):
    '''
    Compute the sliced Wasserstein distance between train and test distributions.
    arguments:
    mol: np.ndarray, shape (n_samples, n_features)
        Molecular embeddings.
    ms: np.ndarray, shape (n_samples, n_features)
        Mass spectrometry embeddings.
    split: np.ndarray, shape (n_samples,)
        np array with 'train' and 'test' labels for each sample.
    normalize: bool, default True
        Whether to normalize the shift by the shift between two random splits of the data.
    n_projections_slice: int, default 100
        Number of projections to use for the sliced Wasserstein distance. Decreasing this number will speed up the computation but may decrease the accuracy of the estimate.
    n_samples: int, default None
        Number of samples to use for the computation. If None, all samples will be used. Set this to a smaller number if the dataset is too large to speed up the computation.
    '''
    # Sample the data if it is too large
    if n_samples is not None:
        if n_samples < mol.shape[0]:
            idx = np.random.choice(mol.shape[0], n_samples, replace=False)
            mol = mol[idx]
            ms = ms[idx]
            split = split[idx]

    # Unit sphere normalization + concatenate the two embeddings
    mol = mol / np.linalg.norm(mol, axis=1, keepdims=True)
    ms = ms / np.linalg.norm(ms, axis=1, keepdims=True)
    x = np.concatenate([mol, ms], axis=1)/np.sqrt(2) # division by sqrt(2) to keep the norm of the concatenated vector equal to 1

    # Compute the sliced Wasserstein distance
    shift = sliced_wasserstein_distance(x[split == 'train'], x[split == 'test'], n_projections=n_projections_slice)
    
    # Normalize the shift by the shift between two random splits of the data
    if normalize:
        random_split = split.copy()
        np.random.shuffle(random_split)
        random_shift = sliced_wasserstein_distance(x[random_split == 'train'], x[random_split == 'test'], n_projections=n_projections_slice)
        shift = shift / random_shift
        
    return shift

######################################## DEMO with synthetic data ########################################

from time import time

def generate_data(n_data=1000, n_features=2):
    '''
    Generate bimodal data.
    split1 = domain shift (train and test from different distributions)
    split2 = no domain shift (train and test from the same distribution)
    '''
    
    test_size = n_data // 4
    train_size = n_data - test_size
    
    x = np.concatenate([
        np.random.normal(loc=-10, scale=1, size=(train_size, 2 * n_features)),
        np.random.normal(loc=10, scale=1, size=(test_size, 2 * n_features))
    ], axis=0)
    
    ms = x[:, :n_features]
    mol = x[:, n_features:]
    split1 = ['train'] * train_size + ['test'] * test_size
    split1 = np.array(split1)
    split2 = split1.copy()
    np.random.shuffle(split2)
    
    return ms, mol, split1, split2

if __name__ == "__main__":
    
    n_data = 10000
    n_features = 2
    n_projections_slice = 100
    n_samples = None

    ms, mol, split1, split2 = generate_data(n_data=n_data, n_features=n_features)

    start_time = time()

    shift1 = compute_shift_sswd(mol, ms, split1, normalize=False, n_projections_slice=n_projections_slice, n_samples=n_samples)
    shift2 = compute_shift_sswd(mol, ms, split2, normalize=False, n_projections_slice=n_projections_slice, n_samples=n_samples)

    print(f"Shift between train and test distributions (split1): {shift1}")
    print(f"Shift between train and test distributions (split2): {shift2}")

    end_time = time()

    print(f"Time taken: {end_time - start_time} seconds")