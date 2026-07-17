import numpy as np
from ot.sliced import sliced_wasserstein_sphere

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



def compute_shift(mol, ms, split, normalize=True, n_projections_slice=100, n_samples=10000):
    '''
    Compute the sliced Wasserstein distance between train and test distributions.
    '''
    # Sample the data if it is too large
    if mol.shape[0] > n_samples:
        idx = np.random.choice(mol.shape[0], n_samples, replace=False)
        mol = mol[idx]
        ms = ms[idx]
        split = split[idx]
    
    # Unit sphere normalization + concatenate the two embeddings
    mol = mol / np.linalg.norm(mol, axis=1, keepdims=True)
    ms = ms / np.linalg.norm(ms, axis=1, keepdims=True)
    x = np.concatenate([mol, ms], axis=1)/np.sqrt(2) # division by sqrt(2) to keep the norm of the concatenated vector equal to 1

    # Compute the sliced Wasserstein distance
    shift = sliced_wasserstein_sphere(x[split == 'train'], x[split == 'test'], n_projections=n_projections_slice)
    
    # Normalize the shift by the shift between two random splits of the data
    if normalize:
        random_split = split.copy()
        np.random.shuffle(random_split)
        random_shift = sliced_wasserstein_sphere(x[random_split == 'train'], x[random_split == 'test'], n_projections=n_projections_slice)
        shift = shift / random_shift
        
    return shift

ms, mol, split1, split2 = generate_data(n_data=1000, n_features=2)

print(f"Train size: {np.sum(split1 == 'train')}, Test size: {np.sum(split1 == 'test')}")
print(ms.shape, mol.shape, split1.shape, split2.shape)

shift1 = compute_shift(mol, ms, split1)
shift2 = compute_shift(mol, ms, split2)

print(f"Shift between train and test distributions (split1): {shift1}")
print(f"Shift between train and test distributions (split2): {shift2}")