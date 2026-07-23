import argparse

import numpy as np
import pandas as pd
from ot.lp.solver_1d import wasserstein_1d

from tqdm import tqdm

import json

import os

BIN_ENCODERS = {"bin", "bins"}
FINGERPRINT_ENCODERS = {"fingerprint", "fingerprints", "morgan", "morgan_fingerprint"}


def bin_spectrum(ms, max_mz=1005, bin_width=0.1):
    mzs = ms[:, 0]
    intensities = ms[:, 1]
    num_bins = int(np.ceil(max_mz / bin_width))
    bin_indices = np.floor(mzs / bin_width).astype(int)

    keep = mzs <= max_mz
    valid_indices = np.clip(bin_indices[keep], 0, num_bins - 1)
    valid_intensities = intensities[keep]

    binned_intensities = np.zeros(num_bins, dtype=np.float32)
    np.add.at(binned_intensities, valid_indices, valid_intensities)

    max_intensity = np.max(binned_intensities)
    if max_intensity > 0:
        binned_intensities = binned_intensities / max_intensity * 999
    return np.rint(np.log10(binned_intensities + 1) / 3 * 255).astype(np.uint8)


def morgan_fingerprint(smiles, fp_size=4096, radius=2):
    from rdkit import RDLogger
    from rdkit.Chem import AllChem as Chem
    from rdkit.Chem import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(fp_size, dtype=np.uint8)

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
        fp = generator.GetFingerprint(mol)

        fp_np = np.zeros(fp_size, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, fp_np)
        return fp_np
    except Exception:
        return np.zeros(fp_size, dtype=np.uint8)


def downsample(mol, ms, split, n_samples, seed=None):
    """
    Downsample train and test independently so the final sample contains at
    most ``n_samples`` points.
    """
    if n_samples is None:
        return mol, ms, split
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    rng = np.random.default_rng(seed)
    keep_indices = []
    for fold in ("train", "test"):
        fold_indices = np.where(split == fold)[0]
        fold_sample_size = n_samples // 2
        if len(fold_indices) > fold_sample_size:
            fold_indices = rng.choice(fold_indices, size=fold_sample_size, replace=False)
        if len(fold_indices) > 0:
            keep_indices.append(fold_indices)

    keep = np.sort(np.concatenate(keep_indices)) if keep_indices else np.array([], dtype=int)
    return mol[keep], ms[keep], split[keep]


def compute_row_norms(x, chunk_size=2048):
    norms = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        chunk = x[start:end].astype(np.float32, copy=False)
        norms[start:end] = np.sqrt(np.sum(chunk * chunk, axis=1, dtype=np.float32))
    return norms


def random_unit_projections(dim, n_projections, rng, dtype=np.float32):
    projections = rng.standard_normal((dim, n_projections)).astype(dtype, copy=False)
    norms = np.sqrt(np.sum(projections.astype(np.float32) ** 2, axis=0, keepdims=True))
    projections = np.divide(
        projections,
        norms,
        out=np.zeros_like(projections, dtype=dtype),
        where=norms > 0,
    )
    return projections


def project_normalized(x, projections, row_norms, chunk_size=2048, dtype=np.float32):
    projected = np.empty((x.shape[0], projections.shape[1]), dtype=dtype)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        chunk = x[start:end].astype(dtype, copy=False)
        projected_chunk = chunk @ projections
        chunk_norms = row_norms[start:end, None]
        projected[start:end] = np.divide(
            projected_chunk,
            chunk_norms,
            out=np.zeros_like(projected_chunk, dtype=dtype),
            where=chunk_norms > 0,
        )
    return projected


def compute_shift_with_streamed_sliced_ot(
    mol,
    ms,
    split,
    n_projections=100,
    seed=None,
    ot_dtype=np.float32,
    projection_block_size=8,
    chunk_size=2048,
):
    """
    Compute sliced OT without materializing normalized or concatenated pair embeddings.
    """
    train_mask = split == "train"
    test_mask = split == "test"
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("split must contain at least one train and one test sample")

    rng = np.random.default_rng(seed)
    mol_norms = compute_row_norms(mol, chunk_size=chunk_size)
    ms_norms = compute_row_norms(ms, chunk_size=chunk_size)
    total_dim = mol.shape[1] + ms.shape[1]
    projected_emds = []

    for start in range(0, n_projections, projection_block_size):
        block_size = min(projection_block_size, n_projections - start)
        projections = random_unit_projections(total_dim, block_size, rng, dtype=ot_dtype)
        mol_projections = projections[: mol.shape[1]]
        ms_projections = projections[mol.shape[1] :]

        projected = project_normalized(
            mol,
            mol_projections,
            mol_norms,
            chunk_size=chunk_size,
            dtype=ot_dtype,
        )
        projected += project_normalized(
            ms,
            ms_projections,
            ms_norms,
            chunk_size=chunk_size,
            dtype=ot_dtype,
        )
        projected *= ot_dtype(1 / np.sqrt(2))

        block_emds = wasserstein_1d(
            projected[train_mask],
            projected[test_mask],
            p=2,
        )
        projected_emds.append(np.asarray(block_emds, dtype=np.float32))

    projected_emds = np.concatenate(projected_emds)
    value = np.sqrt(np.mean(projected_emds))
    return value.item()


def compute_shift_with_sliced_ot(
    mol,
    ms,
    split,
    n_projections=100,
    n_samples=50000,
    seed=None,
    ot_dtype=np.float32,
    projection_block_size=8,
    chunk_size=2048,
):
    """
    Compute the Sliced Wasserstein distance between train and test samples for a given split.
    """
    # sliced ot scales very well but if there too much samples I recommend to downsample the data to avoid memory issues
    mol, ms, split = downsample(mol, ms, split, n_samples=n_samples, seed=seed)
    
    return compute_shift_with_streamed_sliced_ot(
        mol,
        ms,
        split,
        n_projections=n_projections,
        seed=seed,
        ot_dtype=ot_dtype,
        projection_block_size=projection_block_size,
        chunk_size=chunk_size,
    )


def get_downsample_indices(split, n_samples, seed=None):
    indices = np.arange(len(split))
    if n_samples is None:
        return indices[np.isin(split, ("train", "test"))]
    sampled_indices, _, _ = downsample(indices, indices, split, n_samples=n_samples, seed=seed)
    return sampled_indices.astype(int)


def load_spectra_embeddings(labelled_dataset_name, encoder_spectra, indices=None, bin_width=0.1, max_mz=1005):
    encoder_key = encoder_spectra.lower()
    if encoder_key in BIN_ENCODERS:
        spectra = np.load(f"data/{labelled_dataset_name}/spectra.npy", mmap_mode="r")
        spectra = spectra if indices is None else spectra[indices]
        spectra_embeddings = np.empty((len(spectra), int(np.ceil(max_mz / bin_width))), dtype=np.uint8)
        for i, spectrum in enumerate(spectra):
            spectra_embeddings[i] = bin_spectrum(spectrum, max_mz=max_mz, bin_width=bin_width)
        return spectra_embeddings

    spectra_emb = np.load(
        f"data/{labelled_dataset_name}/spectra_embeddings/{encoder_spectra}.npy",
        mmap_mode="r",
    )
    return np.asarray(spectra_emb if indices is None else spectra_emb[indices], dtype=np.float32)


def load_mol_embeddings(
    labelled_dataset_name,
    encoder_mol,
    spectra_to_smiles,
    indices=None,
    fingerprint_size=4096,
    fingerprint_radius=2,
):
    smiles_indices = spectra_to_smiles if indices is None else spectra_to_smiles[indices]
    encoder_key = encoder_mol.lower()
    if encoder_key in FINGERPRINT_ENCODERS:
        unique_smiles = pd.read_csv(f"data/{labelled_dataset_name}/unique_smiles.csv")["smiles"].values
        used_smiles_indices = np.unique(smiles_indices)
        fingerprint_by_idx = {
            smiles_idx: morgan_fingerprint(
                unique_smiles[smiles_idx],
                fp_size=fingerprint_size,
                radius=fingerprint_radius,
            )
            for smiles_idx in used_smiles_indices
        }
        mol_embeddings = np.empty((len(smiles_indices), fingerprint_size), dtype=np.uint8)
        for i, smiles_idx in enumerate(smiles_indices):
            mol_embeddings[i] = fingerprint_by_idx[smiles_idx]
        return mol_embeddings

    mol_emb = np.load(
        f"data/{labelled_dataset_name}/mol_embeddings/{encoder_mol}.npy",
        mmap_mode="r",
    )
    return np.asarray(mol_emb[smiles_indices], dtype=np.float32)


def compute_real_data_shift(
    labelled_dataset_name,
    encoder_spectra,
    encoder_mol,
    split,
    spectra_to_smiles,
    n_projections=100,
    n_samples=50000,
    seed=None,
    fingerprint_size=4096,
    fingerprint_radius=2,
    bin_width=0.1,
    max_mz=1005,
    ot_dtype=np.float32,
    projection_block_size=8,
    chunk_size=2048,
):
    indices = get_downsample_indices(split, n_samples=n_samples, seed=seed)
    sampled_split = split[indices]
    ms = load_spectra_embeddings(
        labelled_dataset_name,
        encoder_spectra,
        indices=indices,
        bin_width=bin_width,
        max_mz=max_mz,
    )
    mol = load_mol_embeddings(
        labelled_dataset_name,
        encoder_mol,
        spectra_to_smiles,
        indices=indices,
        fingerprint_size=fingerprint_size,
        fingerprint_radius=fingerprint_radius,
    )
    return compute_shift_with_sliced_ot(
        mol,
        ms,
        sampled_split,
        n_projections=n_projections,
        n_samples=None,
        seed=seed,
        ot_dtype=ot_dtype,
        projection_block_size=projection_block_size,
        chunk_size=chunk_size,
    )


def demo_real_data(
    labelled_dataset_name="massspecgym",
    encoder_spectra="dreams",
    encoder_mol="chemberta_13M",
    n_projections=100,
    n_samples=50000,
    seed=0,
    fingerprint_size=4096,
    fingerprint_radius=2,
    bin_width=0.1,
    max_mz=1005,
    ot_dtype=np.float32,
    projection_block_size=8,
    chunk_size=2048,
    n_seeds=5,
):
    
    split_methods = ["as_provided", "formula", "murcko", "murcko_hist"]
    
    spectra_to_smiles = pd.read_csv(f"data/{labelled_dataset_name}/metadata.csv")["unique_smiles_idx"].values
    
    splits = [pd.read_csv(f"data/{labelled_dataset_name}/splits/{method}.csv")["fold"].values for method in split_methods]
    split_random = pd.read_csv(f"data/{labelled_dataset_name}/splits/random.csv")["fold"].values

    common_kwargs = dict(
        labelled_dataset_name=labelled_dataset_name,
        encoder_spectra=encoder_spectra,
        encoder_mol=encoder_mol,
        spectra_to_smiles=spectra_to_smiles,
        n_projections=n_projections,
        n_samples=n_samples,
        seed=seed,
        fingerprint_size=fingerprint_size,
        fingerprint_radius=fingerprint_radius,
        bin_width=bin_width,
        max_mz=max_mz,
        ot_dtype=ot_dtype,
        projection_block_size=projection_block_size,
        chunk_size=chunk_size,
    )

    shift_list = {method: [] for method in split_methods}
    shift_random_list = []
    for _ in tqdm(range(n_seeds), desc="Computing shifts for multiple seeds"):
        common_kwargs["seed"] = common_kwargs["seed"] + 1
        for method, split in zip(split_methods, splits):
            shift_list[method].append(compute_real_data_shift(split=split, **common_kwargs))
        shift_random_list.append(compute_real_data_shift(split=split_random, **common_kwargs))
        
    normalization_factor = np.array(shift_random_list).mean()

    shift_list['random'] = shift_random_list
    shift_list = {method: np.array(shifts) / normalization_factor for method, shifts in shift_list.items()}

    # save as json
    results = {method: {"mean": float(np.mean(shifts)), "std": float(np.std(shifts))} for method, shifts in shift_list.items()}

    os.makedirs('data/results', exist_ok=True)
    with open(f"data/results/shift_results_{labelled_dataset_name}_{encoder_spectra}_{encoder_mol}.json", "w") as f:
        json.dump(results, f, indent=4) 


def parse_args():
    parser = argparse.ArgumentParser(description="Compute train/test shift with sliced OT.")
    parser.add_argument("--labelled_dataset_name", type=str, default="massspecgym")
    parser.add_argument("--encoder_spectra", type=str, default="dreams")
    parser.add_argument("--encoder_mol", type=str, default="chemberta_13M")
    parser.add_argument("--n_projections", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fingerprint_size", type=int, default=4096)
    parser.add_argument("--fingerprint_radius", type=int, default=2)
    parser.add_argument("--bin_width", type=float, default=0.1)
    parser.add_argument("--max_mz", type=float, default=1005)
    parser.add_argument("--ot_dtype", type=str, choices=("float32", "float16"), default="float32")
    parser.add_argument("--projection_block_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of seeds to run for averaging the shift computation.")
    args = parser.parse_args()
    if args.projection_block_size <= 0:
        raise ValueError("--projection_block_size must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    args.ot_dtype = np.dtype(args.ot_dtype).type
    return args

if __name__ == "__main__":
    args = parse_args()
    demo_real_data(**vars(args))
