import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from rdkit import RDLogger
from tqdm import tqdm
from preprocessing.utils.rdkit_mp import normalize_smiles
import os
import pandas as pd
import json
import h5py
from .definitions import MAX_TOKENS_CHEMBERTA

RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings


class ChembertaEncoder:
    
    MODEL_CONFIGS = {
        "13M": {
            "model_name": "Derify/ChemBERTa_augmented_pubchem_13m",
            "model_class": AutoModel,
            "use_hidden_states": False,
        },
        "100M": {
            "model_name": "DeepChem/ChemBERTa-100M-MLM",
            "model_class": AutoModelForMaskedLM,
            "use_hidden_states": True,
        },
    }

    def __init__(self, version="13M", device=None):
        if version not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown version '{version}'. Choose from: {list(self.MODEL_CONFIGS.keys())}")

        self.version = version
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        config = self.MODEL_CONFIGS[version]
        self.use_hidden_states = config["use_hidden_states"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model = config["model_class"].from_pretrained(config["model_name"]).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, smiles_list, batch_size=32, return_torch=True, tqdm_disable=True, renormalize_smiles=False):
        if renormalize_smiles:
            smiles_list = [normalize_smiles(smi) for smi in smiles_list]

        embeddings = []
        max_length = MAX_TOKENS_CHEMBERTA -2 # for special tokens added by tokenizer
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Encoding", disable=tqdm_disable):
            batch_smiles = smiles_list[i:i + batch_size]
            tokens = self.tokenizer(batch_smiles, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(self.device)

            if self.use_hidden_states:
                # 100M: MLM model — extract CLS from last hidden state
                outputs = self.model(**tokens, output_hidden_states=True)
                cls_embeddings = outputs.hidden_states[-1][:, 0]
            else:
                # 13M: base model — extract CLS directly from last_hidden_state
                outputs = self.model(**tokens)
                cls_embeddings = outputs.last_hidden_state[:, 0]

            if not return_torch:
                cls_embeddings = cls_embeddings.cpu().numpy()

            embeddings.append(cls_embeddings)

        if return_torch:
            return torch.cat(embeddings)
        else:
            return np.concatenate(embeddings, axis=0)
        
        
def get_chemberta_embeddings(dataset_name, batch_size=32, n_workers=4, version="13M", overwrite=False):
    if os.path.exists(f"data/{dataset_name}/mol_embeddings/chemberta_{version}.npy") and not overwrite:
        print(f"Encoded embeddings already exist for {dataset_name} with version {version}. Skipping encoding.")
        return
    smiles = pd.read_csv(f"data/{dataset_name}/unique_smiles.csv")['smiles'].tolist()
    encoder = ChembertaEncoder(version=version)
    embeddings = encoder.encode(smiles, batch_size=batch_size, return_torch=False, tqdm_disable=False, renormalize_smiles=False)
    os.makedirs(f"data/{dataset_name}/mol_embeddings", exist_ok=True)
    np.save(f"data/{dataset_name}/mol_embeddings/chemberta_{version}.npy", embeddings)
    

def get_chemberta_embeddings_for_candidates(
    dataset_name,
    candidate_map_name,
    batch_size=32,
    n_workers=4,
    version="13M",
    overwrite=False,
    chunk_size=32,
):
    output_h5 = f"data/{dataset_name}/candidates/{candidate_map_name}/chemberta_{version}.h5"
    if os.path.exists(output_h5) and not overwrite:
        print(f"Encoded embeddings already exist for {dataset_name} candidates with version {version}. Skipping encoding.")
        return

    candidate_map = json.load(open(f"data/{dataset_name}/candidates/{candidate_map_name}/map.json", "r"))
    smiles = pd.read_csv(f"data/{dataset_name}/unique_smiles.csv")['smiles'].tolist()

    dataset_size = len(smiles)
    n_candidates = max(len(candidate_map[s]) for s in smiles) 
    dim = 768

    encoder = ChembertaEncoder(version=version)

    with h5py.File(output_h5, 'w') as f:
        dataset = f.create_dataset(
            "candidates_embeddings",
            shape=(dataset_size, n_candidates, dim),
            dtype='float32',
            chunks=(chunk_size, n_candidates, dim),
        )
        mask_dataset = f.create_dataset(
            "candidate_mask",
            shape=(dataset_size, n_candidates),
            dtype='bool',
            chunks=(chunk_size, n_candidates),
        )

        for chunk_start in tqdm(range(0, dataset_size, chunk_size), desc=f"Encoding candidates, dataset={dataset_name}, candidate_map={candidate_map_name}"):
            chunk_smiles = smiles[chunk_start:chunk_start + chunk_size]
            chunk_buffer = np.zeros((len(chunk_smiles), n_candidates, dim), dtype=np.float32)
            mask_buffer = np.zeros((len(chunk_smiles), n_candidates), dtype=bool)

            for j, s in enumerate(chunk_smiles):
                candidate_smiles = candidate_map[s]
                assert candidate_smiles[0] == s, (
                    f"First candidate should be the original molecule itself. "
                    f"Found {candidate_smiles[0]} instead of {s}."
                )
                assert s not in candidate_smiles[1:], (
                    f"Original molecule {s} should not appear in candidates other than the first one."
                )

                embeddings = encoder.encode(
                    candidate_smiles,
                    batch_size=batch_size,
                    return_torch=False,
                    tqdm_disable=True,
                    renormalize_smiles=False,
                )
                chunk_buffer[j, :len(candidate_smiles)] = embeddings
                mask_buffer[j, :len(candidate_smiles)] = True

            dataset[chunk_start:chunk_start + len(chunk_smiles)] = chunk_buffer
            mask_dataset[chunk_start:chunk_start + len(chunk_smiles)] = mask_buffer # True for valid candidates, False for padding