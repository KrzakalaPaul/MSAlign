import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from rdkit import RDLogger
from tqdm import tqdm
import torch
from .rdkit import normalize_smiles
RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings


class ChembertaEncoder:
    def __init__(self, model_name="Derify/ChemBERTa_augmented_pubchem_13m", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    
    @torch.no_grad()
    def encode(self, smiles_list, batch_size=32, return_torch=True, tqdm_disable=True, renormalize_smiles=False):
        if renormalize_smiles:
            smiles_list = [normalize_smiles(smi) for smi in smiles_list]
        embeddings = []
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Encoding", disable=tqdm_disable):
            batch_smiles = smiles_list[i:i+batch_size]
            tokens = self.tokenizer(batch_smiles, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**tokens)
                cls_embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            if not return_torch:
                cls_embeddings = cls_embeddings.cpu().numpy()
            embeddings.append(cls_embeddings)
        return torch.concat(embeddings) if return_torch else np.stack(embeddings)
