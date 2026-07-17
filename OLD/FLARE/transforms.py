import numpy as np
import torch
from rdkit.Chem import AllChem as Chem
import dgllife.utils as chemutils
from .params import CHEM_ELEMS_MS, CHEM_ELEMS_MS_ABUNDANCE, CHEM_ELEMS_MOL
import re

def parse_formula(formula: str, elements: list[str]) -> dict[str, int]:
    """Parse a chemical formula string into element counts."""
    # Match element symbol (capital + optional lowercase) followed by optional count
    pattern = r'([A-Z][a-z]?)(\d*)'
    counts = {elem: 0 for elem in elements}
    
    for match in re.finditer(pattern, formula):
        symbol, count = match.group(1), match.group(2)
        if symbol in counts:
            counts[symbol] += int(count) if count else 1
    
    return counts

class MSPreprocessor():
    def __init__(
        self,
        add_mz = True,
        add_intensity = True
    ):
        self.add_mz = add_mz
        self.add_intensity = add_intensity

        
    def __call__(self, mz_list, intensities_list, formulas_list):

        n = len(mz_list)
        token_dim = 2 + len(CHEM_ELEMS_MS)
        tokens = np.zeros((n, token_dim), dtype=np.float32)

        for i, (mz, intensity, formula) in enumerate(zip(mz_list, intensities_list, formulas_list)):
            tokens[i, 0] = mz
            tokens[i, 1] = intensity

            elem_counts = parse_formula(formula, CHEM_ELEMS_MS)
            for j, elem in enumerate(CHEM_ELEMS_MS):
                tokens[i, 2 + j] = elem_counts[elem]
                
        tokens[:, 0] = tokens[:, 0] / 1000.0 if self.add_mz else 0.0
        tokens[:, 1] = tokens[:, 1] if self.add_intensity else 0.0
        tokens[:, 2:] /= CHEM_ELEMS_MS_ABUNDANCE
        return tokens

  
    
import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
# remove rdkit warnings (e.g. "Explicit valence for atom # 0 C, is greater than permitted")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def morgan_fp(smiles, fp_size=2048, radius=2, to_np=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return np.zeros(fp_size, dtype=np.int32) if to_np else None

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
        fp = generator.GetFingerprint(mol)
        
        if to_np:
            fp_np = np.zeros(fp_size, dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fp = fp_np
        
        return fp

    except Exception as e:
        # Log which SMILES caused the issue, return a zero vector as fallback
        print(f"Warning: failed to encode SMILES '{smiles}': {e}", flush=True)
        return np.zeros(fp_size, dtype=np.int32) if to_np else None

class MOLPreprocessor():
    def __init__ (self, atom_feature: str = "full", bond_feature: str = "full", element_list: list = CHEM_ELEMS_MOL):
        self.atom_feature = atom_feature
        self.bond_feature = bond_feature
        self.node_featurizer = self._get_atom_featurizer(element_list=element_list) 
        self.edge_featurizer = self._get_bond_featurizer()
    
    def __call__(self, smiles:str):
        mol = Chem.MolFromSmiles(smiles)
        graph = chemutils.mol_to_bigraph(mol, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer, add_self_loop = True,
                             num_virtual_nodes = 0, canonical_atom_order=False)

        return graph

    def _get_atom_featurizer(self, element_list) -> dict:
        feature_mode = self.atom_feature
        atom_mass_fun = chemutils.ConcatFeaturizer(
            [chemutils.atom_mass]
        )

        def atom_bond_type_one_hot(atom):
            bs = atom.GetBonds()
            if not bs:
                return [False] * 4
            bt = np.array([chemutils.bond_type_one_hot(b) for b in bs])
            return [any(bt[:, i]) for i in range(bt.shape[1])]

        def atom_type_one_hot(atom):
            return chemutils.atom_type_one_hot(
                atom, allowable_set = element_list, encode_unknown = True
            )
        
        if feature_mode == 'light':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                atom_type_one_hot
            ])
        elif feature_mode == 'full':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                atom_type_one_hot, 
                atom_bond_type_one_hot,
                chemutils.atom_degree_one_hot, 
                chemutils.atom_total_degree_one_hot,
                chemutils.atom_explicit_valence_one_hot,
                chemutils.atom_implicit_valence_one_hot,
                chemutils.atom_hybridization_one_hot,
                chemutils.atom_total_num_H_one_hot,
                chemutils.atom_formal_charge_one_hot,
                chemutils.atom_num_radical_electrons_one_hot,
                chemutils.atom_is_aromatic_one_hot,
                chemutils.atom_is_in_ring_one_hot,
                chemutils.atom_chiral_tag_one_hot
            ])
        elif feature_mode == 'medium':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                atom_type_one_hot, 
                atom_bond_type_one_hot,
                chemutils.atom_total_degree_one_hot,
                chemutils.atom_total_num_H_one_hot,
                chemutils.atom_is_aromatic_one_hot,
                chemutils.atom_is_in_ring_one_hot,
            ])
        return chemutils.BaseAtomFeaturizer(
        {"h": atom_featurizer_funs, 
        "m": atom_mass_fun}
    )

    def _get_bond_featurizer(self, self_loop=True) -> dict:
        feature_mode = self.bond_feature
        if feature_mode == 'light':
            return chemutils.BaseBondFeaturizer(
                featurizer_funcs = {'e': chemutils.ConcatFeaturizer([
                    chemutils.bond_type_one_hot
                ])}, self_loop = self_loop
            )
        elif feature_mode == 'full':
            return chemutils.CanonicalBondFeaturizer(
                bond_data_field='e', self_loop = self_loop
            )