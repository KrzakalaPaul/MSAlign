The ultimate goal of the preprocessing pipeline is to produce the following set of files. Not all files are required for each pipeline. Exemple: MSAlign does not require spectra_formulas.json

- data
    - dataset_name_1
        - splits
            - split_name_1.npy
            - split_name_2.npy
            ...
        - candidates
            - candidates_set_name_1
                - map.json
                - encoder_name_1.npy
                - encoder_name_2.npy
                ...
        - spectra_embeddings
                - encoder_name_1.npy
                - encoder_name_2.npy
                ...
        - molecular_embeddings
                - encoder_name_1.npy
                - encoder_name_2.npy
                ...
        - spectra.npy                   # (N x N_HIGHEST_PEAKS x 2) array of (mz, intensity) pairs
        - spectra_formulas.json         # N lists of lists (peak formulas annotations)
        - metadata.csv                  # N rows of spectra metadata (smiles, precursor m/z, collision energy, etc.)
        - unique_smiles.csv             # n rows, columns =  'smiles', 'spectra_idx_start', 'spectra_idx_end' (mapping from unique SMILES to spectra indices) and 'mass'
    - dataset_name_2
        ...
    - candidate_source_1
        - smiles.csv                   
    - candidate_source_2
        - smiles.csv                   
        
        
The files are created in that order:
# CPU preprocessing & Download
1. download_dataset_name() is dataset dependant and produces

    - data
        - dataset_name_1
            - spectra.npy
            - metadata.csv

2. preprocess_smiles(dataset_name) canonicalizes all smiles strings, produces unique_smiles.csv and reorders spectra and metadata to match the order of unique_smiles.csv. 

3. split_dataset(dataset_name, split_method) produces the splits in data/dataset_name/splits
4. create_candidates(dataset_name, candidate_source, n_candidates) produces 
    - data 
        - dataset_name_1
            - candidates
                - {candidate_source}_{n_candidates} # This is the name of the candidate set
                    - map.json
        - candidate_source
            - smiles.csv
5. peak_subformula_annotation(dataset_name) produces spectra_formulas.json

# GPU preprocessing (embeddings precomputations)
6. encode_spectra(dataset_name, encoder_name) produces the spectra embeddings in data/dataset_name/spectra_embeddings
7. encode_molecules(dataset_name, encoder_name) produces the molecule embeddings in data/dataset_name/molecular_embeddings
8. encode_candidates(dataset_name, candidate_set_name, encoder_name) produces the candidate embeddings in data/dataset_name/candidates/{candidate_set_name}