from preprocessing import *

if __name__ == "__main__":
    
    dataset_name = "massspecgym"
    split_method = "formula"
    n_candidates = 16
    sources = ['1M', '4M']
    candidate_selection_method = "mass"
    overwrite = False

    if dataset_name == "massspecgym":
        download_massspecgym(overwrite=overwrite)
    else:
        download_spectraverse(overwrite=overwrite)
        
    process_smiles(dataset_name=dataset_name, overwrite=overwrite, n_threads=16, chunk_size=4096)

    split(dataset_name=dataset_name, split_method=split_method, overwrite=overwrite)
    
    prepare_candidates(dataset_name=dataset_name,
                       n_candidates=n_candidates,
                       kind=candidate_selection_method,
                       sources = sources,
                       overwrite=True,
                       seed=42)

    annotate_peaks(dataset_name=dataset_name, n_threads=8, chunksize=256, overwrite=overwrite)