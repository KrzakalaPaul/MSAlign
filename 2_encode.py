from preprocessing import *

if __name__ == "__main__":
    
    dataset_name = "massspecgym"
    split_method = "formula"
    n_candidates = 16
    sources = ['1M', '4M']
    candidate_selection_method = "mass"
    overwrite = False
    
    get_dreams_embeddings(dataset_name=dataset_name, batch_size=32, n_workers=4, overwrite=overwrite)
