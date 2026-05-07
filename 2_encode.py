from preprocessing import *

if __name__ == "__main__":
    
    dataset_name = "massspecgym"
    candidate_map_name = "1M_4M_16candidates_formula"
    overwrite = True
    
    #get_chemberta_embeddings_for_candidates(dataset_name=dataset_name,candidate_map_name=candidate_map_name,overwrite=overwrite, batch_size=32,chunk_size=32, version="13M")
    #get_chemberta_embeddings(dataset_name=dataset_name, batch_size=32, n_workers=4, version="13M", overwrite=overwrite)
    get_dreams_embeddings(dataset_name=dataset_name, batch_size=32, n_workers=4, overwrite=overwrite)
