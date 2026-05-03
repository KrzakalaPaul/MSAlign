from preprocessing import download_massspecgym, download_spectraverse, process_smiles, split

if __name__ == "__main__":
    
    dataset_name = "massspecgym"
    split_method = "formula"
    overwrite = False
    
    
    if dataset_name == "massspecgym":
        download_massspecgym(overwrite=overwrite)
    else:
        download_spectraverse(overwrite=overwrite)
        
    process_smiles(dataset_name=dataset_name, overwrite=overwrite)
    split(dataset_name=dataset_name, split_method=split_method, overwrite=overwrite)