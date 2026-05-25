from models.FLARE.datamodule import PairDataset

dataset = PairDataset(labelled_dataset_name="massspecgym", split_method="formula", fold="train")
print(len(dataset))