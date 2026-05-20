# MSAlign
MSAlign: Lightweight Alignment of Unimodal Foundation Models for Metabolite Identification

> ⚠️ **Work In Progress — Reviewers Only. Please do not share or redistribute.**

---

## Overview

MSAlign is a contrastive alignment framework that bridges mass spectra and molecular embeddings for metabolite identification. It learns a shared embedding space between MS/MS spectra (encoded via DreaMS) and molecular structures (encoded via ChemBERTa or other molecular encoders), enabling retrieval of candidate molecules from a large pool given a query spectrum.

---

## Installation

```bash
git clone https://github.com/pluskal-lab/DreaMS.git
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Usage

You can test the code with a small subset of candidate using this pipeline:

### 1. Prepare Data

Downloads raw data and builds candidate maps. See [`preprocessing/README.md`](preprocessing/README.md) for full details.

```bash
python 1_prepare.py \
  --dataset_name massspecgym \
  --split_method formula \
  --n_candidates 32 \
  --candidate_selection_method mass \
```

### 2. Precompute Embeddings

Encodes all spectra and molecules into their respective embedding spaces. This step only needs to be run once dataset and candidate map.
```bash
python 2_encode.py \
  --dataset_name massspecgym \
  --candidate_map_name 32_candidates_by_mass \
  --version 13M
```

### 3. Train MSAlign

```bash
python 3_train.py \
  --labelled_dataset_name massspecgym \
  --candidate_map_name 32_candidates_by_mass \
  --split_method formula \
  --encoder_mol chemberta_13M \
  --encoder_spectra dreams \
  --k_candidates 64 \
  --d_shared 256 \
  --max_epochs 100
```

---

## To-Do

- [ ] Share preprocessed candidate pools and use that by default (still provide the code to produce them separately)
- [ ] Use case 1: reproductibility = provide the code the train the models and all baselines. (main calls train/finetune/eval)
- [ ] Use case 2: inference only = provide notebook for this. Only need to work for MSAlign but takes raw spectra/molecules as inputs.


## High level idea of the demo.ipynb

Section 1: Basic demo with one sample

candidates = ['CCH', 'CCCOH' ... ]
spectra = np.array([665, 1.1],
		               [334, 0.8])

model = load_model()
prediction = model(candidates, spectra)

Section 2:

dataset = load_massspecgym(fold='test')
prediction = model(dataset)

print(R@1 = ...)

---

