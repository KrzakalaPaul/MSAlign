# MSAlign
MSAlign: Lightweight Alignment of Unimodal Foundation Models for Metabolite Identification

```bash
git clone https://github.com/pluskal-lab/DreaMS.git
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt# MSAlign: Aligning Molecule and Mass Spectra Foundation Models for Metabolite Identification
```

> ⚠️ **Work In Progress — Reviewers Only. Please do not share or redistribute.**

---

## Overview

MSAlign is a contrastive alignment framework that bridges mass spectra and molecular embeddings for metabolite identification. It learns a shared embedding space between MS/MS spectra (encoded via DreaMS) and molecular structures (encoded via ChemBERTa or other molecular encoders), enabling retrieval of candidate molecules from a large pool given a query spectrum.

---

## Installation

```bash
git clone https://github.com/pluskal-lab/DreaMS.git
cd DreaMS

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
  --sources 1M 4M
```

### 2. Precompute Embeddings

Encodes all spectra and molecules into their respective embedding spaces. This step only needs to be run once dataset and candidate map.
```bash
python 2_encode.py \
  --dataset_name massspecgym \
  --candidate_map_name 1M_4M_64candidates_mass \
  --version 13M
```

### 3. Train MSAlign

```bash
python 3_train.py \
  --labelled_dataset_name massspecgym \
  --candidate_map_name 1M_4M_64candidates_mass \
  --split_method formula \
  --encoder_mol chemberta_13M \
  --encoder_spectra dreams \
  --k_candidates 64 \
  --d_shared 256 \
  --max_epochs 100
```

---

## To-Do

- [ ] Implement baselines
- [ ] Share Zenodo link for candidate maps and pretrained model weights
- [ ] Add default configs
- [ ] Add demo notebook

---

