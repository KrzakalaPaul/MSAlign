#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --gpus=0

python ot_shift_rebutal.py  --encoder_spectra bins --encoder_mol fingerprints --bin_width 0.5 --n_projections 50