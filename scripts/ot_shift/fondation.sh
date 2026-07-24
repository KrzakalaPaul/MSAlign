#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --gpus=0

python ot_shift_rebutal.py  --encoder_spectra dreams --encoder_mol chemberta_13M --n_projections 50