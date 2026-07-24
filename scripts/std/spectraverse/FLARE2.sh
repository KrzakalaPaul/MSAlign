#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=V100
#SBATCH --gpus=1

set -x
ulimit -n 65536 # Fix for FLARE dataloader workers

srun python -u 3_train.py --labelled_dataset_name spectraverse --candidate_map_name 256_candidates_by_mass --model FLARE --config default --wandb_run_name FLARE_spectraverse_formula_seed2 --split_method formula_seed2