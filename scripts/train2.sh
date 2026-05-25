#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=V100
#SBATCH --gpus=1

set -x
ulimit -n 65536 # Fix for FLARE dataloader workers

srun python -u 3_train.py --labelled_dataset_name massspecgym --candidate_map_name official_candidates_by_mass --model FLARE --config default --wandb_run_name FLARE_dev
