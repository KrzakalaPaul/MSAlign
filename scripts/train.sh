#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=V100
#SBATCH --gpus=1

set -x

srun python -u 3_train.py --labelled_dataset_name massspecgym --candidate_map_name official_candidatates_by_mass --model MSAlign --config default --wandb_run_name dev
