#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=V100
#SBATCH --gpus=1

set -x

srun python -u 3_train.py --labelled_dataset_name massspecgym --candidate_map_name 256_candidatates_by_mass --k_candidates 128 --encoder_mol chemberta_13M --wandb_run_name dev_13M

srun python -u 3_train.py --labelled_dataset_name massspecgym --candidate_map_name 256_candidatates_by_mass --k_candidates 128 --encoder_mol chemberta_100M --wandb_run_name dev_100M