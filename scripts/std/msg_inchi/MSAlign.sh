#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=V100
#SBATCH --gpus=1

set -x

srun python -u 3_train.py --labelled_dataset_name massspecgym --candidate_map_name official_candidates_by_mass --model MSAlign --config default --wandb_run_name MSAlign_inchi_inchi_seed3 --split_method inchi_seed3

srun python -u 3_train.py --labelled_dataset_name massspecgym --candidate_map_name official_candidates_by_mass --model MSAlign --config default --wandb_run_name MSAlign_inchi_inchi_seed1 --split_method inchi_seed1

srun python -u 3_train.py --labelled_dataset_name massspecgym --candidate_map_name official_candidates_by_mass --model MSAlign --config default --wandb_run_name MSAlign_inchi_inchi_seed2 --split_method inchi_seed2
