#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --gpus=0

set -x

srun python -u 1_prepare_many_splits.py --dataset_name spectraverse --split_method formula
