#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=H100
#SBATCH --gpus=1

set -x

srun python -u 2_encode.py --candidate_map_name 256_candidates_by_mass