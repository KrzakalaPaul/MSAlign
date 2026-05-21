#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=V100
#SBATCH --gpus=1

set -x

srun python -u 2_encode.py --candidate_map_name official_candidates_by_mass 