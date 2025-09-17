#!/bin/bash
#SBATCH --job-name=collect_bandit
#SBATCH --partition=long
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load anaconda/3
export MPLBACKEND=Agg
cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
python3 collect_data.py \
  --env bandit \
  --envs 100000 \
  --H 500 \
  --dim 5 \
  --var 0.3 \
  --cov 0.0 \
  --envs_eval 200
