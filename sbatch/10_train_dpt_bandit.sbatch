#!/bin/bash
#SBATCH --job-name=train_dpt_bandit
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load anaconda/3 cudatoolkit/11.8
# If you need your conda env, uncomment:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate dpt
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
python3 train.py \
  --env bandit \
  --envs 100000 \
  --H 500 \
  --dim 5 \
  --var 0.3 \
  --cov 0.0 \
  --lr 1e-4 \
  --layer 6 \
  --head 1 \
  --embd 64 \
  --dropout 0.0 \
  --shuffle \
  --seed 0 \
  --num_epochs 400
