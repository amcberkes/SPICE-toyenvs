#!/bin/bash
#SBATCH --job-name=train_spice_bandit
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load anaconda/3 cudatoolkit/11.8
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate dpt
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
python3 spice/train_spice.py \
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
  --dropout 0.1 \
  --num_epochs 400 \
  --spice_heads 11 \
  --spice_prior 0.1 \
  --alphaA 1.0 \
  --lambda_sigma 0.2 \
  --c_sigma 2.0 \
  --lambda_q 1.0 \
  --seed 0
