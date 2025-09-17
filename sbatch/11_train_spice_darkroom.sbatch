#!/bin/bash
#SBATCH --job-name=11_train_spice_darkroom
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_spice_darkroom_%j.out
#SBATCH --export=NONE

set -euo pipefail
mkdir -p logs models figs
module purge
module load anaconda/3
module load cuda/11.8
source activate dpt

export REPO=/home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH=$REPO:$PYTHONPATH
cd $REPO

# IMPORTANT:
# - Keep spice_heads=7 to match the default in fig scripts (avoids load mismatch).
# - Turn ON value loss (lambda_q=0.5) and use gamma=0.99 for Darkroom MC targets.
python3 spice/train_spice.py \
  --env darkroom_heldout --envs 100000 --H 100 --dim 10 \
  --lr 1e-4 --layer 6 --head 1 --embd 64 --dropout 0.0 \
  --shuffle --seed 0 --num_epochs 300 \
  --spice_heads 7 --spice_prior 0.1 \
  --alphaA 0.5 --lambda_sigma 1.0 --c_sigma 3.0 \
  --lambda_q 0.5 --gamma 0.99
