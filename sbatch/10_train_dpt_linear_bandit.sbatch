#!/bin/bash
#SBATCH --job-name=10_train_dpt_linear_bandit
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_dpt_linear_bandit_%j.out
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

python3 train.py \
  --env linear_bandit --envs 1000000 --H 200 --dim 10 --lin_d 2 --var 0.3 --cov 0.0 \
  --lr 1e-4 --layer 6 --head 1 --embd 64 --dropout 0.0 \
  --shuffle --seed 0 --num_epochs 200
