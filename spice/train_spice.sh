#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=spice-train
#SBATCH --output=logs/spice_train_%j.out

module load python/3.10  # adapt for Mila
source activate dpt      # or conda activate dpt

python3 spice/train_spice.py \
  --env bandit --envs 10000 --H 200 --dim 5 --var 0.3 --cov 0.0 \
  --lr 1e-4 --layer 4 --head 1 --embd 32 --dropout 0.0 \
  --shuffle --seed 0 --num_epochs 50 \
  --spice_heads 7 --spice_prior 0.1 \
  --alphaA 0.5 --lambda_sigma 1.0 --c_sigma 3.0 \
  --lambda_q 0.5
