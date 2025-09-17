#!/bin/bash
#SBATCH --job-name=20_eval_linear_bandit_figs
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_linear_bandit_%j.out
#SBATCH --export=NONE

set -euo pipefail
mkdir -p logs figs

module purge
module load anaconda/3
module load cuda/11.8
source activate dpt

export REPO=/home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH=$REPO:$PYTHONPATH
cd $REPO

python3 eval.py \
  --env linear_bandit --envs 1000000 --H 200 --dim 10 --lin_d 2 --var 0.3 --cov 0.0 \
  --embd 64 --layer 6 --head 1 --dropout 0.0 \
  --epoch -1 --n_eval 200 --seed 0
