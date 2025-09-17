#!/bin/bash
#SBATCH --job-name=dpt_weak
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
export PS1="${PS1-}"

module load anaconda/3 cudatoolkit/11.8
if ! conda activate dpt 2>/dev/null; then
  eval "$(conda shell.bash hook)"; conda activate dpt
fi

cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH="$PWD:$PYTHONPATH"
SEED=${SLURM_ARRAY_TASK_ID}

python train.py \
  --env bandit --envs 100000 --hists 1 --samples 1 \
  --H 500 --dim 5 --var 0.3 --cov 0.5 \
  --embd 64 --layer 6 --head 1 --dropout 0.0 \
  --lr 1e-4 --num_epochs 300 --shuffle True \
  --seed ${SEED} --wandb False \
  --dataset_tag weak \
  --models_dir models_weak
