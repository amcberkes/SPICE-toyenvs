#!/bin/bash
#SBATCH --job-name=spice_weaklast
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Chatty + safe shell
export PS1="${PS1-}"
set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

module load anaconda/3
module load cudatoolkit/11.8

cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p logs figs/loss models_weaklast

# Robust conda activation
if ! conda activate dpt 2>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate dpt
fi

pwd; hostname; date
nvidia-smi || true
which python; python -V

# Seeds to sweep
SEEDS=(0 1 2)
IDX=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$IDX]}

python -u spice/train_spice.py \
  --env bandit --envs 100000 --hists 1 --samples 1 \
  --H 500 --dim 5 --var 0.3 --cov 0.5 \
  --embd 64 --layer 6 --head 1 --dropout 0.0 \
  --lr 1e-4 --num_epochs 300 --shuffle True \
  --seed ${SEED} --dataset_tag weaklast --models_dir models_weaklast \
  --spice_heads 7 --spice_prior 0.1

echo "Done seed ${SEED}"
