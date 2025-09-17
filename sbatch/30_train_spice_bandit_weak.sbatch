#!/bin/bash
#SBATCH --job-name=spice_weak
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

# Environment
module load anaconda/3
module load cudatoolkit/11.8

cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH="$PWD:$PYTHONPATH"

# Robust conda activation
if ! conda activate dpt 2>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate dpt
fi

pwd; hostname; date
nvidia-smi || true
which python; python -V

mkdir -p logs figs/loss models_weak

# ------- sweep seeds (edit if you want more/others) -------
SEEDS=(0 1 2)
IDX=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$IDX]}

# ------- training call (chatty via -u and PYTHONUNBUFFERED) -------
python -u spice/train_spice.py \
  --env bandit --envs 100000 --hists 1 --samples 1 \
  --H 500 --dim 5 --var 0.3 --cov 0.5 \
  --embd 64 --layer 6 --head 1 --dropout 0.0 \
  --lr 1e-4 --num_epochs 300 --shuffle True \
  --seed ${SEED} --dataset_tag weak --models_dir models_weak \
  --spice_heads 7 --spice_prior 0.1

echo "Done seed ${SEED}"
