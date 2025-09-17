#!/bin/bash
#SBATCH --job-name=spice_sweep
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=1-8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

module load anaconda/3 cudatoolkit/11.8
export PS1="${PS1-}"

if ! conda activate dpt 2>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate dpt
fi

export PYTHONPATH="/home/mila/a/anaisb/scratch/SPICE-toyenvs:${PYTHONPATH-}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Set your W&B project defaults (these can be overridden in the shell)
export WANDB_ENTITY="${WANDB_ENTITY:-david-rolnick}"
export WANDB_PROJECT="${WANDB_PROJECT:-SPICE-toyenvs-spice}"

cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
mkdir -p logs

if [ $# -lt 1 ]; then
  echo "Usage: sbatch sbatch/40_wandb_agent_spice.sbatch <SWEEP_ID or entity/project/id>"
  exit 1
fi
SWEEP_ID_RAW="$1"

# Accept either short id or full triplet
if [[ "$SWEEP_ID_RAW" == */*/* ]]; then
  SWEEP_ID="$SWEEP_ID_RAW"
else
  SWEEP_ID="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID_RAW}"
fi

# DPT baseline for micro-probe (your epoch-300 model)
export DPT_CKPT="/home/mila/a/anaisb/scratch/SPICE-toyenvs/models_notperhead/bandit_shufTrue_lr0.0001_do0.0_embd64_layer6_head1_envs100000_hists1_samples1_var0.3_cov0.0_H500_d5_seed0_epoch300.pt"
if [ ! -f "$DPT_CKPT" ]; then
  echo "ERROR: DPT_CKPT not found at $DPT_CKPT"
  exit 3
fi

echo "Node:       $(hostname)"
echo "Date:       $(date --iso-8601=seconds)"
python - <<'PY'
import sys, importlib.util
print("Python version:", sys.version.replace("\n"," "))
print("Find 'spice'  :", bool(importlib.util.find_spec('spice')))
print("Find 'envs'   :", bool(importlib.util.find_spec('envs')))
PY
echo "SWEEP_ID:   $SWEEP_ID"
echo "DPT_CKPT:   $DPT_CKPT"
echo "WANDB_ENTITY=$WANDB_ENTITY"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "PYTHONPATH: $PYTHONPATH"

# Launch the W&B agent
wandb agent "$SWEEP_ID"

