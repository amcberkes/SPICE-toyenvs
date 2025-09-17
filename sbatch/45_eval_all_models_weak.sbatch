#!/bin/bash
#SBATCH --job-name=eval_weak_all
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-999%24
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
export PS1="${PS1-}"

module load anaconda/3
if ! conda activate dpt 2>/dev/null; then
  eval "$(conda shell.bash hook)"; conda activate dpt
fi
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH="$PWD:$PYTHONPATH"

# Collect SPICE checkpoints
mapfile -t CKPTS < <(ls -1 models_weak/*spice*epoch*.pt 2>/dev/null | sort || true)
N=${#CKPTS[@]}
if (( N == 0 )); then
  echo "No SPICE checkpoints in models_weak/." >&2
  exit 3
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= N )); then
  echo "Index $IDX >= $N; nothing to do."; exit 0
fi

SP="${CKPTS[$IDX]}"
SP_BASE="$(basename "$SP")"

# Extract epoch number if present
EPOCH_LABEL="final"
if [[ "$SP_BASE" =~ epoch([0-9]+) ]]; then
  E="${BASH_REMATCH[1]}"; EPOCH_LABEL="epoch${E}"
else
  E="300"
fi

# Find a DPT checkpoint in models_weak with the same epoch; fallback to most recent DPT
DPT_MATCH=$(ls -1 models_weak/*_epoch${E}_*.pt 2>/dev/null | grep -v spice | sort | tail -n1 || true)
if [[ -z "${DPT_MATCH}" ]]; then
  DPT_MATCH=$(ls -1 models_weak/*_epoch*.pt 2>/dev/null | grep -v spice | sort | tail -n1 || true)
fi
if [[ -z "${DPT_MATCH}" ]]; then
  echo "No DPT checkpoint found in models_weak/ (needed for paired figs)."; exit 4
fi

echo "[eval] DPT     : $DPT_MATCH"
echo "[eval] SPICE   : $SP"
echo "[eval] epoch   : $EPOCH_LABEL"

# (A) Full figure set
python -m spice.fig2_bandit_pair \
  --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
  --dpt_ckpt "$DPT_MATCH" \
  --spice_ckpt "$SP" \
  --epoch_label "${EPOCH_LABEL}_weak_full" \
  --dataset_tag weak

# (B) Posterior-only
python -m spice.fig2_bandit_pair \
  --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
  --dpt_ckpt "$DPT_MATCH" \
  --spice_ckpt "$SP" \
  --epoch_label "${EPOCH_LABEL}_weak_posterior_only" \
  --spice_include posterior_ctx \
  --dataset_tag weak

# (C) Minimal: DPT vs SPICE(posterior_ctx) only
python -m spice.fig2_bandit_pair \
  --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
  --dpt_ckpt "$DPT_MATCH" \
  --spice_ckpt "$SP" \
  --epoch_label "${EPOCH_LABEL}_weak_posterior_vs_dpt" \
  --spice_include posterior_ctx \
  --skip_baselines TS,Emp,UCB1.0 \
  --dataset_tag weak
