#!/bin/bash
#SBATCH --job-name=eval_models_new_all
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-999%24
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

export PS1="${PS1-}"
set -euo pipefail

module load anaconda/3
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH="$PWD:$PYTHONPATH"

if ! conda activate dpt 2>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate dpt
fi

# --- Collect SPICE checkpoints from your new folder ---
mapfile -t CKPTS < <(ls -1 models_new_badeu/bandit_spice_*.pt 2>/dev/null | sort || true)
N=${#CKPTS[@]}
if (( N == 0 )); then
  echo "No SPICE checkpoints found in models_new_badeu/." >&2
  exit 3
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= N )); then
  echo "Index $IDX >= $N; nothing to do." ; exit 0
fi

SP="${CKPTS[$IDX]}"
SP_BASE="$(basename "$SP")"

# --- Derive epoch label from filename ---
if [[ "$SP_BASE" =~ epoch([0-9]+) ]]; then
  E="${BASH_REMATCH[1]}"
  EPOCH_LABEL="epoch${E}"
else
  E="300"
  EPOCH_LABEL="final"
fi

# --- Pick matching DPT checkpoint (same epoch if available; else error) ---
DPT_DIR="/home/mila/a/anaisb/scratch/SPICE-toyenvs/models_notperhead"
DPT_300="${DPT_DIR}/bandit_shufTrue_lr0.0001_do0.0_embd64_layer6_head1_envs100000_hists1_samples1_var0.3_cov0.0_H500_d5_seed0_epoch300.pt"
if [[ ! -f "$DPT_300" ]]; then
  echo "Cannot find reference DPT @300: $DPT_300" >&2
  exit 4
fi
DPT_PREFIX="${DPT_300%_epoch300.pt}"
DPT="${DPT_PREFIX}_epoch${E}.pt"
if [[ ! -f "$DPT" ]]; then
  echo "Missing matching DPT checkpoint for E=${E}: $DPT" >&2
  exit 5
fi

echo "[eval] DPT  : $DPT"
echo "[eval] SPICE: $SP"
echo "[eval] epoch_label = ${EPOCH_LABEL}"

# (A) Full figure set (all variants)
python -m spice.fig2_bandit_pair \
  --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
  --dpt_ckpt "$DPT" \
  --spice_ckpt "$SP" \
  --epoch_label "$EPOCH_LABEL"

# (B) Posterior-only (keep all non-SPICE baselines)
python -m spice.fig2_bandit_pair \
  --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
  --dpt_ckpt "$DPT" \
  --spice_ckpt "$SP" \
  --epoch_label "${EPOCH_LABEL}_posterior_only" \
  --spice_include posterior_ctx

# (C) Minimal: DPT vs SPICE(posterior_ctx) only
python -m spice.fig2_bandit_pair \
  --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
  --dpt_ckpt "$DPT" \
  --spice_ckpt "$SP" \
  --epoch_label "${EPOCH_LABEL}_posterior_vs_dpt" \
  --spice_include posterior_ctx \
  --skip_baselines TS,Emp,UCB1.0
