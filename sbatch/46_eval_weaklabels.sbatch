#!/bin/bash
#SBATCH --job-name=eval_weaklabels
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-999%24
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# If you want GPU (much faster), replace the partition/mem with something like:
# #SBATCH --partition=long
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16G
# and `module load cudatoolkit/11.8` below.

set -euo pipefail
export PS1="${PS1-}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

module load anaconda/3
# module load cudatoolkit/11.8   # uncomment if using a GPU partition
cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH="$PWD:$PYTHONPATH"

# Ensure conda activation works even in non-interactive shells
if ! conda activate dpt 2>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate dpt
fi

# ---------- Collect SPICE checkpoints ----------
SP_DIR="models_weak"
mapfile -t CKPTS < <(ls -1 ${SP_DIR}/bandit_spice_epoch*.pt 2>/dev/null | sort || true)
N=${#CKPTS[@]}
if (( N == 0 )); then
  echo "No SPICE checkpoints found in ${SP_DIR}/." >&2
  exit 3
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= N )); then
  echo "Index $IDX >= $N; nothing to do." ; exit 0
fi

SP="${CKPTS[$IDX]}"
SP_BASE="$(basename "$SP")"

# Extract epoch label from the SPICE filename
if [[ "$SP_BASE" =~ epoch([0-9]+) ]]; then
  E="${BASH_REMATCH[1]}"
  EPOCH_LABEL="epoch${E}"
else
  E="final"
  EPOCH_LABEL="final"
fi

# ---------- Find a matching DPT checkpoint ----------
# Strategy: search by epoch E under models_weak/ first (adjust to your preferred dir)
DPT_DIR="/home/mila/a/anaisb/scratch/SPICE-toyenvs/models_weak"
if [[ "$E" != "final" ]]; then
  # Try DPT-weak (cov=0.5) first, then any DPT with that epoch
  DPT=$(ls -1 ${DPT_DIR}/bandit_shufTrue_*_epoch${E}_*.pt 2>/dev/null | head -n1 || true)
else
  DPT=$(ls -1 ${DPT_DIR}/bandit_shufTrue_*_final_*.pt 2>/dev/null | head -n1 || true)
fi
if [[ -z "${DPT}" ]]; then
  echo "Could not find a DPT checkpoint for epoch=${E} under ${DPT_DIR}/" >&2
  echo "Set DPT_DIR or hardcode a path." >&2
  exit 5
fi

# ---------- Use the weak eval dataset directly ----------
EVAL_PATH="/home/mila/a/anaisb/scratch/SPICE-toyenvs/datasets/trajs_bandit_envs200_H500_d5_var0.3_cov0.5_eval_weak.pkl"

echo "[eval] DPT  : $DPT"
echo "[eval] SPICE: $SP"
echo "[eval] epoch_label = ${EPOCH_LABEL}"
echo "[eval] eval_path   = ${EVAL_PATH}"

# Full set of figures
python -m spice.fig2_bandit_pair \
  --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
  --dpt_ckpt "$DPT" \
  --spice_ckpt "$SP" \
  --eval_path "$EVAL_PATH" \
  --epoch_label "$EPOCH_LABEL"

# Posterior-only
python -m spice.fig2_bandit_pair \
  --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
  --dpt_ckpt "$DPT" \
  --spice_ckpt "$SP" \
  --eval_path "$EVAL_PATH" \
  --epoch_label "${EPOCH_LABEL}_posterior_only" \
  --spice_include posterior_ctx

# Minimal: DPT vs SPICE(posterior_ctx) (skip baselines)
python -m spice.fig2_bandit_pair \
  --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
  --dpt_ckpt "$DPT" \
  --spice_ckpt "$SP" \
  --eval_path "$EVAL_PATH" \
  --epoch_label "${EPOCH_LABEL}_posterior_vs_dpt" \
  --spice_include posterior_ctx \
  --skip_baselines TS,Emp,UCB1.0
