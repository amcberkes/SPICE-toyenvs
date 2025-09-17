#!/bin/bash
#SBATCH --job-name=eval_spiceweak_vs_dptweak_last
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-999%6
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# ---- robust shell setup ----
export PS1="${PS1-}"
set -euo pipefail
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

module load anaconda/3
cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH="$PWD:$PYTHONPATH"

# robust conda activation
if ! conda activate dpt 2>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate dpt
fi

# ---------- Config ----------
CKPT_DIR="models_weaklast"   # <<<<<< use your 'last' folder
OUTDIR_PREFIX="figs/weaklast"
EVAL_PATH="/home/mila/a/anaisb/scratch/SPICE-toyenvs/datasets/trajs_bandit_envs200_H500_d5_var0.3_cov0.5_eval_weaklast.pkl"

: "${DPT_SEED:=}"                 # optional seed to select specific DPT among matches
# Optional speed knobs (export before submit if you want):
# export SKIP_ROBUSTNESS=1
# export SKIP_ONLINE=1
# export SKIP_BASELINES="Emp,UCB1.0,TS,LCB"

# ---------- Collect SPICE-weak checkpoints ----------
mapfile -t SP_LIST < <(ls -1 ${CKPT_DIR}/bandit_spice_epoch*.pt 2>/dev/null | sort || true)
N=${#SP_LIST[@]}
if (( N == 0 )); then
  echo "No SPICE checkpoints found in ${CKPT_DIR}/." >&2
  exit 3
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= N )); then
  echo "Index $IDX >= $N; nothing to do."
  exit 0
fi

SP="${SP_LIST[$IDX]}"
SP_BASE="$(basename "$SP")"

# Extract epoch from SPICE filename
E=$(echo "$SP_BASE" | sed -n 's/.*epoch\([0-9]\+\).*/\1/p')
if [[ -z "$E" ]]; then
  echo "Cannot parse epoch from $SP_BASE"
  exit 4
fi

# Find matching DPT-weak (cov0.5) with the same epoch (optionally match seed)
DPT=""
if [[ -n "${DPT_SEED}" ]]; then
  DPT=$(ls -1 ${CKPT_DIR}/bandit_shufTrue_*_cov0.5_*_seed${DPT_SEED}_epoch${E}_*.pt 2>/dev/null | head -n1 || true)
fi
if [[ -z "$DPT" ]]; then
  DPT=$(ls -1 ${CKPT_DIR}/bandit_shufTrue_*_cov0.5_*_epoch${E}_*.pt 2>/dev/null | head -n1 || true)
fi
if [[ -z "$DPT" ]]; then
  echo "No DPT-weak found for epoch ${E} in ${CKPT_DIR}/"
  exit 5
fi

# Distinct epoch label per SPICE checkpoint to avoid confusion
SP_TAG="${SP_BASE%.pt}"
EPOCH_LABEL="E${E}_weak_${SP_TAG}"

echo "[PAIR] epoch=$E"
echo "  SPICE: $SP"
echo "  DPT-w: $DPT"
echo "  eval : $EVAL_PATH"
echo "  out  : $OUTDIR_PREFIX/$EPOCH_LABEL/$(basename "$DPT" .pt)/"

# Build common flags
FLAGS=( --H 500 --dim 5 --n_eval 200 --eval_path "$EVAL_PATH" --outdir_prefix "$OUTDIR_PREFIX" )

[[ -n "${SKIP_ROBUSTNESS:-}" ]] && FLAGS+=( --skip_robustness )
[[ -n "${SKIP_ONLINE:-}"      ]] && FLAGS+=( --skip_online )
[[ -n "${SKIP_BASELINES:-}"   ]] && FLAGS+=( --skip_baselines "$SKIP_BASELINES" )

# (A) Full figure set (baselines + all SPICE variants)
python -m spice.fig2_bandit_weak \
  --dpt_weak_ckpt "$DPT" \
  --spice_ckpt    "$SP" \
  --epoch_label   "${EPOCH_LABEL}" \
  "${FLAGS[@]}"

# (B) Posterior-only (keep non-SPICE baselines)
python -m spice.fig2_bandit_weak \
  --dpt_weak_ckpt "$DPT" \
  --spice_ckpt    "$SP" \
  --spice_include posterior_ctx \
  --epoch_label   "${EPOCH_LABEL}_posterior_only" \
  "${FLAGS[@]}"

# (C) Minimal: DPT-weak vs SPICE(posterior_ctx) only
python -m spice.fig2_bandit_weak \
  --dpt_weak_ckpt "$DPT" \
  --spice_ckpt    "$SP" \
  --spice_include posterior_ctx \
  --skip_baselines Emp,UCB1.0,TS,LCB \
  --epoch_label   "${EPOCH_LABEL}_posterior_vs_dptweak" \
  "${FLAGS[@]}"
