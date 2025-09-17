#!/bin/bash
#SBATCH --job-name=20_eval_darkroom_figs
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_darkroom_figs_%j.out
#SBATCH --export=NONE

set -euo pipefail
mkdir -p logs figs

module purge
module load anaconda/3
module load cuda/11.8
source activate dpt

export REPO=/home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH="$REPO:$PYTHONPATH"
cd "$REPO"

# -----------------------------------------------------------------------------
# How to supply inputs:
#   1) Positional args:
#        sbatch sbatch/20_eval_darkroom_figs.sbatch <DPT_CKPT> <SPICE_CKPT> <EVAL_PKL> [OUTDIR]
#   2) Env vars:
#        sbatch --export=DPT_CKPT=...,SPICE_CKPT=...,EVAL_PKL=...,OUTDIR=figs/darkroom_paper \
#               sbatch/20_eval_darkroom_figs.sbatch
# If both provided, positional args take precedence.
# -----------------------------------------------------------------------------

ARG_DPT="${1-}"
ARG_SPICE="${2-}"
ARG_EVAL="${3-}"
ARG_OUTDIR="${4-}"

DPT_CKPT="${ARG_DPT:-${DPT_CKPT-}}"
SPICE_CKPT="${ARG_SPICE:-${SPICE_CKPT-}}"
EVAL_PKL="${ARG_EVAL:-${EVAL_PKL-}}"
OUTDIR="${ARG_OUTDIR:-${OUTDIR-figs}}"

if [[ -z "${DPT_CKPT}" || -z "${SPICE_CKPT}" || -z "${EVAL_PKL}" ]]; then
  echo "USAGE:"
  echo "  sbatch sbatch/20_eval_darkroom_figs.sbatch <DPT_CKPT> <SPICE_CKPT> <EVAL_PKL> [OUTDIR]"
  echo "  # or"
  echo "  sbatch --export=DPT_CKPT=/path/dpt.pt,SPICE_CKPT=/path/spice.pt,EVAL_PKL=/path/eval.pkl,OUTDIR=figs/darkroom_paper \\"
  echo "         sbatch/20_eval_darkroom_figs.sbatch"
  exit 64
fi

[[ -f "$DPT_CKPT"   ]] || { echo "[ERR] DPT_CKPT not found:   $DPT_CKPT" >&2;   exit 66; }
[[ -f "$SPICE_CKPT" ]] || { echo "[ERR] SPICE_CKPT not found: $SPICE_CKPT" >&2; exit 66; }
[[ -f "$EVAL_PKL"   ]] || { echo "[ERR] EVAL_PKL not found:   $EVAL_PKL" >&2;   exit 66; }

mkdir -p "$OUTDIR"

echo "Using DPT:    $DPT_CKPT"
echo "Using SPICE:  $SPICE_CKPT"
echo "Using EVAL:   $EVAL_PKL"
echo "Output dir:   $OUTDIR"

# If fig_darkroom.py supports an --outdir flag in the future, add it below.
python3 spice/fig_darkroom.py \
  --dpt_ckpt "$DPT_CKPT" \
  --spice_ckpt "$SPICE_CKPT" \
  --H 100 --dim 10 --Heps 40 --n_eval 20 \
  --eval_pkl "$EVAL_PKL"
