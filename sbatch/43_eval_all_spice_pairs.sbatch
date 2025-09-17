#!/bin/bash
#SBATCH --job-name=eval_all_spice_pairs
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load anaconda/3
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH="$PWD:$PYTHONPATH"

DPT=models/bandit_shufTrue_lr0.0001_do0.0_embd64_layer6_head1_envs100000_hists1_samples1_var0.3_cov0.0_H500_d5_seed0_epoch300.pt

echo "[eval] DPT  : $DPT"

for SP in models/bandit_spice_*.pt; do
  echo "[eval] SPICE: $SP"
  # Infer epoch tag from filename
  TAG=$(basename "$SP" .pt | sed -E 's/.*_(epoch[0-9]+).*/\1/;t; s/.*/manual/')
  python3 -m spice.fig2_bandit_pair \
    --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
    --dpt_ckpt "$DPT" \
    --spice_ckpt "$SP" \
    --epoch_label "$TAG"
done

echo "[eval] done."
