#!/bin/bash
#SBATCH --job-name=eval_fig2_pairs
#SBATCH --partition=long-cpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load anaconda/3
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH="$PWD:$PYTHONPATH"



echo "[eval] epoch50 pair..."
python3 -m spice.fig2_bandit_pair \
  --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
  --dpt_ckpt models/bandit_shufTrue_lr0.0001_do0.0_embd64_layer6_head1_envs100000_hists1_samples1_var0.3_cov0.0_H500_d5_seed0_epoch100.pt \
  --spice_ckpt models/bandit_spice_epoch100.pt \
  --epoch_label epoch100 \


echo "[eval] epoch250 pair..."
python3 -m spice.fig2_bandit_pair \
  --env bandit --H 500 --dim 5 --var 0.3 --n_eval 200 \
  --dpt_ckpt models/bandit_shufTrue_lr0.0001_do0.0_embd64_layer6_head1_envs100000_hists1_samples1_var0.3_cov0.0_H500_d5_seed0_epoch250.pt \
  --spice_ckpt models/bandit_spice_epoch250.pt \
  --epoch_label epoch250 \


echo "[eval] done. Figures are in ${OUTDIR}"
