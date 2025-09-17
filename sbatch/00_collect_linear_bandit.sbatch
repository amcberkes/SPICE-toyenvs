#!/bin/bash
#SBATCH --job-name=00_collect_linear_bandit
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/collect_linear_bandit_%j.out
#SBATCH --export=NONE

set -euo pipefail
mkdir -p logs datasets

module purge
module load anaconda/3
source activate dpt

export REPO=/home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH=$REPO:$PYTHONPATH
cd $REPO

# Paper-scale (heavy). If queues are tight, reduce ENVS to 100000.
python3 collect_data.py \
  --env linear_bandit --envs 1000000 --envs_eval 200 \
  --H 200 --dim 10 --lin_d 2 --var 0.3 --cov 0.0 \
  --hists 1 --samples 1
