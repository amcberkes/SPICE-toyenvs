#!/bin/bash
#SBATCH --job-name=00_collect_darkroom
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/collect_darkroom_%j.out
#SBATCH --export=NONE

set -euo pipefail
mkdir -p logs datasets
module purge
module load anaconda/3
source activate dpt

export REPO=/home/mila/a/anaisb/scratch/SPICE-toyenvs
export PYTHONPATH=$REPO:$PYTHONPATH
cd $REPO

# Darkroom data per paper setup: 10x10, H=100, 100k train shards + 100 eval
python3 collect_data.py \
  --env darkroom_heldout --envs 100000 --envs_eval 100 \
  --H 100 --dim 10 --hists 1 --samples 1
