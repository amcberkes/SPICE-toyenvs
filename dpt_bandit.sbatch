#!/bin/bash
#SBATCH --job-name=dpt-bandit
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH -o %x-%j.out
# If you want a GPU, uncomment the next line:
# #SBATCH --gres=gpu:1

module load anaconda/3
source activate dpt

# put caches & temps on scratch
export XDG_CACHE_HOME=~/scratch/caches/xdg
export PIP_CACHE_DIR=~/scratch/caches/pip
export TORCH_HOME=~/scratch/caches/torch
export HF_HOME=~/scratch/caches/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export CUDA_CACHE_PATH=~/scratch/caches/cuda
export TMPDIR=~/scratch/caches/tmp

# sane threading; fall back to 4 outside SLURM
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_THREADING_LAYER=GNU
export GYM_DISABLE_DEPRECATED_WARNING=1

cd ~/scratch/SPICE-toyenvs

# ensure data dir is on scratch if the code writes to ./data
[ -e data ] || ln -s ~/scratch/dpt_data data
mkdir -p ~/scratch/dpt_runs

# 3-phase pipeline: collect -> train -> eval
bash ./run_bandit.sh
