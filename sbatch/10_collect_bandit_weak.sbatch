#!/bin/bash
#SBATCH --job-name=collect_bandit_weak
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
export PS1="${PS1-}"
module load anaconda/3

cd /home/mila/a/anaisb/scratch/SPICE-toyenvs
mkdir -p logs
export PYTHONPATH="$PWD:$PYTHONPATH"

# robust conda activation
if ! conda activate dpt 2>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate dpt
fi

# Generate weak-label datasets (behavior-argmax empirical; fixed omega 0.5)
#python collect_data.py \
#  --env bandit --envs 100000 --envs_eval 200 \
#  --hists 1 --samples 1 --H 500 --dim 5 --var 0.3 \
#  --bias random --fixed_omega 0.5 \
  #--label_mode beh_argmax_emp --label_mix_q 0.0 \
 # --tag weak

python collect_data.py \
  --env bandit --envs 100000 --envs_eval 200 \
  --hists 1 --samples 1 --H 500 --dim 5 --var 0.3 \
  --bias random --fixed_omega 0.5 \
  --label_mode beh_last --tag weaklast
