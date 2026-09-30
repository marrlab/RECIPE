#!/bin/bash
#SBATCH --job-name=recipe
#SBATCH --output=logs_recipe/prepare_%j.out
#SBATCH --error=logs_recipe/prepare_%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Run one scenario end to end: the preparation stages here, then emit and submit
# the training array as a dependent job.
#
#   sbatch scripts/run_scenario.sh transfer configs/transfer.yaml
#   sbatch scripts/run_scenario.sh size     configs/size.yaml
#   sbatch scripts/run_scenario.sh augment  my_augment.yaml
#
# The `#SBATCH` lines above size THIS job (codebook fitting + generation). The
# training array is sized by the `train.slurm_*` fields of the config.
#
# `augment` needs a GPU here (it trains the MC-dropout selector); `transfer` and
# `size` do not, but the transfer codebook fit on a large source cohort wants
# plenty of RAM -- submit it to a high-memory CPU partition if it runs short.
set -euo pipefail

SCENARIO="${1:?usage: run_scenario.sh <transfer|size|augment> <config.yaml>}"
CFG="${2:?usage: run_scenario.sh <transfer|size|augment> <config.yaml>}"

# `set -u` is disabled around the profile: stock /etc/bashrc reads unset
# variables and would abort the job before conda is even available.
set +u
source ~/.bashrc
conda activate "${RECIPE_ENV:-recipe}"
set -u
cd "$(dirname "$0")/.."
mkdir -p logs_recipe
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

TAG="$(basename "${CFG%.yaml}")"
ARRAY_SCRIPT="scripts/_train_${TAG}.sh"

echo "=== [1/2] prepare: $SCENARIO ($CFG)"
python -m recipe "$SCENARIO" run -c "$CFG"

echo "=== [2/2] emit + submit the training array -> $ARRAY_SCRIPT"
python -m recipe "$SCENARIO" train -c "$CFG" --emit-slurm "$ARRAY_SCRIPT"
sbatch "$ARRAY_SCRIPT"

echo
echo "When the array finishes:"
echo "    python -m recipe $SCENARIO evaluate -c $CFG"
