#!/usr/bin/env bash
#SBATCH --job-name=verify_hornet_parity
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --output=slurm-%x.out
#SBATCH --error=slurm-%x.err
# HORNet selection parity and planted-frame probe. CPU only: the policy is
# TimeSformerTiny and Eagle never loads. Both code paths pick the same device, so a
# CPU node keeps the comparison consistent and off the GPU queue.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1
PY=/scratch/${USER}/envs/eagle/bin/python
"$PY" scripts/verify_hornet_parity.py \
  --checkpoint "${HORNET_CKPT:-/scratch/${USER}/checkpoints/hornet/checkpoints/short/checkpoint-0.550.pt}" \
  --timeblind-root "${TIMEBLIND_ROOT:-/scratch/${USER}/datasets/timeblind}" \
  --motionblind-root "${MOTIONBLIND_ROOT:-/scratch/${USER}/datasets/motionblind}" \
  --limit "${LIMIT:-10}" --planted --out results/hornet_parity.json
