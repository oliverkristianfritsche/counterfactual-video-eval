#!/usr/bin/env bash
#SBATCH --job-name=frame_selection_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=96GB
#SBATCH --ntasks=8
#SBATCH --array=0-29
#SBATCH --output=slurm-%x-%a.out
#SBATCH --error=slurm-%x-%a.err
# Frame-selection sweep: 2 datasets x 3 selectors x 5 budgets, one cell per array task.
# The hornet arm selects with the authors' own VisionGRPOPolicy + get_action_by_k, imported
# unmodified from HORNet/ (cloned, see README); its cells (array 10-14 timeblind, 25-29 motionblind)
# read the checkpoint from HORNET_CKPT. Roll up with: python scripts/make_table.py results
set -euo pipefail
PYTHON="${PYTHON:-python}"
DATASETS=(timeblind motionblind); SELECTORS=(uniform random hornet); BUDGETS=(1 4 8 16 24)

i="${SLURM_ARRAY_TASK_ID:?submit with sbatch --array=0-29}"
k="${BUDGETS[$((i % 5))]}"; sel="${SELECTORS[$(((i / 5) % 3))]}"; ds="${DATASETS[$((i / 15))]}"

if [ "$ds" = timeblind ]; then
  root="${TIMEBLIND_ROOT:?set TIMEBLIND_ROOT}"; prefix=eagle-tbnat
else
  root="${MOTIONBLIND_ROOT:?set MOTIONBLIND_ROOT}"; prefix=eagle-mbnat
fi

extra=()
if [ "$sel" = hornet ]; then
  extra+=(--set video.hornet_checkpoint="${HORNET_CKPT:?set HORNET_CKPT to the checkpoint-0.550.pt path for the hornet arm}")
fi

"$PYTHON" run.py --config "configs/eagle_${ds}_sweep_native.yaml" \
  --set dataset.root="$root" --set video.frame_source="$sel" --set video.top_k="$k" \
  "${extra[@]}" \
  --wandb-mode disabled --name "${prefix}-${sel}-k${k}"
