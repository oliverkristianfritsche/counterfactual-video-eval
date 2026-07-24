#!/usr/bin/env bash
#SBATCH --job-name=pairwise_argmax
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=96GB
#SBATCH --ntasks=8
#SBATCH --output=slurm-%x.out
#SBATCH --error=slurm-%x.err
# Report section 4: pairwise argmax. Produce the real arms, then decode.
set -euo pipefail
PYTHON="${PYTHON:-python}"
TB="${TIMEBLIND_ROOT:?set TIMEBLIND_ROOT}"
MB="${MOTIONBLIND_ROOT:?set MOTIONBLIND_ROOT}"
OUT="$PWD/arms"

"$PYTHON" run.py --config configs/eagle_timeblind_calib.yaml   --set dataset.root="$TB" --set eval.results_dir="$OUT/tb_real"    --wandb-mode disabled --name eagle-tb-real
"$PYTHON" run.py --config configs/eagle_timeblind_shuffle.yaml --set dataset.root="$TB" --set eval.results_dir="$OUT/tb_shuffle" --wandb-mode disabled --name eagle-tb-shuffle-s1
"$PYTHON" run.py --config configs/eagle_motionblind_calib.yaml --set dataset.root="$MB" --set eval.results_dir="$OUT/mb_real"    --wandb-mode disabled --name eagle-mb-real

"$PYTHON" scripts/pair_decode.py \
  --tb-real "$OUT"/tb_real/run_*/predictions.jsonl \
  --tb-pert "$OUT"/tb_shuffle/run_*/predictions.jsonl \
  --mb-real "$OUT"/mb_real/run_*/predictions.jsonl
