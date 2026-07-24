#!/usr/bin/env bash
#SBATCH --job-name=motioncd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=96GB
#SBATCH --ntasks=8
#SBATCH --output=slurm-%x.out
#SBATCH --error=slurm-%x.err
# Report section 6: MotionCD vs ours. Produce the real and reverse arms, then decode.
set -euo pipefail
PYTHON="${PYTHON:-python}"
TB="${TIMEBLIND_ROOT:?set TIMEBLIND_ROOT}"
MB="${MOTIONBLIND_ROOT:?set MOTIONBLIND_ROOT}"
OUT="$PWD/arms"

"$PYTHON" run.py --config configs/eagle_timeblind_calib.yaml   --set dataset.root="$TB" --set eval.results_dir="$OUT/tb_real"    --wandb-mode disabled --name eagle-tb-real
"$PYTHON" run.py --config configs/eagle_timeblind_reverse.yaml --set dataset.root="$TB" --set eval.results_dir="$OUT/tb_reverse" --wandb-mode disabled --name eagle-tb-reverse
"$PYTHON" run.py --config configs/eagle_motionblind_calib.yaml   --set dataset.root="$MB" --set eval.results_dir="$OUT/mb_real"    --wandb-mode disabled --name eagle-mb-real
"$PYTHON" run.py --config configs/eagle_motionblind_reverse.yaml --set dataset.root="$MB" --set eval.results_dir="$OUT/mb_reverse" --wandb-mode disabled --name eagle-mb-reverse

echo "==== timeblind: MotionCD vs baseline vs ours (reverse arm) ===="
"$PYTHON" scripts/motioncd_decode.py \
  --real "$OUT"/tb_real/run_*/predictions.jsonl \
  --pert "$OUT"/tb_reverse/run_*/predictions.jsonl \
  --dataset timeblind

echo "==== motionblind: MotionCD vs baseline vs ours (reverse arm) ===="
"$PYTHON" scripts/motioncd_decode.py \
  --real "$OUT"/mb_real/run_*/predictions.jsonl \
  --pert "$OUT"/mb_reverse/run_*/predictions.jsonl \
  --dataset motionblind
