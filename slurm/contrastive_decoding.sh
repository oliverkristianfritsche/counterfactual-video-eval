#!/usr/bin/env bash
#SBATCH --job-name=contrastive_decoding
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --ntasks=8
#SBATCH --output=slurm-%x.out
#SBATCH --error=slurm-%x.err
# Report section 5: contrastive decoding. Produce the real arm and each perturbation sibling, then decode.
set -euo pipefail
PYTHON="${PYTHON:-python}"
TB="${TIMEBLIND_ROOT:?set TIMEBLIND_ROOT}"
MB="${MOTIONBLIND_ROOT:?set MOTIONBLIND_ROOT}"
OUT="$PWD/arms"

# TimeBlind arms
"$PYTHON" run.py --config configs/eagle_timeblind_calib.yaml     --set dataset.root="$TB" --set eval.results_dir="$OUT/tb_real"    --wandb-mode disabled --name eagle-tb-real
"$PYTHON" run.py --config configs/eagle_timeblind_shuffle.yaml   --set dataset.root="$TB" --set eval.results_dir="$OUT/tb_shuffle" --wandb-mode disabled --name eagle-tb-shuffle
"$PYTHON" run.py --config configs/eagle_timeblind_reverse.yaml   --set dataset.root="$TB" --set eval.results_dir="$OUT/tb_reverse" --wandb-mode disabled --name eagle-tb-reverse
"$PYTHON" run.py --config configs/eagle_timeblind_segswap.yaml   --set dataset.root="$TB" --set eval.results_dir="$OUT/tb_segswap" --wandb-mode disabled --name eagle-tb-segswap
"$PYTHON" run.py --config configs/eagle_timeblind_blockshuf.yaml --set dataset.root="$TB" --set eval.results_dir="$OUT/tb_block"   --wandb-mode disabled --name eagle-tb-blockshuffle

# MotionBlind arms
"$PYTHON" run.py --config configs/eagle_motionblind_calib.yaml     --set dataset.root="$MB" --set eval.results_dir="$OUT/mb_real"    --wandb-mode disabled --name eagle-mb-real
"$PYTHON" run.py --config configs/eagle_motionblind_shuffle.yaml   --set dataset.root="$MB" --set eval.results_dir="$OUT/mb_shuffle" --wandb-mode disabled --name eagle-mb-shuffle
"$PYTHON" run.py --config configs/eagle_motionblind_reverse.yaml   --set dataset.root="$MB" --set eval.results_dir="$OUT/mb_reverse" --wandb-mode disabled --name eagle-mb-reverse
"$PYTHON" run.py --config configs/eagle_motionblind_segswap.yaml   --set dataset.root="$MB" --set eval.results_dir="$OUT/mb_segswap" --wandb-mode disabled --name eagle-mb-segswap
"$PYTHON" run.py --config configs/eagle_motionblind_blockshuf.yaml --set dataset.root="$MB" --set eval.results_dir="$OUT/mb_block"   --wandb-mode disabled --name eagle-mb-blockshuffle

# TimeBlind: real vs each sibling
for arm in tb_block tb_shuffle tb_segswap tb_reverse; do
  echo "==== TimeBlind real vs ${arm} ===="
  "$PYTHON" scripts/contrastive_decode.py \
    --real "$OUT"/tb_real/run_*/predictions.jsonl \
    --pert "$OUT"/$arm/run_*/predictions.jsonl
done

# MotionBlind: real vs each sibling (per-category breakdown printed by the script)
for arm in mb_block mb_shuffle mb_segswap mb_reverse; do
  echo "==== MotionBlind real vs ${arm} ===="
  "$PYTHON" scripts/contrastive_decode.py \
    --real "$OUT"/mb_real/run_*/predictions.jsonl \
    --pert "$OUT"/$arm/run_*/predictions.jsonl
done
