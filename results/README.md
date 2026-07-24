# Results

Logged outputs of the 46 published Eagle2.5-8B runs; every reported number recomputes from these
files alone. Each run dir holds `predictions.jsonl` (ids, pair role, category, ground truth
answer, model answer, first-token option log-masses) and `summary.json`. Fresh harness runs also
land here as git-ignored `run_*/` dirs; the tracked `sweep/` and `decode/` trees are the shipped
set.

## `sweep/` (30 runs)

`{dataset}_{selector}_k{budget}_native`: both benchmarks x {uniform, random, hornet} x
k in {1, 4, 8, 16, 24}, native answer resolution, 32-candidate pool. Roll up:

```
python scripts/make_table.py results/sweep
```

## `decode/` (16 runs)

Real and perturbed arms (uniform k24, native, logits logged). Frames are selected before
perturbation, so a real arm and a perturbed arm share the exact frame set.

| | TimeBlind (2400) | MotionBlind (388) |
|---|---|---|
| unperturbed | `timeblind_real` | `motionblind_real` |
| shuffled (4 seeds) | `timeblind_shuffle_seed1..4` | `motionblind_shuffle_seed1..4` |
| reversed | `timeblind_reverse` | `motionblind_reverse` |
| segment-swapped | `timeblind_segswap` | `motionblind_segswap` |
| block-shuffled | `timeblind_blockshuffle` | `motionblind_blockshuffle` |

## Reproducing the published numbers

```
python scripts/pair_decode.py \
    --tb-real results/decode/timeblind_real/predictions.jsonl \
    --tb-pert results/decode/timeblind_shuffle_seed1/predictions.jsonl \
              results/decode/timeblind_shuffle_seed2/predictions.jsonl \
              results/decode/timeblind_shuffle_seed3/predictions.jsonl \
              results/decode/timeblind_shuffle_seed4/predictions.jsonl \
    --mb-real results/decode/motionblind_real/predictions.jsonl

python scripts/contrastive_decode.py \
    --real results/decode/timeblind_real/predictions.jsonl \
    --pert results/decode/timeblind_reverse/predictions.jsonl
# MotionBlind: same, with motionblind_real vs each motionblind_* arm

python scripts/motioncd_decode.py \
    --real results/decode/timeblind_real/predictions.jsonl \
    --pert results/decode/timeblind_reverse/predictions.jsonl --dataset timeblind
# and --dataset motionblind with the motionblind arms
```

## Regenerating on a GPU

Each `summary.json` records its recipe; the matching config in `configs/` reproduces it once
`dataset.root` points at the data (see `slurm/`). Two caveats:

- **Precision.** Published runs are bf16 on H200. A different Ampere-class GPU can shift a few
  borderline items; pre-Ampere cards fall back to fp16 and will not match exactly.
- **Timestamps.** Eagle2.5's bundled code drops caller-supplied timestamps on the frame-list
  path (prompts show `-1.00s`). The published runs patched the cached model code; without that
  patch, frame-selection numbers will not match. The full-video path is unaffected.

Question text in these files comes from the TimeBlind and MotionBlind benchmarks and remains
under their licenses.
