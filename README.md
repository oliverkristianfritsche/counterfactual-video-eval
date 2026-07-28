# counterfactual-video-eval

Evaluation harness for frozen **Eagle2.5-8B** on two counterfactual video benchmarks:
**TimeBlind** (2400 questions, temporal order) and **MotionBlind** (388 questions, physical
motion). Each question pairs two clips whose ground truth answers are opposite by construction, so
accuracy measures whether the model reads the queried signal rather than a static shortcut.
Offline decision-rule scripts operate on the logged first-answer-token logits: pairwise argmax,
contrastive decoding, and an exact MotionCD replication. No fine-tuning anywhere.

## Layout

```
run.py       CLI entrypoint (python run.py --config configs/<name>.yaml)
src/         harness: datasets, models, selectors, metrics, runner
configs/     experiment YAMLs, one per experiment per benchmark
scripts/     offline decision-rule and table scripts
slurm/       one job script per experiment
results/     logged predictions of the published runs, for offline reproduction
HORNet/      the authors' HORNet repo (cloned dependency, gitignored; see below)
```

## Quick start

```
pip install -r requirements.txt
python run.py --config configs/smoke.yaml            # mock model + data: no GPU, no downloads
```

`configs/smoke_decode.yaml` smoke-tests the decode path the same way (4-role layout + logits):

```
python run.py --config configs/smoke_decode.yaml
python run.py --config configs/smoke_decode.yaml --set 'eval.perturb={"mode":"shuffle","seed":1234}'
python scripts/contrastive_decode.py --real REAL/predictions.jsonl --pert PERT/predictions.jsonl
```

## Running evaluations

Real runs need a GPU, the gated `nvidia/Eagle2.5-8B` weights, and local benchmark data. Point
`dataset.root` at the benchmark directory, then:

```
python run.py --config configs/eagle_timeblind_sweep_native.yaml
```

Each run writes `predictions.jsonl` + `summary.json` under `results/`. Useful flags:
`--limit N`, `--selector uniform|random|hornet`, `--wandb-mode disabled`,
`--set section.field=value`. `scripts/compare_runs.py` compares two runs with paired statistics.

At model load the harness patches Eagle's cached remote code in the HF cache so
pre-extracted frame lists carry true per-frame timestamps in the prompt (upstream hardcodes
-1.00 s on that path); idempotent, detailed in `src/models/eagle.py`.

### HORNet frame selector

`frame_source: hornet` runs the authors' cloned HORNet implementation from a `HORNet/`
directory at the repo root (needs `torch` + `transformers`, both already in requirements):

```
git clone https://github.com/ostadabbas/HORNet.git HORNet
python run.py --config configs/eagle_timeblind_sweep_native.yaml \
  --set video.frame_source=hornet --set video.top_k=8 \
  --set video.hornet_checkpoint=/path/to/hornet/checkpoint-0.550.pt
```

`python scripts/verify_hornet.py [ckpt] [clips...]` checks the selector loads and selects.
The sweep driver reads the checkpoint from `HORNET_CKPT`. `uniform` and `random` need no setup.

## Decision-rule scripts

All run offline on logged first-token option logits. The perturbation configs produce the
inputs; frames are selected before perturbation, so real and perturbed arms share frames.
Metrics: Acc (per item), Q_Acc / V_Acc (both items of a question / video pair correct),
I_Acc (all four roles of a group correct), sticky (same answer for both members of a pair).

- `contrastive_decode.py`: real arm vs perturbed arm per item. Pure logit difference,
  VCD-style weighted contrast, and plausibility-floor rules, with a per-category breakdown.
  Works on either benchmark.
- `pair_decode.py`: pairwise argmax within each counterfactual pair, TimeBlind and MotionBlind
  in one call, with paired significance (McNemar + clustered bootstrap).
- `motioncd_decode.py`: exact first-token MotionCD (alpha=20, beta=0.1), asserted equal to its
  reduced form, compared against the native/pure/floor rules. `--dataset` picks the benchmark.
- `paired_stats.py`: shared McNemar + bootstrap helpers. `make_table.py`: sweep CSV rollup.

## Reproducing the results

`results/` ships the predictions of all 46 published runs; every reported number recomputes
offline with no GPU, weights, or videos. Full command set in `results/README.md`; for example:

```
python scripts/make_table.py results/sweep
python scripts/contrastive_decode.py \
    --real results/decode/timeblind_real/predictions.jsonl \
    --pert results/decode/timeblind_reverse/predictions.jsonl
```

To regenerate runs on a GPU cluster, `slurm/` has one self-contained script per experiment;
see `slurm/README.md`.
