# counterfactual-video-eval

Evaluation harness for the frozen **Eagle2.5-8B** model on two counterfactual video
benchmarks: **TimeBlind** (2400 questions, temporal order) and **MotionBlind** (388
questions, physical motion). Each question pairs two clips with opposite ground truth
answers by construction. Accuracy thus measures whether the model reads the queried
signal, not a static shortcut. The decision-rule scripts run offline on the logged logits
of the first answer token: pairwise argmax, contrastive decoding, and an exact MotionCD
replication. No fine-tuning anywhere.

## Layout

```
run.py       CLI entrypoint (python run.py --config configs/<name>.yaml)
src/         harness: datasets, models, selectors, metrics, runner
configs/     experiment YAMLs, one per experiment per benchmark
scripts/     offline decision-rule and table scripts
slurm/       one job script per experiment
results/     logged predictions of the published runs, for offline reproduction
analysis/    committed figures, generated from results/ by the plot scripts
HORNet/      the authors' HORNet repo (cloned dependency, gitignored; see below)
```

## Quick start

```
pip install -r requirements.txt
python run.py --config configs/smoke.yaml            # mock model + data: no GPU, no downloads
```

The config `configs/smoke_decode.yaml` tests the decode path the same way (4-role layout
and logits):

```
python run.py --config configs/smoke_decode.yaml
python run.py --config configs/smoke_decode.yaml --set 'eval.perturb={"mode":"shuffle","seed":1234}'
python scripts/contrastive_decode.py --real REAL/predictions.jsonl --pert PERT/predictions.jsonl
```

## Running evaluations

Real runs need a GPU, the gated `nvidia/Eagle2.5-8B` weights, and local benchmark data.
Point `dataset.root` at the benchmark directory, then:

```
python run.py --config configs/eagle_timeblind_sweep_native.yaml
```

Each run writes `predictions.jsonl` and `summary.json` under `results/`. Useful flags:
`--limit N`, `--selector uniform|random|hornet`, `--wandb-mode disabled`,
`--set section.field=value`. `scripts/compare_runs.py` compares two runs with paired
statistics.

At model load, the harness patches Eagle's cached remote code in the HF cache. The patch
makes pre-extracted frame lists carry true per-frame timestamps in the prompt; the
upstream code hardcodes -1.00 s on that path. The patch is idempotent. Details are in
`src/models/eagle.py`.

### HORNet frame selector

The setting `frame_source: hornet` runs the authors' own HORNet code from a `HORNet/`
directory at the repo root. It needs `torch` and `transformers`; both are in the
requirements.

```
git clone https://github.com/ostadabbas/HORNet.git HORNet
python run.py --config configs/eagle_timeblind_sweep_native.yaml \
  --set video.frame_source=hornet --set video.top_k=8 \
  --set video.hornet_checkpoint=/path/to/hornet/checkpoint-0.550.pt
```

`python scripts/verify_hornet.py [ckpt] [clips...]` checks that the selector loads and
selects. `scripts/verify_hornet_parity.py` compares the harness selection against the
authors' own `select_frames` pipeline, per clip and per frame budget. It writes
`results/hornet_parity.json`. `scripts/plot_hornet_parity.py` draws the figures in
`analysis/figs/`. The full verification page, with the evidence and the figures, is
`analysis/hornet_verification.md`. The sweep driver reads the checkpoint from
`HORNET_CKPT`. `uniform` and `random` need no setup.

## Decision-rule scripts

All scripts run offline on the logged first-token option logits. The perturbation configs
produce the inputs. The harness selects the frames before the perturbation, so the real
and perturbed arms share the same frames. Metrics: Acc (per item), Q_Acc / V_Acc (both
items of a question / video pair correct), I_Acc (all four roles of a group correct),
sticky (the same answer for both members of a pair).

- `contrastive_decode.py` compares the real arm against a perturbed arm per item. It
  scores the pure logit difference, the VCD-style weighted contrast, and the
  plausibility-floor rules, with a per-category breakdown. It works on either benchmark.
- `pair_decode.py` applies pairwise argmax within each counterfactual pair. It runs
  TimeBlind and MotionBlind in one call, with paired significance (McNemar and clustered
  bootstrap).
- `motioncd_decode.py` runs the exact first-token MotionCD rule (alpha=20, beta=0.1). It
  asserts the rule equal to its reduced form and compares it against the native, pure,
  and floor rules. `--dataset` picks the benchmark.
- `paired_stats.py` holds the shared McNemar and bootstrap helpers. `make_table.py` makes
  the sweep CSV rollup.

## Reproducing the results

`results/` ships the predictions of all 46 published runs. Every reported number
recomputes offline with no GPU, weights, or videos. The full command set is in
`results/README.md`; for example:

```
python scripts/make_table.py results/sweep
python scripts/contrastive_decode.py \
    --real results/decode/timeblind_real/predictions.jsonl \
    --pert results/decode/timeblind_reverse/predictions.jsonl
```

To regenerate runs on a GPU cluster, `slurm/` has one self-contained script per
experiment; see `slurm/README.md`.
