# SLURM replication scripts

One self-contained script per experiment; each produces its runs and prints that experiment's
tables. Submit from the repo root; adjust the `#SBATCH` directives for your site.

| Script | Reproduces |
|---|---|
| `frame_selection_sweep.sh` | baseline sweep + invariance rate (30-cell array) |
| `pairwise_argmax.sh` | pairwise argmax |
| `contrastive_decoding.sh` | contrastive decoding |
| `motioncd.sh` | MotionCD comparison |

| Variable | Meaning |
|---|---|
| `PYTHON` | interpreter with the deps installed (default `python`) |
| `TIMEBLIND_ROOT` / `MOTIONBLIND_ROOT` | benchmark data dirs |
| `HF_HOME` | optional HF cache; Eagle2.5-8B (~16 GB) downloads here |
| `HORNET_CKPT` | HORNet checkpoint path (hornet sweep cells only) |

```
sbatch slurm/frame_selection_sweep.sh    # hornet cells are array 10-14 (TB), 25-29 (MB)
sbatch slurm/pairwise_argmax.sh
sbatch slurm/contrastive_decoding.sh
sbatch slurm/motioncd.sh
```

The three decode scripts run fine without a scheduler (`bash slurm/motioncd.sh`); only the sweep
needs `sbatch` for its array. The sweep writes to `results/`, decode arms to `arms/`. Published
runs used bf16 on H200; request an Ampere-or-newer GPU for an exact reproduction (pre-Ampere
falls back to fp16 and will not match bit for bit).
