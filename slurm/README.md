# SLURM replication scripts

One self-contained script per experiment. Each script produces its runs and prints the
tables of that experiment. Submit from the repo root. Adjust the `#SBATCH` directives for
your site.

| Script | Reproduces |
|---|---|
| `frame_selection_sweep.sh` | baseline sweep + invariance rate (30-cell array) |
| `pairwise_argmax.sh` | pairwise argmax |
| `contrastive_decoding.sh` | contrastive decoding |
| `motioncd.sh` | MotionCD comparison |
| `verify_hornet_parity.sh` | HORNet selection parity and planted-frame probe (CPU only: the policy is small and Eagle never loads); writes `results/hornet_parity.json` |

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

The three decode scripts also run without a scheduler (`bash slurm/motioncd.sh`). Only
the sweep needs `sbatch` for its array. The sweep writes to `results/`, the decode arms
write to `arms/`. The published runs used bf16 on H200. Request an Ampere-or-newer GPU
for an exact reproduction; pre-Ampere cards fall back to fp16 and will not match bit for
bit.
