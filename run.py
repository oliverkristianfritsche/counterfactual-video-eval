#!/usr/bin/env python3
"""CLI entrypoint for the eval harness.

Examples:
    python run.py --config configs/smoke.yaml                              # mock end-to-end, wandb off
    python run.py --config configs/eagle_timeblind_sweep_native.yaml       # frame-selection sweep
    python run.py --config configs/eagle_timeblind_sweep_native.yaml --limit 20 --wandb-mode offline
    python run.py --config configs/eagle_timeblind_sweep_native.yaml --selector random   # frame-source override
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.runner import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="eval — uniform VLM benchmark harness")
    ap.add_argument("--config", required=True, help="path to an experiment YAML")
    ap.add_argument("--limit", type=int, default=None, help="cap number of samples")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N samples (chunked runs; keep multiples of 4 on TimeBlind)")
    ap.add_argument("--model", default=None, help="override model.kind (e.g. mock, eagle)")
    ap.add_argument("--dataset", default=None, help="override dataset.kind")
    ap.add_argument("--selector", default=None, choices=["uniform", "random", "hornet", "all"],
                    help="override video.frame_source")
    ap.add_argument("--seed", type=int, default=None, help="override eval.seed (default 1234)")
    ap.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"],
                    help="override wandb.mode")
    ap.add_argument("--name", default=None, help="override the run name")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="section.field=value",
                    help="generic config override, repeatable (value parsed as JSON, "
                         "falls back to string), e.g. --set video.native_decode=false")
    args = ap.parse_args()

    overrides = {}
    if args.model:
        overrides["model.kind"] = args.model
    if args.dataset:
        overrides["dataset.kind"] = args.dataset
    if args.selector:
        overrides["video.frame_source"] = args.selector
    if args.seed is not None:
        overrides["eval.seed"] = args.seed
    if args.wandb_mode:
        overrides["wandb.mode"] = args.wandb_mode
    if args.name:
        overrides["wandb.name"] = args.name
    import json as _json
    for item in args.sets:
        key, _, val = item.partition("=")
        try:
            overrides[key] = _json.loads(val)
        except _json.JSONDecodeError:
            overrides[key] = val

    run(args.config, limit=args.limit, offset=args.offset, overrides=overrides)


if __name__ == "__main__":
    main()
