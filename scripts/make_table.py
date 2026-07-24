#!/usr/bin/env python3
"""Aggregate each results/run_*/ and results/merged_*/ summary.json for the frame-selection sweep
into one CSV row per cell (dataset, selector, sampling, and the accuracy columns).

Re-runs that also sit in results/ for their own analysis (contrastive, calibration, and
perturbation arms) and local mock smoke runs are excluded so the rollup stays the sweep table.

Usage: python scripts/make_table.py [results_dir] [run_name_filter]
       results_dir defaults to results/; run_name_filter is an optional substring on run_name.
Writes <results_dir>/TABLE.csv (and echoes it to stdout).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def load_rows(results: Path) -> list[dict]:
    rows = []
    # any immediate subdirectory holding a summary.json: the timestamped run_*/merged_* dirs a live
    # run writes, and the descriptively-named results/ directories shipped with the repo.
    for p in sorted(results.glob("*/summary.json")):
        s = json.loads(p.read_text())
        m = s.get("metrics", {})
        rows.append({
            "dataset": s.get("dataset") or "?",
            "model": s.get("model"),
            "sampling": s.get("sampling"),
            "selector": s.get("frame_source"),
            "task": s.get("task"),
            "technique": (s.get("experiment") or {}).get("technique"),
            "acc": m.get("accuracy") if s.get("task") in ("choice", "yesno") else m.get("f1_lev"),
            "q_acc": m.get("q_acc"),
            "v_acc": m.get("v_acc"),
            "i_acc": m.get("i_acc", m.get("instance_accuracy")),
            "n": m.get("total"),
            "invalid": m.get("invalid_rate"),
            "errors": m.get("errors"),
            "sel_s": m.get("frame_sel_s/mean"),
            "vlm_s": m.get("vlm_proc_s/mean"),
            "frames": m.get("avg_frames"),
            "run": s.get("run_name"),
        })
    return rows


def is_rollup(r: dict) -> bool:
    """A row belongs in the sweep rollup iff it is a frame-selection sweep cell (technique
    'frame-selection sweep[...]'). Everything else in results/ — contrastive, calibration, and
    perturbation re-runs (technique 'video-perturbation-*'), mock smoke — is a per-experiment run
    with its own artifact and is dropped from this table."""
    return (r.get("technique") or "").startswith("frame-selection sweep")


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results"
    only = sys.argv[2] if len(sys.argv) > 2 else None  # substring filter on run_name
    found = load_rows(results)
    rows = [r for r in found if is_rollup(r)]  # sweep rollup only
    if only:
        rows = [r for r in rows if only in (r.get("run") or "")]
    if not rows:
        if found:
            print(f"{len(found)} run(s) under {results}, none in the sweep rollup"
                  + (f" matching '{only}'" if only else ""))
        else:
            print(f"no run_*/summary.json under {results}")
        return

    cols = ["dataset", "model", "selector", "sampling", "task", "technique",
            "acc", "q_acc", "v_acc", "i_acc", "invalid", "errors", "sel_s", "vlm_s", "frames", "n"]
    rows = sorted(rows, key=lambda x: (str(x["dataset"]), str(x["selector"]), str(x["sampling"])))

    out_path = results / "TABLE.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") if r.get(c) is not None else "" for c in cols])

    # echo the same rows to stdout so a run is readable without opening the file
    print(",".join(cols))
    for r in rows:
        print(",".join(str(r.get(c, "") if r.get(c) is not None else "") for c in cols))
    print(f"\n(written to {out_path})")


if __name__ == "__main__":
    main()
