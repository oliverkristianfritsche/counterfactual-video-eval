"""Paired comparison of two runs on the same items (any A/B of run configs).

Two runs that differ in one setting (tiles, k, sampler) answer the SAME items, so the comparison is
paired: report the accuracy delta with an exact McNemar on the discordant cells and a bootstrap CI
clustered on the counterfactual pair (the 4 roles of a MotionBlind pair share clips and are not
independent draws). Comparing two independent Wilson intervals is the wrong test and is badly
underpowered.

Items present in only one run are dropped (counted), and per-category deltas are reported alongside
the overall one. Per-category MotionBlind results are exploratory (~11-12 pairs each).

  python scripts/compare_runs.py results/run_BASE/predictions.jsonl results/run_NEW/predictions.jsonl \
      --labels tiles1 tiles4 --out results/tiles_t1_vs_t4.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_stats import delta_bootstrap, mcnemar_exact  # noqa: E402  (shared paired stats)


def load(path):
    """predictions.jsonl -> {id: row} with correctness, category and pair for clustering."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            out[str(r.get("id"))] = {
                "id": str(r.get("id")),
                "correct": bool(r.get("correct")),
                "prediction": str(r.get("prediction")).strip(),
                "gold": str(r.get("answer")),
                "category": r.get("category") or "all",
                "pair": r.get("instance_id"),
            }
    return out


def compare(base, new, categories=None):
    ids = sorted(set(base) & set(new), key=lambda x: (len(x), x))
    if categories:
        ids = [i for i in ids if base[i]["category"] in categories]
    rows = [{"id": i, "gold": 1, "pair": base[i]["pair"], "category": base[i]["category"]} for i in ids]
    dec_a = {i: int(base[i]["correct"]) for i in ids}   # "decision == gold" iff correct
    dec_b = {i: int(new[i]["correct"]) for i in ids}
    mc, bs = mcnemar_exact(rows, dec_a, dec_b), delta_bootstrap(rows, dec_a, dec_b)
    same = sum(1 for i in ids if base[i]["prediction"].lower() == new[i]["prediction"].lower())
    return {
        "n": len(ids),
        "acc_base": round(sum(dec_a.values()) / len(ids), 4) if ids else None,
        "acc_new": round(sum(dec_b.values()) / len(ids), 4) if ids else None,
        "identical_predictions": f"{same}/{len(ids)}",
        **bs, **mc,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired run-vs-run comparison")
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--labels", nargs=2, default=["base", "new"])
    ap.add_argument("--categories", default=None, help="comma list to restrict")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base, new = load(args.base), load(args.new)
    cats = set(args.categories.split(",")) if args.categories else None
    only_base, only_new = len(set(base) - set(new)), len(set(new) - set(base))
    if only_base or only_new:
        print(f"[compare] {only_base} items only in {args.labels[0]}, {only_new} only in "
              f"{args.labels[1]} — dropped", file=sys.stderr)

    overall = compare(base, new, cats)
    per_cat = {}
    for c in sorted({r["category"] for r in base.values()} if not cats else cats):
        blk = compare(base, new, {c})
        if blk["n"]:
            per_cat[c] = blk

    a, b = args.labels
    print(f"{'set':<22} {'n':>4} {a:>8} {b:>8} {'delta':>8} {'ci95':>18} {'McNemar p':>10} {'identical':>10}")
    def line(name, r):
        print(f"{name:<22} {r['n']:>4} {r['acc_base']:>8.4f} {r['acc_new']:>8.4f} {r['delta']:>+8.4f} "
              f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] {r['p']:>10} {r['identical_predictions']:>10}")
    line("OVERALL", overall)
    for c, r in per_cat.items():
        line(f"  {c}", r)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"base": args.base, "new": args.new, "labels": args.labels,
             "overall": overall, "by_category": per_cat}, indent=2))
        print(f"[compare] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
