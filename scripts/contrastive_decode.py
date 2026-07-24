#!/usr/bin/env python3
"""Video-side contrastive decoding. Combine a REAL run's and a signal-PERTURBED run's
first-token option logit masses per item, decide contrastively, and report all four TimeBlind
accuracies (Acc / Q / V / I). Handles both halves: yes/no items use logit_yes/logit_no, A/B
(multiple_choice) items use logit_a/logit_b (auto per item). Pass --category to score one half only.

Two decision rules:
  * pure difference (parameter-free): option1 iff log-odds(real) > log-odds(perturbed) — the perturbed
    run carries the shared shortcuts (static content, language prior, answer bias), so the difference
    cancels them and leaves the queried signal. No tuning, nothing to overfit.
  * VCD contrastive (alpha, beta): (1+alpha)*lo_real - alpha*lo_pert > 0, with an adaptive-plausibility
    floor beta (only contrast where the real run is uncertain). alpha, beta selected on a held-out
    split of GROUPS. alpha=0 is the real baseline; alpha->inf -> pure difference.
  * plausibility-floor fallback (tau): where the perturbation barely moves the logit
    (|lo_real - lo_pert| < tau — a perturbation-INVARIANT relation, so the contrast is degenerate),
    keep the native answer; else pure difference. tau selected on the same held-out group split.
    tau=0 recovers the pure difference; tau->inf recovers the native baseline. Guards the reverse /
    segment-swap arms, where reversal-invariant properties (duration, co-occurrence) give lo_real ~
    lo_pert and an unguarded pure difference would decide on noise.

The diagnostic reports whether the perturbation shifts the logits — if not, CD cannot help. A
|Δlog-odds|-stratified table (and a per-question-type split) then shows where the contrast has signal
(large |Δlo|, perturbation-sensitive relations) vs where it degenerates (small |Δlo|).

    python scripts/contrastive_decode.py --real REAL/predictions.jsonl --pert PERT/predictions.jsonl
    python scripts/contrastive_decode.py ... --category yes_no        # one half only
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import timeblind_pair_metrics   # noqa: E402


def load(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                d[o["id"]] = o
    return d


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def item_lo(rec):
    """(log-odds of option1, gold_is_option1). option1 = 'yes' (yes/no) or 'A' (multiple_choice)."""
    if rec.get("category") == "multiple_choice":
        return rec["logit_a"] - rec["logit_b"], str(rec["answer"]).strip().lower() == "a"
    return rec["logit_yes"] - rec["logit_no"], str(rec["answer"]).strip().lower() == "yes"


def choose1(lo_r, lo_p, alpha, beta):
    """Return True iff option1 is chosen. beta = adaptive-plausibility floor (VCD): where the real run
    is confident (an option's prob < beta*max), keep the real argmax; else contrast."""
    p1 = sigmoid(lo_r)
    p2 = 1.0 - p1
    mx = max(p1, p2)
    if p1 >= beta * mx and not (p2 >= beta * mx):
        return True
    if p2 >= beta * mx and not (p1 >= beta * mx):
        return False
    return (1.0 + alpha) * lo_r - alpha * lo_p > 0


def choose1_floor(lo_r, lo_p, tau):
    """Plausibility-floor fallback: return True iff option1 is chosen under the |Δlog-odds|
    floor. Where the perturbation barely moved the logit (|lo_real - lo_pert| < tau, a
    perturbation-INVARIANT relation whose contrast is degenerate), keep the native argmax
    (lo_real > 0); otherwise apply the parameter-free pure difference (lo_real - lo_pert > 0).
    tau=0 => always pure (|Δ| >= 0); tau=inf => always native."""
    if abs(lo_r - lo_p) < tau:
        return lo_r > 0
    return lo_r - lo_p > 0


def _metrics_for(real, lop, ids, pick):
    """The four TimeBlind accuracies + sticky for a decision function pick(lo_real, lo_pert)->bool,
    through the shipped timeblind_pair_metrics. Shared by every rule below."""
    recs, correct = [], 0
    for i in ids:
        r = real[i]
        lo_r, gold1 = item_lo(r)
        pick1 = pick(lo_r, lop[i])
        ok = pick1 == gold1
        correct += ok
        recs.append({"instance_id": r["instance_id"], "role": r["role"], "category": r.get("category"),
                     "correct": ok, "error": False, "parsed_choice": 1 if pick1 else 0})
    m = timeblind_pair_metrics(recs)
    g = lambda k: m.get(k) if m.get(k) is not None else float("nan")
    return {"acc": correct / len(ids), "q_acc": g("q_acc"), "v_acc": g("v_acc"),
            "i_acc": g("i_acc"), "sticky": g("sticky_rate")}


def four_acc(real, lop, ids, alpha, beta):
    return _metrics_for(real, lop, ids, lambda lo_r, lo_p: choose1(lo_r, lo_p, alpha, beta))


def four_acc_floor(real, lop, ids, tau):
    return _metrics_for(real, lop, ids, lambda lo_r, lo_p: choose1_floor(lo_r, lo_p, tau))


def _item_acc(real, lop, ids, pick):
    """Item-level accuracy for a decision function — always defined, unlike the group metrics which
    drop incomplete groups; used for the |Δlog-odds|-stratified bins that split groups."""
    if not ids:
        return float("nan")
    c = 0
    for i in ids:
        lo_r, gold1 = item_lo(real[i])
        c += (pick(lo_r, lop[i]) == gold1)
    return c / len(ids)


def stratify_shift(real, lop, ids, edges):
    """Diagnostic: bin items by |Δlog-odds| = |lo_real - lo_pert| (reversal / segment-swap
    move the logit a lot on perturbation-SENSITIVE relations, ~0 on invariant ones) and report native
    vs pure-difference item-accuracy per bin — where the contrast has signal vs where it degenerates.
    edges: ascending interior boundaries; bins are [0,e0), [e0,e1), ..., [e_last, inf)."""
    cuts = [0.0] + list(edges) + [float("inf")]
    nat = lambda lo_r, lo_p: lo_r > 0
    pur = lambda lo_r, lo_p: lo_r - lo_p > 0
    rows = []
    for lo_e, hi_e in zip(cuts[:-1], cuts[1:]):
        sub = [i for i in ids if lo_e <= abs(item_lo(real[i])[0] - lop[i]) < hi_e]
        row = {"lo": lo_e, "hi": hi_e, "n": len(sub)}
        if sub:
            row["mean"] = sum(abs(item_lo(real[i])[0] - lop[i]) for i in sub) / len(sub)
            row["nat_acc"] = _item_acc(real, lop, sub, nat)
            row["pur_acc"] = _item_acc(real, lop, sub, pur)
        rows.append(row)
    return rows


def fmt(d):
    return (f"Acc={d['acc']:.3f}  Q={d['q_acc']:.3f}  V={d['v_acc']:.3f}  "
            f"I={d['i_acc']:.3f}  sticky={d['sticky']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="real-video run predictions.jsonl")
    ap.add_argument("--pert", required=True, nargs="+",
                    help="perturbed-video run predictions.jsonl; pass several => average lo_pert "
                         "over them (shuffle-ensembling / variance reduction)")
    ap.add_argument("--optimize", default="v_acc", choices=["v_acc", "i_acc", "acc", "q_acc"])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--category", help="keep only items with this category (yes_no | multiple_choice); "
                                       "omit to decode the whole benchmark (both halves)")
    args = ap.parse_args()

    real = load(args.real)
    perts = [load(p) for p in args.pert]
    if args.category:
        real = {k: v for k, v in real.items() if v.get("category") == args.category}
        perts = [{k: v for k, v in p.items() if v.get("category") == args.category} for p in perts]
    shared = set(real)
    for p in perts:
        shared &= set(p)
    ids = sorted(shared, key=lambda x: int(x) if str(x).isdigit() else x)
    assert ids, "no shared item ids across the runs"
    # shuffle-ensembling: average lo_pert over the perturbed runs (variance reduction)
    lop = {i: sum(item_lo(p[i])[0] for p in perts) / len(perts) for i in ids}

    lo = lambda d, i: item_lo(d[i])[0]
    shifts = sorted(abs(lo(real, i) - lop[i]) for i in ids)
    flip = sum((lo(real, i) > 0) != (lop[i] > 0) for i in ids) / len(ids)
    print(f"n={len(ids)} shared items"
          + (f" (category={args.category})" if args.category else " (both halves)"))
    print(f"DIAGNOSTIC — perturbation logit shift: mean|Δlog-odds|={sum(shifts)/len(shifts):.2f} "
          f"median={shifts[len(shifts)//2]:.2f}; real-argmax flip rate={flip:.3f}")
    print("  (near 0 => the perturbation didn't move the model => CD has no signal to amplify)\n")

    gids = sorted({real[i]["instance_id"] for i in ids})
    rng = random.Random(args.split_seed)
    rng.shuffle(gids)
    half = set(gids[:len(gids) // 2])
    train = [i for i in ids if real[i]["instance_id"] in half]
    test = [i for i in ids if real[i]["instance_id"] not in half]

    base = four_acc(real, lop, test, 0.0, 1.0)
    pure = four_acc(real, lop, test, 1e9, 0.0)
    pure_full = four_acc(real, lop, ids, 1e9, 0.0)

    alphas = [0, 0.25, 0.5, 1, 1.5, 2, 3, 5]
    betas = [0.0, 0.05, 0.1, 0.2, 0.5]
    best = None
    for a in alphas:
        for b in betas:
            v = four_acc(real, lop, train, a, b)[args.optimize]
            if best is None or (v == v and v > best[0]):
                best = (v, a, b)
    _, a, b = best
    te = four_acc(real, lop, test, a, b)

    print("baseline (real argmax), held-out test:")
    print(f"  {fmt(base)}")
    print("\nPURE DIFFERENCE (parameter-free), held-out test:")
    print(f"  {fmt(pure)}   [full set: {fmt(pure_full)}]")
    print(f"\nVCD (alpha={a}, beta={b}; selected on train to max {args.optimize}), held-out test:")
    print(f"  {fmt(te)}")

    print(f"\nalpha curve at beta={b} (full set, {args.optimize}); alpha->inf approaches pure diff:")
    for a2 in alphas + [1e9]:
        tag = " pure" if a2 == 1e9 else f"{a2:>5}"
        print(f"  alpha={tag}: {args.optimize}={four_acc(real, lop, ids, a2, b)[args.optimize]:.3f}")

    # plausibility-floor fallback: below |Δlo|<tau keep the native answer (the perturbation
    # barely moved the logit -> a perturbation-invariant relation, degenerate contrast). tau selected
    # on the SAME held-out split of groups as the VCD alpha/beta.
    taus = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 1e18]
    btau, bval = None, None
    for t in taus:
        v = four_acc_floor(real, lop, train, t)[args.optimize]
        if bval is None or (v == v and v > bval):
            bval, btau = v, t
    tau_tag = "inf" if btau >= 1e18 else f"{btau:g}"
    print(f"\nPLAUSIBILITY-FLOOR FALLBACK (|Δlo|<tau -> keep native; tau={tau_tag} selected on train "
          f"to max {args.optimize}), held-out test:")
    print(f"  {fmt(four_acc_floor(real, lop, test, btau))}   "
          f"[full set: {fmt(four_acc_floor(real, lop, ids, btau))}]")

    # |Δlog-odds|-stratified breakdown: the contrast should help only where reversal /
    # segment-swap actually moved the logit; small |Δlo| = perturbation-invariant = degenerate.
    print("\n|Δlog-odds|-STRATIFIED (full set) — native vs pure-difference item-accuracy per bin:")
    print(f"  {'|Δlo| bin':>13}  {'n':>5}  {'mean':>5}  {'nat Acc':>8}  {'pure Acc':>8}  {'Δ':>7}")
    for row in stratify_shift(real, lop, ids, [0.25, 0.5, 1.0, 2.0]):
        hi = "inf" if row["hi"] == float("inf") else f"{row['hi']:.2f}"
        rng_s = f"[{row['lo']:.2f},{hi})"
        if not row["n"]:
            print(f"  {rng_s:>13}  {0:>5}  {'-':>5}  {'-':>8}  {'-':>8}  {'-':>7}")
            continue
        d = row["pur_acc"] - row["nat_acc"]
        print(f"  {rng_s:>13}  {row['n']:>5}  {row['mean']:>5.2f}  {row['nat_acc']:>8.3f}  "
              f"{row['pur_acc']:>8.3f}  {d:>+7.3f}")

    # per-question-type split (category), when both halves are present
    cats = sorted({real[i].get("category") for i in ids} - {None})
    if len(cats) > 1:
        print("\nper-question-type (category) — native vs pure-difference:")
        for cat in cats:
            cids = [i for i in ids if real[i].get("category") == cat]
            print(f"  {cat} (n={len(cids)}): native {fmt(four_acc(real, lop, cids, 0.0, 1.0))}")
            print(f"  {' ' * len(cat)}          pure   {fmt(four_acc(real, lop, cids, 1e9, 0.0))}")


if __name__ == "__main__":
    main()
