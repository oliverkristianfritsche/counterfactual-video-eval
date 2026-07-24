#!/usr/bin/env python3
"""MotionCD (MHBench, AAAI 2025) exact-rule replication, offline, first-token.

MotionCD (Motion Contrastive Decoding) contrasts a VideoLLM's next-token logits under the original
video against the same video played BACKWARDS (the "amateur"), then keeps only the tokens the original
run finds plausible. Source of truth is the released code
(reference/motioncd/baselines/video_mcd_sample.py, lines 117-126), which implements the paper's
simplified form (eq. 5/7):

    alpha_eff = mcd_alpha / 2                                  # the /2 is the folded (orig+rev)/2 amateur
    cutoff    = log(mcd_beta) + max_w logit_orig(w)            # adaptive plausibility constraint (APC)
    diffs     = (1 + alpha_eff) * logit_orig - alpha_eff * logit_rev
    mcd_logit = diffs where logit_orig >= cutoff, else -inf    # tokens below the APC are removed
    next_token = argmax(mcd_logit)

with the released defaults mcd_alpha=20 (so alpha_eff=10), mcd_beta=0.1. Our benchmark answers are a
single option token, so generation-level contrast reduces to a first-token contrast over the two option
logits (yes/no or A/B). Under the standard forced-choice assumption that the model's top-1 first token
is one of the two option tokens, max_w logit_orig(w) = max(l1, l2), and the APC removes the LOSING
option exactly when |log-odds_orig| > log(1/beta) (= 2.3026 for beta=0.1) — i.e. MotionCD keeps the
native argmax whenever the original run is confident, and applies the contrast otherwise. That makes
MotionCD's reduced rule identical to our VCD choose1(alpha=alpha_eff, beta=mcd_beta); the
literal per-option implementation here is asserted equal to it item by item so the reduction is
auditable, not assumed.

Four rules compared on identical logged inputs (TimeBlind full-2400 / MotionBlind full-388):
  * native      — real argmax (option1 iff lo_real > 0).
  * pure        — our parameter-free pure difference (option1 iff lo_real - lo_rev > 0).
  * floor       — our plausibility-floor variant: |lo_real - lo_rev| < tau -> keep native,
                  else pure; tau selected on a held-out split of GROUPS (seed 1234) to max I_Acc.
  * motioncd    — MotionCD's EXACT rule at their defaults (mcd_alpha=20 -> alpha_eff=10, mcd_beta=0.1).

Reports all four accuracies (Acc / Q_Acc / V_Acc / I_Acc) + sticky per rule, and the paired exact
McNemar + group-clustered bootstrap (seed 1234, clustered on instance_id) for MotionCD vs pure — the
question the replication answers: does their formulation beat/match ours on counterfactual QA?

    python scripts/motioncd_decode.py --real REAL.jsonl --pert REVERSE.jsonl --dataset timeblind \
        --out results/motioncd_decode.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import timeblind_pair_metrics                             # noqa: E402
from scripts.contrastive_decode import (choose1, choose1_floor, four_acc,          # noqa: E402
                                        four_acc_floor, item_lo, load)
from scripts.paired_stats import delta_bootstrap, mcnemar_exact                    # noqa: E402

TAUS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 1e18]
MCD_ALPHA_DEFAULT = 20.0   # main.py / run.sh default; halved to alpha_eff=10 inside the sampler
MCD_BETA_DEFAULT = 0.1     # adaptive plausibility constraint truncation


def item_logits(rec):
    """(logit_option1, logit_option2, gold_is_option1). option1 = 'yes' (yes/no) or 'A' (mult-choice).
    Mirrors contrastive_decode.item_lo but keeps the two option logits separate, which MotionCD's
    adaptive-plausibility cutoff needs (it compares each option logit against log(beta)+max)."""
    if rec.get("category") == "multiple_choice":
        return rec["logit_a"], rec["logit_b"], str(rec["answer"]).strip().lower() == "a"
    return rec["logit_yes"], rec["logit_no"], str(rec["answer"]).strip().lower() == "yes"


def motioncd_pick1(l1r, l2r, l1p, l2p, mcd_alpha=MCD_ALPHA_DEFAULT, mcd_beta=MCD_BETA_DEFAULT):
    """MotionCD's exact next-token rule (reference/motioncd/baselines/video_mcd_sample.py:117-126),
    reduced to the two option logits. Returns True iff option1 is chosen.

    real = (l1r, l2r) original-video option logits; pert = (l1p, l2p) reversed-video option logits.
    alpha_eff = mcd_alpha/2 (the paper's folded amateur); cutoff = log(mcd_beta) + max(l1r, l2r)
    (the APC, with the full-vocab max standing in as the larger option logit); an option below the
    cutoff is removed (-inf); argmax of the surviving contrasted logits, ties -> option2 (matching
    the pure/native `> 0` convention)."""
    a = mcd_alpha / 2.0
    # cutoff = log(beta) + max(logits); beta<=0 disables the APC (log(0) -> -inf, matching the
    # released torch.log(tensor(0.)) = -inf rather than raising).
    cutoff = -math.inf if mcd_beta <= 0 else math.log(mcd_beta) + max(l1r, l2r)
    d1 = (1.0 + a) * l1r - a * l1p if l1r >= cutoff else -math.inf
    d2 = (1.0 + a) * l2r - a * l2p if l2r >= cutoff else -math.inf
    return d1 > d2


def _pick_metrics(real, pert, ids, pick_rec):
    """Four TimeBlind accuracies + sticky for a per-record decision pick_rec(real_rec, pert_rec)->bool,
    through the shipped timeblind_pair_metrics — the same grouped scorer the other rules use."""
    recs, correct = [], 0
    for i in ids:
        r = real[i]
        _, _, gold1 = item_logits(r)
        pick1 = pick_rec(r, pert[i])
        ok = pick1 == gold1
        correct += ok
        recs.append({"instance_id": r["instance_id"], "role": r["role"], "category": r.get("category"),
                     "correct": ok, "error": False, "parsed_choice": 1 if pick1 else 0})
    m = timeblind_pair_metrics(recs)
    g = lambda k: m.get(k) if m.get(k) is not None else float("nan")
    return {"acc": correct / len(ids), "q_acc": g("q_acc"), "v_acc": g("v_acc"),
            "i_acc": g("i_acc"), "sticky": g("sticky_rate")}


def motioncd_four_acc(real, pert, ids, mcd_alpha=MCD_ALPHA_DEFAULT, mcd_beta=MCD_BETA_DEFAULT):
    """All four accuracies for MotionCD's literal rule."""
    def pick(r, p):
        l1r, l2r, _ = item_logits(r)
        l1p, l2p, _ = item_logits(p)
        return motioncd_pick1(l1r, l2r, l1p, l2p, mcd_alpha, mcd_beta)
    return _pick_metrics(real, pert, ids, pick)


def _decisions(real, pert, lop, ids, tau, mcd_alpha, mcd_beta):
    """Per-item option1/option2 (1/0) decision maps for every rule, keyed by item id, plus gold."""
    gold, native, pure, floor, mcd = {}, {}, {}, {}, {}
    for i in ids:
        r = real[i]
        l1r, l2r, gold1 = item_logits(r)
        l1p, l2p, _ = item_logits(pert[i])
        lo_r, lo_p = item_lo(r)[0], lop[i]
        gold[i] = int(gold1)
        # each map uses the SAME rule as its four_acc / four_acc_floor / motioncd_four_acc row, so the
        # McNemar/bootstrap deltas equal the reported table differences exactly (incl. the tie
        # convention: choose1's alpha->inf residual sends an exact lo_r==lo_p tie to the native argmax).
        native[i] = int(choose1(lo_r, lo_p, 0.0, 1.0))
        pure[i] = int(choose1(lo_r, lo_p, 1e9, 0.0))
        floor[i] = int(choose1_floor(lo_r, lo_p, tau))
        mcd[i] = int(motioncd_pick1(l1r, l2r, l1p, l2p, mcd_alpha, mcd_beta))
    return {"gold": gold, "native": native, "pure": pure, "floor": floor, "motioncd": mcd}


def _instance_correct(real, ids, dec, gold):
    """Per-instance correctness under a decision map: an instance (complete 4-role group) is correct
    iff all four roles are correct — the timeblind_pair_metrics I_Acc definition."""
    groups: dict = {}
    for i in ids:
        r = real[i]
        groups.setdefault(r["instance_id"], {})[r["role"]] = (dec[i] == gold[i])
    return {g: int(all(roles.values())) for g, roles in groups.items() if len(roles) == 4}


def paired_stats(real, ids, dec, seed=1234, n_boot=2000):
    """Paired exact McNemar + group-clustered bootstrap for MotionCD vs pure, at item level (Acc,
    clustered on instance_id) and instance level (I_Acc). dec_a = pure, dec_b = motioncd, so the
    reported delta is MotionCD - pure."""
    a, b, gold = dec["pure"], dec["motioncd"], dec["gold"]
    rows_item = [{"id": i, "gold": gold[i], "pair": real[i]["instance_id"]} for i in ids]
    mc_item = mcnemar_exact(rows_item, b, a)
    bs_item = delta_bootstrap(rows_item, a, b, n_boot=n_boot, seed=seed)   # obs = acc(b) - acc(a)

    ic_a, ic_b = (_instance_correct(real, ids, dec["pure"], gold),
                  _instance_correct(real, ids, dec["motioncd"], gold))
    gids = sorted(ic_a)
    rows_inst = [{"id": g, "gold": 1, "pair": g} for g in gids]
    mc_inst = mcnemar_exact(rows_inst, ic_b, ic_a)
    bs_inst = delta_bootstrap(rows_inst, ic_a, ic_b, n_boot=n_boot, seed=seed)
    return {"acc": {"mcnemar_b_mcd_better": mc_item["b"], "mcnemar_c_pure_better": mc_item["c"],
                    "mcnemar_p": mc_item["p"], **bs_item},
            "i_acc": {"mcnemar_b_mcd_better": mc_inst["b"], "mcnemar_c_pure_better": mc_inst["c"],
                      "mcnemar_p": mc_inst["p"], **bs_inst}}


def select_tau(real, lop, train_ids, optimize="i_acc"):
    best = None
    for t in TAUS:
        v = four_acc_floor(real, lop, train_ids, t)[optimize]
        if best is None or (v == v and v > best[0]):
            best = (v, t)
    return best[1]


def diagnostic(real, lop, ids):
    shifts = sorted(abs(item_lo(real[i])[0] - lop[i]) for i in ids)
    flip = sum((item_lo(real[i])[0] > 0) != (lop[i] > 0) for i in ids) / len(ids)
    return {"n": len(ids), "mean_abs_dlo": sum(shifts) / len(shifts),
            "median_abs_dlo": shifts[len(shifts) // 2], "flip_rate": flip}


def run(real_path, pert_paths, category, mcd_alpha, mcd_beta, seed=1234, n_boot=2000):
    real = load(real_path)
    perts = [load(p) for p in pert_paths]
    if category:
        real = {k: v for k, v in real.items() if v.get("category") == category}
        perts = [{k: v for k, v in p.items() if v.get("category") == category} for p in perts]
    shared = set(real)
    for p in perts:
        shared &= set(p)
    ids = sorted(shared, key=lambda x: int(x) if str(x).isdigit() else x)
    assert ids, "no shared item ids across the runs"
    # average the reversed-arm log-odds if several arms are passed (variance reduction; usually one)
    lop = {i: sum(item_lo(p[i])[0] for p in perts) / len(perts) for i in ids}
    pert0 = perts[0]

    gids = sorted({real[i]["instance_id"] for i in ids})
    rng = random.Random(seed)
    rng.shuffle(gids)
    train = [i for i in ids if real[i]["instance_id"] in set(gids[:len(gids) // 2])]
    tau = select_tau(real, lop, train, "i_acc")

    rules = {
        "native": four_acc(real, lop, ids, 0.0, 1.0),
        "pure": four_acc(real, lop, ids, 1e9, 0.0),
        "floor": four_acc_floor(real, lop, ids, tau),
        "motioncd": motioncd_four_acc(real, pert0, ids, mcd_alpha, mcd_beta),
    }

    # invariant: MotionCD's literal per-option rule equals our choose1(alpha_eff, beta) on the
    # log-odds, item by item (the reduction the notes rely on), and the four accuracies match.
    a_eff = mcd_alpha / 2.0
    for i in ids:
        l1r, l2r, _ = item_logits(real[i])
        l1p, l2p, _ = item_logits(pert0[i])
        lit = motioncd_pick1(l1r, l2r, l1p, l2p, mcd_alpha, mcd_beta)
        red = choose1(item_lo(real[i])[0], lop[i], a_eff, mcd_beta)
        assert lit == red, f"MotionCD literal rule != choose1(alpha_eff, beta) on item {i}"
    mcd_via_choose1 = four_acc(real, lop, ids, a_eff, mcd_beta)
    for k in ("acc", "q_acc", "v_acc", "i_acc"):
        if rules["motioncd"][k] == rules["motioncd"][k]:  # skip NaN
            assert abs(rules["motioncd"][k] - mcd_via_choose1[k]) <= 1e-9, \
                f"MotionCD four_acc {k} != choose1 reduction"

    dec = _decisions(real, pert0, lop, ids, tau, mcd_alpha, mcd_beta)
    stats = paired_stats(real, ids, dec, seed=seed, n_boot=n_boot)

    return {"real": str(real_path), "pert": [str(p) for p in pert_paths],
            "category": category or "all", "n_items": len(ids), "seed": seed,
            "mcd_alpha": mcd_alpha, "mcd_alpha_eff": a_eff, "mcd_beta": mcd_beta, "tau": tau,
            "apc_native_threshold_logodds": -math.log(mcd_beta),
            "diagnostic": diagnostic(real, lop, ids), "rules": rules,
            "motioncd_vs_pure": stats}


def _fmt(d):
    return (f"Acc={d['acc']:.4f} Q={d['q_acc']:.4f} V={d['v_acc']:.4f} "
            f"I={d['i_acc']:.4f} st={d['sticky']:.2f}")


def _pfmt(p):
    return "<1e-6" if p == 0.0 else f"{p:.2e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="original-video run predictions.jsonl")
    ap.add_argument("--pert", required=True, nargs="+",
                    help="reversed-video run predictions.jsonl (the MotionCD amateur); several => "
                         "average their log-odds")
    ap.add_argument("--dataset", default="", help="label for the report (timeblind | motionblind)")
    ap.add_argument("--category", help="keep only this category (yes_no | multiple_choice); omit for "
                                       "the whole benchmark")
    ap.add_argument("--mcd-alpha", type=float, default=MCD_ALPHA_DEFAULT)
    ap.add_argument("--mcd-beta", type=float, default=MCD_BETA_DEFAULT)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", help="write the JSON rollup here")
    a = ap.parse_args()

    out = run(a.real, a.pert, a.category, a.mcd_alpha, a.mcd_beta, a.seed, a.n_boot)

    tag = (a.dataset or "dataset") + (f" [{a.category}]" if a.category else " [all]")
    print(f"=== MotionCD exact-rule replication — {tag} (n={out['n_items']}) ===")
    print(f"mcd_alpha={a.mcd_alpha} (alpha_eff={out['mcd_alpha_eff']}), mcd_beta={a.mcd_beta} "
          f"-> APC keeps native argmax when |lo_real| > {out['apc_native_threshold_logodds']:.4f}; "
          f"our floor tau={out['tau']}")
    dg = out["diagnostic"]
    print(f"DIAGNOSTIC — reversed-arm shift: mean|Δlo|={dg['mean_abs_dlo']:.2f} "
          f"median={dg['median_abs_dlo']:.2f} argmax-flip={dg['flip_rate']:.3f}\n")
    for name in ("native", "pure", "floor", "motioncd"):
        print(f"  {name:9s} {_fmt(out['rules'][name])}")
    st = out["motioncd_vs_pure"]
    print("\nMotionCD vs pure-difference (paired; delta = MotionCD - pure):")
    for lvl in ("acc", "i_acc"):
        s = st[lvl]
        print(f"  {lvl:5s}: delta={s['delta']:+.4f} CI95={s['ci95']} "
              f"McNemar p={_pfmt(s['mcnemar_p'])} "
              f"(MotionCD-better={s['mcnemar_b_mcd_better']}, pure-better={s['mcnemar_c_pure_better']})")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        prev = {}
        if Path(a.out).exists():
            prev = json.loads(Path(a.out).read_text())
        key = f"{a.dataset or 'dataset'}:{out['category']}"
        prev[key] = out
        Path(a.out).write_text(json.dumps(prev, indent=2))
        print(f"\nWrote {a.out} [{key}]")


if __name__ == "__main__":
    main()
