#!/usr/bin/env python3
"""Pair-constrained (structure-constrained) decoding, offline from logged logits.

Every per-item decision rule (native argmax, global calibration, contrastive decoding) thresholds
each item independently. An information-flow probe measured that the raw log-odds already order
the counterfactual pairs correctly far above the realized accuracy (TimeBlind temporal order 0.807,
speed 0.875, magnitude 0.833; force/direction 0.500). This rule converts that ordering directly into
accuracy: within each question-pair (same question, two counterfactual clips, gold exactly-one-yes)
assign yes/A to the clip with the higher yes/no (A/B) log-odds and no/B to the other.

This CHANGES THE INFERENCE PROTOCOL — each answer conditions on the sibling item of the benchmark's
counterfactual pair, which is only possible because the benchmark publishes the pairs. It is a
labeled diagnostic that quantifies what the absolute-criterion failure costs, not a drop-in for the
deployed per-item pipeline, and not comparable to per-item rows without the label.

Pair semantics (verified against the data): each instance_id groups 4 roles; question-pairs are
roles {0,1} and {2,3}; videos are roles {0,2} and {1,3}. TimeBlind yes/no gold yes,no,no,yes; A/B
gold A,B,B,A; MotionBlind the same 4-role structure. Under the pair rule Q_Acc == the within-pair
sign-test rate by construction, and the sticky-rate is 0 by construction (one yes, one no per pair).

Rules:
  (a) native argmax        per item, option1 iff lo_real > 0            (reproduces the baseline)
  (b) pair raw             pair-constrained on lo_real                  (== the sign test as Q_Acc)
  (c) pair CD (1 shuffle)  pair-constrained on lo_real - lo_pert
  (d) pair CD (4 seeds)    pair-constrained on lo_real - mean_k lo_pert
Reference for deltas: the per-item CD rule (option1 iff lo_real - lo_pert > 0).

For the A/B half the score is d = logit_a - logit_b (item_lo handles the channel per record).
MotionBlind runs (a) and (b) only — the freeze perturbation is OOD-poisoned, so no CD arm.

    python scripts/pair_decode.py \
        --tb-real REAL.jsonl --tb-pert PERT1.jsonl [PERT2 PERT3 PERT4] \
        --mb-real MB.jsonl --out results/pair_decode_analysis.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import timeblind_pair_metrics   # noqa: E402
from scripts.contrastive_decode import item_lo           # noqa: E402

QPAIRS = ((0, 1), (2, 3))   # question-pairs
VPAIRS = ((0, 2), (1, 3))   # video-pairs


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def binom_p(k, n):
    """Two-sided exact binomial (McNemar) p for k successes of n at 0.5."""
    return min(1.0, 2 * sum(comb(n, x) for x in range(min(k, n - k) + 1)) / 2 ** n) if n else 1.0


# ---------------------------------------------------------------- decisions
def complete_groups(recs):
    """{instance_id: {role: rec}} for the complete 4-role groups only (others counted, dropped)."""
    byg = defaultdict(dict)
    for r in recs.values():
        byg[r["instance_id"]][r["role"]] = r
    full = {g: v for g, v in byg.items() if set(v) == {0, 1, 2, 3}}
    return full, len(byg) - len(full)


def decide_item(groups, score, thr=0.0):
    """Per-item threshold rule (native / per-item CD). Returns {id: (pick1, correct)}."""
    out = {}
    for roles in groups.values():
        for rec in roles.values():
            g1 = item_lo(rec)[1]
            p1 = score[rec["id"]] > thr
            out[rec["id"]] = (p1, p1 == g1)
    return out


def decide_pair(groups, score):
    """Pair-constrained rule: within each question-pair assign option1 to the higher score, option0
    to the other (exactly one yes/A per pair). Returns {id: (pick1, correct)}.

    Exact score ties (the model gives the two clips identical log-odds — no within-pair ordering
    signal) are scored as a miss on both roles, matching the strict `>` convention of the
    within-pair sign test. An arbitrary one-yes-one-no split keeps the exactly-one-yes
    structure (sticky 0), but neither role is credited. This is the conservative choice and it makes
    pair-raw Q_Acc equal the within-pair sign-test rate exactly (< 1% of pairs are ties)."""
    out = {}
    for roles in groups.values():
        for a, b in QPAIRS:
            ra, rb = roles[a], roles[b]
            sa, sb = score[ra["id"]], score[rb["id"]]
            if sa == sb:                                       # no ordering signal -> both miss
                out[ra["id"]] = (True, False)
                out[rb["id"]] = (False, False)
            else:
                hi_a = sa > sb
                out[ra["id"]] = (hi_a, hi_a == item_lo(ra)[1])
                out[rb["id"]] = (not hi_a, (not hi_a) == item_lo(rb)[1])
    return out


# ---------------------------------------------------------------- summarise
def group_cells(decisions, groups):
    """Per-group {cat, roles:{role:(pick1,correct)}} for aggregation and the bootstrap."""
    cells = {}
    for gid, roles in groups.items():
        cat = roles[0]["category"]
        cells[gid] = {"cat": cat,
                      "r": {role: decisions[rec["id"]] for role, rec in roles.items()}}
    return cells


def _agg(cells, gids):
    n = len(gids)
    if not n:
        return {}
    acc = q = v = i = sticky = 0
    for g in gids:
        r = cells[g]["r"]
        c = {role: r[role][1] for role in range(4)}
        p = {role: r[role][0] for role in range(4)}
        acc += sum(c.values())
        qs = (c[0] and c[1]) + (c[2] and c[3])
        vs = (c[0] and c[2]) + (c[1] and c[3])
        q += qs
        v += vs
        i += int(qs == 2 and vs == 2)
        sticky += (p[0] == p[1]) + (p[2] == p[3])
    return {"acc": acc / (4 * n), "q_acc": q / (2 * n), "v_acc": v / (2 * n),
            "i_acc": i / n, "sticky_rate": sticky / (2 * n), "n_groups": n}


def summarize(decisions, groups, per_category=False):
    cells = group_cells(decisions, groups)
    out = {"overall": _agg(cells, list(cells))}
    if per_category:
        cats = sorted({c["cat"] for c in cells.values()})
        out["per_category"] = {
            cat: _agg(cells, [g for g in cells if cells[g]["cat"] == cat]) for cat in cats}
    return out, cells


def official_cross_check(decisions, groups, label):
    """Recompute Q/V/I/sticky through the official eval metric and assert agreement with the
    from-scratch aggregation, locking the pair semantics to the shipped metric."""
    recs = [{"instance_id": roles[role]["instance_id"], "role": role,
             "category": roles[role]["category"],
             "correct": decisions[roles[role]["id"]][1], "error": False,
             "parsed_choice": 1 if decisions[roles[role]["id"]][0] else 0}
            for roles in groups.values() for role in roles]
    m = timeblind_pair_metrics(recs)
    mine = _agg(group_cells(decisions, groups), list(groups))
    for k in ("q_acc", "v_acc", "i_acc", "sticky_rate"):
        if k in m:                      # the grouped scorer omits keys on an empty subset
            assert abs(m[k] - mine[k]) < 1e-9, f"{label}: {k} {m[k]} vs {mine[k]}"
    return mine


# ---------------------------------------------------------------- bootstrap / mcnemar
def bootstrap(cells_rule, cells_ref, seed=1234, n=2000):
    """Group-clustered bootstrap of the (rule - ref) deltas on Acc / Q / V / I_Acc."""
    gids = sorted(cells_rule)
    rng = random.Random(seed)
    keys = ("acc", "q_acc", "v_acc", "i_acc")
    dist = {k: [] for k in keys}
    for _ in range(n):
        samp = [rng.choice(gids) for _ in gids]
        ru = _agg(cells_rule, samp)
        rf = _agg(cells_ref, samp)
        for k in keys:
            dist[k].append(ru[k] - rf[k])
    out = {}
    for k, d in dist.items():
        s = sorted(d)
        out[k] = {"mean": sum(d) / len(d), "ci95": [s[n // 40], s[-n // 40 - 1]],
                  "p_le_0": sum(x <= 0 for x in d) / len(d)}
    return out


def mcnemar(dec_rule, dec_ref):
    b = sum(dec_rule[i][1] and not dec_ref[i][1] for i in dec_rule)   # ref wrong -> rule right
    c = sum(dec_ref[i][1] and not dec_rule[i][1] for i in dec_rule)   # ref right -> rule wrong
    return {"b_gain": b, "c_loss": c, "n_discordant": b + c, "p_exact": binom_p(min(b, c), b + c)}


# ---------------------------------------------------------------- sign test (independent check)
def sign_test(groups):
    """Independent within-pair sign test: does lo_real order each pair (yes>no)? Returns rate/ok/tot
    per category and overall (the number the pair-raw Q_Acc must equal by construction)."""
    per = defaultdict(lambda: [0, 0])
    for roles in groups.values():
        cat = roles[0]["category"]
        for a, b in QPAIRS:
            la, ga = item_lo(roles[a])
            lb, gb = item_lo(roles[b])
            if ga == gb:
                continue
            yes_lo, no_lo = (la, lb) if ga else (lb, la)
            per[cat][0] += yes_lo > no_lo
            per[cat][1] += 1
    out = {c: {"ok": ok, "tot": tot, "rate": ok / tot} for c, (ok, tot) in per.items()}
    tot_ok = sum(v["ok"] for v in out.values())
    tot_n = sum(v["tot"] for v in out.values())
    out["overall"] = {"ok": tot_ok, "tot": tot_n, "rate": tot_ok / tot_n}
    return out


# ---------------------------------------------------------------- drivers
def fmt(d):
    return (f"Acc={d['acc']:.4f}  Q={d['q_acc']:.4f}  V={d['v_acc']:.4f}  "
            f"I={d['i_acc']:.4f}  sticky={d['sticky_rate']:.3f}")


def score_real(recs):
    return {i: item_lo(r)[0] for i, r in recs.items()}


def score_cd(real, perts, ids):
    """lo_real - mean_k lo_pert over the given perturbed runs, on the shared ids."""
    return {i: item_lo(real[i])[0] - sum(item_lo(p[i])[0] for p in perts) / len(perts) for i in ids}


def run_timeblind(real, perts, out):
    lo_real = score_real(real)
    ids_all = set(real)
    for p in perts:
        ids_all &= set(p)
    assert len(ids_all) == len(real), (
        f"the perturbed arms cover {len(ids_all)} of the real arm's {len(real)} items; every arm "
        "must run the same item set")
    cd1 = score_cd(real, perts[:1], ids_all)     # single shuffle
    cd4 = score_cd(real, perts, ids_all)         # 4-seed mean
    st = sign_test(complete_groups(real)[0])
    res = {"n_shuffle_seeds": len(perts), "sign_test": st, "subsets": {}}

    print(f"\n=== TimeBlind ({len(real)} items) ===")
    for sub, cat in [("yes_no", "yes_no"), ("multiple_choice", "multiple_choice"), ("full", None)]:
        recs = {i: r for i, r in real.items() if cat is None or r["category"] == cat}
        groups, dropped = complete_groups(recs)
        if not groups:                  # dataset carries no items of this category
            print(f"-- {sub}: no complete groups, skipped --")
            continue
        rules = {
            "native": decide_item(groups, lo_real, 0.0),
            "cd_item": decide_item(groups, cd1, 0.0),          # per-item CD reference
            "pair_raw": decide_pair(groups, lo_real),
            "pair_cd1": decide_pair(groups, cd1),
            "pair_cd4": decide_pair(groups, cd4),
        }
        cells = {}
        block = {"n_groups": len(groups), "dropped_incomplete": dropped}
        for name, dec in rules.items():
            official_cross_check(dec, groups, f"TB/{sub}/{name}")
            summ, cells[name] = summarize(dec, groups, per_category=(cat is None))
            block[name] = summ["overall"]
            if cat is None:
                block[name + "_per_half"] = summ["per_category"]
        # bootstrap deltas: pair rules vs native and vs the per-item CD
        block["bootstrap"] = {}
        for rule in ("pair_raw", "pair_cd1", "pair_cd4"):
            block["bootstrap"][f"{rule}_vs_native"] = bootstrap(cells[rule], cells["native"])
            block["bootstrap"][f"{rule}_vs_cd_item"] = bootstrap(cells[rule], cells["cd_item"])
        block["mcnemar"] = {
            "pair_raw_vs_native": mcnemar(rules["pair_raw"], rules["native"]),
            "pair_cd1_vs_native": mcnemar(rules["pair_cd1"], rules["native"]),
            "pair_raw_vs_cd_item": mcnemar(rules["pair_raw"], rules["cd_item"]),
        }
        res["subsets"][sub] = block
        print(f"-- {sub} (n={len(groups)} groups) --")
        for name in rules:
            print(f"   {name:9s} {fmt(block[name])}")

    # invariant (holds on any data by construction): the pair rule's Q_Acc per half equals the
    # strict within-pair sign-test rate.
    for half in ("yes_no", "multiple_choice"):
        if half in res["subsets"] and half in st:
            assert abs(res["subsets"][half]["pair_raw"]["q_acc"] - st[half]["rate"]) < 1e-9, \
                f"pair_raw {half} Q_Acc != within-pair sign test"
    out["timeblind"] = res


def run_motionblind(real, out):
    lo_real = score_real(real)
    groups, dropped = complete_groups(real)
    st = sign_test(groups)
    rules = {"native": decide_item(groups, lo_real, 0.0),
             "pair_raw": decide_pair(groups, lo_real)}
    cells = {}
    block = {"n_groups": len(groups), "dropped_incomplete": dropped, "sign_test": st}
    for name, dec in rules.items():
        official_cross_check(dec, groups, f"MB/{name}")
        summ, cells[name] = summarize(dec, groups, per_category=True)
        block[name] = summ["overall"]
        block[name + "_per_category"] = summ["per_category"]
    block["bootstrap"] = {"pair_raw_vs_native": bootstrap(cells["pair_raw"], cells["native"])}
    block["mcnemar"] = {"pair_raw_vs_native": mcnemar(rules["pair_raw"], rules["native"])}
    out["motionblind"] = block

    print(f"\n=== MotionBlind ({len(real)} items) ===")
    for name in rules:
        print(f"   {name:9s} {fmt(block[name])}")
    print("   per-category pair_raw Q_Acc (== within-pair sign test):")
    for cat in sorted(block["pair_raw_per_category"]):
        print(f"      {cat:22s} Q={block['pair_raw_per_category'][cat]['q_acc']:.3f}  "
              f"I={block['pair_raw_per_category'][cat]['i_acc']:.3f}  "
              f"(native I={block['native_per_category'][cat]['i_acc']:.3f})")

    # invariant (holds on any data by construction): per-category pair_raw Q_Acc equals the
    # within-pair sign-test rate.
    for cat in sorted(block["pair_raw_per_category"]):
        if cat in st:
            assert abs(block["pair_raw_per_category"][cat]["q_acc"] - st[cat]["rate"]) < 1e-9, \
                f"MB {cat} pair_raw Q_Acc != within-pair sign test"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tb-real", required=True)
    ap.add_argument("--tb-pert", required=True, nargs="+",
                    help="perturbed (shuffle) runs; first is the single-shuffle CD arm, all are "
                         "averaged for the 4-seed CD arm")
    ap.add_argument("--mb-real", required=True)
    ap.add_argument("--out", default="results/pair_decode_analysis.json")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    out = {"inputs": {"tb_real": a.tb_real, "tb_pert": a.tb_pert, "mb_real": a.mb_real},
           "seed": a.seed, "n_boot": a.n_boot}

    run_timeblind(load(a.tb_real), [load(p) for p in a.tb_pert], out)
    run_motionblind(load(a.mb_real), out)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
