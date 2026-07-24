"""Paired significance for two decision maps over the same items.

Both the run comparison and the decode scripts threshold one per-item quantity two different ways
and need the same paired statistics: an exact McNemar test on the discordant pairs, and an
accuracy-delta confidence interval bootstrapped over clusters. They live here so every caller
shares one implementation.

Each `rows` entry is a dict with keys `id`, `gold`, and `pair` (the cluster key, or None to treat
the item as its own cluster). Each decision map is `{id: predicted_label}`.
"""
from __future__ import annotations

from collections import defaultdict


def mcnemar_exact(rows, dec_a, dec_b):
    """Paired comparison of two decision maps over the SAME items -- the correct test here, since
    the arms are two thresholds of one per-item d and agree everywhere outside the discordant band.
    b = a-correct/b-wrong, c = a-wrong/b-correct; exact two-sided binomial on b+c (the discordant
    count is small, so the chi-square approximation is not used)."""
    from math import comb
    b = c = 0
    for r in rows:
        ca, cb = dec_a[r["id"]] == r["gold"], dec_b[r["id"]] == r["gold"]
        if ca and not cb:
            b += 1
        elif cb and not ca:
            c += 1
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p": 1.0}
    k = min(b, c)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return {"b": b, "c": c, "n_discordant": n, "p": round(p, 6)}


def delta_bootstrap(rows, dec_a, dec_b, n_boot=5000, seed=1234):
    """Accuracy delta (b - a) with a CI bootstrapped over CLUSTERS (pairs): the 4 roles of a
    MotionBlind pair share clips and are not independent draws, so pairs are resampled, not items."""
    import random
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["pair"] if r["pair"] is not None else r["id"]].append(r)
    keys = list(groups)
    acc = lambda rs, d: sum(1 for r in rs if d[r["id"]] == r["gold"]) / len(rs)
    obs = acc(rows, dec_b) - acc(rows, dec_a)
    deltas = []
    for _ in range(n_boot):
        samp = [r for _ in keys for r in groups[keys[rng.randrange(len(keys))]]]
        if samp:
            deltas.append(acc(samp, dec_b) - acc(samp, dec_a))
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))] if deltas else 0.0
    hi = deltas[int(0.975 * len(deltas))] if deltas else 0.0
    return {"delta": round(obs, 4), "ci95": [round(lo, 4), round(hi, 4)], "n_clusters": len(keys)}
