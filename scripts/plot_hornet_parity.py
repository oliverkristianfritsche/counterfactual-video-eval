#!/usr/bin/env python3
"""HORNet parity figures. Renders results/hornet_parity.json (verify_hornet_parity.py).

  python scripts/plot_hornet_parity.py results/hornet_parity.json --out-dir analysis/figs

Four figures. The matrix shows one cell per clip and frame budget, green for identical
selection. The profiles show per-clip keep probability with both pipelines' top 8
marked; overlapping markers mean identical selection. The slots figure shows keep
probability by frame position for both pipelines. The planted figure counts the planted
unrelated frames that stay in the top 8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

N_CAND = 32


def load(path):
    d = json.loads(Path(path).read_text())
    parity = [r for r in d["parity"] if "error" not in r]
    return d, parity


def _clean_axis(ax):
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    ax.set_xticks(range(0, N_CAND, 8))
    ax.margins(x=0.02)


def _tie_explained(rec, v):
    """A disagreement counts as a tie when each pair of swapped slots has a keep
    probability gap no larger than the measured score noise between the two pipelines
    for that clip (kp_diff), with the JSON rounding step as the floor."""
    kp = rec["keep_prob"]
    tol = max(rec.get("kp_diff", 0.0), 1e-6)
    h, u = set(v["harness"]), set(v["upstream"])
    return all(abs(kp[a] - kp[b]) <= tol for a in (h - u) for b in (u - h))


def fig_matrix(d, parity, out_dir):
    """Draw one cell per clip and frame budget. Green means both pipelines selected
    identical frames. Yellow means the only difference is equally scored tied frames."""
    ks = [str(k) for k in d["ks"]]
    grid, tie_cells = [], 0
    for r in parity:
        row = []
        for k in ks:
            v = r["ks"][k]
            if set(v["harness"]) == set(v["upstream"]):
                row.append(0)
            elif _tie_explained(r, v):
                row.append(1)
                tie_cells += 1
            else:
                row.append(2)
        grid.append(row)
    n = len(grid) * len(ks)
    ident = sum(row.count(0) for row in grid)
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(grid) + 2.2))
    colors = {0: "#4daf4a", 1: "#ffcf4d", 2: "#e41a1c"}
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            ax.add_patch(plt.Rectangle((x, y), 0.92, 0.92, color=colors[val]))
    ax.set_xlim(0, len(ks))
    ax.set_ylim(len(grid), 0)
    ax.set_xticks([x + 0.46 for x in range(len(ks))])
    ax.set_xticklabels([f"k={k}" for k in ks], fontsize=9)
    ax.set_yticks([y + 0.46 for y in range(len(grid))])
    ax.set_yticklabels([f'{r["dataset"]}: {r["clip"].replace(".mp4", "")}'
                        for r in parity], fontsize=7)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"Identical frame selection in {ident} of {n} checks",
                 fontsize=13, pad=28)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in (0, 1, 2)]
    labels = ["identical frames", "tied frames swapped (equal scores)",
              "different frames"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9)
    fig.text(0.5, 0.015, "", ha="center")
    fig.tight_layout(rect=(0, 0.05, 1, 0.99))
    out = out_dir / "hornet_parity_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_scores(d, parity, out_dir):
    """Scatter each frame's keep probability from the harness against the same frame's
    keep probability from the original code. Points on the diagonal mean equal scores."""
    xs, ys = [], []
    for r in parity:
        if "keep_prob_upstream" in r:
            xs += r["keep_prob"]
            ys += r["keep_prob_upstream"]
    if not xs:
        return
    maxd = max(abs(a - b) for a, b in zip(xs, ys))
    lo, hi = min(xs + ys), max(xs + ys)
    pad = (hi - lo) * 0.06
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="C7", linewidth=1,
            zorder=1, label="equal scores")
    ax.scatter(xs, ys, s=16, color="C0", alpha=0.5, zorder=2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    ax.set_xlabel("keep probability, our harness")
    ax.set_ylabel("keep probability, original HORNet code")
    ax.set_title(f"The two pipelines give every frame the same score\n"
                 f"({len(xs)} frames, largest difference {maxd:.1e})", fontsize=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(0.5, 0.015, "one point per frame (20 clips x 32 frames);\na point on the "
             "diagonal has the same score from both pipelines", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = out_dir / "hornet_parity_scores.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_profiles(d, parity, out_dir, per_ds=3):
    picks = []
    for ds in ("timeblind", "motionblind", "cli"):
        picks += [r for r in parity if r["dataset"] == ds][:per_ds]
    if not picks:
        return
    ncol = 2
    nrow = (len(picks) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.2 * nrow),
                             sharex=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, r in enumerate(picks):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        kp = r["keep_prob"]
        k8 = r["ks"]["8"]
        ax.plot(range(N_CAND), kp, color="C7", linewidth=1, zorder=1)
        ax.scatter(k8["harness"], [kp[j] for j in k8["harness"]], s=42, color="C0",
                   zorder=3, label="our harness" if i == 0 else None)
        ax.scatter(k8["upstream"], [kp[j] for j in k8["upstream"]], s=115,
                   facecolors="none", edgecolors="C1", linewidths=1.4, zorder=2,
                   label="original HORNet code" if i == 0 else None)
        ax.set_title(f'{r["dataset"]}: {r["clip"]}', fontsize=10)
        _clean_axis(ax)
    for ax in axes[-1]:
        ax.set_xlabel("frame position (1 of 32)")
    fig.suptitle("Both pipelines select the same top 8 frames", fontsize=13, y=0.995)
    fig.text(0.5, 0.945, "gray line: keep probability per frame; markers: the 8 frames "
             "each pipeline keeps; overlapping markers mean identical selection",
             ha="center", fontsize=9)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045, 1, 0.915))
    out = out_dir / "hornet_parity_profiles.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_slots(d, parity, out_dir):
    dss = [ds for ds in ("timeblind", "motionblind", "cli")
           if any(r["dataset"] == ds for r in parity)]
    fig, axes = plt.subplots(1, len(dss), figsize=(5.4 * len(dss), 3.1), squeeze=False)
    for c, ds in enumerate(dss):
        ax = axes[0][c]
        recs = [r for r in parity if r["dataset"] == ds]
        for r in recs:
            ax.plot(range(N_CAND), r["keep_prob"], color="C0", alpha=0.2, linewidth=1)
        mean_h = [sum(r["keep_prob"][j] for r in recs) / len(recs)
                  for j in range(N_CAND)]
        ax.plot(range(N_CAND), mean_h, color="C0", linewidth=2.6,
                label="our harness (mean)" if c == 0 else None)
        if all("keep_prob_upstream" in r for r in recs):
            mean_u = [sum(r["keep_prob_upstream"][j] for r in recs) / len(recs)
                      for j in range(N_CAND)]
            ax.plot(range(N_CAND), mean_u, color="C1", linewidth=1.6, linestyle="--",
                    label="original HORNet code (mean)" if c == 0 else None)
        ax.set_title(f"{ds}, {len(recs)} clips", fontsize=11)
        ax.set_xlabel("frame position (1 of 32)")
        _clean_axis(ax)
        if c == 0:
            ax.set_ylabel("keep probability")
    fig.suptitle("Both pipelines produce the same keep probabilities",
                 fontsize=13, y=0.99)
    fig.text(0.5, 0.89, f'checkpoint {Path(d["checkpoint"]).name}; thin lines: single '
             "clips; the dashed line covering the solid one means equal scores",
             ha="center", fontsize=9)
    fig.legend(loc="lower center", frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 0.85))
    out = out_dir / "hornet_parity_slots.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_planted(d, out_dir):
    planted = d.get("planted") or []
    kinds = [k for k in ("donor", "noise", "black") if any(r["kind"] == k for r in planted)]
    if not planted or not kinds:
        return
    clips = []
    for r in planted:
        if r["clip"] not in clips:
            clips.append(r["clip"])
    count = {(r["clip"], r["kind"]): r["in_top8"] for r in planted}
    label = {"donor": "frames from another video", "noise": "uniform noise frames",
             "black": "black frames"}
    color = {"donor": "C1", "noise": "C3", "black": "C7"}
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(12.5, 3.7),
                                  gridspec_kw={"width_ratios": [3.4, 1]})

    w = 0.8 / len(kinds)
    for j, kind in enumerate(kinds):
        xs = [i + (j - (len(kinds) - 1) / 2) * w for i in range(len(clips))]
        ax.bar(xs, [count.get((c, kind), 0) for c in clips], width=w,
               color=color[kind], label=label[kind])
    ax.axhline(1.0, color="gray", linewidth=1.2, linestyle="--", label="chance")
    ax.set_ylim(0, 4.3)
    ax.set_yticks(range(5))
    ax.set_ylabel("kept in top 8 (of 4 planted)")
    ax.set_title("per clip: how many planted frames stay in the top 8", fontsize=10)
    ax.set_xticks(range(len(clips)))
    ax.set_xticklabels([c.replace(".mp4", "") for c in clips],
                       rotation=45, ha="right", fontsize=7)

    for j, kind in enumerate(kinds):
        ranks = [x + 1 for r in planted if r["kind"] == kind for x in r["planted_ranks"]]
        axr.bar(j, sum(ranks) / len(ranks), width=0.6, color=color[kind])
    axr.axhline(16.5, color="gray", linewidth=1.2, linestyle="--")
    axr.set_ylim(0, 32)
    axr.set_yticks([1, 16.5, 32])
    axr.set_yticklabels(["1 (best)", "16.5", "32 (worst)"])
    axr.set_xticks(range(len(kinds)))
    axr.set_xticklabels([kind for kind in kinds], fontsize=9)
    axr.set_title("average rank of the planted frames\namong the 32 candidates",
                  fontsize=9)

    fig.suptitle("Does the policy reject planted unrelated frames", fontsize=13, y=0.98)
    fig.text(0.5, 0.885, "4 unrelated frames replace 4 of a clip's 32 candidates before "
             "scoring (noise: per-pixel uniform on 0 to 1, fixed seed); dashed lines: a "
             "policy that ignores content (1 of 4 kept, middle rank 16.5)",
             ha="center", fontsize=9)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 0.86))
    out = out_dir / "hornet_parity_planted.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="hornet_parity.json from verify_hornet_parity.py --out")
    ap.add_argument("--out-dir", default="analysis/figs")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d, parity = load(a.report)
    print(d["verdict"])
    fig_matrix(d, parity, out_dir)
    fig_scores(d, parity, out_dir)
    fig_profiles(d, parity, out_dir)
    fig_slots(d, parity, out_dir)
    fig_planted(d, out_dir)


if __name__ == "__main__":
    main()
