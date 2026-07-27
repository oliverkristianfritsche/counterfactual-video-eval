#!/usr/bin/env python3
"""Plot accuracy and instance accuracy for the report's three decision rules, one
image per rule plus a combined head-to-head, TimeBlind in yellow and MotionBlind
in blue, dotted uniform-k24 baselines (results/sweep/{dataset}_uniform_k24_native,
labeled with their values), percent labels with up to two decimals.

The rules are the shipped implementations, imported so the exact published code
scores both benchmarks:

  * pairwise argmax — pair_decode.decide_pair on lo_real: within each annotated
    question-pair, yes/A goes to the clip with the higher log-odds (report §4).
    Uses the benchmark's labeled pairs; no perturbed arm involved.
  * plain difference — contrastive_decode.four_acc with alpha->inf: yes iff
    lo_real - lo_pert > 0 (report §5), across the four perturbed arms.
  * MotionCD — motioncd_decode.motioncd_four_acc at the released defaults
    (mcd_alpha=20 -> alpha_eff=10, mcd_beta=0.1; report §6), across the same arms
    with each arm as the amateur (the paper itself uses reverse only).

The combined image compares all three at the report's head-to-head condition
(the reverse arm for the two CD rules). Shuffle bars are the mean over the four
seeded arms.

  python scripts/plot_cd_rules.py [--decode-dir results/decode] [--out-dir analysis/figs]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.contrastive_decode import load, item_lo, four_acc                    # noqa: E402
from scripts.motioncd_decode import motioncd_four_acc                             # noqa: E402
from scripts.pair_decode import complete_groups, decide_pair, score_real, summarize  # noqa: E402

ARMS = [
    ("reverse", ["reverse"], "Reverse"),
    ("shuffle", [f"shuffle_seed{i}" for i in range(1, 5)], "Shuffle\n(4 seeds)"),
    ("segswap", ["segswap"], "Segment\nswap"),
    ("blockshuffle", ["blockshuffle"], "Block\nshuffle"),
]
DATASETS = [("timeblind", "TimeBlind", "#eda100"), ("motionblind", "MotionBlind", "#2a78d6")]
PANELS = [("acc", "Accuracy"), ("i_acc", "Instance accuracy")]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def pct(d):
    return {m: 100 * d[m] for m, _ in PANELS}


def fmt_pct(v, signed=False):
    """Percent label with up to two decimals (trailing zeros trimmed)."""
    s = f"{v:+.2f}" if signed else f"{v:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s + "%"


def collect(decode_dir: Path, sweep_dir: Path, dataset: str):
    """Per-dataset values in percent: uniform-k24 baseline, pairwise argmax, and the
    two CD rules per arm (a list over runs — one entry per shuffle seed)."""
    real = load(decode_dir / f"{dataset}_real" / "predictions.jsonl")
    with open(sweep_dir / f"{dataset}_uniform_k24_native" / "summary.json") as f:
        sm = json.load(f)["metrics"]
    native = {"acc": 100 * sm["accuracy"], "i_acc": 100 * sm["instance_accuracy"]}

    groups, _ = complete_groups(real)
    argmax = pct(summarize(decide_pair(groups, score_real(real)), groups)[0]["overall"])

    arms = {"difference": {}, "motioncd": {}}
    for arm, runs, _ in ARMS:
        diffs, mcds = [], []
        for run in runs:
            pert = load(decode_dir / f"{dataset}_{run}" / "predictions.jsonl")
            pids = sorted(set(real) & set(pert), key=lambda x: int(x) if str(x).isdigit() else x)
            lop = {i: item_lo(pert[i])[0] for i in pids}
            diffs.append(pct(four_acc(real, lop, pids, 1e9, 0.0)))
            mcds.append(pct(motioncd_four_acc(real, pert, pids)))
        arms["difference"][arm] = diffs
        arms["motioncd"][arm] = mcds
    return {"native": native, "argmax": argmax, "arms": arms}


def draw_figure(plt, Line2D, data, title, categories, values, out, delta=False):
    """One two-panel (Acc | I_Acc) grouped-bar figure. values[ds][cat] is a list of
    runs (multi-run categories are averaged). Absolute mode draws dotted baseline
    lines labeled with their values; delta mode anchors the bars on the baseline
    itself (the zero line), above or below."""
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 4.5))
    fig.patch.set_facecolor(SURFACE)
    width = 0.34

    for ax, (metric, panel_title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        top, bot = 0.0, 0.0

        for m, (ds, _, color) in enumerate(DATASETS):
            xs = [i + (m - 0.5) * (width + 0.02) for i in range(len(categories))]
            native = data[ds]["native"][metric]
            for x, runs in zip(xs, values[ds]):
                vals = [r[metric] for r in runs]
                mean = sum(vals) / len(vals)
                if delta:
                    mean -= native
                ax.bar(x, mean, width=width, color=color, zorder=2)
                lab = fmt_pct(mean, signed=delta)
                box = dict(facecolor=SURFACE, edgecolor="none", pad=1)
                if mean >= 0:
                    ax.text(x, mean + 1.4, lab, ha="center", va="bottom",
                            fontsize=9, color=INK, zorder=6, bbox=box)
                    top = max(top, mean + 1.4)
                else:
                    ax.text(x, mean - 1.4, lab, ha="center", va="top",
                            fontsize=9, color=INK, zorder=6, bbox=box)
                    bot = min(bot, mean - 1.4)
            if not delta:
                ax.axhline(native, color=color, linestyle=(0, (2, 2.5)),
                           linewidth=1.4, zorder=4)
                ax.text(len(categories) - 1 + 0.93, native, fmt_pct(native),
                        ha="right", va="center", fontsize=8.5, color=INK2, zorder=5,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
                top = max(top, native)

        if delta:
            ax.axhline(0, color=MUTED, linewidth=1.2, zorder=3)
            ax.set_ylim(min(-5.0, bot - 8), max(10.0, top + 8))
            ax.set_ylabel("Δ percent correct vs baseline" if metric == "acc" else None,
                          fontsize=10, color=INK2)
        else:
            ax.set_ylim(0, max(40.0, top + 9))
            ax.set_ylabel("Percent correct" if metric == "acc" else None,
                          fontsize=10, color=INK2)
        ax.set_title(panel_title, fontsize=11.5, color=INK, pad=8)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=9.5, color=INK2)
        ax.set_xlim(-0.65, len(categories) - 1 + (0.95 if not delta else 0.65))
        ax.tick_params(colors=MUTED, length=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in DATASETS]
    labels = [lbl for _, lbl, _ in DATASETS]
    if delta:
        handles.append(Line2D([], [], color=MUTED, linewidth=1.2))
        labels.append("uniform k24 baseline (zero line)")
    else:
        handles.append(Line2D([], [], color=MUTED, linestyle=(0, (2, 2.5)), linewidth=1.4))
        labels.append("uniform k24 baseline")
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False,
               fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(title, fontsize=12.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-dir", type=Path, default=Path("results/decode"))
    ap.add_argument("--sweep-dir", type=Path, default=Path("results/sweep"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/figs"))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = {ds: collect(a.decode_dir, a.sweep_dir, ds) for ds, _, _ in DATASETS}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    arm_labels = [lbl for _, _, lbl in ARMS]

    figures = [
        ("cd_rule_argmax",
         "Pairwise argmax  (annotated pairs: yes → clip with the higher log-odds)",
         ["Annotated\npairs"],
         {ds: [[data[ds]["argmax"]]] for ds, _, _ in DATASETS}),
        ("cd_rule_difference",
         "Plain difference  (pairwise vs the perturbed copy: yes iff lo_real > lo_pert)",
         arm_labels,
         {ds: [data[ds]["arms"]["difference"][arm] for arm, _, _ in ARMS]
          for ds, _, _ in DATASETS}),
        ("cd_rule_motioncd",
         "MotionCD  (α = 20, β = 0.1; contrasted against the same video perturbed)",
         arm_labels,
         {ds: [data[ds]["arms"]["motioncd"][arm] for arm, _, _ in ARMS]
          for ds, _, _ in DATASETS}),
        ("cd_rules_combined",
         "Decision rules head-to-head  (CD rules on the reverse arm)",
         ["Pairwise argmax\n(annotated pairs)", "Plain difference\n(reverse)",
          "MotionCD\n(reverse)"],
         {ds: [[data[ds]["argmax"]],
               data[ds]["arms"]["difference"]["reverse"],
               data[ds]["arms"]["motioncd"]["reverse"]] for ds, _, _ in DATASETS}),
    ]
    for name, title, categories, values in figures:
        draw_figure(plt, Line2D, data, title, categories, values,
                    a.out_dir / f"{name}.png")
        draw_figure(plt, Line2D, data, f"{title} — Δ vs baseline", categories, values,
                    a.out_dir / f"{name}_delta.png", delta=True)


if __name__ == "__main__":
    main()
