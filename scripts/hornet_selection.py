#!/usr/bin/env python3
"""HORNet frame-selection analysis: dump per-clip selections, plot the picked-slot
distribution, and render a frame gallery for a single clip.

  python scripts/hornet_selection.py dump --checkpoint CKPT \
      --timeblind-root /path/to/timeblind --motionblind-root /path/to/motionblind \
      --out selection.jsonl [--limit N]
  python scripts/hornet_selection.py plot selection.jsonl --k 8 --out-dir figs
  python scripts/hornet_selection.py plot selection.jsonl --all-k --out-dir figs
  python scripts/hornet_selection.py gallery --video clip.mp4 --out gallery.png \
      [--checkpoint CKPT | --selection selection.jsonl]

dump needs torch, decord, the cloned HORNet/ (see README) and the benchmark data; plot needs
only matplotlib. gallery scores the clip with the policy (--checkpoint) or reuses a stored
ranking (--selection, no torch needed). Candidates follow the scoring protocol: 32 uniform
frames at 288px; the ranking is the policy keep-probability with deterministic top-k (ties to
the earlier slot), so one dump yields the distribution at every k.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_CAND, SCORE_RES = 32, 288
KS = [1, 4, 8, 16, 24]


class _Cfg:
    def __init__(self, ckpt):
        self.video = {"hornet_checkpoint": ckpt}
        self.eval = {}


def _selector(ckpt):
    from src.selectors import HornetSelector
    return HornetSelector(_Cfg(ckpt))


def load_frames(path):
    """The scoring protocol: 32 uniform candidates, resized to 288 (torch tensors)."""
    import decord
    import numpy as np
    import torch
    vr = decord.VideoReader(str(path))
    idx = np.linspace(0, len(vr) - 1, N_CAND).astype(int)
    f = torch.from_numpy(vr.get_batch(idx).asnumpy()).float() / 255.0
    f = torch.nn.functional.interpolate(f.permute(0, 3, 1, 2), size=(SCORE_RES, SCORE_RES),
                                        mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    return [f[i] for i in range(f.shape[0])]


def load_frames_np(path, res=256):
    """Display loader: 32 uniform candidates as uint8 arrays (no torch)."""
    import decord
    import numpy as np
    from PIL import Image
    vr = decord.VideoReader(str(path))
    idx = np.linspace(0, len(vr) - 1, N_CAND).astype(int)
    batch = vr.get_batch(idx).asnumpy()
    return [np.asarray(Image.fromarray(fr).resize((res, res), Image.BILINEAR)) for fr in batch]


def rank_clip(sel, frames):
    """(keep_prob[32], ranking[32] descending, ties to the earlier slot)."""
    import numpy as np
    import torch
    with torch.no_grad():
        videos = torch.stack(frames).unsqueeze(0).float().to(sel.device)
        kp = sel.model(videos)["keep_prob"][0].float().cpu().numpy()
    return kp, np.argsort(-kp, kind="stable").tolist()


def unique_clips(ann_path, root, strip_prefix=None):
    out, seen = [], set()
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vp = str(json.loads(line).get("video_path") or "")
            if strip_prefix and vp.startswith(strip_prefix):
                vp = vp.split("/", 1)[1]
            p = str(Path(root) / vp)
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def cmd_dump(a):
    sel = _selector(a.checkpoint)
    datasets = []
    if a.timeblind_root:
        datasets.append(("timeblind", unique_clips(Path(a.timeblind_root) / "data.jsonl",
                                                   a.timeblind_root, "TimeBlind/")))
    if a.motionblind_root:
        datasets.append(("motionblind", unique_clips(Path(a.motionblind_root) / "data_db.jsonl",
                                                     a.motionblind_root)))
    assert datasets, "pass --timeblind-root and/or --motionblind-root"
    n = 0
    with open(a.out, "w") as fout:
        for ds, clips in datasets:
            if a.limit:
                clips = clips[:a.limit]
            print(f"[{ds}] {len(clips)} unique clips")
            for i, path in enumerate(clips):
                try:
                    kp, ranking = rank_clip(sel, load_frames(path))
                except Exception as e:
                    print(f"  skip {Path(path).name}: {type(e).__name__}")
                    continue
                fout.write(json.dumps({"dataset": ds, "clip": Path(path).name,
                                       "checkpoint": Path(a.checkpoint).name,
                                       "keep_prob": [round(float(v), 6) for v in kp],
                                       "ranking": ranking}) + "\n")
                n += 1
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(clips)}")
    print(f"wrote {n} clips -> {a.out}")


def _freqs(selection_path):
    """({dataset: {"n": count, k: [fraction per slot]}}, checkpoint_name) from a dump file."""
    per, ckpt = {}, ""
    for line in open(selection_path):
        d = json.loads(line)
        per.setdefault(d["dataset"], []).append(d["ranking"])
        ckpt = ckpt or d.get("checkpoint", "")
    out = {}
    for ds, rankings in per.items():
        out[ds] = {"n": len(rankings)}
        for k in KS:
            freq = [0] * N_CAND
            for r in rankings:
                for p in r[:k]:
                    freq[p] += 1
            out[ds][k] = [c / len(rankings) for c in freq]
    return out, ckpt


def _hornet_label(ckpt_name):
    return f"HORNet ({Path(ckpt_name).stem})" if ckpt_name else "HORNet"


def _bars(ax, frac, k, label_uniform=True):
    ax.bar(range(N_CAND), frac, color="C0")
    ax.axhline(k / N_CAND, color="gray", linewidth=1, linestyle="--",
               label=f"uniform ({k}/{N_CAND})" if label_uniform else None)
    ax.set_xticks(range(0, N_CAND, 4))
    ax.margins(x=0.01)


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, ckpt = _freqs(a.selection)
    label = _hornet_label(a.checkpoint_label or ckpt)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.all_k:
        cols = [ds for ds in ("timeblind", "motionblind") if ds in data] or sorted(data)
        fig, axes = plt.subplots(len(KS), len(cols), figsize=(6.5 * len(cols), 2.0 * len(KS)),
                                 sharex=True, squeeze=False)
        for r, k in enumerate(KS):
            ymax = max(max(data[ds][k]) for ds in cols) * 1.18
            for c, ds in enumerate(cols):
                ax = axes[r][c]
                _bars(ax, data[ds][k], k, label_uniform=False)
                ax.set_ylim(0, ymax)
                if c == 0:
                    ax.set_ylabel(f"k={k}", fontsize=11)
                if r == 0:
                    ax.set_title(f"{ds} (n={data[ds]['n']})", fontsize=12)
                if r == len(KS) - 1:
                    ax.set_xlabel("candidate slot (temporal order)")
        fig.suptitle(f"{label} picked-slot distribution: share of clips per slot in the top-k "
                     "(dashed = uniform k/32)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = out_dir / "hornet_positions_all_k.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")
        return
    for ds in data:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        _bars(ax, data[ds][a.k], a.k)
        ax.set_xlabel("candidate slot (temporal order)")
        ax.set_ylabel(f"share of clips in top-{a.k}")
        ax.set_title(f"{label} picked-slot distribution: {ds} (n={data[ds]['n']}, k={a.k})")
        ax.legend(frameon=False)
        fig.tight_layout()
        out = out_dir / f"hornet_positions_{ds}_k{a.k}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def clip_questions(ann_path, clip_name, limit=2):
    """Unique questions asked of this clip, in annotation order."""
    out = []
    for line in open(ann_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if Path(str(d.get("video_path") or "")).name == clip_name:
            q = (d.get("question") or "").strip()
            if q and q not in out:
                out.append(q)
    return out[:limit]


def cmd_gallery(a):
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name = Path(a.video).name
    ckpt = a.checkpoint_label or ""
    if a.selection:
        ranking = None
        for line in open(a.selection):
            d = json.loads(line)
            if d["clip"] == name:
                ranking = d["ranking"]
                ckpt = ckpt or d.get("checkpoint", "")
                break
        assert ranking is not None, f"{name} not found in {a.selection}"
        frames = load_frames_np(a.video)
    else:
        assert a.checkpoint, "pass --checkpoint or --selection"
        sel = _selector(a.checkpoint)
        tframes = load_frames(a.video)
        _, ranking = rank_clip(sel, tframes)
        frames = [(f.numpy() * 255).astype("uint8") for f in tframes]
        ckpt = ckpt or Path(a.checkpoint).name
    ks = [int(x) for x in str(a.ks).split(",")]
    questions = clip_questions(a.annotations, name) if a.annotations else []
    label = _hornet_label(ckpt)

    def header_lines(title, fig_w):
        lines = [(title, 12)]
        for j, q in enumerate(questions):
            for line in textwrap.wrap(f"Q{j + 1}: {q}", max(60, int(fig_w / 0.095))):
                lines.append((line, 10.5))
        return lines

    def draw_header(fig, fig_h, lines):
        """Reserved header band: returns the top figure-fraction for the axes grid."""
        y_in = fig_h - 0.18
        for text, size in lines:
            fig.text(0.5, y_in / fig_h, text, ha="center", va="top", fontsize=size)
            y_in -= 0.26
        return (y_in - 0.16) / fig_h

    def strip_row(fig, sub_gs, picked, borders=2.0, ticks=True):
        for i in range(N_CAND):
            ax = fig.add_subplot(sub_gs[0, i])
            ax.imshow(frames[i], alpha=1.0 if i in picked else 0.35)
            ax.set_xticks([])
            ax.set_yticks([])
            if i in picked:
                for s in ax.spines.values():
                    s.set_edgecolor("C0")
                    s.set_linewidth(borders)
            else:
                for s in ax.spines.values():
                    s.set_visible(False)
            if ticks and (i % 4 == 0 or i == N_CAND - 1):
                ax.set_xlabel(str(i), fontsize=7, color="gray", labelpad=2)
            yield i, ax

    # figure 1: one strip per k (all frames always visible, picks highlighted)
    fig_w = 0.58 * N_CAND
    lines = header_lines(f"{name}: {label} top-k per row; highlighted = selected, "
                         "faded = unselected", fig_w)
    head_in = 0.18 + 0.26 * len(lines) + 0.16
    fig_h = 0.78 * len(ks) + head_in + 0.34
    fig = plt.figure(figsize=(fig_w, fig_h))
    top = draw_header(fig, fig_h, lines)
    gs = fig.add_gridspec(len(ks), 1, top=top, bottom=0.34 / fig_h,
                          left=0.045, right=0.995, hspace=0.34)
    for r, k in enumerate(ks):
        picked = set(ranking[:k])
        row = gs[r].subgridspec(1, N_CAND, wspace=0.05)
        for i, ax in strip_row(fig, row, picked, ticks=(r == len(ks) - 1)):
            if i in picked:
                ax.set_title(str(i), fontsize=6.5, color="C0", pad=1.5)
            else:
                pos = ax.get_position()
                dw, dh = pos.width * 0.16, pos.height * 0.16
                ax.set_position((pos.x0 + dw / 2, pos.y0 + dh / 2,
                                 pos.width - dw, pos.height - dh))
            if i == 0:
                ax.set_ylabel(f"k={k}", fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=16)
    fig.savefig(a.out, dpi=140)
    plt.close(fig)
    print(f"wrote {a.out}")

    # figures 2..: per k, selected frames zoomed in rows of <=8 + that k's strip below
    if not a.per_k:
        return
    import math
    out_path = Path(a.out)
    for k in ks:
        picked_sorted = sorted(ranking[:k])
        rank_of = {p: r + 1 for r, p in enumerate(ranking[:k])}
        nrows = math.ceil(k / 8)
        fig_w = 12.8
        lines = header_lines(f"{name}: {label} top-{k} = slots {picked_sorted}", fig_w)
        head_in = 0.18 + 0.26 * len(lines) + 0.16
        strip_in, cell_in, bottom_in = 0.62, 1.55, 0.34
        fig_h = nrows * cell_in + strip_in + head_in + bottom_in + 0.25
        fig = plt.figure(figsize=(fig_w, fig_h))
        top = draw_header(fig, fig_h, lines)
        outer = fig.add_gridspec(nrows + 1, 1,
                                 height_ratios=[1.0] * nrows + [strip_in / cell_in],
                                 top=top, bottom=bottom_in / fig_h,
                                 left=0.03, right=0.985, hspace=0.32)
        for rrow in range(nrows):
            chunk = picked_sorted[rrow * 8:(rrow + 1) * 8]
            row = outer[rrow].subgridspec(1, 8, wspace=0.05)
            for j, slot in enumerate(chunk):
                ax = fig.add_subplot(row[0, j])
                ax.imshow(frames[slot])
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor("C0")
                    s.set_linewidth(2.2)
                ax.set_title(f"slot {slot}  #{rank_of[slot]}", fontsize=9, color="C0", pad=2)
        strip = outer[nrows].subgridspec(1, N_CAND, wspace=0.06)
        for _ in strip_row(fig, strip, set(picked_sorted), borders=1.8):
            pass
        out_k = out_path.with_name(f"{out_path.stem}_k{k}{out_path.suffix}")
        fig.savefig(out_k, dpi=140)
        plt.close(fig)
        print(f"wrote {out_k}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="score every unique clip, write selection.jsonl")
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--timeblind-root")
    d.add_argument("--motionblind-root")
    d.add_argument("--out", default="selection.jsonl")
    d.add_argument("--limit", type=int, help="clips per dataset (testing)")
    d.set_defaults(fn=cmd_dump)

    p = sub.add_parser("plot", help="bar charts of the picked-slot distribution")
    p.add_argument("selection", help="selection.jsonl from dump")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--all-k", action="store_true",
                   help="one combined figure: datasets as columns, k values as rows")
    p.add_argument("--checkpoint-label", help="checkpoint name for the title "
                   "(read from the dump records when present)")
    p.add_argument("--out-dir", default="figs")
    p.set_defaults(fn=cmd_plot)

    g = sub.add_parser("gallery", help="selected frames on top, full 32-frame strip below")
    g.add_argument("--video", required=True)
    g.add_argument("--checkpoint", help="score the clip with the policy")
    g.add_argument("--selection", help="reuse a stored ranking from selection.jsonl (no torch)")
    g.add_argument("--annotations", help="dataset jsonl; renders the clip's question(s) in the title")
    g.add_argument("--checkpoint-label", help="checkpoint name for the title "
                   "(read from the dump record when present)")
    g.add_argument("--ks", default="1,4,8,16,24",
                   help="comma-separated k values, one strip row per k")
    g.add_argument("--per-k", action="store_true",
                   help="also write one figure per k: selected frames zoomed in rows of 8 "
                        "with that k's strip below (<out>_k<K>.png)")
    g.add_argument("--out", default="gallery.png")
    g.set_defaults(fn=cmd_gallery)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
