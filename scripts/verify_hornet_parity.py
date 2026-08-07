#!/usr/bin/env python3
"""HORNet parity check — the harness frame selection against the upstream select_frames.

For each clip, the script computes the top-k frame indices twice from one
checkpoint-loaded policy:

  harness   The sweep-protocol path of src/models/eagle.py _select_clip for
            frame_source: hornet. The script binds the real methods, so no Eagle
            weights load: EagleModel._decode (32 uniform candidates, native
            resolution), then EagleModel._resize to 288, then HornetSelector.select.
  upstream  HORNet/lmms_eval_utils/hornet.py select_frames: its load_frames (32
            uniform candidates resized to 288), then the policy forward, then its
            get_action_by_k(keep_prob, 1, k, random_sample=False), then nonzero.
            select_frames returns frames, not indices. The script re-runs the index
            step with the upstream functions. It then calls the real select_frames
            and byte-compares the returned frames against candidates[idx].

One policy instance serves both paths. HornetSelector fails hard unless every
checkpoint tensor matches. The comparison therefore isolates the pipelines, not the
weights. The script also reports the max abs difference between the two decoded
candidate stacks and between the two keep_prob vectors. These numbers localize a
mismatch. One known benign difference: the harness divides by 255 before the bilinear
resize, upstream divides after it. The pixel difference is about one float ulp. When
keep probabilities tie, that ulp can swap a top-k boundary frame. This check exists to
catch exactly that.

--planted adds a behavioral probe. It never changes the exit code. Unrelated frames of
three kinds replace four fixed slots of the candidate pool: four frames from a
different clip, four fixed-seed noise frames (per-pixel uniform on [0,1]), and four
black frames. The probe compares the keep_prob of the planted slots against the rest.
It counts the planted slots that stay in the top-8 (content-blind chance: 1.0 of 4).

  python scripts/verify_hornet_parity.py --checkpoint CKPT clip1.mp4 clip2.mp4
  python scripts/verify_hornet_parity.py --checkpoint CKPT \
      --timeblind-root /path/to/timeblind --motionblind-root /path/to/motionblind \
      [--limit 10] [--planted] [--out parity.json]

The script needs torch, decord, transformers, PIL, and HORNet/ cloned at the repo root
(see README). Exit code 0 means every clip matches at every k in {1, 4, 8, 16, 24}.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KS = [1, 4, 8, 16, 24]
PLANT_SLOTS = [3, 11, 19, 27]


def _ensure_transformers():
    """Stub transformers only when it is absent. HORNet/model.py and
    lmms_eval_utils/hornet.py import it at module level, but the selection path does
    not use it. The stub base accepts any constructor arguments, because
    VisionGRPOConfig subclasses PretrainedConfig."""
    try:
        import transformers  # noqa: F401
        return
    except Exception:
        pass
    import types

    class _Any:
        def __init__(self, *a, **k):
            pass

    tf = types.ModuleType("transformers")
    for n in ("PreTrainedModel", "PretrainedConfig", "AutoProcessor",
              "AutoModelForImageTextToText"):
        setattr(tf, n, _Any)
    sys.modules["transformers"] = tf


class _Cfg:
    """The sweep-protocol video block (configs/eagle_*_sweep_native.yaml)."""

    def __init__(self, ckpt):
        self.video = {"hornet_checkpoint": ckpt, "frame_source": "hornet",
                      "candidates": 32, "candidate_resize": None,
                      "hornet_score_resize": 288}
        self.eval = {}


def _harness(ckpt):
    """Bind the real harness methods to a stub, so no Eagle checkpoint is necessary."""
    import torch
    from src.models.eagle import EagleModel
    from src.selectors import HornetSelector

    class _Stub:
        _decode = EagleModel._decode
        _resize = EagleModel._resize

    stub = _Stub()
    stub.torch = torch
    stub.cfg = _Cfg(ckpt)
    stub._hornet = HornetSelector(stub.cfg)
    return stub


def harness_score_frames(stub, path):
    """Prepare the candidate frames as _select_clip does for HornetSelector: native
    decode, then a resize to hornet_score_resize when candidate_resize differs."""
    v = stub.cfg.video
    candidates, _ts = stub._decode(str(path), mode="uniform", num=int(v["candidates"]),
                                   resize=v["candidate_resize"])
    score_res = int(v["hornet_score_resize"])
    if v["candidate_resize"] == score_res:
        return candidates
    return stub._resize(candidates, score_res)


def _upstream():
    """Import HORNet/lmms_eval_utils/hornet.py. Call _ensure_transformers first."""
    f = ROOT / "HORNet" / "lmms_eval_utils" / "hornet.py"
    if not f.exists():
        raise RuntimeError(f"upstream HORNet not found at {f} — clone it (see README)")
    spec = importlib.util.spec_from_file_location("_hornet_upstream", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def upstream_indices(up, model, videos, k):
    """Run the select_frames index computation, line for line, with upstream functions."""
    import torch
    with torch.no_grad():
        keep_prob = model(videos.unsqueeze(0))["keep_prob"]
        actions = up.get_action_by_k(keep_prob, 1, k, random_sample=False)
        idx = torch.nonzero(actions[0][0]).squeeze(-1).tolist()
    return idx, keep_prob[0].float().cpu()


def select_frames_agrees(up, model, path, k, videos, idx):
    """Run the real upstream select_frames and byte-compare its frames against
    candidates[idx]. This closes the gap that select_frames returns frames, not
    indices. On disagreement, the function matches each returned frame back to a
    candidate slot (-1 when no candidate matches). A boundary flip then points to
    specific slots."""
    import numpy as np
    frames, _sel_time, _total = up.select_frames(str(path), model, top_k=k)
    ours = up.fit_video_for_qwen(videos.cpu()[idx])
    if len(frames) == len(ours) and all(
            np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(frames, ours)):
        return True, list(idx)
    cand = [np.asarray(f) for f in up.fit_video_for_qwen(videos.cpu())]
    got = [next((j for j, c in enumerate(cand) if np.array_equal(np.asarray(f), c)), -1)
           for f in frames]
    return False, got


def check_clip(stub, up, model, path):
    import torch
    score = harness_score_frames(stub, path)
    with torch.no_grad():
        h_stack = torch.stack(score).float()
        h_kp = model(h_stack.unsqueeze(0).to(up.device))["keep_prob"][0].float().cpu()
        u_videos, _total = up.load_frames(str(path))
        u_videos = u_videos.to(up.device)

    frame_diff = (h_stack - u_videos.cpu()).abs().max().item()
    mismatches, per_k = [], {}
    for k in KS:
        h_idx = stub._hornet.select(score, k)
        u_idx, u_kp = upstream_indices(up, model, u_videos, k)
        f_ok, sf_slots = select_frames_agrees(up, model, path, k, u_videos, u_idx)
        ok = h_idx == u_idx and f_ok
        per_k[k] = {"match": ok, "harness": h_idx, "upstream": u_idx,
                    "select_frames_bytes_ok": f_ok,
                    **({} if f_ok else {"select_frames_slots": sf_slots})}
        tag = "match" if ok else "MISMATCH"
        detail = f"{h_idx}" if h_idx == u_idx else f"harness={h_idx} upstream={u_idx}"
        print(f"  k={k:>2}: {tag}  {detail}"
              + ("" if f_ok else f"  [select_frames returned slots {sorted(sf_slots)}]"))
        if not ok:
            mismatches.append(k)
    kp_diff = (h_kp - u_kp).abs().max().item()
    print(f"  decode |Δ|max {frame_diff:.3e}   keep_prob |Δ|max {kp_diff:.3e}")
    return mismatches, {"frame_diff": frame_diff, "kp_diff": kp_diff,
                        "keep_prob": [round(float(v), 6) for v in h_kp.tolist()],
                        "keep_prob_upstream": [round(float(v), 6) for v in u_kp.tolist()],
                        "ks": per_k}


def planted_probe(stub, model, clips):
    """Replace PLANT_SLOTS of the candidate pool with unrelated frames: donor frames
    from a different clip, fixed-seed uniform noise, and black frames. Report the
    keep_prob of the planted slots against the rest, and count the planted slots that
    stay in the top-8. The probe is informational and never changes the exit code."""
    import torch
    dev = stub._hornet.device
    stacks = []
    for ds, path in clips:
        try:
            stacks.append((ds, path, torch.stack(harness_score_frames(stub, path)).float()))
        except Exception as e:
            print(f"[planted] skip {path.name}: {type(e).__name__}")
    if not stacks:
        return []
    torch.manual_seed(0)
    noise = torch.rand(len(PLANT_SLOTS), *stacks[0][2].shape[1:])
    records = []
    print(f"\n[planted] slots {PLANT_SLOTS} replaced; content-blind chance for "
          f"in-top8 is {len(PLANT_SLOTS) * 8 / 32:.1f}/4")
    for i, (ds, path, base) in enumerate(stacks):
        variants = [("noise", noise), ("black", torch.zeros_like(noise))]
        if len(stacks) > 1:
            donor = stacks[(i + 1) % len(stacks)][2]
            variants.insert(0, ("donor", donor[[0, 8, 16, 24]]))
        for kind, plants in variants:
            planted = base.clone()
            planted[PLANT_SLOTS] = plants
            with torch.no_grad():
                kp = model(planted.unsqueeze(0).to(dev))["keep_prob"][0].float().cpu()
            order = torch.argsort(-kp, stable=True).tolist()
            ranks = sorted(order.index(s) for s in PLANT_SLOTS)
            in_top8 = sum(1 for r in ranks if r < 8)
            p_mean = kp[PLANT_SLOTS].mean().item()
            o_mean = kp[[j for j in range(32) if j not in PLANT_SLOTS]].mean().item()
            print(f"  [{kind:>5}] {path.name}: kp planted {p_mean:.4f} vs rest {o_mean:.4f}"
                  f"  in-top8 {in_top8}/4  ranks {ranks}")
            records.append({"dataset": ds, "clip": path.name, "kind": kind,
                            "kp_planted_mean": p_mean, "kp_rest_mean": o_mean,
                            "in_top8": in_top8, "planted_ranks": ranks,
                            "keep_prob": [round(float(v), 6) for v in kp.tolist()]})
    for kind in ("donor", "noise", "black"):
        rs = [r for r in records if r["kind"] == kind]
        if rs:
            mt = sum(r["in_top8"] for r in rs) / len(rs)
            md = sum(r["kp_planted_mean"] - r["kp_rest_mean"] for r in rs) / len(rs)
            print(f"[planted] {kind}: mean in-top8 {mt:.2f}/4 (chance 1.0), "
                  f"mean kp delta {md:+.4f} over {len(rs)} clips")
    return records


def gather_clips(args):
    clips = [("cli", Path(v)) for v in args.videos]
    if args.timeblind_root or args.motionblind_root:
        spec = importlib.util.spec_from_file_location(
            "_hornet_selection", Path(__file__).resolve().parent / "hornet_selection.py")
        hs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hs)
        if args.timeblind_root:
            paths = hs.unique_clips(Path(args.timeblind_root) / "data.jsonl",
                                    args.timeblind_root, "TimeBlind/")
            clips += [("timeblind", Path(p)) for p in paths[:args.limit]]
        if args.motionblind_root:
            paths = hs.unique_clips(Path(args.motionblind_root) / "data_db.jsonl",
                                    args.motionblind_root)
            clips += [("motionblind", Path(p)) for p in paths[:args.limit]]
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", help="explicit clip paths")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--timeblind-root")
    ap.add_argument("--motionblind-root")
    ap.add_argument("--limit", type=int, default=10, help="clips per dataset root")
    ap.add_argument("--planted", action="store_true",
                    help="also run the unrelated-frame keep_prob probe")
    ap.add_argument("--out", help="write a JSON report here")
    args = ap.parse_args()

    clips = gather_clips(args)
    if not clips:
        ap.error("pass clip paths and/or --timeblind-root/--motionblind-root")

    _ensure_transformers()
    stub = _harness(args.checkpoint)
    up = _upstream()
    model = stub._hornet.model  # one checkpoint-loaded policy for both paths

    bad, records = [], []
    for ds, path in clips:
        print(f"[{ds}] {path.name}")
        try:
            mism, rec = check_clip(stub, up, model, path)
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")
            bad.append((path.name, "error"))
            records.append({"dataset": ds, "clip": path.name, "error": str(e)})
            continue
        records.append({"dataset": ds, "clip": path.name, **rec})
        if mism:
            bad.append((path.name, mism))

    planted = planted_probe(stub, model, clips) if args.planted else []

    n = len(clips)
    verdict = (f"parity FAILED on {len(bad)}/{n} clips: "
               + ", ".join(f"{name} (k={m})" for name, m in bad)) if bad else (
        f"parity OK: {n} clips x k in {KS} — harness and upstream select "
        "identical frame indices")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoint": args.checkpoint, "ks": KS, "n_clips": n,
            "verdict": verdict, "parity": records,
            "planted_slots": PLANT_SLOTS if planted else None,
            "planted": planted}, indent=1))
        print(f"report -> {out}")
    print(f"\n{verdict}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
