#!/usr/bin/env python3
"""Validate the HORNet selector (frame_source: hornet), which imports the authors' own
VisionGRPOPolicy + get_action_by_k unmodified from HORNet/ (cloned, see README).

Part A (no args): construct HornetSelector with no checkpoint (random-init policy) and check
select() returns k sorted-unique in-range indices deterministically, and that the upstream
get_action_by_k(keep_prob, 1, k, random_sample=False) reduction equals torch.topk.

Part B (checkpoint [+ videos]): load the checkpoint (the selector asserts every tensor
matched) and select frames — from the given videos if provided, else random — reporting the
picks at each k in {1,4,8,16,24}. Run with the main checkpoint-0.550.pt to confirm it
loads and selects non-degenerately on real clips.

  python scripts/verify_hornet.py
  python scripts/verify_hornet.py /path/to/hornet/checkpoint-0.550.pt clip1.mp4 clip2.mp4

Needs HORNet/ cloned at the repo root (see README) and torch; Part B also needs decord when
videos are passed.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.selectors import HornetSelector  # noqa: E402

KS = [1, 4, 8, 16, 24]


class _Cfg:
    def __init__(self, ckpt=None):
        self.video = {"hornet_checkpoint": ckpt} if ckpt else {}
        self.eval = {}


def _load_frames(path, n=32, size=288):
    import decord
    import numpy as np
    vr = decord.VideoReader(str(path))
    total = len(vr)
    if total >= n:
        idx = np.linspace(0, total - 1, n).astype(int)
    else:
        idx = np.array(list(range(total)) + [total - 1] * (n - total))
    f = torch.from_numpy(vr.get_batch(idx).asnumpy()).float()
    f = torch.nn.functional.interpolate(f.permute(0, 3, 1, 2), size=(size, size),
                                        mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    return list(f / 255.0)


def part_a():
    torch.manual_seed(0)
    sel = HornetSelector(_Cfg())
    frames = [torch.rand(288, 288, 3) for _ in range(32)]
    for k in KS:
        idx = sel.select(frames, k)
        assert len(idx) == k and idx == sorted(set(idx)) and all(0 <= i < 32 for i in idx), (k, idx)
        assert sel.select(frames, k) == idx, ("nondeterministic", k)
        with torch.no_grad():
            kp = sel.model(torch.stack(frames).unsqueeze(0).float().to(sel.device))["keep_prob"]
        assert idx == sorted(torch.topk(kp[0], k).indices.tolist()), ("reduction != topk", k)
    print("part A: HornetSelector returns k sorted-unique deterministic indices; "
          "get_action_by_k(random_sample=False) == top-k — OK")


def part_b(ckpt, videos):
    sel = HornetSelector(_Cfg(ckpt))  # raises unless every checkpoint tensor matches
    if videos:
        clips = [(Path(v).name, _load_frames(v)) for v in videos]
    else:
        torch.manual_seed(1)
        clips = [("random", [torch.rand(288, 288, 3) for _ in range(32)])]
    for name, frames in clips:
        for k in KS:
            idx = sel.select(frames, k)
            assert len(idx) == k and idx == sorted(set(idx)), (name, k, idx)
            print(f"  {name} k={k:>2}: {idx}")
    print("part B: checkpoint loaded and frames selected — OK")


if __name__ == "__main__":
    part_a()
    if len(sys.argv) > 1:
        part_b(sys.argv[1], sys.argv[2:])
