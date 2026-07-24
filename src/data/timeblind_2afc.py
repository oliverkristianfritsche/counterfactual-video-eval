"""TimeBlind single-clip 2AFC loader (response-format arm).

Reformats TimeBlind's yes/no half into a single-clip two-alternative forced choice, holding the CLIP
and the CONTENT constant and changing only the response format. This is the clean format comparison (the benchmark's own yes/no and multiple-choice halves use
different video files, so their gap confounds format with difficulty): a 2AFC over ONE clip,
with no second clip present.

Source is the same data.jsonl as the yes/no loader (src/data/timeblind.py). Each yes/no group
(sample_id = index // 4) is:
    role 0 = q0 on video_0 (gold yes)   role 1 = q0 on video_1 (gold no)
    role 2 = q1 on video_0 (gold no)    role 3 = q1 on video_1 (gold yes)
so q0 and q1 are two mutually-exclusive temporal hypotheses, video_0's true hypothesis is q0 and
video_1's is q1. For each clip we build a forced choice between q0 and q1 in BOTH option orders
(A=q0/B=q1 and A=q1/B=q0) to cancel position bias, mirroring the benchmark's own MCQ counterbalance:
4 items per group (2 clips x 2 orders), 1200 total, matching the 1200 yes/no rows.

The two poles are independently-worded paraphrases, not minimal edits, so they are presented
verbatim as the options under a "for exactly one the answer is yes" frame rather than string-diffed
into a synthetic stem — no wording is introduced by us. Scoring reuses the A/B path
(category = multiple_choice -> score_timeblind extract_ab); the runner appends "Please output A or B."
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .base import DatasetLoader, Sample
from ..registry import register_dataset

_PROMPT = ("For exactly one of the following questions about the video the answer is yes:\n"
           "A: {a}\nB: {b}\nWhich one is it?")


@register_dataset("timeblind_2afc")
class TimeBlind2AFCDataset(DatasetLoader):
    def load(self, limit: Optional[int] = None) -> list[Sample]:
        root = Path(self.cfg.dataset.get("root", "."))
        ann = root / self.cfg.dataset.get("annotations", "data.jsonl")
        if not ann.exists():
            raise FileNotFoundError(
                f"TimeBlind annotations not found: {ann} — "
                "hf download BaiqiL/TimeBlind --repo-type dataset --local-dir <root>")

        def vpath(vp):
            vp = str(vp or "")
            if vp.startswith("TimeBlind/"):
                vp = vp.split("/", 1)[1]
            return str(root / vp)

        yn = []
        with open(ann, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                if o.get("type") == "yes_no":       # A/B multiple-choice half is not reformatted here
                    yn.append(o)

        fmt = (self.cfg.dataset.get("format") or "2afc").lower()
        if fmt == "binary":
            # FORMAT-ONLY arm (format-only control): relabel each single yes/no question as an A/B
            # choice between "yes" and "no" — one hypothesis, no second pole, no exclusivity cue — so
            # it differs from native yes/no ONLY in the output token. Both A/B mappings (A=yes/B=no and
            # A=no/B=yes) cancel position bias. Isolates the pure response-format effect; the two-pole
            # 2afc arm layers co-presentation + exclusivity on top.
            out = []
            for o in yn:
                idx = int(o["index"])
                q = o["question"]
                yesans = str(o["answer"]).strip().lower() == "yes"
                for order, (oa, ob) in (("YN", ("yes", "no")), ("NY", ("no", "yes"))):
                    gold = "A" if (yesans == (oa == "yes")) else "B"
                    out.append(Sample(
                        id=f"{idx}|{order}",
                        question=f"{q}\nA: {oa}\nB: {ob}\nWhich is correct?",
                        frames=[], options=["A", "B"], answer=gold, category="multiple_choice",
                        instance_id=str(idx // 4),
                        meta={"video_path": vpath(o["video_path"]), "role": idx % 4,
                              "orig_index": idx, "order": order, "qa_type": "binary"}))
            return out[:limit] if limit else out

        # default: two-pole single-clip 2AFC. group the yes/no rows into 4-role groups
        groups: dict = {}
        for o in yn:
            idx = int(o["index"])
            groups.setdefault(idx // 4, {})[idx % 4] = o

        out: list[Sample] = []
        skipped = 0
        for sid in sorted(groups):
            g = groups[sid]
            if sorted(g) != [0, 1, 2, 3]:
                skipped += 1
                continue
            q0, q1 = g[0]["question"], g[2]["question"]
            # guard the assumed structure: roles 0,2 share a clip; 1,3 share the other; the two
            # poles are DISTINCT (a group with q0==q1 is degenerate — the 2AFC would offer the same
            # option twice, and its yes/no gold is self-contradictory; one such group exists in the
            # real data); and the gold pattern is yes,no,no,yes.
            if (g[0]["video_path"] != g[2]["video_path"] or g[1]["video_path"] != g[3]["video_path"]
                    or q0 != g[1]["question"] or q1 != g[3]["question"]
                    or q0.strip().lower() == q1.strip().lower()
                    or [str(g[r]["answer"]).lower() for r in (0, 1, 2, 3)] != ["yes", "no", "no", "yes"]):
                skipped += 1
                continue
            # video_0 (roles 0,2): true pole is q0 ; video_1 (roles 1,3): true pole is q1
            clips = {"v0": (vpath(g[0]["video_path"]), q0), "v1": (vpath(g[1]["video_path"]), q1)}
            role = 0
            for clip, (path, true_q) in clips.items():
                for order in ("AB", "BA"):
                    a, b = (q0, q1) if order == "AB" else (q1, q0)
                    gold = "A" if a == true_q else "B"
                    out.append(Sample(
                        id=f"{sid}|{clip}|{order}",
                        question=_PROMPT.format(a=a, b=b),
                        frames=[],
                        options=["A", "B"],
                        answer=gold,
                        category="multiple_choice",       # -> score_timeblind uses the A/B path
                        instance_id=str(sid),
                        meta={"video_path": path, "role": role, "clip": clip, "order": order,
                              "true_pole": true_q, "sid": sid, "qa_type": "2afc"},
                    ))
                    role += 1
        if skipped:
            print(f"[timeblind_2afc] skipped {skipped} groups (structure guard); "
                  f"{len(out)} items from {len(out)//4} groups")
        return out[:limit] if limit else out
