"""Contrastive-pair MotionBlind loader.

Builds two-video comparison items from the 97 counterfactual pairs: for each pair and each pole
question, present both clips and ask which one matches that pole *rather than the other*, in both
presentation orders.

Source is the same data_db.jsonl as the isolated loader (src/data/motionblind.py). Per pair
(sample_id = index // 4, role = index % 4, gold pattern yes,no,no,yes):
    clip A = roles 0,2 video; clip B = roles 1,3 video.
    Q1 = roles 0,1 question; Q2 = roles 2,3 question (opposite poles).
    the pole's gold clip is the role answered "yes".
Each pole x each order (AB, BA) yields one first/second item, so up to 4 items per pair. The two
orders of one (pair, pole) share an instance_id.

Rewording is deliberately COMPARATIVE, not absolute: "In which one does the person tap the table
forcefully, rather than gently?" names the opposite pole as the reference, so the model makes a
relative judgment (which clip is more forceful) instead of the absolute one it cannot calibrate in
isolation. The reference phrase is the other pole's distinctive words, so no adjective morphology is
needed and any stem phrasing works. Stems whose leading form is not Does/Do/Is/Are (e.g. "After ...")
are skipped and counted; every force/speed/magnitude stem is Does/Is, so the primary is unaffected.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .base import DatasetLoader, Sample
from ..registry import register_dataset

_LEADING = ("does", "do", "is", "are")
_PROMPT = ("Here are two videos, a first and a second. In which one {lead} {pred}, rather than "
           "{contrast}? Answer with only 'first' or 'second'.")


def _split_stem(stem: str):
    """(leading verb, predicate) from a yes/no stem, or (None, None) if unsupported."""
    s = (stem or "").strip().rstrip("?").strip()
    parts = s.split(None, 1)
    if len(parts) < 2:
        return None, None
    lead = parts[0].lower()
    if lead not in _LEADING:
        return None, None
    return lead, parts[1]


def reword(query_stem: str, contrast_stem: str) -> Optional[str]:
    """Comparative two-video question: which clip matches the queried pole rather than the other.

    Names both poles so the judgment is relative, not absolute. The reference is the contrast pole's
    distinctive words (the tail after the shared action prefix). Returns None if either stem has an
    unsupported leading form.
    """
    lead_q, pred_q = _split_stem(query_stem)
    _, pred_c = _split_stem(contrast_stem)
    if pred_q is None or pred_c is None:
        return None
    q_words, c_words = pred_q.split(), pred_c.split()
    # strip the shared action prefix and the shared trailing words -> the pole's distinctive span
    i = 0
    while i < len(q_words) and i < len(c_words) and q_words[i] == c_words[i]:
        i += 1
    j = 0
    while (j < len(q_words) - i and j < len(c_words) - i
           and q_words[-1 - j] == c_words[-1 - j]):
        j += 1
    contrast = " ".join(c_words[i:len(c_words) - j]) or pred_c
    return _PROMPT.format(lead=lead_q, pred=pred_q, contrast=contrast)


@register_dataset("motionblind_contrastive")
class MotionBlindContrastiveDataset(DatasetLoader):
    def load(self, limit: Optional[int] = None) -> list[Sample]:
        root = Path(self.cfg.dataset.get("root", "."))
        ann = root / self.cfg.dataset.get("annotations", "data_db.jsonl")
        if not ann.exists():
            raise FileNotFoundError(
                "MotionBlind annotations not found: {} — point dataset.root at the MotionBlind "
                "dir (canonical data_db.jsonl).".format(ann)
            )
        only = self.cfg.dataset.get("categories")  # optional list -> restrict to these categories
        only = set(only) if only else None

        # group the 388 rows into 97 four-role pairs
        pairs: dict = {}
        with open(ann, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx = int(obj["index"])
                pairs.setdefault(idx // 4, {})[idx % 4] = obj

        out: list[Sample] = []
        skipped = 0
        for pid in sorted(pairs):
            roles = pairs[pid]
            if sorted(roles) != [0, 1, 2, 3]:
                continue
            cat = str(roles[0].get("video_path", "")).split("/")[0] or "all"
            if only and cat not in only:
                continue
            clip_a = str(root / roles[0]["video_path"])  # roles 0,2
            clip_b = str(root / roles[1]["video_path"])  # roles 1,3
            # guard: the runtime must honour the structure the hygiene check validated offline,
            # so a future data change cannot silently mis-assign the gold clip.
            if str(root / roles[2]["video_path"]) != clip_a or str(root / roles[3]["video_path"]) != clip_b:
                print("[motionblind_contrastive] pair {}: roles 0,2 / 1,3 clip mismatch — skipped".format(pid))
                continue
            # (question, clip-A answer, clip-B answer) per pole
            poles = {
                "Q1": (roles[0]["question"], roles[0]["answer"], roles[1]["answer"]),
                "Q2": (roles[2]["question"], roles[2]["answer"], roles[3]["answer"]),
            }
            for pole, (stem, ans_a, ans_b) in poles.items():
                contrast_stem = poles["Q2"][0] if pole == "Q1" else poles["Q1"][0]
                prompt = reword(stem, contrast_stem)
                if prompt is None:
                    skipped += 1
                    continue
                yes_clip = "A" if str(ans_a).lower() == "yes" else "B"
                for order in ("AB", "BA"):
                    vids, first_clip = ([clip_a, clip_b], "A") if order == "AB" else ([clip_b, clip_a], "B")
                    gold = "first" if yes_clip == first_clip else "second"
                    out.append(Sample(
                        id="{}|{}|{}".format(pid, pole, order),
                        question=prompt,
                        frames=[],
                        options=["first", "second"],
                        answer=gold,
                        category=cat,
                        instance_id="{}|{}".format(pid, pole),   # the both-orders unit
                        meta={"video_paths": vids, "pair_id": pid, "pole": pole,
                              "order": order, "yes_clip": yes_clip, "raw_question": stem,
                              "clip_a": clip_a, "clip_b": clip_b},
                    ))
        if skipped:
            print("[motionblind_contrastive] skipped {} pole-stems with unsupported leading "
                  "form (non Does/Do/Is/Are)".format(skipped))
        return out[:limit] if limit else out
