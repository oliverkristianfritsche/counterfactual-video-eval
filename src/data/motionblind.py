"""MotionBlind loader — the lab's counterfactual-pair motion/physics benchmark
(CV4Smalls 2026 challenge; distributed as a multi-part Google Drive export).

Structure mirrors TimeBlind's yes/no half (verified 2026-07-17): 388 records / 97 pairs,
4 records per pair, sample_id = index//4, role = index%4, gold pattern (yes,no,no,yes),
balanced 194 yes / 194 no. I_Acc requires all 4 of a group correct. Chance = 50%.

video_path's leading folder is one of 9 physics-reasoning categories (force, speed,
direction, magnitude, duration, state-transitions, causal_contingency,
fine-grained_action, temporal-topology). We set Sample.category to that so metrics break
I_Acc/Acc down per reasoning-category. Everything is yes/no,
so scoring uses the yes/no path (category is never "multiple_choice").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .base import DatasetLoader, Sample
from ..registry import register_dataset


@register_dataset("motionblind")
class MotionBlindDataset(DatasetLoader):
    def load(self, limit: Optional[int] = None) -> list[Sample]:
        root = Path(self.cfg.dataset.get("root", "."))
        ann = root / self.cfg.dataset.get("annotations", "data_db.jsonl")
        if not ann.exists():
            raise FileNotFoundError(
                f"MotionBlind annotations not found: {ann} — extract the Drive export and "
                "point dataset.root at the 'MotionBlind Dataset' dir. Use data_db.jsonl "
                "(canonical), NOT data.jsonl (stale draft)."
            )
        if limit and limit % 4 != 0:
            print(f"[motionblind] WARN: limit={limit} not divisible by 4 — last group incomplete, "
                  "pair metrics (Q/V/I_Acc) drop it")
        # optional category restriction, e.g. dataset.categories: [force, speed, magnitude] to run
        # only the blind categories. Whole 4-role groups are kept, so pair metrics stay valid.
        only = self.cfg.dataset.get("categories")
        only = set(only) if only else None

        out = []
        with open(ann, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx = int(obj["index"])
                vp = str(obj.get("video_path") or "")
                cat = vp.split("/")[0] if "/" in vp else "all"   # physics-reasoning category
                if only and cat not in only:
                    continue
                out.append(Sample(
                    id=str(idx),
                    question=obj.get("question", ""),
                    frames=[],
                    options=None,
                    answer=obj.get("answer"),            # "yes" | "no"
                    category=cat,                        # per-category I_Acc breakdown
                    instance_id=str(idx // 4),           # counterfactual-pair grouping
                    meta={"video_path": str(root / vp), "role": idx % 4,
                          "qa_type": obj.get("type", "yes_no")},
                ))
                if limit and len(out) >= limit:
                    break
        return out
