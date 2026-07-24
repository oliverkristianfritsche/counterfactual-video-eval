"""TimeBlind loader — the official BaiqiL/TimeBlind benchmark (arXiv 2602.00288).

Structure (reference/timeblind/utils.py): samples come in groups of 4 consecutive
indices — sample_id = index // 4, role = index % 4:
    role 0 = q0 on video_0 (gold yes)   role 1 = q0 on video_1 (gold no)
    role 2 = q1 on video_0 (gold no)    role 3 = q1 on video_1 (gold yes)
instance_id = the group; I_Acc requires all four correct. Keep limits divisible by 4
so groups stay intact (the runner's --limit is applied as-is).

Get the data: hf download BaiqiL/TimeBlind --repo-type dataset --local-dir <root>
(jsonl video_path values carry a leading "TimeBlind/" repo prefix — stripped here).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .base import DatasetLoader, Sample
from ..registry import register_dataset


@register_dataset("timeblind")
class TimeBlindDataset(DatasetLoader):
    def load(self, limit: Optional[int] = None) -> list[Sample]:
        root = Path(self.cfg.dataset.get("root", "."))
        ann = root / self.cfg.dataset.get("annotations", "data.jsonl")
        if not ann.exists():
            raise FileNotFoundError(
                f"TimeBlind annotations not found: {ann} — "
                "hf download BaiqiL/TimeBlind --repo-type dataset --local-dir <root>"
            )
        if limit and limit % 4 != 0:
            print(f"[timeblind] WARN: limit={limit} not divisible by 4 — last group will be "
                  "incomplete and pair metrics (Q/V/I_Acc) will drop it")

        out = []
        with open(ann, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx = int(obj["index"])
                vp = str(obj.get("video_path") or "")
                if vp.startswith("TimeBlind/"):
                    vp = vp.split("/", 1)[1]
                out.append(Sample(
                    id=str(idx),
                    question=obj.get("question", ""),
                    frames=[],
                    options=None,
                    answer=obj.get("answer"),            # "yes" | "no"
                    category=obj.get("type"),            # "yes_no" | "multiple_choice"
                    instance_id=str(idx // 4),           # official sample grouping
                    meta={"video_path": str(root / vp), "role": idx % 4},
                ))
                if limit and len(out) >= limit:
                    break
        return out
