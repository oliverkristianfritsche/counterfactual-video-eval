"""Synthetic samples in HORNet's MCQ format (options = choice texts, answer = digit
index) so the smoke test exercises the exact scoring path real runs will use.

With dataset.pair_mode set, emits the counterfactual 4-role yes/no structure instead
(role = index % 4, instance_id = index // 4, gold yes/no/no/yes), matching the layout the
grouped Q/V/I_Acc scorer and the offline decision-rule scripts consume. That lets the
decode path be exercised end to end with no GPU, weights or benchmark data."""
from __future__ import annotations

from typing import Optional

from .base import DatasetLoader, Sample
from ..registry import register_dataset

_CHOICES = ["walking", "running", "jumping", "sitting"]
_CATS = ["motion", "temporal", "counting"]


@register_dataset("mock")
class MockDataset(DatasetLoader):
    def load(self, limit: Optional[int] = None) -> list[Sample]:
        n = int(self.cfg.dataset.get("num_samples", 20))
        if limit:
            n = min(n, limit)
        if self.cfg.dataset.get("pair_mode"):
            return self._pairs(n - n % 4)
        qs_per_video = int(self.cfg.dataset.get("questions_per_video", 2))
        return [
            Sample(
                id=f"mock-{i:04d}",
                question=f"What is the person doing in synthetic clip #{i}?",
                frames=[],
                options=list(_CHOICES),
                answer=i % len(_CHOICES),          # digit index, HORNet MCQ protocol
                category=_CATS[i % len(_CATS)],
                instance_id=f"vid-{i // qs_per_video:03d}",  # exercises instance accuracy
                meta={"synthetic": True},
            )
            for i in range(n)
        ]

    def _pairs(self, n: int) -> list[Sample]:
        """The counterfactual 4-role layout: two questions crossed with two videos whose gold
        answers are opposite by construction (roles 0/3 gold yes, roles 1/2 gold no)."""
        out = []
        for i in range(n):
            g, role = i // 4, i % 4
            q, v = role // 2, role % 2
            out.append(Sample(
                id=str(i),
                question=f"Does event {q} happen before event {1 - q} in synthetic clip {g}-{v}?",
                frames=[],
                options=None,
                answer="yes" if role in (0, 3) else "no",
                category="yes_no",
                instance_id=str(g),
                meta={"synthetic": True, "role": role},
            ))
        return out
