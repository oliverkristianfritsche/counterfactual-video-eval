"""Dataset adapter interface + the common Sample type every loader yields."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sample:
    id: str
    question: str
    frames: list = field(default_factory=list)   # decoded frames (adapters usually decode themselves)
    options: Optional[list] = None               # choice texts for MCQ
    answer: Optional[str] = None                 # digit index (choice) or gold text (open-ended)
    category: Optional[str] = None               # e.g. "motion" / "temporal" — enables per-cat acc
    instance_id: Optional[str] = None            # groups questions about the same video/instance —
                                                 # drives instance accuracy + num_videos
    meta: dict = field(default_factory=dict)     # anything extra (video_path, etc.)


class DatasetLoader(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def load(self, limit: Optional[int] = None) -> list[Sample]:
        """Return the eval samples (respecting `limit` if given)."""
