"""Model adapter interface. Add a new VLM = one subclass + @register_model.

The two-stage split (select_frames -> generate) mirrors HORNet's pipeline and lets the
runner time frame selection and VLM processing separately — the paper's "Frame Sel. (s)"
and "Qwen Proc. (s)" columns come from exactly this split.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class VLMAdapter(ABC):
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def load(self) -> None:
        """Load weights / processor. Called once before the loop."""

    def select_frames(self, sample) -> list:
        """Decode candidates + pick the frames the VLM will see. Timed as selection.

        Default: pass through whatever the dataset already attached. Real adapters decode
        the video (HORNet protocol: 32 uniform candidates @ 288x288) and apply the
        configured selector (uniform / random / hornet).
        """
        return list(sample.frames or [])

    @abstractmethod
    def generate(self, sample, frames: list, user_prompt: str, system_prompt: str) -> str:
        """Answer one sample given the selected frames. Timed as VLM processing."""

    def unload(self) -> None:
        """Free resources (optional)."""
