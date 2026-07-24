"""Mock model — deterministic, no weights, no GPU. Validates the full pipeline:
selection timing, MCQ digit protocol, invalid-answer handling, metrics, W&B, results.

Behavior (seeded by sample id): ~70% correct digit, ~20% wrong digit, ~10% unparseable
text (exercises invalid_rate).
"""
from __future__ import annotations

import hashlib

from .base import VLMAdapter
from ..registry import register_model


@register_model("mock")
class MockModel(VLMAdapter):
    name = "mock"

    def load(self) -> None:
        self._loaded = True

    def select_frames(self, sample) -> list:
        # stand-in frame selection: k integer indices, recorded as selected_indices
        # in predictions.jsonl
        k = int(self.cfg.model.get("top_k", 4))
        return list(range(k))

    def _perturb_attenuation(self, sample) -> float:
        """Stand-in for what a temporal perturbation does to the answer signal: with eval.perturb
        set, most items keep a weakened version of the signal and a seeded minority flip, so a
        perturbed arm differs from the real one and the offline decode scripts have two arms to
        contrast. Real adapters perturb the frames themselves; the mock has none."""
        pert = self.cfg.eval.get("perturb") if hasattr(self.cfg, "eval") else None
        if not pert:
            return 1.0
        seed = pert.get("seed", 0) if isinstance(pert, dict) else 0
        h = int(hashlib.md5(f"{sample.id}:{seed}".encode()).hexdigest(), 16)
        return 0.3 if h % 10 < 7 else -0.3

    def generate(self, sample, frames, user_prompt, system_prompt) -> str:
        self.last_yesno_logits = None
        h = int(hashlib.md5(sample.id.encode()).hexdigest(), 16)
        roll = h % 10
        ans = str(sample.answer).strip().lower() if sample.answer is not None else ""
        if ans in ("yes", "no"):
            # synthetic yes/no logits with an injected affirmative bias, so bias-calibration
            # (offline bias calibration) has something to correct: ~70% aligned with gold, plus a yes-bias.
            gold = 1.0 if ans == "yes" else -1.0
            signal = 0.6 * (gold if roll < 7 else -gold)   # noisy toward gold
            signal *= self._perturb_attenuation(sample)
            yes_bias = 0.8
            lyes, lno = signal + yes_bias, -signal
            self.last_yesno_logits = {"yes": lyes, "no": lno}
            return "Yes" if lyes >= lno else "No"
        if sample.options is not None and sample.answer is not None:
            try:
                gold = int(sample.answer)
            except (TypeError, ValueError):
                # word options (contrastive "first"/"second"): answer with the option text
                opts = [str(o) for o in sample.options]
                gold_i = opts.index(str(sample.answer)) if str(sample.answer) in opts else 0
                if roll < 7:
                    return f"The {opts[gold_i]} one."
                if roll < 9:
                    return f"The {opts[(gold_i + 1) % len(opts)]} one."
                return "I am not sure about this one."  # unparseable -> invalid_rate
            if roll < 7:
                return f"{gold}."
            if roll < 9:
                return f"{(gold + 1) % len(sample.options)}."
            return "I am not sure about this one."  # unparseable -> invalid_rate
        # open-ended fallback
        return str(sample.answer) if roll < 7 else "something else entirely"
