"""Frame-selection strategies — mirrors the paper's Table 5 ablation (Random / Uniform /
HORNet). Real adapters decode candidate frames the way HORNet does (32 uniform candidates
at 288x288, see reference/hornet/lmms_eval_utils/hornet.py load_frames) and then apply one
of these to pick top_k.
"""
from __future__ import annotations

import random as _random


def uniform_indices(num_candidates: int, k: int) -> list[int]:
    """Evenly spaced k of num_candidates (numpy.linspace semantics, dependency-free)."""
    if num_candidates <= 0:
        return []
    if k >= num_candidates:
        return list(range(num_candidates))
    if k == 1:
        return [num_candidates // 2]  # single frame: take the midpoint
    step = (num_candidates - 1) / (k - 1)
    return sorted(set(int(round(i * step)) for i in range(k)))


def random_indices(num_candidates: int, k: int, seed: int) -> list[int]:
    rng = _random.Random(seed)
    k = min(k, num_candidates)
    return sorted(rng.sample(range(num_candidates), k)) if num_candidates else []


class HornetSelector:
    """HORNet frame selection through the authors' OWN code, imported unmodified.

    Constructs the upstream VisionGRPOPolicy from HORNet/model.py and applies their
    get_action_by_k reduction from HORNet/util.py, so the selected frames come from the
    published HORNet code path. Run with the trained policy checkpoint (checkpoint-0.550.pt).

    HORNet/ is the authors' repository, cloned as a dependency (see README); it is not
    vendored into this release. The two selection-irrelevant modules model.py pulls (vp, and
    util's `from util import *`) are stubbed; the frozen answerer the constructor demands is
    a dummy nn.Linear, never used for selection. get_action_by_k is loaded from the real
    util.py source (transformers stubbed only if unavailable — that function does not use
    it), so the top-k reduction is upstream code too. With n_samples=1 and
    random_sample=False it reduces to deterministic top-k, matching their
    lmms_eval_utils/hornet.py select_frames.

    Checkpoint: config video.hornet_checkpoint. checkpoint=None -> random init (tests only).
    """

    def __init__(self, cfg):
        import sys
        import types
        import importlib.util as ilu
        from pathlib import Path
        import torch
        import torch.nn as nn
        self.torch = torch
        ref = Path(__file__).resolve().parent.parent / "HORNet"
        if not (ref / "model.py").exists():
            raise RuntimeError(
                f"upstream HORNet not found at {ref} — clone it there (see README) before "
                "using frame_source: hornet")

        # stub the selection-irrelevant imports model.py pulls
        if "vp" not in sys.modules:
            vp = types.ModuleType("vp")
            vp.TorchVideoPrism = object          # only touched when encoder_name is not None
            vp.FrozenVideoPrismEncoder = object
            sys.modules["vp"] = vp
        if "util" not in sys.modules:
            u = types.ModuleType("util")          # model.py does `from util import *`
            u.__all__ = []
            sys.modules["util"] = u
        if str(ref) not in sys.path:
            sys.path.insert(0, str(ref))
        from model import VisionGRPOPolicy  # their implementation, unmodified

        # their get_action_by_k, from the real util.py source (transformers stubbed if absent)
        try:
            import transformers  # noqa: F401
        except Exception:
            tf = types.ModuleType("transformers")
            tf.AutoProcessor = object
            tf.AutoModelForImageTextToText = object
            sys.modules["transformers"] = tf
        spec = ilu.spec_from_file_location("_hornet_util", ref / "util.py")
        _hu = ilu.module_from_spec(spec)
        spec.loader.exec_module(_hu)
        self._get_action_by_k = _hu.get_action_by_k

        dummy_qwen = nn.Linear(1, 1)  # constructor freezes it; never used for selection
        self.model = VisionGRPOPolicy(encoder_name=None, feat_dim=768, action_dim=1,
                                      qwen_model=dummy_qwen, qwen_processor=None)
        ckpt_path = (cfg.video.get("hornet_checkpoint") if hasattr(cfg, "video") else None)
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            result = self.model.load_state_dict(state, strict=False)
            loaded = len(state) - len(result.unexpected_keys)
            print(f"[hornet] checkpoint {ckpt_path}: matched {loaded}/{len(state)} "
                  f"tensors (unexpected: {len(result.unexpected_keys)})")
            if loaded != len(state):
                raise RuntimeError(
                    f"upstream HORNet checkpoint only matched {loaded}/{len(state)} tensors — "
                    "key mismatch vs HORNet/model.py VisionGRPOPolicy")
        else:
            print("[hornet] WARNING: no checkpoint — random policy (tests only)")
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def select(self, candidates, k: int) -> list[int]:
        """candidates: list of [H,W,3] float tensors in [0,1] -> top-k indices, sorted.

        Runs the upstream policy forward for keep_prob, then the upstream
        get_action_by_k(keep_prob, 1, k, random_sample=False) reduction (deterministic
        top-k), exactly as HORNet/lmms_eval_utils/hornet.py select_frames."""
        with self.torch.no_grad():
            videos = self.torch.stack(list(candidates)).unsqueeze(0).float().to(self.device)
            keep_prob = self.model(videos)["keep_prob"]           # [1, T]
            k = min(k, keep_prob.shape[1])
            actions = self._get_action_by_k(keep_prob, 1, k, random_sample=False)  # [1, 1, T]
            idx = self.torch.nonzero(actions[0][0]).squeeze(-1).tolist()
        return sorted(idx)
