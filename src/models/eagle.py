"""Eagle answerer adapter — Eagle as the frozen VLM behind (or without) HORNet selection.

Written to mirror HORNet's own answerer recipe (reference/hornet/util.py:
load_qwen_model + qwen_answer_question) so model behavior is comparable across runs:
  * bf16, device_map=auto, trust_remote_code (Eagle loads via AutoModel, see below)
  * greedy decode, max_new_tokens=128

Frame resolution is decoupled by design: candidates are drawn from a 32-position pool
(HORNet paper Table-5 protocol), but Eagle answers at video.candidate_resize (default
null = native, to match the native-resolution protocol) while the hornet selector
scores 288x288 copies for selection only. See select_frames().

Loading/inference follows the model card's own video example verbatim (cached snapshot
README, "single video" section): AutoModel + trust_remote_code, AutoProcessor with
use_fast=True, tokenizer.padding_side="left", and the processor's OWN
process_vision_info (no qwen-vl-utils dependency). Requires transformers==4.55.4 —
the card's pin; config.json's 4.51.0 stamp is stale (code needs auto_docstring, >=4.52)
and 5.x removes DefaultFastImageProcessorKwargs. Checkpoint:
nvidia/Eagle2.5-8B — GATED on HF (accept license + `hf auth login`), ~16.2 GB bf16.
dtype auto-selects: bf16 on Ampere+ (A100/H200), fp16 on older GPUs (e.g. V100, sm70).
The published runs used bf16; fp16 is a different numeric path and can shift borderline logits.
"""
from __future__ import annotations

from .base import VLMAdapter
from .. import selectors
from ..registry import register_model


@register_model("eagle")
class EagleModel(VLMAdapter):
    name = "eagle"

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModel, AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                "Eagle needs torch + transformers==4.55.4 (+ decord for video): "
                "pip install torch 'transformers==4.55.4' accelerate decord"
            ) from e

        ckpt = self.cfg.model.get("checkpoint")
        if not ckpt:
            raise ValueError("Set model.checkpoint (nvidia/Eagle2.5-8B).")
        self.torch = torch
        self._patch_frame_timestamps(ckpt)
        # bf16 needs Ampere+ (A100/H200); older cards (V100, sm70) fall back to fp16
        dtype = torch.bfloat16 if (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        ) else torch.float16
        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True, use_fast=True)
        self.processor.tokenizer.padding_side = "left"
        # Tile cap: the processor's video path falls back to
        # image_processor.max_dynamic_tiles (12) whenever kwargs merging drops the
        # videos_kwargs default of 1. Mutate the instance default so the fallback
        # itself is the video-correct value. 12 tiles/frame OOMs a V100-32GB.
        tiles = int(self.cfg.video.get("max_dynamic_tiles", 1))
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None and hasattr(ip, "max_dynamic_tiles"):
            ip.max_dynamic_tiles = tiles
            print(f"[eagle] image_processor.max_dynamic_tiles -> {tiles}")

        # Eagle's config.json pins flash_attention_2 with autoset:false, so the plain
        # attn_implementation kwarg loses — force it on the config object at all levels.
        # (The vision tower additionally HARDCODES flash in the remote code; run
        # remote code once per machine to make it inherit instead.)
        attn = self.cfg.model.get("attn_implementation", "sdpa")
        config = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
        for c in [config, getattr(config, "text_config", None), getattr(config, "vision_config", None)]:
            if c is None:
                continue
            c._attn_implementation = attn
            try:
                c._attn_implementation_autoset = True
            except Exception:
                pass
        self.model = AutoModel.from_pretrained(
            ckpt,
            config=config,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        # self-report the ACTUAL backends so every run's log proves its configuration
        # (requesting flash without the lib hard-crashes, so this can't silently lie)
        print(f"[eagle] dtype={dtype} | attn(text)="
              f"{getattr(self.model.config, '_attn_implementation', '?')} | attn(vision)="
              f"{getattr(getattr(self.model.config, 'vision_config', None), '_attn_implementation', '?')}")

    def _patch_frame_timestamps(self, ckpt) -> None:
        """Rewrite Eagle's cached processing_eagle2_5_vl.py so the frame-list branch of
        fetch_video passes caller-supplied per-frame timestamps through instead of
        hardcoding -1, which renders literally as "Frame i sample at -1.00s" in prompts
        (see README, "Eagle timestamps patch"). Must run BEFORE transformers imports the
        remote code; transformers 4.55 filecmp-syncs the patched snapshot into
        transformers_modules, so patching the snapshot is sufficient (existing module
        copies are patched too in case a stale one predates this load). Idempotent;
        no-ops once upstream accepts timestamps itself; raises if upstream changed shape,
        so a frame-list run can never silently mislabel its temporal metadata."""
        from pathlib import Path
        marker = "patched: timestamps passthrough"
        old_pop = (
            '        process_info = ele.copy()\n'
            '        process_info.pop("type", None)\n'
            '        process_info.pop("video", None)'
        )
        # pop, not get: **process_info is splatted into fetch_image per frame below
        new_pop = old_pop + '\n        _caller_ts = process_info.pop("timestamps", None)  # ' + marker
        old_ts = '        timestamps = [-1 for i in range(nframes)] # not sure about this'
        # nframes may be rounded up with the last frame duplicated; pad timestamps to match
        new_ts = (
            '        if _caller_ts is not None:  # ' + marker + '\n'
            '            timestamps = list(_caller_ts) + [_caller_ts[-1]] * (nframes - len(_caller_ts))\n'
            '        else:\n    ' + old_ts
        )
        fname = "processing_eagle2_5_vl.py"
        targets = []
        if Path(ckpt).is_dir():
            targets.append(Path(ckpt) / fname)
        else:
            try:
                from huggingface_hub import hf_hub_download
                targets.append(Path(hf_hub_download(ckpt, fname)))
            except Exception:  # offline node: fall back to the warm snapshot cache
                from huggingface_hub.constants import HF_HUB_CACHE
                repo = "models--" + str(ckpt).replace("/", "--")
                targets += sorted(Path(HF_HUB_CACHE).glob(f"{repo}/snapshots/*/{fname}"))
        try:
            from transformers.utils import HF_MODULES_CACHE
            targets += sorted(Path(HF_MODULES_CACHE).glob(f"transformers_modules/**/{fname}"))
        except ImportError:
            pass
        if not targets:
            raise RuntimeError(f"cannot locate {fname} for {ckpt} (no download, no cache)")
        for f in targets:
            if not f.exists():
                continue
            s = f.read_text()
            if marker in s or 'process_info.pop("timestamps"' in s:
                continue  # already patched / upstream ships the fix
            if old_pop not in s or old_ts not in s:
                raise RuntimeError(
                    f"{f}: fetch_video changed upstream — timestamps passthrough cannot "
                    "be applied, frame-list runs would mislabel temporal metadata")
            f.write_text(s.replace(old_pop, new_pop).replace(old_ts, new_ts))
            print(f"[eagle] timestamps passthrough patched: {f}")

    # ---- selection stage (timed separately by the runner) ----
    def select_frames(self, sample) -> list:
        """Decode + pick frames per the config.

        Sampling (video.sampling) decides what gets decoded:
          uniform -> num_frames evenly spaced ("Uniform, 16 frames" rows)
          fps     -> fixed-rate at video.fps, optionally video.frame_cap
                     ("Fixed-rate, 8 fps" / "Frame cap, up to 64 frames" rows)
        A selector (video.frame_source) can subselect top_k on top:
          all (default, = the plain baseline) | uniform | random | hornet
        frame_source: hornet ignores video.sampling — the policy's own protocol is fixed
        (32 candidates @ 288x288, reference/hornet/lmms_eval_utils/hornet.py).
        """
        self.last_frame_timestamps = None  # reset before ANY path (incl. native return)
        meta = sample.meta or {}

        # TWO-VIDEO (contrastive task): meta.video_paths holds the two clips in
        # presentation order. Select k frames per clip with the same sweep protocol so
        # generate() hands the processor both -> <video-1>/<video-2> = "first"/"second".
        paths = meta.get("video_paths")
        if paths:
            if bool(self.cfg.video.get("text_only", False)):
                self.last_frames_used = 0        # sanity floor: the question alone, no videos
                return {"two_video": []}
            clips = [self._select_clip(p, sample) for p in paths]  # [(frames, timestamps), ...]
            self.last_frames_used = sum(len(f) for f, _ in clips)
            return {"two_video": clips}

        video_path = meta.get("video_path")
        if not video_path:
            raise ValueError(f"sample {sample.id} has no meta.video_path")
        source = (self.cfg.video.get("frame_source") or "all").lower()

        # NATIVE DECODE (default for the plain baseline): let Eagle's own fetch_video
        # sample the video from its PATH. This is the only route that produces real
        # per-frame timestamps and the true fps in the prompt ("Frame i sample at Xs",
        # "sampled with 8.00 fps"); the explicit-frames path hardcodes timestamps=-1
        # and fps=2.0 upstream, which mislabels the temporal metadata on a timing
        # benchmark. Matches the "timestamps: corrected" methodology.
        if source == "all" and bool(self.cfg.video.get("native_decode", True)):
            # A null-input arm must NOT ride this path: the sentinel hands Eagle the file and
            # its own fetch_video decodes the real video, so blank_frames/text_only would be
            # silently ignored and the null-arm delta would come from a content-bearing pass.
            # Fail loudly rather than emit a no-op null arm.
            if self.cfg.video.get("blank_frames") or self.cfg.video.get("text_only"):
                raise ValueError(
                    "video.blank_frames/text_only are ignored on the native-decode path "
                    "(frame_source=all + native_decode): the null arm would silently see the "
                    "real video. Use frame_source=uniform for null/floor arms.")
            return [video_path]  # sentinel: generate() builds a path message

        if source in ("uniform", "random", "hornet"):
            frames, ts = self._select_clip(video_path, sample)
            self.last_frame_timestamps = ts
            return frames

        # frame_source == all, non-native: decode per the sampling config
        candidates, cand_ts = self._decode(
            video_path,
            mode=(self.cfg.video.get("sampling") or "uniform").lower(),
            num=int(self.cfg.video.get("num_frames", 16)),
            fps=float(self.cfg.video.get("fps", 8)),
            cap=self.cfg.video.get("frame_cap"),
            resize=self.cfg.video.get("resize"),   # None = native resolution
        )
        if bool(self.cfg.video.get("pass_timestamps", True)):
            self.last_frame_timestamps = list(cand_ts)
        return candidates

    def _select_clip(self, video_path, sample):
        """Sweep-protocol selection for one clip: top-k of a 32-position uniform candidate pool
        (HORNet paper Table 5), so uniform/random/HORNet stay comparable (same pool, only the
        strategy differs). Candidates decode at candidate_resize for Eagle to answer on
        (null => native); HORNet scores 288 copies for selection only. Returns (frames,
        timestamps); timestamps is None when pass_timestamps is off. Shared by the single-video
        sweep path and the two-video contrastive path."""
        source = (self.cfg.video.get("frame_source") or "all").lower()
        n_cand = int(self.cfg.video.get("candidates", 32))
        answer_res = self.cfg.video.get("candidate_resize")  # None => native
        candidates, cand_ts = self._decode(video_path, mode="uniform", num=n_cand, resize=answer_res)
        k = int(self.cfg.video.get("top_k", 8))
        if source == "random":
            # per-sample deterministic seed: eval seed combined with sample id
            idx = selectors.random_indices(len(candidates), k,
                                           int(self.cfg.eval.get("seed", 1234)) * 100003
                                           + int(sample.id) if str(sample.id).isdigit()
                                           else int(self.cfg.eval.get("seed", 1234)))
        elif source == "hornet":  # upstream HORNet code path; score at 288, answer on native
            self._hornet = getattr(self, "_hornet", None) or selectors.HornetSelector(self.cfg)
            score_res = int(self.cfg.video.get("hornet_score_resize", 288))
            score_frames = (candidates if answer_res == score_res
                            else self._resize(candidates, score_res))
            idx = self._hornet.select(score_frames, k)
        else:  # uniform (and the default)
            idx = selectors.uniform_indices(len(candidates), k)
        pass_ts = bool(self.cfg.video.get("pass_timestamps", True))
        ts = [cand_ts[i] for i in idx] if pass_ts else None
        sel = [candidates[i] for i in idx]
        if bool(self.cfg.video.get("blank_frames", False)):
            sel = [f * 0.0 for f in sel]  # black-frame floor: same geometry, no content
        return sel, ts

    def _resize(self, frames, size):
        """Resize a list of [H,W,3] float frames to (size,size). Used to feed HORNet's
        policy its required 288 input while Eagle answers on the native frames."""
        t = self.torch.stack([f.permute(2, 0, 1) for f in frames])  # [N,3,H,W]
        t = self.torch.nn.functional.interpolate(
            t, size=(int(size), int(size)), mode="bilinear", align_corners=False)
        return [t[i].permute(1, 2, 0) for i in range(t.shape[0])]

    def _decode(self, video_path: str, mode: str, num: int = 16,
                fps: float = 8.0, cap=None, resize=None):
        """decord decode; uniform mode w/ resize=288,num=32 == HORNet's load_frames().
        Returns (frames, timestamps_seconds) — both aligned to the sampled indices."""
        try:
            import decord
            import numpy as np
        except ImportError as e:
            raise RuntimeError("video decode needs decord + numpy: pip install decord numpy") from e
        vr = decord.VideoReader(video_path)
        total = len(vr)
        native_fps = float(vr.get_avg_fps()) or 30.0
        if mode == "fps":
            step = max(native_fps / fps, 1.0)
            indices = np.arange(0, total, step).astype(int)
            if cap and len(indices) > int(cap):
                indices = np.linspace(0, total - 1, int(cap)).astype(int)
        else:  # uniform
            if total >= num:
                indices = np.linspace(0, total - 1, num).astype(int)
            else:
                indices = np.array(list(range(total)) + [total - 1] * (num - total))
        timestamps = [round(float(i) / native_fps, 2) for i in indices]
        frames = self.torch.from_numpy(vr.get_batch(indices).asnumpy()).float()
        if resize:
            frames = self.torch.nn.functional.interpolate(
                frames.permute(0, 3, 1, 2), size=(int(resize), int(resize)),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)
        return list(frames / 255.0), timestamps  # [H,W,3] floats in [0,1] + seconds

    # ---- answering stage ----
    def _maybe_mark_context(self, sample, video_contents, user_prompt):
        """Visual marks: when eval.mark_context is set, append the pre-rendered summary frame
        (comet-tail traces for this clip) as an extra image and prepend the pointer sentence."""
        mc = self.cfg.eval.get("mark_context") if hasattr(self.cfg, "eval") else None
        if not mc:
            return video_contents, user_prompt
        from pathlib import Path as _P
        frames_dir = mc.get("frames_dir") if isinstance(mc, dict) else mc
        vp = str((getattr(sample, "meta", None) or {}).get("video_path", ""))
        img = _P(frames_dir) / (_P(vp).stem + ".png")
        if not vp or not img.exists():
            return video_contents, user_prompt          # no summary frame for this clip -> unchanged
        from PIL import Image
        pointer = (mc.get("pointer") if isinstance(mc, dict) else None) or (
            "An extra image shows the same scene with each moving object's path over the clip drawn as "
            "a colored trail; the white dots along a trail are evenly spaced in time (earlier near the "
            "letter tag, later near the arrow).")
        return (video_contents + [{"type": "image", "image": Image.open(img).convert("RGB")}],
                f"{pointer}\n{user_prompt}")

    def _maybe_frame_marks(self, frames):
        """In-pixel frame time labels (ViKey-style): when eval.frame_index_marks is set, burn a bold ordinal + timestamp
        label into each sampled frame (in-pixel time labels), sized to survive the ->448 tile. Tests
        whether VISUAL time labels help temporal order beyond Eagle's existing TEXT timestamps."""
        fm = self.cfg.eval.get("frame_index_marks") if hasattr(self.cfg, "eval") else None
        if not fm or isinstance(frames, dict) or not frames or isinstance(frames[0], str):
            return frames
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        ts = getattr(self, "last_frame_timestamps", None) or list(range(len(frames)))
        out = []
        for i, f in enumerate(frames):
            im = Image.fromarray((f * 255).clamp(0, 255).byte().cpu().numpy())
            fs = max(40, im.height // 14)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", fs)
            except Exception:
                font = ImageFont.load_default()
            t = ts[i] if i < len(ts) else float(i)
            ImageDraw.Draw(im).text((int(fs * 0.3), int(fs * 0.2)), f"#{i + 1}  {t:.1f}s",
                                    fill=(255, 255, 0), font=font, stroke_width=max(3, fs // 12),
                                    stroke_fill=(0, 0, 0))
            out.append(self.torch.from_numpy(np.array(im)).float().div(255))
        return out

    def _maybe_flow_overlay(self, sample, frames):
        """Motion flow overlay: when eval.flow_overlay is set, draw the previous-frame optical
        flow onto each selected frame as short arrows (turbo colour = speed), giving the frozen model
        the per-frame motion its coarse k-frame sampling cannot resolve. Flow is vs each frame's
        IMMEDIATE video predecessor (adjacent-frame -> clean, finer than the ~5-apart selected
        neighbour). eval.flow_overlay.mode: 'flow' draws the real field; 'scramble' shuffles the
        vectors across grid cells (same arrows, wrong places) for the ground-truth-free flip test
        (does the answer change when motion is decoupled from where it happened?); 'off' is the
        matched control -- same downscale + dim, NO arrows -- so the arms differ only in the overlay.
        Drawing is done at a bounded resolution (maxside, default 768): the model downscales to its
        ~448 tile regardless, and this keeps 24-frame Farneback + drawing inside walltime. Only the
        explicit-frames path carries frame tensors; native path-decode and two-video pass through."""
        fo = self.cfg.eval.get("flow_overlay") if hasattr(self.cfg, "eval") else None
        if not fo or isinstance(frames, dict) or not frames or isinstance(frames[0], str):
            return frames
        import numpy as np
        import cv2
        opts = fo if isinstance(fo, dict) else {}
        mode = str(opts.get("mode", "flow")).lower()
        if mode not in ("flow", "scramble", "off"):
            mode = "flow"
        dim = float(opts.get("dim", 0.7))
        seed = int(opts.get("seed", 1234))
        maxside = int(opts.get("maxside", 768))

        cur_rgb = [(f * 255).clamp(0, 255).byte().cpu().numpy() for f in frames]    # RGB uint8
        H0, W0 = cur_rgb[0].shape[:2]
        sc = maxside / max(H0, W0)
        W, H = (int(round(W0 * sc)), int(round(H0 * sc))) if sc < 1.0 else (W0, H0)
        cur = [cv2.resize(a, (W, H)) if (W, H) != (W0, H0) else a for a in cur_rgb]

        def tens(arr):
            return self.torch.from_numpy(np.ascontiguousarray(arr)).float().div(255)

        if mode == "off":                        # matched control: same downscale + dim, no arrows
            self._flow_log_once(mode, dim, 0, 0.0, len(frames))
            return [tens((a.astype(np.float32) * dim).astype(np.uint8)) for a in cur]

        import decord
        vp = str((getattr(sample, "meta", None) or {}).get("video_path", ""))
        ts = getattr(self, "last_frame_timestamps", None)
        if not vp or not ts:
            return frames               # need the source clip + per-frame times to find neighbours
        # decode the neighbour frames DIRECTLY at draw resolution -- decoding native 1080p neighbours
        # then resizing is ~10s/clip (dominates); decord's width/height decode makes it ~2s.
        vr = decord.VideoReader(vp, width=W, height=H)
        nfps = float(vr.get_avg_fps()) or 30.0
        total = len(vr)
        sel_idx = [min(total - 1, max(0, int(round(t * nfps)))) for t in ts]
        prev_small = vr.get_batch([max(0, i - 1) for i in sel_idx]).asnumpy()   # (k,H,W,3) at draw res

        flows = []
        for i in range(len(frames)):
            flows.append(cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(prev_small[i], cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(cur[i], cv2.COLOR_RGB2GRAY),
                None, 0.5, 3, 21, 3, 7, 1.5, 0))
        gmax = float(np.mean([np.percentile(np.hypot(fl[..., 0], fl[..., 1]), 99)
                              for fl in flows])) or 1.0
        STEP = max(14, W // 17)
        THICK = max(2, W // 150)
        scale = (STEP * 0.85) / gmax
        noise = max(0.6, 0.14 * gmax)
        rng = np.random.default_rng(seed)

        _lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1),
                                 cv2.COLORMAP_TURBO).reshape(256, 3)     # BGR LUT, built once

        def turbo_rgb(t):
            b = _lut[int(np.clip(t, 0, 1) * 255)]
            return int(b[2]), int(b[1]), int(b[0])

        pts = [(x, y) for y in range(STEP // 2, H, STEP) for x in range(STEP // 2, W, STEP)]
        out = []
        for i in range(len(frames)):
            fl = flows[i]
            canvas = (cur[i].astype(np.float32) * dim).astype(np.uint8).copy()
            vecs = [(float(fl[y, x, 0]), float(fl[y, x, 1])) for (x, y) in pts]
            if mode == "scramble":
                vecs = [vecs[j] for j in rng.permutation(len(vecs))]
            for (x, y), (dx, dy) in zip(pts, vecs):
                m = (dx * dx + dy * dy) ** 0.5
                if m < noise:
                    continue
                cv2.arrowedLine(canvas, (x, y), (int(x + dx * scale), int(y + dy * scale)),
                                turbo_rgb(m / gmax), THICK, cv2.LINE_AA, tipLength=0.4)
            out.append(tens(canvas))
        self._flow_log_once(mode, dim, len(pts), gmax, len(frames))
        return out

    def _flow_log_once(self, mode, dim, npts, gmax, nframes):
        if not getattr(self, "_flow_logged", False):
            self._flow_logged = True
            print(f"[eagle] flow_overlay active: mode={mode} dim={dim} grid={npts}pts "
                  f"gmax~{gmax:.1f} on {nframes} frames")

    def _maybe_flow_pointer(self, frames, user_prompt):
        """When eval.flow_overlay.pointer is set (and arrows are actually drawn — mode flow/scramble),
        prepend a sentence naming what the overlaid arrows mean. The synthetic probe found the pointer
        scaffold load-bearing for mark conditioning (flip 0.73 with it vs 0.53 without); the first
        flow pass omitted it, so this is the fair retry. pointer: true uses a default sentence; a string overrides it."""
        fo = self.cfg.eval.get("flow_overlay") if hasattr(self.cfg, "eval") else None
        if not isinstance(fo, dict) or str(fo.get("mode", "flow")).lower() == "off":
            return user_prompt
        ptr = fo.get("pointer")
        if not ptr or isinstance(frames, dict) or not frames or isinstance(frames[0], str):
            return user_prompt          # only when the overlay actually drew arrows on frame tensors
        if ptr is True:
            ptr = ("Coloured arrows are drawn on the video showing each region's motion between "
                   "consecutive frames: a longer, redder arrow means faster motion, and an area with "
                   "no arrow is not moving.")
        if not getattr(self, "_flow_ptr_logged", False):
            self._flow_ptr_logged = True
            print(f"[eagle] flow_overlay pointer prepended: {str(ptr)[:80]!r}")
        return f"{ptr}\n{user_prompt}"

    def _perturb_indices(self, n, opts, sample):
        """Perturbation orderings: the output-frame ordering for a perturbation mode — a list `idx` of length n
        where output frame j takes input frame idx[j]. 'shuffle' permutes (never the identity),
        'reverse' flips, 'segswap' swaps the two halves at `cut`, 'blockshuffle' permutes B consecutive
        blocks keeping within-block order, 'freeze' repeats the middle frame. Seeded per
        item and process-STABLE: digit ids are used directly, others via crc32 (Python's hash() of a str
        is PYTHONHASHSEED-randomised, so the 2AFC ids 'sid|clip|order' would not reproduce across runs)."""
        import numpy as np
        import zlib
        mode = str(opts.get("mode", "shuffle")).lower()
        seed = int(opts.get("seed", 0))
        sid = getattr(sample, "id", 0)
        sid = int(sid) if str(sid).isdigit() else (zlib.crc32(str(sid).encode()) & 0x7fffffff)
        rng = np.random.default_rng(seed * 1000003 + sid)
        if mode == "shuffle" and n > 1:
            perm = rng.permutation(n)
            if all(int(perm[i]) == i for i in range(n)):        # never the identity
                perm = perm[::-1]
            return [int(i) for i in perm]
        if mode == "reverse":
            return list(range(n - 1, -1, -1))
        if mode == "segswap" and n > 1:                         # cut at `cut`, swap the two halves
            c = max(1, min(n - 1, int(round(n * float(opts.get("cut", 0.5))))))
            return list(range(c, n)) + list(range(c))
        if mode == "blockshuffle" and n > 1:                    # permute B consecutive blocks
            nb = max(2, min(int(opts.get("blocks", 4)), n))
            edges = [round(k * n / nb) for k in range(nb + 1)]
            blocks = [list(range(edges[k], edges[k + 1])) for k in range(nb)]
            blocks = [b for b in blocks if b]                   # drop empties (n < nb edge)
            perm = rng.permutation(len(blocks))
            if all(int(perm[i]) == i for i in range(len(blocks))):   # never the identity
                perm = perm[::-1]
            return [i for k in perm for i in blocks[int(k)]]
        if mode == "freeze":
            return [n // 2] * n
        return list(range(n))

    def _maybe_perturb(self, sample, frames):
        """Video-side contrastive decoding: when eval.perturb is set, destroy the QUERIED
        signal in the selected frames while keeping their content, so the perturbed run's logits carry
        every shortcut (static content, language prior, yes-bias) EXCEPT that signal. Subtracting them
        from the real run's logits offline (contrastive decoding, scripts/contrastive_decode.py)
        cancels the shortcuts and leaves the signal. Every mode keeps timestamps ascending (only the
        frame content is reordered), so the time labels no longer match the content. Modes, ordered by
        sibling quality (how cleanly they negate the queried relation without going out of
        distribution): 'shuffle' permutes all frame order (maximal scramble); 'blockshuffle' permutes
        B consecutive blocks, within-block order kept (milder — local motion survives, macro order
        broken); 'reverse' flips the order (a clean opposite before/after relation); 'segswap' cuts the
        sequence and swaps the two halves, each half kept forward (locally-natural motion, macro order
        flipped); 'freeze' repeats one frame (kills repetition/duration). This is the EXPLICIT-frames
        path (uniform-k); the two-video path passes through, and the NATIVE-decode path is handled by
        _maybe_perturb_decoded (its frames don't exist as tensors yet — Eagle's fetch_video decodes
        them inside process_vision_info)."""
        pb = self.cfg.eval.get("perturb") if hasattr(self.cfg, "eval") else None
        if not pb or isinstance(frames, dict) or not frames or isinstance(frames[0], str):
            return frames
        opts = pb if isinstance(pb, dict) else {"mode": pb}
        idx = self._perturb_indices(len(frames), opts, sample)
        out = [frames[i] for i in idx]
        if not getattr(self, "_perturb_logged", False):
            self._perturb_logged = True
            print(f"[eagle] perturb active: mode={str(opts.get('mode', 'shuffle')).lower()} "
                  f"on {len(frames)} frames (timestamps kept ascending)")
        return out

    def _maybe_perturb_decoded(self, sample, frames, video_inputs):
        """Flagship CD (native-decode path): the flagship recipe (frame_source: all +
        native_decode) hands Eagle a video PATH, and its own fetch_video decodes the frames INSIDE
        process_vision_info — so _maybe_perturb (which acts on selected frame tensors) silently no-ops
        here, the same class of silent no-op the blank-frames guard fails fast on. process_vision_info returns the
        raw per-frame tensor(s) as video_inputs PLUS a SEPARATE ascending timestamps list in
        video_kwargs; the processor labels frame i 'Frame i sample at t_i' by position. So permuting the
        decoded frames while leaving the timestamps ascending is the SAME shuffle the explicit path does
        (content reordered, time labels ascending → labels no longer match the content). Applies ONLY on
        the native path (frames == [path]); the explicit path is already perturbed by _maybe_perturb, so
        there is no double-shuffle."""
        pb = self.cfg.eval.get("perturb") if hasattr(self.cfg, "eval") else None
        if (not pb or not video_inputs
                or not (isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], str))):
            return video_inputs
        opts = pb if isinstance(pb, dict) else {"mode": pb}
        out, shapes = [], []
        for v in video_inputs:
            n = int(v.shape[0]) if hasattr(v, "shape") else len(v)
            idx = self._perturb_indices(n, opts, sample)
            out.append(v[idx] if hasattr(v, "shape") else [v[i] for i in idx])
            shapes.append(tuple(v.shape) if hasattr(v, "shape") else n)
        if not getattr(self, "_perturb_dec_logged", False):
            self._perturb_dec_logged = True
            print(f"[eagle] perturb(native-decode) active: mode={str(opts.get('mode', 'shuffle')).lower()} "
                  f"on decoded frames {shapes} (timestamps kept ascending)")
        return out

    def _maybe_spatial_scale(self, frames):
        """Magnitude anchor (spatial): eval.spatial_scale center-crops each selected frame by a
        factor f then resizes back to the original size, changing the APPARENT pixel displacement of the
        same motion. The information-flow probe put magnitude in ‖Δe‖ — the spatial frame-difference — so this is the
        spatial analogue of eval.retime, which anchored speed via the timestamp channel. f<1 crops a
        smaller central window and upsamples → zoom IN (apparent displacement ×1/f > 1, the 'larger'
        anchor); f>1 shrinks the frame into a letterboxed central region → zoom OUT (×1/f < 1, the
        'smaller' anchor). Only the explicit-frames path carries frame tensors."""
        ss = self.cfg.eval.get("spatial_scale") if hasattr(self.cfg, "eval") else None
        if ss is None or isinstance(frames, dict) or not frames or isinstance(frames[0], str):
            return frames
        f = float(ss.get("factor") if isinstance(ss, dict) else ss)
        if f <= 0 or abs(f - 1.0) < 1e-9:
            return frames
        torch = self.torch

        def _resize(fr, h, w):
            return torch.nn.functional.interpolate(
                fr.permute(2, 0, 1).unsqueeze(0), size=(int(h), int(w)),
                mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)

        out = []
        for fr in frames:                              # fr: [H, W, 3] float in [0,1]
            H, W = int(fr.shape[0]), int(fr.shape[1])
            if f < 1.0:                                # zoom IN: crop central fH×fW, upscale to H×W
                ch, cw = max(1, round(f * H)), max(1, round(f * W))
                top, left = (H - ch) // 2, (W - cw) // 2
                res = _resize(fr[top:top + ch, left:left + cw, :], H, W)
            else:                                      # zoom OUT: shrink to H/f×W/f, letterbox-pad
                sh, sw = max(1, round(H / f)), max(1, round(W / f))
                small = _resize(fr, sh, sw)
                res = torch.zeros_like(fr)
                top, left = (H - sh) // 2, (W - sw) // 2
                res[top:top + sh, left:left + sw, :] = small
            out.append(res)
        if not getattr(self, "_spatial_logged", False):
            self._spatial_logged = True
            kind = "zoom-in crop+upscale" if f < 1 else "zoom-out shrink+letterbox"
            print(f"[eagle] spatial_scale f={f}: {kind} on {len(frames)} frames "
                  f"(apparent displacement x{1.0 / f:.2f})")
        return out

    def _maybe_retime(self):
        """Speed-anchor trick: eval.retime scales the per-frame timestamps by a factor — the SAME
        frames, but presented as if the motion happened in less time (factor<1 → faster) or more
        (factor>1 → slower). Creates synthetic fast/slow speed anchors of a clip, so an offline decode can
        answer an absolute speed question relatively (is the real clip closer to the fast or slow
        anchor?), the one way CD-family logic can touch an absolute-criterion deficit. Only the
        explicit-frames path carries real per-frame timestamps to rescale."""
        rt = self.cfg.eval.get("retime") if hasattr(self.cfg, "eval") else None
        ts = getattr(self, "last_frame_timestamps", None)
        if not rt or not ts:
            return
        f = float(rt)
        self.last_frame_timestamps = [t * f for t in ts]
        if not getattr(self, "_retime_logged", False):
            self._retime_logged = True
            print(f"[eagle] retime x{f}: timestamps scaled (speed anchor; smaller = faster)")

    @staticmethod
    def _emb_record(fields, base, tower_perframe, proj, resid, tower_mean):
        """Assemble the extract-embeddings JSONL record for the requested field set (a pure function of
        already-pooled python lists, so it is unit-tested without a forward pass):

        * 'full'       — per-frame tower (nf x 1152), projector frame-mean proj (3584), and the
                         per-layer answer-position residual resid (L+1 x 3584); the offline
                         information-flow probe reads all three.
        * 'proj_tower' — proj (3584) + tower_mean (frame-pooled SigLIP, 1152) only. This is the compact
                         clip embedding the offline retrieval analysis consumes; it drops the
                         per-frame tower and the whole resid stack (~1/30 the bytes) and needs no
                         hidden-state forward."""
        rec = dict(base)
        if fields == "proj_tower":
            rec["proj"] = proj
            rec["tower_mean"] = tower_mean
        else:
            rec["tower"] = tower_perframe
            rec["proj"] = proj
            rec["resid"] = resid
        return rec

    def _extract_embeddings(self, sample, inputs):
        """Information-flow probe: with eval.extract_embeddings set to an output .jsonl path, run
        ONE forward pass and log the queried variable at every interface — post vision tower (SigLIP
        per-frame), post projector (mlp1, LLM-space, frame-mean), and the answer-position residual per
        LLM layer — per item, so offline probes locate the readout depth per category. Patch/token dims are
        mean-pooled; the answer position (-1) is where the first answer token is predicted. eval.extract_fields
        selects the field set (default 'full'; 'proj_tower' writes proj + frame-pooled tower_mean only —
        the compact retrieval embedding, no residual forward). Returns a dummy answer (the run's accuracy
        is irrelevant here)."""
        import json as _json
        torch = self.torch
        fields = self.cfg.eval.get("extract_fields", "full") if hasattr(self.cfg, "eval") else "full"
        need_resid = fields != "proj_tower"
        caps = {}

        def grab(o):
            o = o[0] if isinstance(o, (tuple, list)) else o
            return getattr(o, "last_hidden_state", o)

        def hook(name):
            def f(_m, _i, o):
                caps[name] = grab(o).detach().float().cpu()
            return f

        handles = [self.model.vision_model.register_forward_hook(hook("tower")),
                   self.model.mlp1.register_forward_hook(hook("proj"))]
        try:
            with torch.no_grad():
                # route through generate (handles image_sizes etc.); 1 token, prefill hidden states.
                # proj_tower needs only the two vision hooks, so skip the per-layer hidden states.
                out = self.model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                          output_hidden_states=need_resid, return_dict_in_generate=True)
        finally:
            for h in handles:
                h.remove()

        nf = int(inputs["pixel_values"].shape[0]) if "pixel_values" in inputs else 1

        def perframe(t):                          # -> (nf, d), robust to (nf,patch,d) or (nf*patch,d)
            t = t if t.dim() == 3 else t.reshape(nf, -1, t.shape[-1])
            return t.mean(dim=1)

        tower = perframe(caps["tower"])           # (nf, 1152) — SigLIP per-frame
        tower_mean = tower.mean(0)                # (1152,)    — frame-pooled SigLIP
        proj = perframe(caps["proj"]).mean(0)     # (3584,)    — projector, frame-mean
        resid = None
        if need_resid:
            hs = out.hidden_states[0]             # first generated step: per-layer (B, prompt_len, D)
            resid = torch.stack([hh[0, -1, :].detach().float().cpu()
                                 for hh in hs])    # (L+1, 3584) answer-position

        if not getattr(self, "_emb_logged", False):
            self._emb_logged = True
            print(f"[eagle] extract SHAPES ({fields}): tower_raw={tuple(caps['tower'].shape)} "
                  f"proj_raw={tuple(caps['proj'].shape)} nf={nf} => tower{tuple(tower.shape)} "
                  f"proj{tuple(proj.shape)} resid={tuple(resid.shape) if resid is not None else None}")

        meta = getattr(sample, "meta", None) or {}
        base = {"id": str(sample.id), "category": getattr(sample, "category", None),
                "answer": sample.answer, "role": meta.get("role"),
                "instance_id": getattr(sample, "instance_id", None)}
        rec = self._emb_record(
            fields, base,
            tower.numpy().round(4).tolist(),                              # per-frame (nf x 1152)
            proj.numpy().round(4).tolist(),                              # frame-mean (3584)
            resid.numpy().round(4).tolist() if resid is not None else None,   # per-layer (L+1 x 3584)
            tower_mean.numpy().round(4).tolist())                        # frame-pooled (1152)
        import os as _os
        path = self.cfg.eval.get("extract_embeddings")
        _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
        with open(path, "a") as fh:
            fh.write(_json.dumps(rec) + "\n")
        return "yes"

    def generate(self, sample, frames, user_prompt, system_prompt) -> str:
        """Model-card video inference pattern: one or two videos, native path or explicit frames."""
        frames = self._maybe_frame_marks(frames)
        frames = self._maybe_flow_overlay(sample, frames)
        frames = self._maybe_perturb(sample, frames)
        frames = self._maybe_spatial_scale(frames)
        self._maybe_retime()
        user_prompt = self._maybe_flow_pointer(frames, user_prompt)
        if isinstance(frames, dict) and "two_video" in frames:
            # contrastive: two clips -> two video blocks -> <video-1>/<video-2>
            video_contents = [self._pil_video_content(cf, cts) for cf, cts in frames["two_video"]]
        elif len(frames) == 1 and isinstance(frames[0], str):
            video_contents = [self._path_video_content(frames[0])]
        else:
            video_contents = [self._pil_video_content(
                frames, getattr(self, "last_frame_timestamps", None))]
        video_contents, user_prompt = self._maybe_mark_context(sample, video_contents, user_prompt)
        messages = self._build_messages(video_contents, user_prompt, system_prompt)
        text_list = [self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )]
        # process_vision_info is where video reading/sampling happens in native mode —
        # time it separately so frame-acquisition cost lands in sel_time, not vlm_time,
        # keeping the "Frame Sel (s)" column comparable across native and frame-list paths
        import time as _time
        _t = _time.perf_counter()
        image_inputs, video_inputs, video_kwargs = self.processor.process_vision_info(
            messages, return_video_kwargs=True
        )
        self.last_decode_s = _time.perf_counter() - _t
        # Flagship CD: on the native-decode path the frames only exist post-decode, so the
        # frame-order shuffle applies to video_inputs here (video_kwargs['timestamps'] stays ascending).
        video_inputs = self._maybe_perturb_decoded(sample, frames, video_inputs)
        # CRITICAL: passing an explicit videos_kwargs clobbers the processor's video
        # default of max_dynamic_tiles=1 and silently falls back to the IMAGE default
        # (12 tiles/frame -> ~27k vision tokens -> OOM on 32GB). Re-assert it here.
        video_kwargs = {
            **(video_kwargs or {}),
            "max_dynamic_tiles": int(self.cfg.video.get("max_dynamic_tiles", 1)),
        }
        inputs = self.processor(
            text=text_list, images=image_inputs, videos=video_inputs,
            videos_kwargs=video_kwargs, padding=True, return_tensors="pt",
        ).to(self.model.device)
        if self.cfg.eval.get("extract_embeddings") if hasattr(self.cfg, "eval") else False:
            return self._extract_embeddings(sample, inputs)
        pv = inputs.get("pixel_values") if hasattr(inputs, "get") else None
        if pv is not None:
            self.last_frames_used = int(pv.shape[0])
        elif not isinstance(frames, dict):
            self.last_frames_used = len(frames)
        # two-video dict with no pixels (text_only): select_frames already recorded it
        # compressed EXPANDED prompt (processor injects fps/timestamps when replacing
        # <video-1>) — recorded per sample so predictions/tables carry prompt evidence
        import re as _re
        full = self.processor.tokenizer.decode(inputs["input_ids"][0][:400])
        self.last_prompt_head = _re.sub(r"(<IMG_CONTEXT>)+", "<IMG…>", full)[:400]
        if not getattr(self, "_shapes_logged", False):
            self._shapes_logged = True
            print(f"[eagle] first sample: input_ids {tuple(inputs['input_ids'].shape)}, "
                  f"pixel_values {tuple(pv.shape) if pv is not None else None}")
            print(f"[eagle] expanded prompt head: {self.last_prompt_head!r}")
        gen = self.cfg.generation
        want_logits = bool(self.cfg.eval.get("log_yesno_logits", False))
        self.last_yesno_logits = None
        try:
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(gen.get("max_new_tokens", 128)),
                do_sample=bool(gen.get("do_sample", False)),
                **({"output_scores": True, "return_dict_in_generate": True} if want_logits else {}),
            )
            # Eagle's remote generate() returns ONLY the new tokens (InternVL-style), so
            # no prompt-trimming — the model card decodes generated_ids directly.
            if want_logits and getattr(output, "scores", None):
                # first new-token logits -> yes/no mass for the offline decode/calibration scripts
                self.last_yesno_logits = self._binary_logit_mass(output.scores[0][0])
            seq = output.sequences if hasattr(output, "sequences") else output
            return self.processor.batch_decode(
                seq, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        finally:
            del inputs  # don't let an OOM'd sample's tensors outlive the call

    def _path_video_content(self, path):
        """Native path-decode video block (Eagle's own fetch_video reads + times the file)."""
        vc = {"type": "video", "video": path, "fps": float(self.cfg.video.get("fps", 8))}
        cap = self.cfg.video.get("frame_cap")
        if cap:
            vc["max_frames"] = int(cap)
        return vc

    def _pil_video_content(self, frames, ts):
        """Explicit-frames video block. With timestamps (passthrough patch) the prompt gets true
        per-frame times + the effective rate from the selected timestamps, not the nominal fps."""
        from PIL import Image
        pil = [Image.fromarray((f * 255).clamp(0, 255).byte().cpu().numpy()) for f in frames]
        if ts:
            span = ts[-1] - ts[0]
            eff_fps = (len(ts) - 1) / span if len(ts) > 1 and span > 0 else 1.0
            return {"type": "video", "video": pil, "fps": round(eff_fps, 2), "timestamps": ts}
        return {"type": "video", "video": pil, "fps": float(self.cfg.video.get("fps", 8))}

    # ---- option-token logit scoring (bias calibration) ----
    def _option_ids(self, words):
        """First-token vocab ids for the answer words, over case / leading-space variants."""
        tok = self.processor.tokenizer
        ids = set()
        for w in words:
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.upper(), " " + w.upper()):
                enc = tok.encode(v, add_special_tokens=False)
                if enc:
                    ids.add(int(enc[0]))
        return sorted(ids)

    def _binary_logit_mass(self, logits):
        """Log-sum-exp mass on the first-token vocab groups for BOTH binary answer vocabularies —
        yes/no and A/B (ids cached). TimeBlind is type-aware (yes_no rows and multiple_choice A/B
        rows), so logging both lets the offline calibration score whichever pair matches an item's
        gold answer, covering the whole benchmark from one run instead of only its yes/no half."""
        if not hasattr(self, "_yesno_ids"):
            self._yesno_ids = (self._option_ids(("yes",)), self._option_ids(("no",)))
            self._ab_ids = (self._option_ids(("a",)), self._option_ids(("b",)))
        yes_ids, no_ids = self._yesno_ids
        a_ids, b_ids = self._ab_ids

        def lse(ids):
            if not ids:
                return float("-inf")
            t = self.torch.tensor(ids, device=logits.device)
            return float(self.torch.logsumexp(logits.index_select(0, t), dim=0))

        return {"yes": lse(yes_ids), "no": lse(no_ids), "a": lse(a_ids), "b": lse(b_ids)}

    @staticmethod
    def _build_messages(video_contents, user_prompt, system_prompt):
        # Card examples use a user-only message; Eagle's template (Qwen2.5 lineage)
        # also accepts a system role. Empty system prompt (e.g. TimeBlind's official
        # protocol, where the format instruction is a question suffix) => omit the role.
        # video_contents is a list: one block, or two for the contrastive first/second task.
        messages = []
        if (system_prompt or "").strip():
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": [
            *video_contents,
            {"type": "text", "text": user_prompt},
        ]})
        return messages

    def unload(self) -> None:
        self.model = None
        self.processor = None
