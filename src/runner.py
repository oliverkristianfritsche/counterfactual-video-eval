"""The one runner every experiment goes through:
seed -> load data -> load model -> (select frames | answer | score | log) per sample
     -> aggregate -> save -> W&B.

Reliability guarantees:
  * seeded (default 1234, same as HORNet's eval generator)
  * predictions.jsonl written incrementally — a crash keeps everything scored so far
  * per-sample errors are recorded, not fatal (eval.fail_fast: true to override)
  * W&B initialized only after data+model load, so config errors never leave orphan runs
  * environment + metric-implementation stamps in summary.json for comparability
"""
from __future__ import annotations

import json
import platform
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import Config
from .metrics import compute_metrics, f1_lev, metric_impl, score_first_second, score_mcq, score_timeblind
from .prompts import build_user_prompt, resolve_system_prompt
from .registry import build_dataset, build_model
from .wandb_logger import WandbLogger

HARNESS_VERSION = "0.2.0"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _environment(seed: int) -> dict:
    env = {
        "harness_version": HARNESS_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "seed": seed,
        "metric_impl": metric_impl(),
    }
    for mod in ("torch", "transformers", "wandb", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    return env


def _sampling_desc(video_cfg: dict, frame_source: str) -> str:
    """Human-readable sampling type, the 'Sampling'/'FPS' table columns."""
    if frame_source in ("uniform", "random", "hornet"):
        # sweep protocol: strategy picks top-k from the shared candidate pool
        k = video_cfg.get("top_k", 8)
        n = video_cfg.get("candidates", 32)
        return f"{frame_source}-k{k}of{n}"
    mode = (video_cfg.get("sampling") or "uniform").lower()
    if mode == "fps":
        desc = f"fps-{video_cfg.get('fps', 8)}"
        if video_cfg.get("frame_cap"):
            desc += f"-cap{video_cfg['frame_cap']}"
    else:
        desc = f"uniform-{video_cfg.get('num_frames', 16)}"
    return desc


def _score(record: dict, sample, task: str, mcq_protocol: str) -> None:
    if task == "choice":
        correct, parsed = score_mcq(record["prediction"], sample.answer, sample.options, mcq_protocol)
        record["correct"] = correct
        record["parsed_choice"] = parsed
    elif task == "yesno":
        # TimeBlind is type-aware: yes_no rows and multiple_choice (A/B) rows
        correct, parsed = score_timeblind(record["prediction"], sample.answer, sample.category)
        record["correct"] = correct
        record["parsed_choice"] = parsed  # 1=yes/A, 0=no/B, None=invalid
    elif task == "contrastive":
        correct, parsed = score_first_second(record["prediction"], sample.answer)
        record["correct"] = correct
        record["parsed_choice"] = parsed  # 1=first, 0=second, None=invalid
    else:
        record["f1_lev"] = f1_lev(record["prediction"], str(sample.answer or ""))
        record["correct"] = record["f1_lev"] >= 0.999  # exact-match indicator

def run(config_path, limit: int | None = None, offset: int = 0,
        overrides: dict | None = None) -> dict:
    cfg = Config.load(config_path)
    for key, val in (overrides or {}).items():
        section, _, field = key.partition(".")
        getattr(cfg, section)[field] = val
        # keep cfg.raw (what W&B logs as config) in sync, so overridden fields
        # (e.g. --set video.top_k=4) log the effective value, not the yaml default
        cfg.raw.setdefault(section, {})[field] = val

    seed = int(cfg.eval.get("seed", 1234))  # 1234 = HORNet's eval seed
    _seed_everything(seed)

    # per-sample watchdog: a hung decode never throws, so try/except alone can't save
    # the run. SIGALRM turns a hang into a recordable per-sample error. Linux
    # main-thread only.
    import signal
    sample_timeout = int(cfg.eval.get("sample_timeout_s", 300))
    watchdog = hasattr(signal, "SIGALRM") and sample_timeout > 0
    if watchdog:
        def _on_alarm(signum, frame):
            raise TimeoutError(f"sample watchdog: exceeded {sample_timeout}s")
        signal.signal(signal.SIGALRM, _on_alarm)

    model_name = cfg.model.get("kind") or cfg.model.get("name") or "model"
    ds_name = cfg.dataset.get("kind") or cfg.dataset.get("name") or "data"
    frame_source = (cfg.video.get("frame_source") or "all").lower()
    sampling = _sampling_desc(cfg.video, frame_source)
    task, system_prompt = resolve_system_prompt(cfg)
    mcq_protocol = (cfg.prompt.get("mcq_protocol") or "digit").lower()
    fail_fast = bool(cfg.eval.get("fail_fast", False))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # uniquify against parallel jobs launched the same second (Slurm array/chunk runs)
    # that would otherwise share a run dir and interleave predictions.jsonl.
    import os
    stamp += f"-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    run_name = cfg.wandb.get("name") or f"{model_name}-{sampling}-{ds_name}-{stamp}"

    print(f"[run] {run_name}: task={task} sampling={sampling} seed={seed}")
    dataset = build_dataset(cfg)
    if offset:
        # chunked runs: load offset+limit, slice off the front. Keep offset & limit
        # multiples of 4 on TimeBlind so groups stay intact.
        samples = dataset.load(limit=(offset + limit) if limit else None)[offset:]
    else:
        samples = dataset.load(limit=limit)
    print(f"[run] {len(samples)} samples (offset={offset})")

    model = build_model(cfg)
    model.load()

    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu = torch
    except ImportError:
        pass

    # W&B only after data + model load succeeded
    logger = WandbLogger(cfg.wandb, cfg, run_name)

    out_dir = cfg.resolve(cfg.eval.get("results_dir", "../results")) / f"run_{stamp}_{model_name}_{frame_source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.raw, indent=2))

    records = []
    prev_errored = False
    log_every = int(cfg.wandb.get("log_every", 10))
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w") as pred_f:
        for i, s in enumerate(samples):
            if prev_errored and gpu is not None:
                # the previous sample's exception traceback pinned its tensors until the
                # handler exited — reclaim now so one OOM doesn't cascade into the rest
                import gc
                gc.collect()
                gpu.cuda.empty_cache()
                prev_errored = False
            rec = {"id": s.id, "instance_id": s.instance_id, "category": s.category,
                   "question": s.question, "options": s.options, "answer": s.answer}
            if (s.meta or {}).get("role") is not None:
                rec["role"] = s.meta["role"]   # TimeBlind 4-role groups -> Q/V/I_Acc
            try:
                if watchdog:
                    signal.alarm(sample_timeout)
                t0 = time.perf_counter()
                frames = model.select_frames(s)
                t1 = time.perf_counter()
                user_prompt = build_user_prompt(s, task)
                rec["prediction"] = model.generate(s, frames, user_prompt, system_prompt)
                t2 = time.perf_counter()
                if watchdog:
                    signal.alarm(0)
                rec["sel_time"] = t1 - t0
                rec["vlm_time"] = t2 - t1
                # adapters that decode inside generate() report that sub-time so frame
                # acquisition is accounted under selection on both code paths
                decode_s = getattr(model, "last_decode_s", None)
                if decode_s:
                    rec["sel_time"] += decode_s
                    rec["vlm_time"] = max(0.0, rec["vlm_time"] - decode_s)
                    model.last_decode_s = None
                # native path-decode: the adapter knows the real frame count post-decode
                rec["frames_used"] = getattr(model, "last_frames_used", None) or len(frames)
                ph = getattr(model, "last_prompt_head", None)
                if ph:
                    rec["prompt_head"] = ph  # rendered-prompt evidence per sample
                yl = getattr(model, "last_yesno_logits", None)
                if yl:  # first-token option log-masses for the offline decode/calibration scripts;
                    # both binary vocabularies are logged so TimeBlind's yes/no and A/B halves
                    # are both calibratable from one run
                    rec["logit_yes"], rec["logit_no"] = round(yl["yes"], 5), round(yl["no"], 5)
                    if yl.get("a") is not None:
                        rec["logit_a"], rec["logit_b"] = round(yl["a"], 5), round(yl["b"], 5)
                if frames and all(isinstance(f, int) for f in frames):
                    rec["selected_indices"] = frames
                _score(rec, s, task, mcq_protocol)
            except Exception as e:
                if watchdog:
                    signal.alarm(0)
                if fail_fast:
                    raise
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["prediction"] = None
                print(f"[run] sample {s.id} failed: {rec['error']}")
                prev_errored = True  # real cleanup happens at next loop top (traceback
                # keeps this sample's tensors alive inside the handler)
            records.append(rec)
            pred_f.write(json.dumps(rec) + "\n")
            pred_f.flush()  # crash-safe partial results
            logger.log_sample(rec)
            if (i + 1) % log_every == 0 or (i + 1) == len(samples):
                done = [r for r in records if not r.get("error")]
                prog = {
                    "running/accuracy": (sum(1 for r in done if r.get("correct")) / len(done)) if done else 0.0,
                    "running/invalid_rate": (sum(1 for r in done if r.get("parsed_choice") is None) / len(done)) if done else 0.0,
                    "running/errors": len(records) - len(done),
                    "running/samples": len(records),
                    "running/sel_s": rec.get("sel_time"),
                    "running/vlm_s": rec.get("vlm_time"),
                }
                if task == "yesno":  # live pair metrics over complete groups so far
                    from .metrics import timeblind_pair_metrics
                    pm = timeblind_pair_metrics(records)
                    for k in ("q_acc", "v_acc", "i_acc"):
                        if k in pm:
                            prog[f"running/{k}"] = pm[k]
                logger.log_progress(prog, step=i + 1)
            if (i + 1) % 25 == 0 or (i + 1) == len(samples):
                print(f"[run] {i + 1}/{len(samples)}")
    model.unload()

    metrics = compute_metrics(records, task)
    if gpu is not None:
        metrics["gpu_peak_gb"] = round(gpu.cuda.max_memory_allocated() / 1e9, 3)
    if frame_source in ("uniform", "random", "hornet"):
        # expose top_k/candidates as metrics so they reach W&B (only the metrics
        # dict is logged there)
        metrics["top_k"] = int(cfg.video.get("top_k", 0))
        metrics["candidates"] = int(cfg.video.get("candidates", 32))

    summary = {
        "run_name": run_name,
        "model": model_name,
        "dataset": ds_name,
        "frame_source": frame_source,
        "sampling": sampling,                          # table column: sampling type
        "top_k": (cfg.video.get("top_k") if frame_source in ("uniform", "random", "hornet")
                  else None),                          # table column: frame selection (k)
        "fps": cfg.video.get("fps") if (cfg.video.get("sampling") or "").lower() == "fps" else None,
        "task": task,
        "experiment": {                                # table columns: Type / Technique
            "type": cfg.experiment.get("type"),
            "technique": cfg.experiment.get("technique"),
        },
        "metrics": metrics,
        "environment": _environment(seed),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.log_summary({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    logger.log_environment(summary["environment"])
    logger.log_environment({"prompt_protocol": {  # rides the same config-update path
        "task": task,
        "system_prompt": system_prompt or "(none — chat template injects its default)",
        "user_prompt_rule": ("question + ' Please output Yes or No.' / ' Please output A or B.'"
                             if task == "yesno" else f"archetype '{task}'"),
        "rendered_example": next((r.get("prompt_head") for r in records if r.get("prompt_head")), None),
    }})
    logger.save_file(pred_path)
    logger.finish()

    headline = (("accuracy", metrics.get("accuracy")) if task in ("choice", "yesno", "contrastive")
                else ("f1_lev", metrics.get("f1_lev")))
    inst = (f", inst_acc {metrics['instance_accuracy']:.4f} over {metrics['num_videos']} videos"
            if "instance_accuracy" in metrics else "")
    print(f"\n[run] {headline[0]} = {headline[1]:.4f}  "
          f"(n={metrics['total']}, errors={metrics['errors']}{inst})")
    if "frame_sel_s/mean" in metrics:
        print(f"[run] frame_sel {metrics['frame_sel_s/mean']:.3f}s | "
              f"vlm {metrics.get('vlm_proc_s/mean', 0):.3f}s | "
              f"avg_frames {metrics.get('avg_frames', 0):.1f}")
    print(f"[run] results -> {out_dir}")
    return metrics
