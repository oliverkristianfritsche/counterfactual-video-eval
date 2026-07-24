"""Weights & Biases logging — optional and non-fatal.

mode: "online" (default), "offline" (log locally, sync later), or "disabled" (no-op).
Nothing here raises if wandb is missing or unconfigured; the eval still runs and writes
local results either way.
"""
from __future__ import annotations


class WandbLogger:
    def __init__(self, wcfg: dict, config, run_name: str):
        self.mode = (wcfg.get("mode") or "online").lower()
        self.enabled = self.mode != "disabled"
        self.run = None
        self.wandb = None
        self._rows: list[list] = []
        self._cols = ["id", "category", "question", "prediction", "answer",
                      "correct", "f1_lev", "sel_s", "vlm_s", "frames", "error",
                      "prompt_head"]
        self._max_rows = int(wcfg.get("log_samples", 50))

        if not self.enabled:
            print("[wandb] disabled")
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] not installed — continuing without logging")
            self.enabled = False
            return

        self.wandb = wandb
        try:
            self.run = wandb.init(
                project=wcfg.get("project", "vlm-benchmark"),
                entity=wcfg.get("entity"),  # None -> default entity
                name=run_name,
                group=wcfg.get("group"),
                tags=wcfg.get("tags"),
                job_type="eval",
                mode="offline" if self.mode == "offline" else "online",
                config=config.raw,
            )
            print(f"[wandb] {self.mode} run: {getattr(self.run, 'url', '(offline)')}")
        except Exception as e:  # auth/network issues shouldn't kill the eval
            print(f"[wandb] init failed ({e}) — continuing without logging")
            self.enabled = False

    def log_sample(self, record: dict) -> None:
        if not self.enabled or len(self._rows) >= self._max_rows:
            return

        def cell(v):  # W&B Table cells: pass through None/bool/int/float, stringify the rest
            if v is None or isinstance(v, (bool, int, float)):
                return v
            return str(v)[:300]

        self._rows.append([cell(record.get(k)) for k in (
            "id", "category", "question", "prediction", "answer",
            "correct", "f1_lev", "sel_time", "vlm_time", "frames_used", "error",
            "prompt_head",
        )])

    def log_progress(self, values: dict, step: int) -> None:
        """Log the numeric values at `step` so W&B charts show per-step lines."""
        if not self.enabled:
            return
        try:
            self.wandb.log({k: v for k, v in values.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)},
                           step=step)
        except Exception as e:
            print(f"[wandb] progress log failed ({e})")

    def log_summary(self, metrics: dict) -> None:
        if not self.enabled:
            return
        try:
            numeric = {k: v for k, v in metrics.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            self.wandb.log(numeric)
            for k, v in metrics.items():
                self.run.summary[k] = v
            if self._rows:
                table = self.wandb.Table(columns=self._cols, data=self._rows)
                self.wandb.log({"samples": table})
        except Exception as e:
            print(f"[wandb] summary log failed ({e}) — local results are authoritative")

    def log_environment(self, env: dict) -> None:
        """Attach the reproducibility stamps (versions, host, seed, metric impls) to the
        run config so W&B carries them, not just the local summary.json."""
        if self.enabled and self.run is not None:
            try:
                self.run.config.update({"environment": env}, allow_val_change=True)
            except Exception as e:
                print(f"[wandb] config env update failed ({e})")

    def save_file(self, path) -> None:
        """Attach a results file (e.g. predictions.jsonl) to the run for provenance."""
        if self.enabled:
            try:
                self.wandb.save(str(path), base_path=str(path.parent), policy="now")
            except Exception as e:
                print(f"[wandb] save_file failed ({e}) — local copy is authoritative")

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()
