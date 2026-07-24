"""Experiment configuration — one YAML file per run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    path: Path
    model: dict
    dataset: dict
    video: dict
    prompt: dict
    generation: dict
    eval: dict
    wandb: dict
    experiment: dict
    raw: dict

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path).resolve()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        def sec(name: str) -> dict:
            return dict(raw.get(name) or {})

        return cls(
            path=path,
            model=sec("model"),
            dataset=sec("dataset"),
            video=sec("video"),
            prompt=sec("prompt"),
            generation=sec("generation"),
            eval=sec("eval"),
            wandb=sec("wandb"),
            experiment=sec("experiment"),
            raw=raw,
        )

    def resolve(self, rel: str | Path) -> Path:
        """Resolve a path relative to the config file's directory."""
        return (self.path.parent / rel).resolve()

    def system_prompt(self) -> str:
        f = self.prompt.get("system_prompt_file")
        if f:
            p = self.resolve(f)
            if p.exists():
                return p.read_text().strip()
        return (self.prompt.get("system_prompt") or "").strip()
