"""Tiny registry so models/datasets plug in by name (config-selectable)."""
from __future__ import annotations

MODELS: dict = {}
DATASETS: dict = {}


def register_model(name: str):
    def deco(cls):
        MODELS[name] = cls
        return cls
    return deco


def register_dataset(name: str):
    def deco(cls):
        DATASETS[name] = cls
        return cls
    return deco


def build_model(cfg):
    kind = cfg.model.get("kind") or cfg.model.get("name")
    if kind not in MODELS:
        raise KeyError(f"Unknown model '{kind}'. Registered: {sorted(MODELS)}")
    return MODELS[kind](cfg)


def build_dataset(cfg):
    kind = cfg.dataset.get("kind") or cfg.dataset.get("name")
    if kind not in DATASETS:
        raise KeyError(f"Unknown dataset '{kind}'. Registered: {sorted(DATASETS)}")
    return DATASETS[kind](cfg)
