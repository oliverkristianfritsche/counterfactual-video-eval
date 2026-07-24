"""eval — an eval harness for VLM benchmarking.

Config-driven: one YAML per experiment, one runner for every model. Model and dataset
backends are pluggable via a registry, so adding a new model (or dataset) is a single
adapter file. Weights & Biases logging is built in and optional (mode: disabled/offline/online).

Importing this package registers the built-in model + dataset adapters.
"""

from . import models as _models  # noqa: F401  (registers model adapters)
from . import data as _data      # noqa: F401  (registers dataset adapters)

__all__ = ["models", "data"]
