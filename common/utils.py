"""Small cross-cutting helpers: seeding, device selection, logging, JSON IO."""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Make a run reproducible (CPU + CUDA + python + numpy)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str = "auto") -> str:
    """Return 'cuda' or 'cpu' given a preference string."""
    if device == "cpu":
        return "cpu"
    try:
        import torch
        if device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        # auto
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """A logger that prints to stdout and (optionally) tees to a file."""
    logger = logging.getLogger(name)
    if logger.handlers:                       # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any):
    """Let json serialise numpy scalars/arrays produced by metrics."""
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
