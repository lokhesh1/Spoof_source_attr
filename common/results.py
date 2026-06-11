"""Helpers to lay out and persist run artefacts consistently.

Directory convention (makes cross-PTM comparison trivial)::

    results/
      single/{dataset}/{ptm}/
        fold_1/ ... fold_5/        (ASVspoof: per-fold metrics + confusion PNGs)
        summary.json               (mean +/- std across folds, or test metrics)
        config.json  training_log.csv  best_model.pt
      fusion/{dataset}/{ptmA}+{ptmB}/
        ...
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from common.utils import save_json


def run_dir(results_dir: str, trial: str, dataset: str, tag: str) -> str:
    path = os.path.join(results_dir, trial, dataset, tag)
    os.makedirs(path, exist_ok=True)
    return path


def save_config(cfg, path: str) -> None:
    save_json(cfg.as_dict(), path)


def save_history_csv(history: List[dict], path: str) -> None:
    if not history:
        return
    import pandas as pd
    pd.DataFrame(history).to_csv(path, index=False)


def save_model(model, path: str) -> None:
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def summarize_folds(fold_metrics: List[Dict]) -> Dict:
    """Aggregate per-fold metric dicts into mean/std for the scalar fields."""
    def _vals(key):
        return [m[key] for m in fold_metrics if key in m and m[key] is not None
                and not (isinstance(m[key], float) and np.isnan(m[key]))]

    scalar_keys = ["accuracy", "eer_macro_ovr", "detection_eer_spoof_vs_bonafide"]
    summary: Dict[str, Dict[str, float]] = {}
    for k in scalar_keys:
        v = _vals(k)
        if v:
            summary[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                          "per_fold": [float(x) for x in v]}

    # mean per-attack accuracy across folds
    attack_acc: Dict[str, List[float]] = {}
    for m in fold_metrics:
        for atk, a in m.get("per_attack_accuracy", {}).items():
            if a is not None and not (isinstance(a, float) and np.isnan(a)):
                attack_acc.setdefault(atk, []).append(float(a))
    summary["per_attack_accuracy_mean"] = {k: float(np.mean(v)) for k, v in attack_acc.items()}
    return summary
