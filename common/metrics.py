"""Evaluation metrics + plots for ADSA.

Reported per run:
  * overall accuracy
  * per-attack-type accuracy (per-class recall)
  * EER  -- attribution is multi-class, so we report the mean one-vs-rest EER
           (and, when bonafide is a class, a spoof-vs-bonafide detection EER)
  * confusion matrix saved as PNG (raw counts + row-normalised)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       class_names: List[str]) -> Dict[str, float]:
    """Per-class recall == accuracy restricted to that attack type."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out: Dict[str, float] = {}
    for c, name in enumerate(class_names):
        mask = y_true == c
        out[name] = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")
    return out


# --------------------------------------------------------------------------- #
# EER
# --------------------------------------------------------------------------- #
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Binary EER from target/non-target scores (higher score = target)."""
    from sklearn.metrics import roc_curve
    scores, labels = np.asarray(scores), np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    return eer, float(thr[i])


def macro_ovr_eer(y_true: np.ndarray, probs: np.ndarray,
                  class_names: List[str]) -> Tuple[float, Dict[str, float]]:
    """Mean one-vs-rest EER across classes (+ per-class breakdown)."""
    y_true = np.asarray(y_true)
    per: Dict[str, float] = {}
    vals: List[float] = []
    for c, name in enumerate(class_names):
        binary = (y_true == c).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            per[name] = float("nan")
            continue
        eer, _ = compute_eer(probs[:, c], binary)
        per[name] = eer
        if not np.isnan(eer):
            vals.append(eer)
    return (float(np.mean(vals)) if vals else float("nan")), per


def detection_eer(y_true: np.ndarray, probs: np.ndarray,
                  bonafide_index: int) -> float:
    """Spoof-vs-bonafide detection EER (only meaningful if bonafide is a class).

    Target = spoof. Score = total spoofing probability = 1 - P(bonafide).
    """
    y_true = np.asarray(y_true)
    is_spoof = (y_true != bonafide_index).astype(int)
    spoof_score = 1.0 - probs[:, bonafide_index]
    eer, _ = compute_eer(spoof_score, is_spoof)
    return eer


# --------------------------------------------------------------------------- #
# Confusion matrix plot
# --------------------------------------------------------------------------- #
def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str], out_path: str,
                          normalize: bool = False, title: Optional[str] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import os

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.55)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True attack type", xlabel="Predicted attack type",
           title=title or ("Confusion matrix" + (" (normalized)" if normalize else "")))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    fmt = ".2f" if normalize else ".0f"
    for i in range(n):
        for j in range(n):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# One-call report
# --------------------------------------------------------------------------- #
def evaluate_report(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                    class_names: List[str], out_dir: str, tag: str = "test",
                    bonafide_name: str = "bonafide") -> Dict:
    """Compute every metric, save confusion-matrix PNGs, return a metrics dict."""
    import os
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probs = np.asarray(probs)

    acc = accuracy(y_true, y_pred)
    per_attack = per_class_accuracy(y_true, y_pred, class_names)
    eer_macro, eer_per = macro_ovr_eer(y_true, probs, class_names)

    metrics = {
        "tag": tag,
        "accuracy": acc,
        "eer_macro_ovr": eer_macro,
        "per_attack_accuracy": per_attack,
        "per_class_eer_ovr": eer_per,
        "num_samples": int(len(y_true)),
        "class_names": class_names,
    }
    if bonafide_name in class_names:
        metrics["detection_eer_spoof_vs_bonafide"] = detection_eer(
            y_true, probs, class_names.index(bonafide_name))

    save_confusion_matrix(y_true, y_pred, class_names,
                          os.path.join(out_dir, f"confusion_matrix_{tag}.png"),
                          normalize=False, title=f"Confusion matrix ({tag})")
    save_confusion_matrix(y_true, y_pred, class_names,
                          os.path.join(out_dir, f"confusion_matrix_{tag}_normalized.png"),
                          normalize=True, title=f"Confusion matrix ({tag}, normalized)")
    return metrics
