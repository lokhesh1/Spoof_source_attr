"""Dataset splitting strategies.

* ASVspoof2019 : all splits combined, then **stratified 5-fold** cross-validation.
* CFAD         : the **official** train/dev/test split (dev carved from train if absent).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from common.datasets import Record


def asvspoof_folds(records: List[Record], label_to_idx: Dict[str, int],
                   n_folds: int = 5, seed: int = 42
                   ) -> List[Tuple[List[Record], List[Record]]]:
    """Return a list of (train_records, val_records) for stratified K-fold CV."""
    from sklearn.model_selection import StratifiedKFold

    y = np.array([label_to_idx[r.label_name] for r in records])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    idx = np.arange(len(records))
    for tr, va in skf.split(idx, y):
        folds.append(([records[i] for i in tr], [records[i] for i in va]))
    return folds


def cfad_train_dev_test(splits: Dict[str, List[Record]], seed: int = 42,
                        dev_frac: float = 0.1
                        ) -> Tuple[List[Record], List[Record], List[Record]]:
    """Return (train, dev, test) records from the official CFAD split.

    If the release has no ``dev`` split, a stratified ``dev_frac`` slice is held
    out from train (so we still have a validation signal for early stopping).
    """
    train = splits["train"]
    test = splits["test"]
    if "dev" in splits and splits["dev"]:
        return train, splits["dev"], test

    from sklearn.model_selection import train_test_split
    y = [r.label_name for r in train]
    tr, dev = train_test_split(train, test_size=dev_frac, random_state=seed, stratify=y)
    return tr, dev, test
