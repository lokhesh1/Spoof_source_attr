"""On-disk memoisation of extracted PTM features.

Extracting features from a 300M-param model for an entire dataset is the
expensive part of these experiments, and we re-use the same features across
folds and across the single/fusion trials. So we cache them once as ``.npy``.

Layout::

    {cache_dir}/{ptm}/{dataset}/{utt_id}.npy   # array of shape (T, D)
"""
from __future__ import annotations

import os
import numpy as np


def _safe_id(utt_id: str) -> str:
    """Make an utterance id safe to use as a filename."""
    return utt_id.replace(os.sep, "_").replace("/", "_").replace("\\", "_")


def feature_path(cache_dir: str, ptm: str, dataset: str, utt_id: str) -> str:
    return os.path.join(cache_dir, ptm, dataset, _safe_id(utt_id) + ".npy")


def get_features(extractor, ptm: str, dataset: str, utt_id: str,
                 waveform_loader, cache_dir: str, use_cache: bool = True) -> np.ndarray:
    """Return features for one utterance, extracting + caching on a miss.

    Parameters
    ----------
    extractor : PTMExtractor
    waveform_loader : callable returning the (already 16 kHz, fixed-length) waveform.
                      Called lazily so we skip audio IO on a cache hit.
    """
    path = feature_path(cache_dir, ptm, dataset, utt_id)
    if use_cache and os.path.exists(path):
        return np.load(path)

    feats = extractor.extract(waveform_loader())
    if use_cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, feats)
    return feats


def is_cached(cache_dir: str, ptm: str, dataset: str, utt_id: str) -> bool:
    return os.path.exists(feature_path(cache_dir, ptm, dataset, utt_id))


def prefetch(dataset, desc: str = "extracting features") -> None:
    """Iterate a dataset once to populate the on-disk feature cache.

    Doing this up-front (with a progress bar) keeps the first training epoch
    fast and decouples the slow PTM forward passes from the training loop.
    ``dataset`` only needs ``__len__`` and ``__getitem__``.
    """
    try:
        from tqdm import tqdm
        rng = tqdm(range(len(dataset)), desc=desc)
    except ImportError:
        rng = range(len(dataset))
    for i in rng:
        dataset[i]
