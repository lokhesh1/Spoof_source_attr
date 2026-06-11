"""Pre-extract and cache PTM features for a dataset (optional convenience).

Both ``main_single.py`` and ``main_fusion.py`` populate the cache lazily, but
running extraction once up-front is handy (e.g. on a GPU node) so that later
training runs are pure cache reads.

Examples
--------
    python extract_features.py --ptm wav2vec2 --dataset asvspoof2019
    python extract_features.py --ptm whisper  --dataset cfad
"""
from __future__ import annotations

import argparse

from config import PTM_REGISTRY, add_common_args, config_from_args
from common.utils import set_seed, resolve_device, get_logger
from common.ptm_extractors import build_extractor
from common.datasets import (SourceAttrDataset, build_label_map,
                             load_asvspoof_all, load_cfad_splits)
from common.feature_cache import prefetch


def main():
    parser = argparse.ArgumentParser(description="Pre-extract PTM features into the cache")
    parser.add_argument("--ptm", required=True, choices=list(PTM_REGISTRY))
    add_common_args(parser)
    args = parser.parse_args()

    cfg = config_from_args(args)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger = get_logger("extract")

    if cfg.dataset == "asvspoof2019":
        records = load_asvspoof_all(cfg.asvspoof_root, cfg.include_bonafide)
    else:
        splits = load_cfad_splits(cfg.cfad_root, cfg.include_bonafide)
        records = splits["train"] + splits.get("dev", []) + splits["test"]

    label_to_idx = build_label_map(records)
    extractor = build_extractor(args.ptm, device=device)
    ds = SourceAttrDataset(records, label_to_idx, [args.ptm], {args.ptm: extractor},
                           cfg.dataset, cache_dir=cfg.cache_dir, target_sr=cfg.target_sr,
                           min_duration=cfg.min_duration, max_duration=cfg.max_duration,
                           use_cache=True)
    logger.info(f"extracting {args.ptm} features for {len(records)} {cfg.dataset} utts "
                f"on {device} -> {cfg.cache_dir}")
    prefetch(ds, desc=f"[{args.ptm}] {cfg.dataset}")
    logger.info("done.")


if __name__ == "__main__":
    main()
