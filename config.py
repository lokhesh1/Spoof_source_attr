"""Central configuration for the ADSA experiments.

Everything that a reader/maintainer might want to tweak lives here:
  * dataset roots (the data is downloaded later -> just point these paths)
  * the PTM registry (the 6 pre-trained models)
  * training / model / loss hyper-parameters
  * a small argparse helper shared by ``main_single.py`` and ``main_fusion.py``

Nothing here imports torch/transformers, so this module is cheap to import.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


# --------------------------------------------------------------------------- #
# Pre-Trained Model (PTM) registry
# --------------------------------------------------------------------------- #
# `kind` selects the extractor implementation in common/ptm_extractors.py.
# `hidden_dim` is informational only (the CNN front-end infers it lazily),
# but it is handy for logging / sanity checks.
PTM_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- Monolingual self-supervised ---------------------------------------
    "wav2vec2": {
        "kind": "hf_wav2vec2",
        "hf_id": "facebook/wav2vec2-base",
        "hidden_dim": 768,
        "group": "monolingual",
    },
    "wavlm": {
        "kind": "hf_wavlm",
        "hf_id": "microsoft/wavlm-base",
        "hidden_dim": 768,
        "group": "monolingual",
    },
    # --- Multilingual self-supervised / ASR --------------------------------
    "xlsr": {
        "kind": "hf_wav2vec2",  # XLS-R shares the Wav2Vec2 architecture
        "hf_id": "facebook/wav2vec2-xls-r-300m",
        "hidden_dim": 1024,
        "group": "multilingual",
    },
    "whisper": {
        "kind": "hf_whisper",
        "hf_id": "openai/whisper-base",
        "hidden_dim": 512,
        "group": "multilingual",
    },
    # --- Para-linguistic / speaker -----------------------------------------
    "wav2vec2_emo": {
        "kind": "sb_emotion",
        "hf_id": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        "hidden_dim": 768,
        "group": "emotion",
    },
    "xvector": {
        "kind": "sb_xvector",
        "hf_id": "speechbrain/spkrec-xvect-voxceleb",
        "hidden_dim": 1500,  # frame-level TDNN features (pre stats-pooling)
        "group": "speaker",
    },
}

VALID_DATASETS = ("cfad", "asvspoof2019")


@dataclass
class Config:
    """Single source of truth for one experiment run."""

    # ---- data ------------------------------------------------------------
    dataset: str = "asvspoof2019"          # 'cfad' | 'asvspoof2019'
    asvspoof_root: str = "/home/hp/Desktop/Spoof_source_attr/LA"
    asvspoof_access: str = "LA"            # 'LA' (logical access) – standard for source attribution
    cfad_root: str = "data/CFAD"
    include_bonafide: bool = False          # treat real/bonafide audio as an extra class
    target_sr: int = 16000
    # Variable-length audio (no fixed truncation). Short clips are padded up to
    # min_duration so pooling never collapses to length 0; max_duration is an
    # optional OOM safety cap (None = keep full length).
    min_duration: float = 1.0
    max_duration: "float | None" = None

    # ---- feature cache ---------------------------------------------------
    cache_dir: str = "/media/hp/36E07D91E07D57D7/cache/features"      # extracted PTM features are memoised here
    use_cache: bool = True

    # ---- model -----------------------------------------------------------
    proj_dim: int = 120                    # the per-PTM embedding size (projection target)
    fcn_dims: tuple = (256, 128, 64)       # three dense layers of the FCN
    dropout: float = 0.0                   # spec does not use dropout; expose anyway

    # ---- loss (fusion only) ---------------------------------------------
    rd_alpha: float = 2.0                  # Renyi divergence order (alpha != 1)
    rd_lambda: float = 0.1                 # weight of the RD alignment term
    rd_symmetric: bool = True              # average D(p||q) and D(q||p)

    # ---- optimisation ----------------------------------------------------
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10                     # early-stopping patience (epochs)
    num_workers: int = 0                   # Windows-friendly default

    # ---- cross-validation (ASVspoof2019) --------------------------------
    n_folds: int = 5

    # ---- bookkeeping -----------------------------------------------------
    seed: int = 42
    results_dir: str = "results"
    device: str = "auto"                   # 'auto' | 'cuda' | 'cpu'

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# argparse helpers (shared by both trials)
# --------------------------------------------------------------------------- #
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments common to single & fusion trials."""
    d = Config()
    parser.add_argument("--dataset", choices=VALID_DATASETS, default=d.dataset)
    parser.add_argument("--asvspoof-root", default=d.asvspoof_root)
    parser.add_argument("--cfad-root", default=d.cfad_root)
    parser.add_argument("--include-bonafide", dest="include_bonafide",
                        action="store_true", default=d.include_bonafide)
    parser.add_argument("--no-bonafide", dest="include_bonafide", action="store_false")
    parser.add_argument("--min-duration", type=float, default=d.min_duration,
                        help="short clips are padded up to this many seconds")
    parser.add_argument("--max-duration", type=float, default=d.max_duration,
                        help="optional cap on clip length (seconds) to avoid OOM; default keeps full length")
    parser.add_argument("--cache-dir", default=d.cache_dir)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", default=d.use_cache)

    parser.add_argument("--proj-dim", type=int, default=d.proj_dim)
    parser.add_argument("--dropout", type=float, default=d.dropout)

    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--batch-size", type=int, default=d.batch_size)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--weight-decay", type=float, default=d.weight_decay)
    parser.add_argument("--patience", type=int, default=d.patience)
    parser.add_argument("--num-workers", type=int, default=d.num_workers)

    parser.add_argument("--n-folds", type=int, default=d.n_folds)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--results-dir", default=d.results_dir)
    parser.add_argument("--device", default=d.device, choices=["auto", "cuda", "cpu"])


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a Config from parsed args, keeping any defaults not exposed on CLI."""
    cfg = Config()
    for k, v in vars(args).items():
        if hasattr(cfg, k) and k not in ("ptm", "ptm_a", "ptm_b"):
            setattr(cfg, k, v)
    return cfg
