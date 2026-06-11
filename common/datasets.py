"""Dataset protocols + torch Dataset for ADSA.

Supports two corpora:

* **ASVspoof2019** (LA): parsed from the official cm-protocol files. For source
  attribution the class label is the spoofing **system id** (A01..A19); bonafide
  audio is an optional extra class. All splits are combined for 5-fold CV.

* **CFAD**: the official train/dev/test split. Because the on-disk layout of the
  CFAD release can vary, the loader is deliberately flexible — it uses a protocol
  file when present, otherwise infers the class from the per-class sub-directory
  name. Adjust :func:`parse_cfad_split` if your local layout differs.

The :class:`SourceAttrDataset` returns *cached PTM features* (not raw audio), so
it is shared unchanged by the single- and fusion-PTM trials (1 or 2 PTMs).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from common.feature_cache import get_features

AUDIO_EXTS = (".flac", ".wav", ".mp3", ".m4a", ".ogg")


# --------------------------------------------------------------------------- #
# Record container + audio loading
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    utt_id: str
    path: str
    label_name: str          # class name (system id / 'bonafide' / cfad method)
    split: str = ""          # 'train' | 'dev' | 'eval' | 'test' | ...
    meta: dict = field(default_factory=dict)


def load_audio(path: str, target_sr: int = 16000,
               min_duration: float = 1.0, max_duration: float | None = None) -> np.ndarray:
    """Load -> mono -> resample to target_sr, keeping the **full length**.

    No fixed-length truncation (so no temporal information is lost). Two guards:
      * very short clips are zero-padded up to ``min_duration`` so the two
        MaxPool(2) layers never collapse the sequence to length 0;
      * ``max_duration`` (optional) caps pathologically long clips to avoid OOM.
    """
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:                              # stereo -> mono
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    if max_duration is not None:
        wav = wav[: int(round(max_duration * target_sr))]
    min_n = int(round(min_duration * target_sr))
    if len(wav) < min_n:
        wav = np.pad(wav, (0, min_n - len(wav)))
    return wav.astype(np.float32)


# --------------------------------------------------------------------------- #
# ASVspoof2019 (LA) protocol parsing
# --------------------------------------------------------------------------- #
_ASV_SPLITS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev":   ("ASVspoof2019_LA_dev",   "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval":  ("ASVspoof2019_LA_eval",  "ASVspoof2019.LA.cm.eval.trl.txt"),
}


def parse_asvspoof_split(root: str, split: str, include_bonafide: bool = True) -> List[Record]:
    """Parse one ASVspoof2019 LA split into attribution records.

    Protocol line format (whitespace separated)::

        SPEAKER  UTT_ID  -  SYSTEM_ID  KEY
        LA_0079  LA_T_1271820  -  A01  spoof
        LA_0079  LA_T_1138215  -  -    bonafide
    """
    audio_subdir, proto_name = _ASV_SPLITS[split]
    proto_path = os.path.join(root, "ASVspoof2019_LA_cm_protocols", proto_name)
    flac_dir = os.path.join(root, audio_subdir, "flac")

    if not os.path.exists(proto_path):
        raise FileNotFoundError(
            f"ASVspoof protocol not found: {proto_path}\n"
            f"Expected the standard ASVspoof2019 LA directory layout under {root}."
        )

    records: List[Record] = []
    with open(proto_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.split()
            if len(tok) < 4:
                continue
            utt_id, system_id, key = tok[1], tok[-2], tok[-1].lower()
            if key == "bonafide":
                if not include_bonafide:
                    continue
                label = "bonafide"
            else:
                label = system_id            # A01..A19
            records.append(Record(
                utt_id=utt_id,
                path=os.path.join(flac_dir, utt_id + ".flac"),
                label_name=label,
                split=split,
            ))
    return records


def load_asvspoof_all(root: str, include_bonafide: bool = True) -> List[Record]:
    """Combine train + dev + eval (used for 5-fold CV)."""
    records: List[Record] = []
    for split in _ASV_SPLITS:
        records.extend(parse_asvspoof_split(root, split, include_bonafide))
    return records


# --------------------------------------------------------------------------- #
# CFAD protocol parsing (official train/dev/test split)
# --------------------------------------------------------------------------- #
# Common label-file names seen across CFAD-style releases; the first match wins.
_CFAD_LABEL_FILES = ("label.txt", "labels.txt", "protocol.txt", "meta.txt")


def parse_cfad_split(split_root: str, split: str,
                     include_bonafide: bool = True) -> List[Record]:
    """Parse one CFAD split directory.

    Strategy
    --------
    1. If a label/protocol file exists (``label.txt`` etc.), parse it. Each line
       is ``<relative_audio_path> <class_label>`` (extra columns ignored).
    2. Otherwise fall back to *directory-per-class*: every immediate sub-folder
       of ``split_root`` is treated as a class, and the audio files inside it as
       members of that class.

    The 'real'/'genuine' class is treated as bonafide and dropped when
    ``include_bonafide`` is False.
    """
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"CFAD split directory not found: {split_root}")

    records: List[Record] = []
    label_file = next((os.path.join(split_root, n) for n in _CFAD_LABEL_FILES
                       if os.path.exists(os.path.join(split_root, n))), None)

    if label_file:
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                tok = line.split()
                if len(tok) < 2:
                    continue
                rel_path, label = tok[0], tok[1]
                if _is_bonafide(label) and not include_bonafide:
                    continue
                apath = rel_path if os.path.isabs(rel_path) else os.path.join(split_root, rel_path)
                records.append(Record(
                    utt_id=os.path.splitext(os.path.basename(rel_path))[0],
                    path=apath,
                    label_name="bonafide" if _is_bonafide(label) else label,
                    split=split,
                ))
    else:
        # directory-per-class fallback
        for class_dir in sorted(d for d in os.listdir(split_root)
                                if os.path.isdir(os.path.join(split_root, d))):
            if _is_bonafide(class_dir) and not include_bonafide:
                continue
            label = "bonafide" if _is_bonafide(class_dir) else class_dir
            cdir = os.path.join(split_root, class_dir)
            for apath in _iter_audio(cdir):
                records.append(Record(
                    utt_id=f"{class_dir}__{os.path.splitext(os.path.basename(apath))[0]}",
                    path=apath,
                    label_name=label,
                    split=split,
                ))
    if not records:
        raise RuntimeError(
            f"No CFAD records parsed from {split_root}. "
            f"Check the layout and adjust parse_cfad_split()."
        )
    return records


def load_cfad_splits(root: str, include_bonafide: bool = True) -> Dict[str, List[Record]]:
    """Return {'train':[...], 'dev':[...], 'test':[...]} using the official split.

    ``dev`` is optional; if absent a small slice of train is used for validation
    by the training script.
    """
    splits: Dict[str, List[Record]] = {}
    for split in ("train", "dev", "test"):
        sdir = os.path.join(root, split)
        if os.path.isdir(sdir):
            splits[split] = parse_cfad_split(sdir, split, include_bonafide)
    if "train" not in splits or "test" not in splits:
        raise FileNotFoundError(
            f"Expected CFAD train/ and test/ directories under {root}. Found: {list(splits)}"
        )
    return splits


def _is_bonafide(name: str) -> bool:
    return name.lower() in {"real", "genuine", "bonafide", "bona-fide", "human", "0"}


def _iter_audio(directory: str):
    for ext in AUDIO_EXTS:
        yield from glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True)


# --------------------------------------------------------------------------- #
# Label mapping
# --------------------------------------------------------------------------- #
def build_label_map(records: Sequence[Record]) -> Dict[str, int]:
    """Deterministic class-name -> index map (sorted for reproducibility)."""
    names = sorted({r.label_name for r in records})
    return {name: i for i, name in enumerate(names)}


# --------------------------------------------------------------------------- #
# torch Dataset
# --------------------------------------------------------------------------- #
class SourceAttrDataset:
    """Returns cached PTM feature tensors for 1 (single) or 2 (fusion) PTMs.

    __getitem__ -> (features, label) where:
        * single  : features is a torch.FloatTensor of shape (T, D)
        * fusion  : features is a list [ (T_a, D_a), (T_b, D_b) ]
    """

    def __init__(self, records: Sequence[Record], label_to_idx: Dict[str, int],
                 ptm_names: Sequence[str], extractors: Dict[str, object],
                 dataset_name: str, cache_dir: str, target_sr: int = 16000,
                 min_duration: float = 1.0, max_duration: float | None = None,
                 use_cache: bool = True):
        import torch  # noqa: F401  (ensure torch present early)
        self.records = list(records)
        self.label_to_idx = label_to_idx
        self.ptm_names = list(ptm_names)
        self.extractors = extractors
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.target_sr = target_sr
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.use_cache = use_cache

    def __len__(self) -> int:
        return len(self.records)

    def _audio_loader(self, rec: Record) -> Callable[[], np.ndarray]:
        return lambda: load_audio(rec.path, self.target_sr,
                                  self.min_duration, self.max_duration)

    def __getitem__(self, idx: int):
        import torch
        rec = self.records[idx]
        loader = self._audio_loader(rec)
        feats = []
        for ptm in self.ptm_names:
            arr = get_features(
                self.extractors[ptm], ptm, self.dataset_name, rec.utt_id,
                loader, self.cache_dir, self.use_cache,
            )
            feats.append(torch.from_numpy(np.ascontiguousarray(arr)).float())
        label = self.label_to_idx[rec.label_name]
        if len(feats) == 1:
            return feats[0], label
        return feats, label


def _pad_stack(seqs):
    """Zero-pad a list of (T_i, D) tensors to (B, T_max, D); return padded + lengths."""
    import torch
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    t_max = int(lengths.max())
    d = seqs[0].size(1)
    out = seqs[0].new_zeros(len(seqs), t_max, d)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out, lengths


def collate_single(batch):
    """Collate for the single-PTM trial -> ((B,T,D), lengths), labels.

    Variable-length sequences are zero-padded to the batch max; ``lengths`` lets
    the model mask the padding out of its temporal average pool.
    """
    import torch
    xs, ys = zip(*batch)
    padded, lengths = _pad_stack(list(xs))
    return (padded, lengths), torch.tensor(ys, dtype=torch.long)


def collate_fusion(batch):
    """Collate for the fusion trial -> ((Xa,La),(Xb,Lb)), labels (each branch padded)."""
    import torch
    feats, ys = zip(*batch)
    a, la = _pad_stack([f[0] for f in feats])
    b, lb = _pad_stack([f[1] for f in feats])
    return ((a, la), (b, lb)), torch.tensor(ys, dtype=torch.long)
