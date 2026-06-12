"""PTM feature extractors.

Each extractor wraps one pre-trained model and exposes a single method::

    extract(waveform_16k: np.ndarray) -> np.ndarray   # shape (T, D)

where ``T`` is the number of frames (time steps) and ``D`` is the hidden size.
We always return the *last hidden state* (a frame-level sequence) so that the
downstream 1D-CNN front-end is identical for every PTM.

Notes on the two non-sequence models
-------------------------------------
* ``xvector`` (speechbrain/spkrec-xvect-voxceleb) normally returns a single
  utterance-level embedding. To keep the CNN pipeline uniform we tap the
  **last 512-d TDNN frame layer** (the 512-d "last hidden state", before the
  1500-d expansion / statistics pooling), giving a (T, 512) sequence.
* ``wav2vec2_emo`` (speechbrain emotion model) is a fine-tuned wav2vec2; we
  return its wav2vec2 backbone's last hidden state (D = 768).

All heavy imports (torch/transformers/speechbrain) are done lazily inside the
constructors so that simply importing this module is cheap and so that a
missing optional dependency only breaks the PTM that needs it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from config import PTM_REGISTRY


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class PTMExtractor:
    """Abstract base. Sub-classes implement :meth:`_extract`."""

    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.spec = PTM_REGISTRY[name]
        self.hf_id = self.spec["hf_id"]
        self.device = device

    @property
    def hidden_dim(self) -> int:
        return self.spec["hidden_dim"]

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """waveform: mono float32 @ 16 kHz -> features (T, D) float32."""
        import torch
        with torch.no_grad():
            feats = self._extract(np.asarray(waveform, dtype=np.float32))
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2:
            raise ValueError(f"{self.name}: expected (T, D) features, got {feats.shape}")
        return feats

    def _extract(self, waveform: np.ndarray) -> np.ndarray:   # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# HuggingFace Wav2Vec2 / XLS-R
# --------------------------------------------------------------------------- #
class HFWav2Vec2Extractor(PTMExtractor):
    def __init__(self, name: str, device: str = "cpu"):
        super().__init__(name, device)
        import torch  # noqa: F401
        from transformers import AutoFeatureExtractor, Wav2Vec2Model
        self.processor = AutoFeatureExtractor.from_pretrained(self.hf_id)
        self.model = Wav2Vec2Model.from_pretrained(self.hf_id).to(self.device).eval()

    def _extract(self, waveform: np.ndarray) -> np.ndarray:
        inputs = self.processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        )
        input_values = inputs["input_values"].to(self.device)
        out = self.model(input_values)
        return out.last_hidden_state.squeeze(0).cpu().numpy()   # (T, D)


# --------------------------------------------------------------------------- #
# HuggingFace WavLM
# --------------------------------------------------------------------------- #
class HFWavLMExtractor(PTMExtractor):
    def __init__(self, name: str, device: str = "cpu"):
        super().__init__(name, device)
        from transformers import AutoFeatureExtractor, WavLMModel
        self.processor = AutoFeatureExtractor.from_pretrained(self.hf_id)
        self.model = WavLMModel.from_pretrained(self.hf_id).to(self.device).eval()

    def _extract(self, waveform: np.ndarray) -> np.ndarray:
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        out = self.model(input_values)
        return out.last_hidden_state.squeeze(0).cpu().numpy()


# --------------------------------------------------------------------------- #
# HuggingFace Whisper (encoder last hidden state)
# --------------------------------------------------------------------------- #
class HFWhisperExtractor(PTMExtractor):
    """Whisper pads to 30 s internally (1500 encoder frames @ 50 fps).

    We slice the encoder output back to the frames that correspond to the
    **actual** audio length so the sequence length matches the real clip (no
    fixed duration). Clips longer than 30 s are capped at Whisper's 1500 frames.
    """

    FPS = 50  # Whisper encoder frames per second

    def __init__(self, name: str, device: str = "cpu"):
        super().__init__(name, device)
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.processor = WhisperFeatureExtractor.from_pretrained(self.hf_id)
        self.model = WhisperModel.from_pretrained(self.hf_id).to(self.device).eval()
        self.encoder = self.model.get_encoder()

    def _extract(self, waveform: np.ndarray) -> np.ndarray:
        import math
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        feats = inputs["input_features"].to(self.device)
        out = self.encoder(feats)
        hidden = out.last_hidden_state.squeeze(0)               # (1500, 512)
        keep = max(1, min(hidden.size(0), math.ceil(len(waveform) / 16000 * self.FPS)))
        return hidden[:keep].cpu().numpy()                      # (~secs*50, 512)


# --------------------------------------------------------------------------- #
# SpeechBrain emotion-recognition wav2vec2 (backbone last hidden state)
# --------------------------------------------------------------------------- #
class SBEmotionExtractor(PTMExtractor):
    def __init__(self, name: str, device: str = "cpu"):
        super().__init__(name, device)
        from speechbrain.inference.interfaces import foreign_class
        self.clf = foreign_class(
            source=self.hf_id,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": self.device},
        )

    def _extract(self, waveform: np.ndarray) -> np.ndarray:
        import torch
        wav = torch.tensor(waveform, dtype=torch.float32, device=self.device).unsqueeze(0)
        wav_lens = torch.ones(1, device=self.device)
        # The speechbrain wrapper exposes the wav2vec2 backbone under `mods`.
        feats = self.clf.mods.wav2vec2(wav, wav_lens)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats.squeeze(0).detach().cpu().numpy()          # (T, 768)


# --------------------------------------------------------------------------- #
# SpeechBrain x-vector (frame-level features before statistics pooling)
# --------------------------------------------------------------------------- #
class SBXVectorExtractor(PTMExtractor):
    """x-vector frame-level **512-d** last hidden state, as a (T, 512) sequence.

    The TDNN frame layers are 512-d; the final block expands to 1500-d right
    before statistics pooling. We tap the **last 512-d frame layer**, so the
    representation is the 512-d "last hidden state" *and* still a time sequence
    -- it flows through the same 1D-CNN + masked pool as every other PTM (the
    temporal averaging happens after the CNN, not here).
    """

    FRAME_DIM = 512  # x-vector TDNN frame-level hidden size

    def __init__(self, name: str, device: str = "cpu"):
        super().__init__(name, device)
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # older speechbrain
            from speechbrain.pretrained import EncoderClassifier
        self.clf = EncoderClassifier.from_hparams(
            source=self.hf_id, run_opts={"device": self.device}
        )
        self._frame_feats = None
        self._register_hooks()

    def _register_hooks(self):
        import torch

        def _hook(_module, _inp, out):
            # Keep the deepest frame-level (B, T, 512) activation = the last
            # 512-d TDNN layer, before the 1500-d expansion / statistics pooling.
            if (isinstance(out, torch.Tensor) and out.dim() == 3
                    and out.shape[-1] == self.FRAME_DIM and out.shape[1] > 1):
                self._frame_feats = out.detach()

        for module in self.clf.mods.embedding_model.modules():
            module.register_forward_hook(_hook)

    def _extract(self, waveform: np.ndarray) -> np.ndarray:
        import torch
        wav = torch.tensor(waveform, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._frame_feats = None
        emb = self.clf.encode_batch(wav)                        # triggers the hooks
        if self._frame_feats is not None:
            return self._frame_feats.squeeze(0).cpu().numpy()   # (T, 512)
        # Fallback: tile the 512-d utterance embedding into a short sequence.
        e = emb.reshape(-1)[: self.FRAME_DIM].cpu().numpy()     # (512,)
        return np.tile(e[None, :], (16, 1))                     # (16, 512)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_KIND_TO_CLASS = {
    "hf_wav2vec2": HFWav2Vec2Extractor,
    "hf_wavlm": HFWavLMExtractor,
    "hf_whisper": HFWhisperExtractor,
    "sb_emotion": SBEmotionExtractor,
    "sb_xvector": SBXVectorExtractor,
}


def build_extractor(name: str, device: str = "cpu") -> PTMExtractor:
    """Instantiate the extractor for a registered PTM name."""
    if name not in PTM_REGISTRY:
        raise KeyError(f"Unknown PTM '{name}'. Valid: {list(PTM_REGISTRY)}")
    kind = PTM_REGISTRY[name]["kind"]
    return _KIND_TO_CLASS[kind](name, device=device)
