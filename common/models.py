"""Neural network components for ADSA.

Two downstream front-ends, each matching the reference paper.

**Single-PTM trial** (``SinglePTMModel`` -> ``PooledCNNFrontEnd``) -- the PTM
last hidden state is average-pooled over time *before* the CNN (paper Appendix
A.2), so the 1D conv slides along the feature axis of the pooled vector::

    features (B, T, D)
      -> masked average pool over time          -> (B, D)         # pool FIRST
      -> (B, 1, D)                                                 # D = conv length, 1 channel
      -> Conv1d(1  -> 256, k=3, pad=1) -> BN -> ReLU -> MaxPool1d(2)
      -> Conv1d(256-> 128, k=3, pad=1) -> BN -> ReLU -> MaxPool1d(2)
      -> flatten                                -> (B, 128 * (D//4))
      -> FCN: 256 -> 128 -> 64 -> num_classes   (logits; softmax at inference)

There is no global temporal pool or 120-d projection in the single trail -- the
conv output is flattened straight into the FCN, exactly as the paper describes.

**Fusion trial** (``FusionModel`` -> ``CNNFrontEnd``) -- length-agnostic
*sequence* conv (conv over time, masked temporal pool, Linear -> 120), kept as
the FINDER design. The padded tail is re-zeroed after every conv so a clip's
embedding is invariant to how much padding its batch-mates force.

``LazyConv1d`` / ``LazyLinear`` infer their input sizes, so the same code works
for every PTM (one forward pass must run before the optimizer is built -- see
common.engine.materialize_lazy).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

# Channels out of the second conv == the pooled embedding width fed to the projection.
_CONV2_CHANNELS = 128


def conv_out_length(lengths: torch.Tensor) -> torch.Tensor:
    """Valid time length after the front-end's two MaxPool1d(2) layers.

    Conv1d(k=3, pad=1) preserves length; each MaxPool1d(2) floors it by 2.
    """
    out = torch.div(torch.div(lengths, 2, rounding_mode="floor"), 2, rounding_mode="floor")
    return out.clamp(min=1)


def _frame_mask(width: int, valid_len: torch.Tensor, device, dtype) -> torch.Tensor:
    """(B, width) float mask: 1 for t < valid_len, else 0."""
    ar = torch.arange(width, device=device).unsqueeze(0)          # (1, width)
    return (ar < valid_len.unsqueeze(1)).to(dtype)               # (B, width)


def _zero_pad_tail(x: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
    """Zero the padded tail of (B, C, T) so conv bias can't leak into padding."""
    mask = _frame_mask(x.size(-1), valid_len, x.device, x.dtype)
    return x * mask.unsqueeze(1)


def _halve(lengths: torch.Tensor) -> torch.Tensor:
    return torch.div(lengths, 2, rounding_mode="floor")


def masked_mean(feats: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
    """Average ``feats`` (B, C, T) over only the first ``valid_len`` time steps."""
    mask = _frame_mask(feats.size(-1), valid_len, feats.device, feats.dtype)
    summed = (feats * mask.unsqueeze(1)).sum(dim=-1)             # (B, C)
    denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)       # (B, 1)
    return summed / denom


class CNNFrontEnd(nn.Module):
    """1D-CNN feature processor + masked temporal pool + projection to ``proj_dim``.

    The padded tail is re-zeroed after every conv (which has a bias) so the
    embedding of a clip is exactly invariant to how much padding its batch-mates
    force -- i.e. deterministic per sample, independent of batch composition.
    """

    def __init__(self, proj_dim: int = 120):
        super().__init__()
        self.conv1 = nn.LazyConv1d(256, kernel_size=3, padding=1)   # in-channels = D (inferred)
        self.conv2 = nn.Conv1d(256, _CONV2_CHANNELS, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.proj = nn.Linear(_CONV2_CHANNELS, proj_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T) for Conv1d (channels = feature dim)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))                   # k3,pad1 preserves length
        x = _zero_pad_tail(x, lengths)
        x = self.pool(x); lengths = _halve(lengths)     # MaxPool(2) floors length
        x = _zero_pad_tail(x, lengths)
        x = torch.relu(self.conv2(x))
        x = _zero_pad_tail(x, lengths)
        x = self.pool(x); lengths = _halve(lengths)
        pooled = masked_mean(x, lengths.clamp(min=1))   # (B, 128)
        return self.proj(pooled)                        # (B, proj_dim)


class FCN(nn.Module):
    """Three dense layers (256, 128, 64) + a classification layer.

    Pass ``in_dim=None`` to make the first layer ``nn.LazyLinear`` so the input
    width is inferred on the first forward pass -- used by the single-PTM model,
    whose flattened conv width (128 * D//4) depends on the PTM hidden size.
    """

    def __init__(self, in_dim: "int | None", num_classes: int,
                 dims: Sequence[int] = (256, 128, 64), dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for i, h in enumerate(dims):
            first = nn.LazyLinear(h) if (i == 0 and in_dim is None) else nn.Linear(prev, h)
            layers += [first, nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)       # logits (softmax via CE / at inference)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


class PooledCNNFrontEnd(nn.Module):
    """Paper-faithful single-trail front-end: average-pool over time *first*,
    then two 1D-conv blocks over the feature axis, then flatten.

    The PTM last hidden state is masked-averaged over the valid time steps to a
    (B, D) vector (paper Appendix A.2), which is treated as a length-D,
    single-channel signal. Each conv block is Conv -> BatchNorm -> ReLU ->
    MaxPool(2); the result is flattened (no global temporal pool, no 120-d
    projection -- those belong to the FINDER fusion design only).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, _CONV2_CHANNELS, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(_CONV2_CHANNELS)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, D, T); average over the valid frames -> (B, D)
        rep = masked_mean(x.transpose(1, 2), lengths.clamp(min=1))   # (B, D)
        h = rep.unsqueeze(1)                                         # (B, 1, D)
        h = self.pool(torch.relu(self.bn1(self.conv1(h))))          # (B, 256, D//2)
        h = self.pool(torch.relu(self.bn2(self.conv2(h))))          # (B, 128, D//4)
        return torch.flatten(h, 1)                                  # (B, 128 * (D//4))


class SinglePTMModel(nn.Module):
    """Trial 1: one PTM representation -> avg-pool -> CNN -> flatten -> FCN."""

    def __init__(self, num_classes: int,
                 fcn_dims: Sequence[int] = (256, 128, 64), dropout: float = 0.0):
        super().__init__()
        self.frontend = PooledCNNFrontEnd()
        self.fcn = FCN(None, num_classes, fcn_dims, dropout)   # lazy in_dim = 128 * D//4

    def forward(self, x: torch.Tensor, lengths: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.frontend(x, lengths)    # (B, 128 * D//4)
        logits = self.fcn(feat)             # (B, num_classes)
        return logits, feat


class FusionModel(nn.Module):
    """Trial 2: two PTMs -> two independent CNN front-ends -> two 120-d
    embeddings -> concatenate (240) -> FCN.

    The two embeddings are also returned so the joint loss can apply Renyi
    divergence alignment between them.
    """

    def __init__(self, num_classes: int, proj_dim: int = 120,
                 fcn_dims: Sequence[int] = (256, 128, 64), dropout: float = 0.0):
        super().__init__()
        self.frontend_a = CNNFrontEnd(proj_dim)
        self.frontend_b = CNNFrontEnd(proj_dim)
        self.fcn = FCN(2 * proj_dim, num_classes, fcn_dims, dropout)

    def forward(self, x_a: torch.Tensor, len_a: torch.Tensor,
                x_b: torch.Tensor, len_b: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_a = self.frontend_a(x_a, len_a)         # (B, 120)
        emb_b = self.frontend_b(x_b, len_b)         # (B, 120)
        fused = torch.cat([emb_a, emb_b], dim=1)    # (B, 240)
        logits = self.fcn(fused)
        return logits, emb_a, emb_b
