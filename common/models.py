"""Neural network components for ADSA.

Pipeline per PTM representation (steps 2-3 of the spec), now **length-agnostic**::

    features (B, T, D)          # variable T -- no fixed 4 s truncation
      -> transpose -> (B, D, T)
      -> Conv1d(D  -> 256, k=3, pad=1) -> ReLU -> MaxPool1d(2)
      -> Conv1d(256-> 128, k=3, pad=1) -> ReLU -> MaxPool1d(2)
      -> masked global average pool over time   -> (B, 128)
      -> Linear(128 -> proj_dim=120)            # the per-PTM embedding

The conv stack is length-agnostic; variable-length clips are zero-padded per
batch and the trailing padded frames are *masked out* of the average pool, so
no temporal information is discarded and padding never biases the embedding.
``LazyConv1d`` still infers the in-channels (D), so the same code works for
every PTM (it must see one forward pass before the optimizer is built --
see common.engine.materialize_lazy).

FCN (step 4)::  embed -> 256 -> 128 -> 64 -> num_classes   (logits; softmax at inference)
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
    """Three dense layers (256, 128, 64) + a classification layer."""

    def __init__(self, in_dim: int, num_classes: int,
                 dims: Sequence[int] = (256, 128, 64), dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)       # logits (softmax via CE / at inference)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


class SinglePTMModel(nn.Module):
    """Trial 1: one PTM representation -> CNN -> 120-d embedding -> FCN."""

    def __init__(self, num_classes: int, proj_dim: int = 120,
                 fcn_dims: Sequence[int] = (256, 128, 64), dropout: float = 0.0):
        super().__init__()
        self.frontend = CNNFrontEnd(proj_dim)
        self.fcn = FCN(proj_dim, num_classes, fcn_dims, dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.frontend(x, lengths)     # (B, 120)
        logits = self.fcn(emb)              # (B, num_classes)
        return logits, emb


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
