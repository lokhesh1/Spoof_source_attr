"""Joint loss = cross-entropy (+ Renyi divergence alignment for the fusion trial).

Renyi divergence of order alpha between two distributions p, q::

    D_alpha(p || q) = 1/(alpha-1) * log( sum_i p_i^alpha * q_i^(1-alpha) )

The two per-PTM embeddings (each 120-d) are turned into probability
distributions with a softmax, and we align them by minimising their Renyi
divergence. We compute it in log-space for numerical stability and (by default)
symmetrise it: 0.5 * (D(p||q) + D(q||p)).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def renyi_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor,
                     alpha: float = 2.0, symmetric: bool = True) -> torch.Tensor:
    """Mean Renyi divergence over the batch (alpha != 1; alpha->1 falls back to KL)."""
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)

    if abs(alpha - 1.0) < 1e-6:                       # KL limit
        d = (log_p.exp() * (log_p - log_q)).sum(-1)
        if symmetric:
            d = 0.5 * (d + (log_q.exp() * (log_q - log_p)).sum(-1))
        return d.mean()

    def _d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # log sum_i exp(alpha*log a_i + (1-alpha)*log b_i)  / (alpha-1)
        t = alpha * a + (1.0 - alpha) * b
        return torch.logsumexp(t, dim=-1) / (alpha - 1.0)

    d = _d(log_p, log_q)
    if symmetric:
        d = 0.5 * (d + _d(log_q, log_p))
    return d.mean()


def single_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Plain cross-entropy for the single-PTM trial."""
    return F.cross_entropy(logits, targets)


def fusion_loss(logits: torch.Tensor, targets: torch.Tensor,
                emb_a: torch.Tensor, emb_b: torch.Tensor,
                rd_alpha: float = 2.0, rd_lambda: float = 0.1,
                rd_symmetric: bool = True):
    """CE + lambda * Renyi-divergence alignment.

    Returns (total, ce, rd) so each term can be logged separately.
    """
    ce = F.cross_entropy(logits, targets)
    rd = renyi_divergence(emb_a, emb_b, alpha=rd_alpha, symmetric=rd_symmetric)
    total = ce + rd_lambda * rd
    return total, ce, rd
