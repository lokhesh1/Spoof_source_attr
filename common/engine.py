"""Training / evaluation engine shared by both trials.

``mode`` is either ``"single"`` (model returns logits, emb) or ``"fusion"``
(model returns logits, emb_a, emb_b and the joint Renyi loss is used).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from common.losses import single_loss, fusion_loss


# --------------------------------------------------------------------------- #
# Lazy-layer materialisation (must run before building the optimizer)
# --------------------------------------------------------------------------- #
def materialize_lazy(model, sample_batch, device: str, mode: str) -> None:
    """Run one forward pass so nn.Lazy* layers create their parameters."""
    import torch
    model.to(device)
    model.eval()
    with torch.no_grad():
        feats, _ = sample_batch
        if mode == "single":
            x, lengths = _move_x(feats, device, mode)
            model(x, lengths)
        else:
            a, la, b, lb = _move_x(feats, device, mode)
            model(a, la, b, lb)


def _move_x(feats, device, mode):
    """Move a batch's features (and their lengths) to ``device``."""
    if mode == "single":
        x, lengths = feats
        return x.to(device), lengths.to(device)
    (a, la), (b, lb) = feats
    return a.to(device), la.to(device), b.to(device), lb.to(device)


# --------------------------------------------------------------------------- #
# One epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, device, mode, loss_cfg) -> Dict[str, float]:
    import torch
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "rd": 0.0}
    n = 0
    for feats, y in loader:
        y = y.to(device)
        optimizer.zero_grad()
        if mode == "single":
            x, lengths = _move_x(feats, device, mode)
            logits, _ = model(x, lengths)
            loss = single_loss(logits, y)
            ce, rd = loss, torch.tensor(0.0)
        else:
            a, la, b, lb = _move_x(feats, device, mode)
            logits, emb_a, emb_b = model(a, la, b, lb)
            loss, ce, rd = fusion_loss(
                logits, y, emb_a, emb_b,
                rd_alpha=loss_cfg["rd_alpha"], rd_lambda=loss_cfg["rd_lambda"],
                rd_symmetric=loss_cfg["rd_symmetric"],
            )
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        n += bs
        totals["loss"] += loss.item() * bs
        totals["ce"] += ce.item() * bs
        totals["rd"] += rd.item() * bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def evaluate(model, loader, device, mode, loss_cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (y_true, y_pred, probs, avg_loss)."""
    import torch
    model.eval()
    ys, preds, probs = [], [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for feats, y in loader:
            y_dev = y.to(device)
            if mode == "single":
                x, lengths = _move_x(feats, device, mode)
                logits, _ = model(x, lengths)
                loss = single_loss(logits, y_dev)
            else:
                a, la, b, lb = _move_x(feats, device, mode)
                logits, emb_a, emb_b = model(a, la, b, lb)
                loss, _, _ = fusion_loss(
                    logits, y_dev, emb_a, emb_b,
                    rd_alpha=loss_cfg["rd_alpha"], rd_lambda=loss_cfg["rd_lambda"],
                    rd_symmetric=loss_cfg["rd_symmetric"],
                )
            p = torch.softmax(logits, dim=-1)
            probs.append(p.cpu().numpy())
            preds.append(p.argmax(-1).cpu().numpy())
            ys.append(y.numpy())
            total_loss += float(loss) * y.size(0)
            n += y.size(0)
    return (np.concatenate(ys), np.concatenate(preds),
            np.concatenate(probs), total_loss / max(n, 1))


# --------------------------------------------------------------------------- #
# Full training loop with early stopping
# --------------------------------------------------------------------------- #
def fit(model, train_loader, val_loader, device, mode, cfg, loss_cfg,
        logger=None) -> Tuple[List[dict], dict]:
    """Train with early stopping on validation accuracy.

    Returns (history, best_info). ``model`` is left holding the best weights.
    """
    import torch
    from common.metrics import accuracy

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[dict] = []
    best_acc, best_state, best_epoch, since_improve = -1.0, None, -1, 0

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, mode, loss_cfg)
        y_true, y_pred, _, val_loss = evaluate(model, val_loader, device, mode, loss_cfg)
        val_acc = accuracy(y_true, y_pred)

        row = {"epoch": epoch, "train_loss": tr["loss"], "train_ce": tr["ce"],
               "train_rd": tr["rd"], "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        if logger:
            logger.info(
                f"epoch {epoch:03d} | train_loss {tr['loss']:.4f} "
                f"(ce {tr['ce']:.4f}, rd {tr['rd']:.4f}) | "
                f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, best_epoch, since_improve = val_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_improve += 1
            if since_improve >= cfg.patience:
                if logger:
                    logger.info(f"early stopping at epoch {epoch} (best {best_acc:.4f} @ {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, {"best_val_acc": best_acc, "best_epoch": best_epoch}


def predict(model, loader, device, mode, loss_cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true, y_pred, probs, _ = evaluate(model, loader, device, mode, loss_cfg)
    return y_true, y_pred, probs
