"""Trial 2 -- Fused two-PTM Audio Deepfake Source Attribution.

Two pre-trained models' last-hidden-state representations are processed by two
independent 1D-CNN front-ends, each projected to a 120-d embedding. The two
embeddings are concatenated (240-d) and classified by the FCN.

Joint loss = cross-entropy + lambda * Renyi-divergence alignment between the
two 120-d embeddings.

Examples
--------
    python main_fusion.py --ptm-a wav2vec2 --ptm-b wavlm   --dataset asvspoof2019
    python main_fusion.py --ptm-a xlsr     --ptm-b whisper --dataset cfad --rd-lambda 0.2

Results are written under ``results/fusion/{dataset}/{ptmA}+{ptmB}/``.
"""
from __future__ import annotations

import argparse
import os

from config import PTM_REGISTRY, add_common_args, config_from_args
from common.utils import set_seed, resolve_device, get_logger, save_json
from common.ptm_extractors import build_extractor
from common.datasets import (SourceAttrDataset, collate_fusion, build_label_map,
                             load_asvspoof_all, load_cfad_splits)
from common.feature_cache import prefetch
from common.splits import asvspoof_folds, cfad_train_dev_test
from common.models import FusionModel
from common.engine import materialize_lazy, fit, predict
from common.metrics import evaluate_report
from common.results import (run_dir, save_config, save_history_csv, save_model,
                            summarize_folds)

TRIAL = "fusion"
MODE = "fusion"


def class_names_from_map(label_to_idx):
    return [n for n, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]


def make_dataset(records, label_to_idx, ptms, extractors, dataset_name, cfg):
    return SourceAttrDataset(
        records, label_to_idx, ptms, extractors, dataset_name,
        cache_dir=cfg.cache_dir, target_sr=cfg.target_sr,
        min_duration=cfg.min_duration, max_duration=cfg.max_duration,
        use_cache=cfg.use_cache,
    )


def make_loader(ds, cfg, shuffle):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      collate_fn=collate_fusion, num_workers=cfg.num_workers)


def loss_cfg_from(cfg):
    return {"rd_alpha": cfg.rd_alpha, "rd_lambda": cfg.rd_lambda,
            "rd_symmetric": cfg.rd_symmetric}


def train_eval_split(train_recs, val_recs, label_to_idx, class_names, ptms,
                     extractors, dataset_name, cfg, device, out_dir, logger, tag):
    os.makedirs(out_dir, exist_ok=True)
    loss_cfg = loss_cfg_from(cfg)

    train_ds = make_dataset(train_recs, label_to_idx, ptms, extractors, dataset_name, cfg)
    val_ds = make_dataset(val_recs, label_to_idx, ptms, extractors, dataset_name, cfg)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    model = FusionModel(num_classes=len(class_names), proj_dim=cfg.proj_dim,
                        fcn_dims=cfg.fcn_dims, dropout=cfg.dropout)
    materialize_lazy(model, next(iter(train_loader)), device, MODE)

    history, best = fit(model, train_loader, val_loader, device, MODE, cfg, loss_cfg, logger)
    save_history_csv(history, os.path.join(out_dir, "training_log.csv"))
    save_model(model, os.path.join(out_dir, "best_model.pt"))

    y_true, y_pred, probs = predict(model, val_loader, device, MODE, loss_cfg)
    metrics = evaluate_report(y_true, y_pred, probs, class_names, out_dir, tag=tag)
    metrics.update(best)
    save_json(metrics, os.path.join(out_dir, "metrics.json"))
    logger.info(f"[{tag}] accuracy={metrics['accuracy']:.4f} "
                f"eer(macro-ovr)={metrics['eer_macro_ovr']:.4f}")
    return metrics


def run_asvspoof(ptms, extractors, cfg, device, base_dir, logger):
    records = load_asvspoof_all(cfg.asvspoof_root, cfg.include_bonafide)
    label_to_idx = build_label_map(records)
    class_names = class_names_from_map(label_to_idx)
    logger.info(f"ASVspoof2019: {len(records)} utts, {len(class_names)} classes")

    full_ds = make_dataset(records, label_to_idx, ptms, extractors, "asvspoof2019", cfg)
    prefetch(full_ds, desc=f"[{'+'.join(ptms)}] caching asvspoof features")

    folds = asvspoof_folds(records, label_to_idx, cfg.n_folds, cfg.seed)
    fold_metrics = []
    for i, (tr, va) in enumerate(folds, 1):
        logger.info(f"===== fold {i}/{cfg.n_folds} (train {len(tr)}, val {len(va)}) =====")
        out_dir = os.path.join(base_dir, f"fold_{i}")
        m = train_eval_split(tr, va, label_to_idx, class_names, ptms, extractors,
                             "asvspoof2019", cfg, device, out_dir, logger, tag=f"fold_{i}")
        fold_metrics.append(m)

    summary = summarize_folds(fold_metrics)
    save_json(summary, os.path.join(base_dir, "summary.json"))
    logger.info(f"CV summary: {summary.get('accuracy', {})}")


def run_cfad(ptms, extractors, cfg, device, base_dir, logger):
    splits = load_cfad_splits(cfg.cfad_root, cfg.include_bonafide)
    all_recs = splits["train"] + splits.get("dev", []) + splits["test"]
    label_to_idx = build_label_map(all_recs)
    class_names = class_names_from_map(label_to_idx)
    logger.info(f"CFAD: {len(all_recs)} utts, {len(class_names)} classes")

    full_ds = make_dataset(all_recs, label_to_idx, ptms, extractors, "cfad", cfg)
    prefetch(full_ds, desc=f"[{'+'.join(ptms)}] caching cfad features")

    train_recs, dev_recs, test_recs = cfad_train_dev_test(splits, cfg.seed)
    logger.info(f"official split -> train {len(train_recs)}, dev {len(dev_recs)}, test {len(test_recs)}")

    train_eval_split(train_recs, dev_recs, label_to_idx, class_names, ptms,
                     extractors, "cfad", cfg, device, base_dir, logger, tag="dev")
    _evaluate_cfad_test(test_recs, label_to_idx, class_names, ptms, extractors,
                        cfg, device, base_dir, logger)


def _evaluate_cfad_test(test_recs, label_to_idx, class_names, ptms, extractors,
                        cfg, device, base_dir, logger):
    import torch
    loss_cfg = loss_cfg_from(cfg)
    test_ds = make_dataset(test_recs, label_to_idx, ptms, extractors, "cfad", cfg)
    test_loader = make_loader(test_ds, cfg, shuffle=False)

    model = FusionModel(num_classes=len(class_names), proj_dim=cfg.proj_dim,
                        fcn_dims=cfg.fcn_dims, dropout=cfg.dropout)
    materialize_lazy(model, next(iter(test_loader)), device, MODE)
    model.load_state_dict(torch.load(os.path.join(base_dir, "best_model.pt"), map_location=device))

    y_true, y_pred, probs = predict(model, test_loader, device, MODE, loss_cfg)
    metrics = evaluate_report(y_true, y_pred, probs, class_names, base_dir, tag="test")
    save_json(metrics, os.path.join(base_dir, "metrics_test.json"))
    logger.info(f"[test] accuracy={metrics['accuracy']:.4f} "
                f"eer(macro-ovr)={metrics['eer_macro_ovr']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fused two-PTM ADSA (Trial 2)")
    parser.add_argument("--ptm-a", dest="ptm_a", required=True, choices=list(PTM_REGISTRY))
    parser.add_argument("--ptm-b", dest="ptm_b", required=True, choices=list(PTM_REGISTRY))
    parser.add_argument("--rd-alpha", dest="rd_alpha", type=float, default=2.0,
                        help="Renyi divergence order (alpha)")
    parser.add_argument("--rd-lambda", dest="rd_lambda", type=float, default=0.1,
                        help="weight of the RD alignment term")
    add_common_args(parser)
    args = parser.parse_args()

    if args.ptm_a == args.ptm_b:
        parser.error("--ptm-a and --ptm-b must be different models for fusion.")

    cfg = config_from_args(args)
    cfg.rd_alpha, cfg.rd_lambda = args.rd_alpha, args.rd_lambda
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    tag = f"{args.ptm_a}+{args.ptm_b}"
    base_dir = run_dir(cfg.results_dir, TRIAL, cfg.dataset, tag)
    logger = get_logger(f"fusion.{tag}.{cfg.dataset}", os.path.join(base_dir, "run.log"))
    logger.info(f"device={device} | fusion={tag} | dataset={cfg.dataset} "
                f"| RD(alpha={cfg.rd_alpha}, lambda={cfg.rd_lambda})")
    save_config(cfg, os.path.join(base_dir, "config.json"))

    ptms = [args.ptm_a, args.ptm_b]
    extractors = {p: build_extractor(p, device=device) for p in ptms}

    if cfg.dataset == "asvspoof2019":
        run_asvspoof(ptms, extractors, cfg, device, base_dir, logger)
    else:
        run_cfad(ptms, extractors, cfg, device, base_dir, logger)
    logger.info("done.")


if __name__ == "__main__":
    main()
