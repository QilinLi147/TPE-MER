"""K-fold song-grouped cross-validation on original TRAIN songs only.

Usage (server):
  python -m gfdp.cv --config configs/main.json --cache-base PATH_TO_CACHE \
      --dataset data_1000songs --folds 5 --seeds 2025 2031 2032 --device cuda --out runs/main
Smoke test (CPU):
  python -m gfdp.cv --config configs/smoke.json --synthetic --folds 3 --seeds 1 --out /tmp/gfdp_smoke

Writes per seed/fold: predictions npz (holdout rows only), checkpoint, history;
and summary.json with pooled holdout metrics, song-bootstrap CIs, per-fold metrics.
Outer VALIDATION and TEST are never opened by this entry point.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_trace_cache, song_folds, split_dev, synthetic_corpus
from .metrics import (CLASSIFICATION_KEYS, REGRESSION_KEYS, song_bootstrap, song_bootstrap_regression,
                      summarise_seeds, task_metrics)
from .model import ModelConfig
from .train import TrainConfig, Trainer, predict


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-base", default="PATH_TO_CACHE")
    ap.add_argument("--dataset", default="data_1000songs")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=20261010)
    ap.add_argument("--only-folds", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2025, 2031, 2032])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic-songs", type=int, default=40)
    ap.add_argument("--stop-file", default=None,
                    help="if this path exists the run stops at the next batch (project STOP policy)")
    ap.add_argument("--memory-cap-bytes", type=int, default=20_000_000_000)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    if args.stop_file:
        import os
        os.environ["GFDP_STOP_FILE"] = args.stop_file
    if args.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_cap_bytes / total), 0)
    cfg = json.loads(Path(args.config).read_text())
    tc = TrainConfig.from_dict(cfg.get("train", {}))
    baseline = cfg.get("model_kind", "gfdp") == "baseline"      # v1.5 black-box baselines
    if baseline:
        from .baselines import BaselineConfig, BaselineTrainer, predict_baseline
        mc = BaselineConfig.from_dict(cfg.get("baseline", {}))
    else:
        mc = ModelConfig.from_dict(cfg.get("model", {}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.get("dataset") and not args.synthetic and cfg["dataset"] != args.dataset:
        raise SystemExit(f"config is for dataset {cfg['dataset']!r} but --dataset is {args.dataset!r}")
    corpus = synthetic_corpus(n_songs=args.synthetic_songs, frames=mc.frames) if args.synthetic else \
        load_trace_cache(args.cache_base, args.dataset, roles=("train",))
    if corpus.frames != mc.frames:
        raise SystemExit(f"cache has {corpus.frames} frames per row but the config expects {mc.frames}")
    folds = song_folds(corpus, args.folds, args.fold_seed)
    fold_manifest = [{"fold": f, "holdout_track_ids": [s["track_id"] for s in fold["holdout"]]}
                     for f, fold in enumerate(folds)]
    (out / "folds.json").write_text(json.dumps(dict(fold_seed=args.fold_seed, folds=fold_manifest,
                                                    corpus=corpus.identity), indent=1))
    log_path = out / "log.jsonl"

    def log(record):
        with log_path.open("a") as f:
            f.write(json.dumps(record, default=float) + "\n")

    from . import __version__
    summary = dict(config=cfg, args=vars(args), host=platform.node(), started=time.time(), gfdp_version=__version__,
                   task=mc.task, frames=mc.frames, model_kind=cfg.get("model_kind", "gfdp"),
                   corpus=corpus.identity, seeds={})
    for seed in args.seeds:
        pooled = {"logit": [], "y": [], "song": [], "row": []}
        per_fold = []
        for f, fold in enumerate(folds):
            if args.only_folds is not None and f not in args.only_folds:
                continue
            fit, dev = split_dev(fold["train"], tc.dev_fraction, seed + 17 * f)
            logger = lambda r, f=f, s=seed: log(dict(r, fold=f, seed=s))  # noqa: E731
            trainer = BaselineTrainer(mc, tc, args.device, log=logger) if baseline else \
                Trainer(mc, tc, args.device, log=logger)
            result = trainer.fit(corpus, fit, dev, seed=seed * 100 + f)
            pred = (predict_baseline if baseline else predict)(result["model"], corpus, fold["holdout"], args.device)
            run_dir = out / f"seed{seed}_fold{f}"
            run_dir.mkdir(exist_ok=True)
            np.savez(run_dir / "holdout_predictions.npz", **pred)
            torch.save(result["model"].checkpoint(dict(seed=seed, fold=f)), run_dir / "model.pt")
            (run_dir / "history.json").write_text(json.dumps(
                {k: v for k, v in result.items() if k != "model"}, default=float, indent=1))
            m = task_metrics(pred["y"], pred["logit"], mc.task)
            per_fold.append(dict(fold=f, metrics=m, best_joint_epoch=result["best_joint_epoch"],
                                 seconds=result["seconds"], parameters=result["parameters"]))
            for k in pooled:
                pooled[k].append(pred[k])
            log(dict(event="fold_done", seed=seed, fold=f, metrics=m))
        pooled = {k: np.concatenate(v) for k, v in pooled.items()}
        np.savez(out / f"seed{seed}_pooled_holdout.npz", **pooled)
        metrics = task_metrics(pooled["y"], pooled["logit"], mc.task)
        boot_fn = song_bootstrap if mc.task == "classification" else song_bootstrap_regression
        boot = {name: boot_fn(pooled["y"], pooled["logit"], pooled["song"], d, seed=seed)
                for d, name in enumerate(("valence", "arousal"))}
        summary["seeds"][str(seed)] = dict(pooled=metrics, bootstrap=boot, folds=per_fold)
    summary["across_seeds"] = {
        name: {k: summarise_seeds([summary["seeds"][str(s)]["pooled"][name][k] for s in args.seeds])
               for k in (CLASSIFICATION_KEYS if mc.task == "classification" else REGRESSION_KEYS)}
        for name in ("valence", "arousal")}
    summary["finished"] = time.time()
    summary["outer_validation_rows_read"] = 0
    summary["test_rows_read"] = 0
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == "__main__":
    main()
