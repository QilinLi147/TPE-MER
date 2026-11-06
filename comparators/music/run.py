"""Run a music comparator on song-disjoint folds of a cached input corpus."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gfdp import baselines
from gfdp.data import load_trace_cache, song_folds, split_dev
from gfdp.metrics import task_metrics
from gfdp.train import TrainConfig
from .adapter_model import AdapterModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-base", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025, 2031, 2032])
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    mc = baselines.BaselineConfig.from_dict(config["baseline"])
    tc = TrainConfig.from_dict(config["train"])
    if mc.kind in ("damff", "dsaml", "damer"):
        baselines.BaselineModel = AdapterModel
    elif mc.kind != "shortchunk":
        raise ValueError("unsupported manuscript comparator")
    corpus = load_trace_cache(args.cache_base, args.dataset, roles=("train",))
    if corpus.frames != mc.frames:
        raise ValueError("cache frame count differs from the configuration")
    folds = song_folds(corpus, 5, 20261010)
    for seed in args.seeds:
        for fold_number in args.folds:
            if fold_number not in range(5):
                parser.error("fold numbers must be between 0 and 4")
            fold = folds[fold_number]
            fit, dev = split_dev(fold["train"], tc.dev_fraction, seed + 17 * fold_number)
            output = Path(args.out) / f"seed{seed}_fold{fold_number}"
            output.mkdir(parents=True, exist_ok=False)
            trainer = baselines.BaselineTrainer(mc, tc, args.device)
            result = trainer.fit(corpus, fit, dev, seed=seed * 100 + fold_number)
            model = result["model"]
            torch.save(model.checkpoint({"seed": seed, "fold": fold_number}), output / "model.pt")
            prediction = baselines.predict_baseline(model, corpus, fold["holdout"], args.device)
            np.savez(output / "holdout_predictions.npz", **prediction)
            (output / "metrics.json").write_text(
                json.dumps(task_metrics(prediction["y"], prediction["logit"], mc.task), indent=2)
            )


if __name__ == "__main__":
    main()
