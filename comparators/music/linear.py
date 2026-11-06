"""v1.5 non-neural reference baselines under the GFDP-MER protocol (CPU).

kind
  majority  classification: constant prediction of the fit-row majority class (logit +-1);
            regression: constant fit-row mean.
  linear    per-row acoustic summary features computed from the same cached input x:
            for each of the 212 channels its time mean and time std (424 values), plus the
            mean of those over the song's rows (song context) -> 848 features, standardised
            with fit-row statistics. Classification: L2 logistic regression per dimension
            (C chosen on the dev songs by accuracy); regression: ridge per dimension (alpha
            chosen on dev by CCC).
Same folds (song_folds, fold seed), same fit/dev split per seed and fold, same outputs as
gfdp.cv (holdout_predictions.npz, seed*_pooled_holdout.npz, folds.json, summary.json), so
gfdp.compare works unchanged. These models are deterministic: the three seeds differ only
through their fit/dev split.

  python -m comparators.music.linear --kind linear --task classification --dataset data_PMEmo --out runs/V15_pmemo_cls_linear
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gfdp import __version__  # noqa: E402
from gfdp.data import load_trace_cache, song_folds, split_dev, synthetic_corpus  # noqa: E402
from gfdp.metrics import (CLASSIFICATION_KEYS, REGRESSION_KEYS, ccc, song_bootstrap,  # noqa: E402
                          song_bootstrap_regression, summarise_seeds, task_metrics)

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def song_features(corpus, songs):
    """-> dict song_index -> (features [n, 848], y [n, 2], rows [n])."""
    out = {}
    for s in songs:
        d = corpus.read_song(s)
        x = d["x"].astype(np.float64)                       # [n, 212, frames]
        row = np.concatenate([x.mean(-1), x.std(-1)], -1)    # [n, 424]
        ctx = np.repeat(row.mean(0, keepdims=True), len(row), 0)
        out[s["song_index"]] = (np.concatenate([row, ctx], -1), d["y"].astype(np.float64), d["rows"])
    return out


def stack(feats, songs):
    X = np.concatenate([feats[s["song_index"]][0] for s in songs])
    Y = np.concatenate([feats[s["song_index"]][1] for s in songs])
    S = np.concatenate([np.full(len(feats[s["song_index"]][0]), s["song_index"]) for s in songs])
    R = np.concatenate([feats[s["song_index"]][2] for s in songs])
    return X, Y, S, R


def logistic(X, y, C, iters=200):
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y.astype(np.float64))
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + (w @ w) / (2 * C * len(y))
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def ridge(X, y, alpha):
    mu = y.mean()
    A = X.T @ X + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ (y - mu))
    return w, float(mu)


def fit_predict(kind, task, Xf, Yf, Xd, Yd, Xh):
    out = np.zeros((len(Xh), 2)); chosen = []
    for d in range(2):
        if kind == "majority":
            if task == "classification":
                out[:, d] = 1.0 if (Yf[:, d] > 0).mean() > 0.5 else -1.0
            else:
                out[:, d] = Yf[:, d].mean()
            chosen.append(None); continue
        best = None
        for p in (C_GRID if task == "classification" else ALPHA_GRID):
            if task == "classification":
                w, b = logistic(Xf, (Yf[:, d] > 0).astype(float), p)
                score = float((((Xd @ w + b) > 0) == (Yd[:, d] > 0)).mean())
            else:
                w, b = ridge(Xf, Yf[:, d], p)
                score = ccc(Yd[:, d], Xd @ w + b)
            if best is None or score > best[0]:
                best = (score, p, w, b)
        out[:, d] = Xh @ best[2] + best[3]
        chosen.append(best[1])
    return out, chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("majority", "linear"), required=True)
    ap.add_argument("--task", choices=("classification", "regression"), required=True)
    ap.add_argument("--cache-base", default="PATH_TO_CACHE")
    ap.add_argument("--dataset", default="data_1000songs")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=20261010)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2025, 2031, 2032])
    ap.add_argument("--dev-fraction", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    corpus = synthetic_corpus(n_songs=40) if args.synthetic else \
        load_trace_cache(args.cache_base, args.dataset, roles=("train",))
    folds = song_folds(corpus, args.folds, args.fold_seed)
    (out / "folds.json").write_text(json.dumps(dict(fold_seed=args.fold_seed, corpus=corpus.identity, folds=[
        {"fold": f, "holdout_track_ids": [s["track_id"] for s in fold["holdout"]]} for f, fold in enumerate(folds)]), indent=1))
    t0 = time.time()
    feats = song_features(corpus, corpus.songs)
    summary = dict(config=dict(kind=args.kind, task=args.task), args=vars(args), task=args.task,
                   model_kind=f"reference_{args.kind}", gfdp_version=__version__, corpus=corpus.identity,
                   seeds={}, outer_validation_rows_read=0, test_rows_read=0, started=t0)
    for seed in args.seeds:
        pooled = {"logit": [], "y": [], "song": [], "row": []}
        per_fold = []
        for f, fold in enumerate(folds):
            fit, dev = split_dev(fold["train"], args.dev_fraction, seed + 17 * f)
            Xf, Yf, _, _ = stack(feats, fit); Xd, Yd, _, _ = stack(feats, dev)
            Xh, Yh, Sh, Rh = stack(feats, fold["holdout"])
            mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
            pred, chosen = fit_predict(args.kind, args.task, (Xf - mu) / sd, Yf, (Xd - mu) / sd, Yd, (Xh - mu) / sd)
            run_dir = out / f"seed{seed}_fold{f}"; run_dir.mkdir(exist_ok=True)
            np.savez(run_dir / "holdout_predictions.npz", logit=pred.astype(np.float32), y=Yh.astype(np.float32),
                     song=Sh, row=Rh)
            per_fold.append(dict(fold=f, chosen=chosen, metrics=task_metrics(Yh, pred, args.task)))
            for k, v in (("logit", pred.astype(np.float32)), ("y", Yh.astype(np.float32)), ("song", Sh), ("row", Rh)):
                pooled[k].append(v)
        pooled = {k: np.concatenate(v) for k, v in pooled.items()}
        np.savez(out / f"seed{seed}_pooled_holdout.npz", **pooled)
        boot_fn = song_bootstrap if args.task == "classification" else song_bootstrap_regression
        summary["seeds"][str(seed)] = dict(pooled=task_metrics(pooled["y"], pooled["logit"], args.task), folds=per_fold,
                                           bootstrap={n: boot_fn(pooled["y"], pooled["logit"], pooled["song"], d, seed=seed)
                                                      for d, n in enumerate(("valence", "arousal"))})
        print(args.kind, args.task, seed, {k: {m: round(v, 4) for m, v in d.items()}
                                           for k, d in summary["seeds"][str(seed)]["pooled"].items()}, flush=True)
    keys = CLASSIFICATION_KEYS if args.task == "classification" else REGRESSION_KEYS
    summary["across_seeds"] = {n: {k: summarise_seeds([summary["seeds"][str(s)]["pooled"][n][k] for s in args.seeds])
                                   for k in keys} for n in ("valence", "arousal")}
    summary["finished"] = time.time()
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
