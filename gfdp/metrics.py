"""Row metrics (same five as the D2 audits) and song-clustered bootstrap.

Rows within a song are strongly correlated (about 61 rows per 1000Songs clip),
so uncertainty must be resampled at the SONG level. A 104-song holdout has a
song-level standard error of roughly 4 percentage points; this is why the old
+0.5 pp promotion gate on 2 folds x 3 seeds could not separate signal from noise.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties (vectorised, v1.1)."""
    y = np.asarray(y).astype(bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(np.asarray(score, dtype=np.float64),
                                   return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    average_rank = upper - (counts - 1) / 2.0           # 1-based average rank of each tie group
    ranks = average_rank[inverse]
    return float((ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def binary_metrics(y_hard: np.ndarray, logit: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_hard).astype(int)
    p = (np.asarray(logit) > 0).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    def f1(a, b, c):
        return 2 * a / (2 * a + b + c) if (2 * a + b + c) else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    pos_f1, neg_f1 = f1(tp, fp, fn), f1(tn, fn, fp)
    return dict(accuracy=(tp + tn) / len(y), balanced_accuracy=(tpr + tnr) / 2,
                macro_f1=(pos_f1 + neg_f1) / 2, positive_f1=pos_f1,
                roc_auc=roc_auc(y, np.asarray(logit, dtype=np.float64)))


def all_metrics(y_cont: np.ndarray, logit: np.ndarray) -> Dict[str, Dict[str, float]]:
    return {name: binary_metrics(y_cont[:, d] > 0, logit[:, d])
            for d, name in enumerate(("valence", "arousal"))}


def song_bootstrap(y_cont: np.ndarray, logit: np.ndarray, song: np.ndarray, d: int,
                   reps: int = 2000, seed: int = 0) -> Dict[str, float]:
    """95% CI of row accuracy with songs resampled with replacement."""
    rng = np.random.default_rng(seed)
    ids = np.unique(song)
    correct = ((logit[:, d] > 0) == (y_cont[:, d] > 0)).astype(np.float64)
    per_song_sum = np.array([correct[song == s].sum() for s in ids])
    per_song_n = np.array([(song == s).sum() for s in ids], dtype=np.float64)
    stats = []
    for _ in range(reps):
        pick = rng.integers(0, len(ids), len(ids))
        stats.append(per_song_sum[pick].sum() / per_song_n[pick].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(accuracy=float(correct.mean()), ci_low=float(lo), ci_high=float(hi),
                songs=int(len(ids)))


def paired_song_bootstrap(y_cont: np.ndarray, logit_a: np.ndarray, logit_b: np.ndarray,
                          song: np.ndarray, d: int, reps: int = 5000, seed: int = 0) -> Dict[str, float]:
    """Accuracy(b) - accuracy(a) with a song-level paired bootstrap CI.
    Both prediction arrays must be on the same rows (same CV holdouts)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(song)
    y = y_cont[:, d] > 0
    ca = ((logit_a[:, d] > 0) == y).astype(np.float64)
    cb = ((logit_b[:, d] > 0) == y).astype(np.float64)
    da = np.array([ca[song == s].sum() for s in ids]); db = np.array([cb[song == s].sum() for s in ids])
    n = np.array([(song == s).sum() for s in ids], dtype=np.float64)
    diffs = []
    for _ in range(reps):
        pick = rng.integers(0, len(ids), len(ids))
        diffs.append((db[pick].sum() - da[pick].sum()) / n[pick].sum())
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(delta=float(cb.mean() - ca.mean()), ci_low=float(lo), ci_high=float(hi),
                p_not_better=float((diffs <= 0).mean()), songs=int(len(ids)))


def summarise_seeds(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)))


# ------------------------------------------------------------------ v1.4: regression
REGRESSION_KEYS = ("ccc", "pearson", "r2", "rmse", "mae", "sign_accuracy")
CLASSIFICATION_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")


def ccc(y: np.ndarray, p: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (population moments)."""
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(), p.var()
    cov = ((y - my) * (p - mp)).mean()
    denom = vy + vp + (my - mp) ** 2
    return float(2 * cov / denom) if denom > 0 else float("nan")


def regression_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    err = p - y
    sy, sp = y.std(), p.std()
    pear = float(((y - y.mean()) * (p - p.mean())).mean() / (sy * sp)) if sy > 0 and sp > 0 else float("nan")
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return dict(ccc=ccc(y, p), pearson=pear, r2=1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                rmse=float(np.sqrt((err ** 2).mean())), mae=float(np.abs(err).mean()),
                sign_accuracy=float(((p > 0) == (y > 0)).mean()))


def task_metrics(y_cont: np.ndarray, out: np.ndarray, task: str = "classification") -> Dict[str, Dict[str, float]]:
    if task == "classification":
        return all_metrics(y_cont, out)
    return {name: regression_metrics(y_cont[:, d], out[:, d]) for d, name in enumerate(("valence", "arousal"))}


def song_bootstrap_regression(y_cont: np.ndarray, pred: np.ndarray, song: np.ndarray, d: int,
                              reps: int = 2000, seed: int = 0) -> Dict[str, float]:
    """95% CI of CCC and RMSE with songs resampled with replacement (rows of a song move together)."""
    rng = np.random.default_rng(seed)
    ids, inverse = np.unique(song, return_inverse=True)
    rows = [np.flatnonzero(inverse == i) for i in range(len(ids))]
    stats_c, stats_r = [], []
    for _ in range(reps):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(ids), len(ids))])
        stats_c.append(ccc(y_cont[idx, d], pred[idx, d]))
        stats_r.append(float(np.sqrt(((pred[idx, d] - y_cont[idx, d]) ** 2).mean())))
    base = regression_metrics(y_cont[:, d], pred[:, d])
    clo, chi = np.nanpercentile(stats_c, [2.5, 97.5])
    rlo, rhi = np.nanpercentile(stats_r, [2.5, 97.5])
    return dict(ccc=base["ccc"], ccc_ci_low=float(clo), ccc_ci_high=float(chi), rmse=base["rmse"],
                rmse_ci_low=float(rlo), rmse_ci_high=float(rhi), songs=int(len(ids)))
