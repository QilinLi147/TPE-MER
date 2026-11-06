"""Statistically certified early decision (v1.3-S, "two-tier certified listening").

v1.2.1 showed that the deterministic certificate (gfdp.certify) is sound but
worst-case: on frozen models it needs ~99% of the segments, and training it
tighter costs valence accuracy (REPORT ch. 16). v1.3-S adds a second tier that
keeps the model frozen (zero accuracy cost from training) and gives a
distribution-free guarantee instead of a worst-case one:

    With probability >= 1 - delta over the calibration songs, the stopping rule
    selected by Learn-then-Test (LTT; Angelopoulos et al. 2021) has
    disagreement risk  P(early decision != full-song decision) <= alpha
    on a new song exchangeable with the calibration songs.

Tier 1 (deterministic, gfdp.certify) is always kept inside every rule: whenever
the worst-case interval already excludes zero the rule stops with the certified
sign (zero risk). The calibrated rule only adds earlier stops on top.

Per song, dimension and acquisition policy a trajectory is recorded once:
    z_t      partial song-level logit from the heard segments (= forward(select=heard))
    lo_t,hi_t  deterministic bounds of the full-song logit (gfdp.certify)
    f        full-song logit (z_n == f)
Stopping rules (one parameter each, grids fixed before any v1.3 run, ordered
from conservative to aggressive so fixed-sequence testing is valid):
    S1  relaxed certificate : stop when |z_t| >= lambda * (hi_t - lo_t)/2       (primary)
    S2  scaled margin       : stop when |z_t| * sqrt(t/n) >= lambda
    B0  fixed prefix        : stop at t = ceil(p * n)                           (non-adaptive baseline)
    B1  stable sign         : stop when sign(z) unchanged for k steps            (v1.2 heuristic, calibrated)
    D   deterministic only  : tier 1 alone (no calibration, zero risk)
Loss = 1[decision != sign(f)]; LTT p-value = exact binomial tail
P(Binom(m, alpha) <= #losses), which is super-uniform under H: risk > alpha
for {0,1} losses. Fixed-sequence testing from the conservative end; the chosen
parameter is the last rejected one; if the first is not rejected the rule falls
back to D (risk 0), so the procedure is always valid.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .certify import SongCertifier, coarse_to_fine_order

EPS_CERT = 1e-6
STAT_POLICIES = ("chronological", "coarse_to_fine")
RULES = ("S1", "S2", "B0", "B1")
RULE_NAMES = {"S1": "relaxed_certificate", "S2": "scaled_margin", "B0": "fixed_prefix",
              "B1": "stable_sign", "D": "deterministic_only"}
# Pre-registered grids (conservative -> aggressive). Do not change after v1.3 runs start.
GRIDS = {
    "S1": np.geomspace(1.0, 1e-3, 61),
    "S2": np.geomspace(32.0, 1e-3, 91),
    "B0": np.round(np.arange(1.00, 0.0, -0.02), 2),
    "B1": np.arange(64, 0, -1).astype(float),
}


def sign(x):
    """Same convention as gfdp.certify.run_policy: > 0 -> +1, otherwise -1."""
    return np.where(np.asarray(x) > 0, 1, -1)


# --------------------------------------------------------------- trajectories
def song_trajectory(cert: SongCertifier, policy: str) -> Dict[str, np.ndarray]:
    n = cert.n
    order = list(range(n)) if policy == "chronological" else coarse_to_fine_order(n)
    seen = np.zeros(n, dtype=bool)
    z = np.empty((n, 2)); lo = np.empty((n, 2)); hi = np.empty((n, 2))
    for step, idx in enumerate(order):
        seen[idx] = True
        z[step] = cert.partial_logits(seen).mean(0)
        lo[step], hi[step] = cert.bounds(seen, "song")
    full = cert.full_logits().mean(0)
    if np.abs(z[-1] - full).max() > 1e-8:
        raise AssertionError("partial logit with every segment heard must equal the full logit")
    if np.any(lo[-1] > full + 1e-8) or np.any(hi[-1] < full - 1e-8):
        raise AssertionError("bounds do not contain the full logit")
    return dict(z=z, lo=lo, hi=hi, full=full, n=n)


# ---------------------------------------------------------------- stop rules
def _first_reaching(score: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """1-based first step whose running max of `score` reaches each grid value (n+1 if never)."""
    run = np.maximum.accumulate(score)
    return np.searchsorted(run, grid, side="left") + 1


def stop_rule(z: np.ndarray, lo: np.ndarray, hi: np.ndarray, f: float, rule: str,
              grid: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """One song, one dimension. Returns (stop_step [G], decision [G]) for every grid value.
    rule 'D' returns arrays of length 1."""
    n = len(z)
    t = np.arange(1, n + 1)
    certified = (lo > EPS_CERT) | (hi <= -EPS_CERT)
    cert_step = int(np.argmax(certified)) + 1 if certified.any() else n + 1
    cert_sign = (1 if lo[cert_step - 1] > EPS_CERT else -1) if cert_step <= n else 0
    if rule == "D":
        step = np.array([min(cert_step, n)])
        dec = np.array([cert_sign if cert_step <= n else int(sign(z[-1]))])
        return step, dec
    grid = GRIDS[rule] if grid is None else grid
    if rule == "S1":
        half = (hi - lo) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(half > 0, np.abs(z) / np.where(half > 0, half, 1.0), np.inf)
        rule_step = _first_reaching(score, grid)
    elif rule == "S2":
        rule_step = _first_reaching(np.abs(z) * np.sqrt(t / n), grid)
    elif rule == "B0":
        rule_step = np.clip(np.ceil(grid * n - 1e-9).astype(int), 1, n)
    elif rule == "B1":
        s = sign(z)
        run = np.ones(n)
        for i in range(1, n):
            run[i] = run[i - 1] + 1 if s[i] == s[i - 1] else 1
        rule_step = _first_reaching(run, grid)
    else:
        raise ValueError(rule)
    rule_step = np.minimum(rule_step, n)
    step = np.minimum(rule_step, cert_step)
    dec = np.where(cert_step <= rule_step, cert_sign, sign(z[step - 1]))
    return step.astype(int), dec.astype(int)


# ------------------------------------------------------------------------ LTT
def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Binom(n, p) <= k), exact (log-space terms)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k + 1)]
    m = max(terms)
    return float(min(1.0, math.exp(m) * sum(math.exp(x - m) for x in terms)))


def ltt_select(losses: np.ndarray, alpha: float, delta: float) -> Tuple[int, np.ndarray]:
    """losses [m, G] in {0,1}, columns conservative -> aggressive.
    Fixed-sequence testing of H_j: risk_j > alpha. Returns (index of the last
    rejected column, or -1 for fallback to the deterministic tier; p-values)."""
    m = losses.shape[0]
    k = losses.sum(0).astype(int)
    pvals = np.array([binom_cdf(int(kj), m, alpha) for kj in k])
    chosen = -1
    for j, p in enumerate(pvals):
        if p <= delta:
            chosen = j
        else:
            break
    return chosen, pvals


def min_feasible_alpha(m: int, delta: float) -> float:
    """Smallest alpha for which zero observed losses in m songs is rejected: (1-alpha)^m <= delta."""
    return 1.0 - delta ** (1.0 / m)


# ------------------------------------------------------------- evaluation
def precompute(trajs: List[Dict], dim: int, rules: Sequence[str] = RULES + ("D",)) -> Dict:
    """Stop steps and decisions of every song for every rule and grid value (alpha-free).
    Returns {rule: (steps [songs, G], decisions [songs, G])} plus 'full_sign' [songs] and 'n' [songs]."""
    out = {"full_sign": np.array([int(sign(tr["full"][dim])) for tr in trajs]),
           "n": np.array([tr["n"] for tr in trajs])}
    for rule in rules:
        res = [stop_rule(tr["z"][:, dim], tr["lo"][:, dim], tr["hi"][:, dim], tr["full"][dim], rule)
               for tr in trajs]
        out[rule] = (np.stack([r[0] for r in res]), np.stack([r[1] for r in res]))
    return out


def calibrate_apply(pre: Dict, rule: str, alpha: float, delta: float,
                    calib: Sequence[int], evals: Sequence[int]) -> Dict:
    """Select the rule parameter by LTT on songs `calib`; apply it to songs `evals`.
    Fallback (no parameter rejected) = deterministic tier D."""
    calib, evals = np.asarray(calib, dtype=int), np.asarray(evals, dtype=int)
    if rule == "D":
        chosen, pvals = 0, np.zeros(1)
    else:
        steps, decs = pre[rule]
        losses = (decs[calib] != pre["full_sign"][calib, None]).astype(int)
        chosen, pvals = ltt_select(losses, alpha, delta)
    if chosen < 0:
        steps, decs = pre["D"]; col = 0
    else:
        steps, decs = pre[rule]; col = chosen
    s, d = steps[evals, col], decs[evals, col]
    return dict(chosen_index=int(chosen), fallback=bool(chosen < 0),
                chosen_param=None if (chosen < 0 or rule == "D") else float(GRIDS[rule][chosen]),
                steps=s, decisions=d, loss=(d != pre["full_sign"][evals]).astype(int),
                fraction=s / pre["n"][evals], calib_songs=int(len(calib)))


def holdout_halves(n_songs: int, analysis_seed: int, seed: int, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic random halves of a fold's holdout songs; identical for every run, so
    comparisons between runs and rules are paired."""
    perm = np.random.default_rng([analysis_seed, seed, fold]).permutation(n_songs)
    h = n_songs // 2
    return np.sort(perm[:h]), np.sort(perm[h:])


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, reps: int = 10000, seed: int = 20260906,
                      weights_cache: Dict = None) -> Dict[str, float]:
    """Mean of `values` with a song-cluster bootstrap 95% CI (all records of a song move together)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values.astype(float), minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    key = (len(uniq), reps, seed)
    if weights_cache is not None and key in weights_cache:
        W = weights_cache[key]
    else:
        rng = np.random.default_rng(seed)
        W = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=reps).astype(np.float32)
        if weights_cache is not None:
            weights_cache[key] = W
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (W @ sums) / np.maximum(W @ cnts, 1e-12)
    return dict(mean=float(values.mean()), ci_low=float(np.percentile(boot, 2.5)),
                ci_high=float(np.percentile(boot, 97.5)), clusters=int(len(uniq)), records=int(len(values)))


# =========================================================================
# v1.3.1: nested (selection-aware) evaluation with cross-fitted calibration
# =========================================================================
# v1.3 chose its rule/policy/alpha before seeing data (S1 was the registered primary) and
# reported the other 47 configurations descriptively. Picking the best of those afterwards on
# the same songs would be optimistic. v1.3.1 evaluates the *procedure* "choose a configuration
# from data, then calibrate it" honestly, inside the TRAIN CV, with three disjoint song sets:
#   S (selection)  : pick the configuration (rule x policy) from the candidate list
#   C (calibration): LTT for the selected configuration only -> fresh, valid guarantee
#   E (evaluation) : report efficiency, disagreement and accuracy
# Sets are thirds of every fold's holdout songs, rotated so every song is in E exactly once per
# seed. Each song is scored by the model of the fold that held it out; the five fold models of a
# seed are pooled ("cross-fitted calibration"), giving ~173 songs per set instead of 52.
NESTED_RULES = ("S1", "S2", "B1", "B0")
NESTED_CANDIDATES = tuple((r, p) for p in STAT_POLICIES for r in NESTED_RULES)   # 8 configurations
B0_CANDIDATES = tuple(("B0", p) for p in STAT_POLICIES)                          # non-adaptive baseline


def holdout_thirds(n_songs: int, analysis_seed: int, seed: int, fold: int) -> np.ndarray:
    """Part label 0/1/2 for each holdout song of one fold (sizes differ by at most one)."""
    perm = np.random.default_rng([analysis_seed, seed, fold, 3]).permutation(n_songs)
    parts = np.empty(n_songs, dtype=int)
    parts[perm] = np.arange(n_songs) % 3
    return parts


def binom_pvalue_guarded(k: int, m: int, level: float) -> float:
    """Binomial tail p-value P(Bin(m, level) <= k), used only when k <= m*level - 1.
    With songs pooled over fold models the losses are independent but not identically
    distributed; by Hoeffding (1956) the Poisson-binomial lower tail is bounded by the binomial
    tail of the same mean for k <= m*mean - 1, so the p-value stays valid in that region.
    Outside it we return 1 (never reject), which is conservative."""
    if k > m * level - 1:
        return 1.0
    return binom_cdf(int(k), m, level)


def ltt_select_joint(loss_mats: Sequence[np.ndarray], levels: Sequence[float], delta: float):
    """Fixed-sequence LTT controlling several risks at once: the p-value of a grid column is the
    maximum of its per-risk p-values (valid for the union null 'any risk above its level').
    loss_mats: list of [m, G] {0,1} arrays, columns conservative -> aggressive."""
    m, G = loss_mats[0].shape
    p = np.zeros(G)
    for L, lev in zip(loss_mats, levels):
        k = L.sum(0)
        p = np.maximum(p, np.array([binom_pvalue_guarded(int(kj), m, lev) for kj in k]))
    chosen = -1
    for j in range(G):
        if p[j] <= delta:
            chosen = j
        else:
            break
    return chosen, p


def pool_folds(fold_trajs: Sequence[List[Dict]], dim: int) -> Dict:
    """Concatenate the holdout songs of the fold models of one seed (one policy, one dimension).
    fold_trajs[f] = list of trajectories (dicts with z, lo, hi, full, n, major, track_id)."""
    pres = [precompute(tr, dim) for tr in fold_trajs]
    out = {"full_sign": np.concatenate([p["full_sign"] for p in pres]),
           "n": np.concatenate([p["n"] for p in pres]),
           "label": np.concatenate([[int(t["major"][dim]) for t in tr] for tr in fold_trajs]).astype(int),
           "track_id": np.concatenate([[t["track_id"] for t in tr] for tr in fold_trajs]),
           "fold": np.concatenate([np.full(len(tr), f) for f, tr in enumerate(fold_trajs)])}
    for rule in RULES + ("D",):
        out[rule] = (np.concatenate([p[rule][0] for p in pres]), np.concatenate([p[rule][1] for p in pres]))
    return out


def _losses(pool: Dict, rule: str, idx: np.ndarray, beta):
    steps, decs = pool[rule]
    dis = (decs[idx] != pool["full_sign"][idx, None]).astype(int)
    mats = [dis]
    if beta is not None:
        harm = ((decs[idx] != pool["label"][idx, None]) &
                (pool["full_sign"][idx, None] == pool["label"][idx, None])).astype(int)
        mats.append(harm)
    return mats


def nested_procedure(pools: Dict[str, Dict], candidates, S: np.ndarray, C: np.ndarray, E: np.ndarray,
                     alpha: float, delta: float, gamma: float, beta=None) -> Dict:
    """pools: {policy: pool_folds(...)} (identical song order across policies).
    1) on S, for every candidate (rule, policy): j_ltt = LTT column on S; then j_S = the most
       aggressive column <= j_ltt whose accuracy change on S (early vs full decision, against the
       song-majority labels) is >= -gamma. Candidates without such a column are ineligible.
       Select the eligible candidate with the smallest mean encoded fraction on S at j_S
       (ties: candidate order). None eligible -> deterministic tier D (chronological).
    2) on C (fresh songs): j_C = LTT column of the selected candidate. Use j = min(j_C, j_S).
       Fixed-sequence testing controls every rejected column jointly, and j_S does not depend on
       C, so column j keeps the guarantee P(risk > alpha) <= delta. j_C < 0 -> D of that policy.
    3) on E: apply and return per-song outcomes."""
    levels = [alpha] + ([beta] if beta is not None else [])
    sel_table = []
    best = None
    for rule, policy in candidates:
        pool = pools[policy]
        j_ltt, _ = ltt_select_joint(_losses(pool, rule, S, beta), levels, delta)
        row = dict(rule=rule, policy=policy, s_ltt_index=int(j_ltt), s_index=-1, eligible=False)
        if j_ltt >= 0:
            steps, decs = pool[rule]
            full_acc = float((pool["full_sign"][S] == pool["label"][S]).mean())
            acc = (decs[S, :j_ltt + 1] == pool["label"][S, None]).mean(0) - full_acc       # [j_ltt+1]
            ok = np.flatnonzero(acc >= -gamma - 1e-12)
            if ok.size:
                j_s = int(ok.max())
                frac = float((steps[S, j_s] / pool["n"][S]).mean())
                row.update(s_index=j_s, s_fraction=frac, s_accuracy_delta=float(acc[j_s]), eligible=True)
                if best is None or frac < best[2] - 1e-12:
                    best = (rule, policy, frac, j_s)
        sel_table.append(row)
    j_s = j_c = -1
    if best is None:
        rule, policy, chosen = "D", "chronological", -1
    else:
        rule, policy, _, j_s = best
        j_c, _ = ltt_select_joint(_losses(pools[policy], rule, C, beta), levels, delta)
        chosen = min(j_c, j_s) if j_c >= 0 else -1
    pool = pools[policy]
    if chosen < 0:
        steps, decs = pool["D"]; col = 0
    else:
        steps, decs = pool[rule]; col = chosen
    s, d = steps[E, col], decs[E, col]
    full, lab = pool["full_sign"][E], pool["label"][E]
    return dict(selected_rule=rule, selected_policy=policy, selection_table=sel_table,
                s_index=int(j_s), c_index=int(j_c), final_index=int(chosen), fallback=bool(chosen < 0),
                param=None if (chosen < 0 or rule == "D") else float(GRIDS[rule][chosen]),
                steps=s, decisions=d, fraction=s / pool["n"][E], disagree=(d != full).astype(int),
                correct=(d == lab).astype(int), full_correct=(full == lab).astype(int),
                harmful=((d != lab) & (full == lab)).astype(int), track_id=pool["track_id"][E])


def deterministic_on(pools: Dict[str, Dict], E: np.ndarray, policy: str = "chronological") -> Dict:
    pool = pools[policy]
    s, d = pool["D"][0][E, 0], pool["D"][1][E, 0]
    full, lab = pool["full_sign"][E], pool["label"][E]
    return dict(selected_rule="D", selected_policy=policy, selection_table=[], s_index=-1, c_index=-1,
                final_index=-1, fallback=False,
                param=None, steps=s, decisions=d, fraction=s / pool["n"][E], disagree=(d != full).astype(int),
                correct=(d == lab).astype(int), full_correct=(full == lab).astype(int),
                harmful=((d != lab) & (full == lab)).astype(int), track_id=pool["track_id"][E])


def load_traj(path) -> Dict:
    """Read a traj_<run>.npz written by `stat_certify.py traj`.
    Returns {(seed, fold, policy): {"holdout": [...], "dev": [...]}} of trajectory dicts.
    v1.3.1: every array is materialised ONCE. Indexing an NpzFile (f["z"]) decompresses the whole
    member on each access; the v1.3 loader did that per trajectory, which made `calib` take
    105 min instead of minutes (REPORT 17.7). Values are identical."""
    with np.load(Path(path)) as f:
        a = {k: f[k] for k in f.files}
    units = defaultdict(lambda: {"holdout": [], "dev": []})
    for i in range(len(a["n"])):
        o, n = int(a["offset"][i]), int(a["n"][i])
        tr = dict(z=a["z"][o:o + n], lo=a["lo"][o:o + n], hi=a["hi"][o:o + n], full=a["full"][i], n=n,
                  major=a["major"][i], track_id=str(a["track_id"][i]))
        units[(int(a["seed"][i]), int(a["fold"][i]), str(a["policy"][i]))][str(a["split"][i])].append(tr)
    return units
