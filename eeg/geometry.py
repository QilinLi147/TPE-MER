"""FIT-only deterministic geometry. No labels, DEV or evaluation arguments."""
import numpy as np


def stable(x):
    return np.concatenate([x.mean(2, dtype=np.float64), x.std(2, dtype=np.float64)],
                          axis=2).reshape(len(x), -1)


def native(x):
    return x.mean(2, dtype=np.float64).reshape(len(x), -1)


def mse(a, q):
    a, q = np.asarray(a, np.float64), np.asarray(q, np.float64)
    # Same finite quadratic distance as the parent regional implementation.
    d = (np.square(a).sum(1)[:, None] + np.square(q).sum(1)[None, :] - 2 * a @ q.T)
    return d.clip(0) / a.shape[1]


def median_scale(distances, source_rows):
    d = np.array(distances, dtype=np.float64, copy=True)
    assert np.isfinite(d).all() and (d >= 0).all()
    rows = np.asarray(source_rows, int)
    assert len(set(rows.tolist())) == len(rows) and d.shape[1] == len(rows)
    assert (rows >= 0).all() and (rows < len(d)).all()
    # Each distinct FIT row appears once as a source in this family. Self-pairs
    # and every exact zero distance are excluded; numerical positive distances
    # are retained. No target-dependent sampling or width search.
    d[rows, np.arange(len(rows))] = 0.
    positive = d[d > 0]
    tau = max(float(np.median(positive)), 1e-12) if len(positive) else 1.
    return tau, dict(positive_pairs=int(len(positive)), distinct_sources=len(rows),
                     self_pairs_excluded=len(rows), all_zero_fallback=not len(positive),
                     tau=tau)


def scale_state(z, x, bank):
    v = (stable(x) - bank["mu"]) / bank["sd"]
    z = np.asarray(z, np.float64)
    scales, stats = [], []
    for typ in range(4):
        rows = np.unique(bank["rows"][bank["types"] == typ])
        assert len(rows), "empty evidence family"
        if typ == 0:
            d = (2 - 2 * z @ z[rows].T).clip(0, 4)
        else:
            mask = bank["masks"][typ - 1]
            d = mse(v[:, mask], v[rows][:, mask])
        tau, info = median_scale(d, rows)
        scales.append(tau)
        stats.append(info)
    return {"tau": np.asarray(scales)}, stats


def native_state(x, bank):
    a = native(x)
    mu, sd = a.mean(0), a.std(0).clip(1e-3)
    a = (a - mu) / sd
    rows = np.unique(bank["rows"])
    tau, stats = median_scale(mse(a, a[rows]), rows)
    state = dict(mu=mu, sd=sd, sources=a[bank["rows"]], tau=np.array([tau]))
    return state, [stats]


def features(arm, z, x, bank, state):
    N, (D, K) = len(x), bank["rows"].shape
    out = np.empty((N, D, K), np.float64)
    if arm == "A_scale":
        v = (stable(x) - bank["mu"]) / bank["sd"]
        z = np.asarray(z, np.float64)
        for d in range(D):
            for typ in range(4):
                ix = np.flatnonzero(bank["types"][d] == typ)
                if typ == 0:
                    dist = (2 - 2 * z @ bank["z"][d, ix].T).clip(0, 4)
                else:
                    mask = bank["masks"][typ - 1]
                    dist = mse(v[:, mask], bank["v"][d, ix][:, mask])
                # IEEE float64 underflow to exactly zero is accepted. No
                # lower clipping or evaluation-dependent rescaling.
                out[:, d, ix] = np.exp(-dist / state["tau"][typ])
    elif arm == "B_native":
        a = (native(x) - state["mu"]) / state["sd"]
        for d in range(D):
            out[:, d] = np.exp(-mse(a, state["sources"][d]) / state["tau"][0])
    else:
        raise PermissionError("unregistered arm")
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    return out


def provenance(bank, fit):
    rows = bank["rows"]
    assert rows.ndim == 2 and rows.shape[1] == 28
    assert rows.min() >= 0 and rows.max() < len(fit["x"])
    for name in ("original_index", "subject", "session", "trial"):
        assert np.array_equal(fit[name][rows], bank["source_" + name]), name
    identities = np.stack([fit[k] for k in ("subject", "session", "trial", "original_index")], 1)
    assert len(np.unique(identities, axis=0)) == len(identities), "ambiguous FIT row identities"
    return dict(nonfit_sources=0, slots_per_class=28,
                distinct_sources=int(len(np.unique(rows))),
                distinct_sources_per_class=[len(np.unique(r)) for r in rows])
