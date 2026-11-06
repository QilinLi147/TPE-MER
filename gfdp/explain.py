"""Ante-hoc + post-hoc explanation suite.

ANTE-HOC (exact, from the model's own structure)
  * contribution_ledger: bias + local(h,k) + glance(k) (+ residual) == logit,
    with each prototype's real TRAIN source (song, row, time).
  * segment_attribution: because every evidence term is linear in a[k](s), the
    logit of row t splits EXACTLY over the song's segments s:
        local:  sum_h sum_k W_loc[d,h,k] K_h(t,s) a[k](s)
        glance: sum_k W_glb[d,k] alpha_k(s) a[k](s)
    (alpha is treated as the realised attention weight, as in attention rollout;
     the identity is exact for the computed forward pass.)

POST-HOC (about the black-box parts: encoder and, in arm R, the residual)
  * integrated_gradients over the input spectrogram views (completeness checked)
  * band_occlusion over registered frequency-index bands
  * prototype_deletion: remove prototypes by |contribution| vs random order
  * randomisation_sanity (Adebayo et al. 2018): re-initialised encoder must change IG
  * ante_post_agreement: Spearman between ante-hoc segment attribution and
    post-hoc IG mass per segment
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .model import DIMENSIONS, GFDPModel


def _forward(model: GFDPModel, batch: Dict, **kw):
    return model(batch["x"], batch["times"], batch["songs"], batch["song_slices"], **kw)


@torch.no_grad()
def contribution_ledger(model: GFDPModel, batch: Dict, row: int, d: int, top: int = 5) -> Dict:
    model.eval()
    out = _forward(model, batch, decompose=True)
    loc = out["local_terms"][row, d]                    # [h, K]
    glb = out["glance_terms"][row, d]                   # [K]
    total = float(out["bias"][d] + loc.sum() + glb.sum() + out["residual"][row, d])
    per_proto = (loc.sum(0) + glb).cpu().numpy()
    meta = model.prototypes.metadata()
    order = np.argsort(-np.abs(per_proto))
    shown = order[:top]
    return dict(dimension=DIMENSIONS[d], logit=float(out["logit"][row, d]),
                reconstruction_error=abs(total - float(out["logit"][row, d])),
                bias=float(out["bias"][d]), residual=float(out["residual"][row, d]),
                local_by_scale=loc.sum(-1).cpu().tolist(), glance_total=float(glb.sum()),
                top_prototypes=[dict(meta[k], contribution=float(per_proto[k]),
                                     local=float(loc[:, k].sum()), glance=float(glb[k]))
                                for k in shown],
                remainder=float(per_proto[order[top:]].sum()))


@torch.no_grad()
def segment_attribution(model: GFDPModel, batch: Dict, song_index: int, d: int) -> Dict[str, np.ndarray]:
    """[rows_in_song, rows_in_song] matrix M with logit(t) = bias + residual(t) + sum_s M[t,s]."""
    model.eval()
    out = _forward(model, batch, decompose=True)
    sl = batch["song_slices"][song_index]
    a = out["activation"][sl]                           # [n, K]
    mask = model.readout_mask[d]
    w_loc = model.w_loc[d] * mask[None]                 # [h, K]
    w_glb = model.w_glb[d] * mask
    kernels = out["kernels"][song_index]                # [h, n, n]
    local = torch.einsum("hts,sk,hk->ts", kernels, a, w_loc)
    alpha = out["glance_attention"][sl]                 # [n, K]
    glance_seg = (alpha * a * w_glb[None]).sum(-1)      # [n] same for every t
    matrix = local + glance_seg[None, :]
    recon = out["bias"][d] + out["residual"][sl, d] + matrix.sum(-1)
    return dict(matrix=matrix.cpu().numpy(), local=local.cpu().numpy(),
                glance=glance_seg.cpu().numpy(),
                max_error=float((recon - out["logit"][sl, d]).abs().max()))


def integrated_gradients(model: GFDPModel, batch: Dict, d: int, target_rows: Optional[torch.Tensor] = None,
                         steps: int = 32, baseline: str = "batch_mean", prototype: Optional[int] = None,
                         adaptive: bool = True, max_steps: int = 512, rel_tol: float = 0.05,
                         abs_tol: float = 1e-2) -> Dict:
    """IG of sum_t logit_d(t) (or of activation a[prototype]) w.r.t. the input x.

    v1.1: nested trapezoid rule. If the completeness gap |sum(IG) - (f(x)-f(b))|
    exceeds max(abs_tol, rel_tol*|f(x)-f(b)|), the number of intervals is doubled
    (re-using every previous gradient) up to `max_steps`. Sharp prototype and
    glance softmaxes made a fixed 32-step rule insufficient on real songs.
    Prototype embeddings are held fixed at their exact values."""
    model.eval()
    x = batch["x"].detach()
    # A baseline that is constant along frames sits on the LayerNorm(frames)
    # singularity of the encoder and breaks IG completeness; the default is the
    # mean spectrogram of the explained batch, which varies along frames.
    if baseline == "zeros":
        base = torch.zeros_like(x)
    elif baseline == "batch_mean":
        base = x.mean(0, keepdim=True).expand_as(x).clone()
    else:
        raise ValueError("baseline must be 'batch_mean' or 'zeros'")
    with torch.no_grad():
        proto = model.prototypes.embed(model.encoder)
    rows = target_rows if target_rows is not None else torch.ones(len(x), dtype=torch.bool)

    def objective(inp):
        out = model(inp, batch["times"], batch["songs"], batch["song_slices"],
                    proto_embedding=proto, decompose=prototype is not None)
        if prototype is None:
            return out[rows, d].sum()
        return out["activation"][rows, prototype].sum()

    def grad_at(alpha):
        inp = (base + (x - base) * alpha).requires_grad_(True)
        g, = torch.autograd.grad(objective(inp), inp)
        return g

    with torch.no_grad():
        f_x, f_b = float(objective(x)), float(objective(base))
    delta = f_x - f_b
    tolerance = max(abs_tol, rel_tol * abs(delta))
    n = max(1, int(steps))
    total = 0.5 * (grad_at(0.0) + grad_at(1.0))
    for i in range(1, n):
        total = total + grad_at(i / n)
    history = []
    while True:
        attribution = (x - base) * total / n
        gap = abs(float(attribution.sum()) - delta)
        history.append(dict(intervals=n, completeness_gap=gap))
        if not adaptive or gap <= tolerance or n * 2 > max_steps:
            break
        for j in range(1, n + 1):                       # new midpoints of every interval
            total = total + grad_at((2 * j - 1) / (2 * n))
        n *= 2
    return dict(attribution=attribution, completeness_gap=gap,
                relative_gap=gap / abs(delta) if delta != 0 else float("inf"),
                completeness_passes=gap <= tolerance, tolerance=tolerance, intervals=n,
                refinement_history=history, objective_input=f_x, objective_baseline=f_b)


@torch.no_grad()
def band_occlusion(model: GFDPModel, batch: Dict, d: int,
                   bands: Sequence[tuple] = tuple((i * 16, (i + 1) * 16) for i in range(8))
                   + tuple((128 + i * 21, 128 + (i + 1) * 21) for i in range(4))) -> List[Dict]:
    """Mean logit change when a frequency-index band is replaced by its song mean."""
    model.eval()
    ref = _forward(model, batch)[:, d]
    out = []
    for lo, hi in bands:
        x = batch["x"].clone()
        for sl in batch["song_slices"]:
            x[sl, lo:hi] = x[sl, lo:hi].mean((0, 2), keepdim=True)
        new = model(x, batch["times"], batch["songs"], batch["song_slices"])[:, d]
        out.append(dict(band=[lo, hi], view="mel" if hi <= 128 else "second",
                        mean_abs_change=float((new - ref).abs().mean()),
                        mean_change=float((new - ref).mean())))
    return out


@torch.no_grad()
def prototype_deletion(model: GFDPModel, batch: Dict, d: int, seed: int = 0, n_random: int = 20) -> Dict:
    """Faithfulness of the ante-hoc ledger: cumulatively zero the readout weights
    of this dimension's prototypes in order of mean |contribution| and track the
    fraction of decisions that flip. v1.1: compared with the MEAN of `n_random`
    random orders (not a single permutation); faithful requires a strictly
    larger mean flip rate, and the share of random orders beaten is reported."""
    model.eval()
    base = _forward(model, batch, decompose=True)
    idx = model.prototypes.indices(d)
    contrib = (base["local_terms"][:, d].sum(1) + base["glance_terms"][:, d])[:, idx].abs().mean(0)
    ref_sign = base["logit"][:, d] > 0

    def curve(order):
        m = copy.deepcopy(model)
        flips = []
        for k in order.tolist():
            m.w_loc.data[d, :, k] = 0
            m.w_glb.data[d, k] = 0
            new = _forward(m, batch)[:, d] > 0
            flips.append(float((new != ref_sign).float().mean()))
        return flips

    by_contribution = curve(idx[torch.argsort(contrib, descending=True)])
    gen = torch.Generator().manual_seed(seed)
    random_curves = [curve(idx[torch.randperm(len(idx), generator=gen)]) for _ in range(n_random)]
    mean_contrib = float(np.mean(by_contribution))
    random_means = np.array([np.mean(c) for c in random_curves])
    return dict(curve_by_contribution=by_contribution,
                curve_random_mean=np.mean(random_curves, 0).tolist(),
                mean_flip_rate=dict(by_contribution=mean_contrib, random=float(random_means.mean())),
                share_random_orders_beaten=float((mean_contrib > random_means).mean()),
                informative=bool(mean_contrib > 0 or random_means.max() > 0),
                faithful=bool(mean_contrib > random_means.mean()), n_random=n_random)


def randomisation_sanity(model: GFDPModel, batch: Dict, d: int, steps: int = 16) -> Dict:
    """Saliency maps only are compared here, so completeness refinement is off."""
    ig = integrated_gradients(model, batch, d, steps=steps, adaptive=False)["attribution"].abs().sum(1).flatten()
    rnd = copy.deepcopy(model)
    for module in rnd.encoder.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    ig_r = integrated_gradients(rnd, batch, d, steps=steps, adaptive=False)["attribution"].abs().sum(1).flatten()
    rho = _spearman(ig.detach().numpy(), ig_r.detach().numpy())
    return dict(spearman_original_vs_randomised=rho, passes=bool(np.isfinite(rho) and rho < 0.5))


def ante_post_agreement(model: GFDPModel, batch: Dict, song_index: int, d: int, row: int,
                        steps: int = 32) -> Dict:
    """Does post-hoc IG locate the same segments as the exact ante-hoc split?"""
    sl = batch["song_slices"][song_index]
    seg = segment_attribution(model, batch, song_index, d)["matrix"][row - sl.start]
    target = torch.zeros(len(batch["x"]), dtype=torch.bool)
    target[row] = True
    ig = integrated_gradients(model, batch, d, target_rows=target, steps=steps)
    ig_seg = ig["attribution"][sl].sum((1, 2)).detach().numpy()
    return dict(spearman=_spearman(seg, ig_seg), ante=seg.tolist(), post=ig_seg.tolist(),
                ig_completeness_passes=ig["completeness_passes"], ig_intervals=ig["intervals"])


def _average_ranks(v: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(np.asarray(v, dtype=np.float64), return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman with average ranks for ties (v1.1); NaN if either input is constant."""
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")
