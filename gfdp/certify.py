"""Certified early decision ("certified adaptive listening") for GFDP-MER (v1.2).

Why this is possible only because of the interpretable structure
----------------------------------------------------------------
For a song with segments s = 1..n, the FULL-SONG logit of row t, dimension d is

    logit_d(t) = b_d + sum_s sum_k C_d[t,s,k] a_k(s) + sum_k w_d[k] G_k,
    C_d[t,s,k] = sum_h W_loc[d,h,k] K_h(t,s)          (public: times + weights)
    G_k        = sum_s softmax_s(a_k(s)/tau_g) a_k(s)  (glance, a weighted mean)

and every activation a_k(s) depends on segment s alone (per-segment encoder,
frozen centre, fixed prototypes) and lies in a known range [L_k, U_k]:
    differential: [-lambda_d, 1] per prototype, and for the local part the exact
      simplex bound max(c) - lambda*min(c) (s1, s2 are softmaxes over prototypes);
    cosine_kernel: [exp(-2/tau_d), 1];
    protopnet_log: [log(5/4.0001), log(1e4)]; a prototype from the same song: 0.

So after encoding only a subset S of segments, the full-song logit is bounded:
  * local part: exact for s in S, interval arithmetic for s not in S (linear);
  * glance part: G_k is a softmax-weighted mean. Its maximum over the unseen box
    is attained with every unseen value at L_k or U_k (the weighted mean is
    decreasing-then-increasing along each coordinate), so the upper bound is an
    exact enumeration over how many unseen values sit at U_k. The lower bound
    uses the always-valid min(seen values, L_k).
If the bound interval excludes zero, the full-song decision is CERTIFIED: no
content of the unheard segments can change it. Bounds shrink monotonically as
segments are revealed and collapse to the exact logit when S is complete.

Nothing here requires a black box to be explained afterwards; the guarantee is a
property of the additive, bounded evidence structure. Residual arms (R1) are not
certifiable and are rejected.

Targets
  * "song": the song-level decision sign(mean_t logit_d(t)) (labels are ~96%
    song-level in 1000Songs, see Stage 0);
  * "rows": every row decision of the song.
Policies (which segment to encode next; none of them peeks at unheard content)
  * chronological : listening order;
  * coarse_to_fine: glance-like spread (end points, middle, quarters, ...);
  * greedy        : AdaFocus-like adaptive choice of the unseen segment with the
                    largest public bound width over still-uncertified targets.
Baseline (NOT certified): stop when the partial model's song decision has been
unchanged for `stable_steps` consecutive acquisitions.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch

from .model import GFDPModel, gaussian_time_kernel

POLICIES = ("chronological", "coarse_to_fine", "greedy")


def activation_range(model: GFDPModel) -> np.ndarray:
    """[K, 2] valid lower/upper bound of each prototype activation."""
    c = model.config
    k = model.prototypes.size
    out = np.zeros((k, 2))
    dims = model.prototypes.proto_dim.cpu().numpy()
    for j in range(k):
        d = int(dims[j])
        if c.activation == "differential":
            out[j] = (-float(model.lam()[d].detach()), 1.0)
        elif c.activation == "cosine_kernel":
            tau = float(model.log_tau[d, 0].detach().clamp(math.log(0.02), math.log(5.0)).exp())
            out[j] = (math.exp(-4.0 / (2 * tau)), 1.0)
        elif c.activation == "protopnet_log":
            out[j] = (math.log(5.0 / 4.0001), math.log(1.0 / 1e-4))
        else:
            raise ValueError("unknown activation")
    return out


def coarse_to_fine_order(n: int) -> List[int]:
    order, seen = [], set()
    step = n
    while len(order) < n:
        for i in range(0, n, max(1, step)):
            for j in (i, min(n - 1, i + step // 2) if step > 1 else i):
                if j not in seen:
                    seen.add(j); order.append(j)
        if step == 1:
            break
        step = max(1, step // 2)
    order += [i for i in range(n) if i not in seen]
    return order


class SongCertifier:
    """All arithmetic in float64 numpy on quantities computed once per song."""

    def __init__(self, model: GFDPModel, x: torch.Tensor, times: torch.Tensor, song_id: int,
                 activations: np.ndarray = None, range_slack: float = 1e-6) -> None:
        """range_slack widens [L, U] by slack*(1+|bound|) so that activations
        computed in float32 (e.g. on GPU, raw distance marginally above 4) can
        never fall outside the range used by the float64 bound arithmetic."""
        if model.config.residual:
            raise ValueError("arms with a residual bypass are not certifiable")
        model.eval()
        self.n = int(x.shape[0])
        t = times.to(torch.float64)
        if activations is None:
            with torch.no_grad():
                z = model.encoder(x)
                p = model.prototypes.embed(model.encoder)
                allowed = (torch.full((self.n,), song_id, dtype=torch.long, device=x.device)[:, None]
                           != model.prototypes.source_song[None, :])
                activations = model.activations(model.distances(z, p), allowed).double().cpu().numpy()
        self.a = activations                                                   # [n, K]
        self.allowed = (model.prototypes.source_song.cpu().numpy() != song_id)  # [K]
        self.kind = model.config.activation
        self.lam = model.lam().detach().double().cpu().numpy()
        self.dim_of = model.prototypes.proto_dim.cpu().numpy()
        rng = activation_range(model)
        self.slack = float(range_slack)
        lo_r = rng[:, 0] - range_slack * (1 + np.abs(rng[:, 0]))
        hi_r = rng[:, 1] + range_slack * (1 + np.abs(rng[:, 1]))
        self.L = np.where(self.allowed, lo_r, 0.0)
        self.U = np.where(self.allowed, hi_r, 0.0)
        if (self.a < self.L - 1e-12).any() or (self.a > self.U + 1e-12).any():
            raise ValueError("an activation lies outside its certified range; certification is unsound")
        mask = model.readout_mask.double().cpu().numpy()                       # [2, K]
        w_loc = model.w_loc.detach().double().cpu().numpy() * mask[:, None, :]  # [2, H, K]
        self.w_glb = model.w_glb.detach().double().cpu().numpy() * mask        # [2, K]
        if not model.config.glance:
            self.w_glb = np.zeros_like(self.w_glb)
        self.bias = model.bias.detach().double().cpu().numpy()
        self.tau_g = float(model.config.glance_tau)
        kernels = np.stack([gaussian_time_kernel(t, float(s)).numpy() for s in model.config.time_scales])
        self._kernels, self._w_loc = kernels, w_loc
        self.C = np.einsum("dhk,hts->dtsk", w_loc, kernels)                    # [2, n, n, K]
        self.Cbar = self.C.mean(1)                                            # [2, n, K] song-mean target

    # ----------------------------------------------------------------- exact
    @staticmethod
    def _glance(values: np.ndarray, tau: float) -> np.ndarray:
        """values [m, K] -> softmax-weighted mean over m, per K."""
        z = values / tau
        z = z - z.max(0, keepdims=True)
        w = np.exp(z)
        return (w * values).sum(0) / w.sum(0)

    def full_logits(self) -> np.ndarray:
        G = self._glance(self.a, self.tau_g)
        local = np.einsum("dtsk,sk->td", self.C, self.a)
        return self.bias[None] + local + (self.w_glb @ G)[None]

    def partial_logits(self, seen: np.ndarray) -> np.ndarray:
        """Heuristic (uncertified) prediction from heard segments only, exactly as
        GFDPModel.forward(select=seen): kernels renormalised over heard segments,
        glance softmax over heard segments, unheard activations treated as absent."""
        seen = np.asarray(seen, dtype=bool)
        a = self.a * seen[:, None]
        k = self._kernels * seen[None, None, :]
        k = k / np.clip(k.sum(-1, keepdims=True), 1e-12, None)
        local = np.einsum("dhk,hts,sk->td", self._w_loc, k, a)
        G = self._glance(self.a[seen], self.tau_g) if seen.any() else np.zeros(self.a.shape[1])
        return self.bias[None] + local + (self.w_glb @ G)[None]

    # ---------------------------------------------------------------- bounds
    def _glance_bounds(self, seen: np.ndarray):
        """Per prototype: valid [lo, hi] for G_k given heard values."""
        K = self.a.shape[1]
        n_u = int((~seen).sum())
        if n_u == 0:
            g = self._glance(self.a, self.tau_g)
            return g, g
        sv = self.a[seen]                                      # [m, K]
        lo = self.L.copy()
        if sv.shape[0]:
            lo = np.minimum(lo, sv.min(0))
        hi = np.empty(K)
        m_up = np.arange(n_u + 1)                              # how many unseen at U
        for k in range(K):
            L, U = self.L[k], self.U[k]
            ref = max(U, sv[:, k].max() if sv.shape[0] else U)
            es = np.exp((sv[:, k] - ref) / self.tau_g) if sv.shape[0] else np.zeros(0)
            num_s, den_s = float((es * sv[:, k]).sum()), float(es.sum())
            eU, eL = math.exp((U - ref) / self.tau_g), math.exp((L - ref) / self.tau_g)
            num = num_s + m_up * eU * U + (n_u - m_up) * eL * L
            den = den_s + m_up * eU + (n_u - m_up) * eL
            hi[k] = float((num / den).max())
        return lo, hi

    def bounds(self, seen: np.ndarray, target: str = "rows"):
        """target 'rows' -> (lo [n,2], hi [n,2]); 'song' -> (lo [2], hi [2])."""
        seen = np.asarray(seen, dtype=bool)
        C = self.C if target == "rows" else self.Cbar[:, None]           # [2, T, n, K]
        exact = np.einsum("dtsk,sk->dt", C[:, :, seen], self.a[seen])
        Cu = C[:, :, ~seen]
        if self.kind == "differential":
            # a = s1 - lambda*s2 with s1, s2 on the probability simplex over the
            # allowed prototypes of the dimension: the exact extreme of
            # sum_k c_k a_k is max(c) - lambda*min(c) (and min(c) - lambda*max(c)).
            up = np.zeros(Cu.shape[:2]); dn = np.zeros(Cu.shape[:2])
            for d in range(2):
                idx = np.flatnonzero((self.dim_of == d) & self.allowed)
                if idx.size == 0 or Cu.shape[2] == 0:
                    continue
                cd = Cu[d][..., idx]                                       # [T, u, k]
                pad = self.slack * (1 + self.lam[d]) * np.abs(cd).sum((-1, -2))   # float32 simplex slack
                up[d] = (cd.max(-1) - self.lam[d] * cd.min(-1)).sum(-1) + pad
                dn[d] = (cd.min(-1) - self.lam[d] * cd.max(-1)).sum(-1) - pad
        else:
            up = np.maximum(Cu * self.U, Cu * self.L).sum((2, 3))
            dn = np.minimum(Cu * self.U, Cu * self.L).sum((2, 3))
        g_lo, g_hi = self._glance_bounds(seen)
        gw_hi = np.maximum(self.w_glb * g_hi, self.w_glb * g_lo).sum(1)    # [2]
        gw_lo = np.minimum(self.w_glb * g_hi, self.w_glb * g_lo).sum(1)
        lo = self.bias[:, None] + exact + dn + gw_lo[:, None]
        hi = self.bias[:, None] + exact + up + gw_hi[:, None]
        if target == "song":
            return lo[:, 0], hi[:, 0]
        return lo.T, hi.T

    def width_scores(self, seen: np.ndarray, active: np.ndarray, target: str) -> np.ndarray:
        """Public bound width each unseen segment contributes to active targets."""
        C = self.C if target == "rows" else self.Cbar[:, None]
        span = (self.U - self.L)
        per = np.abs(C) * span                                             # [2, T, n, K]
        score = np.einsum("dtsk,dt->s", per, active.astype(float))
        score[seen] = -np.inf
        return score


def certified(lo, hi, eps: float = 1e-6):
    """+1 certified positive (lo > eps), -1 certified negative (hi <= -eps), 0 not yet."""
    return np.where(lo > eps, 1, np.where(hi <= -eps, -1, 0))


def run_policy(cert: SongCertifier, policy: str, target: str, stable_steps: int = 3) -> Dict:
    n = cert.n
    full = cert.full_logits()
    full_target = full if target == "rows" else full.mean(0, keepdims=True)     # [T, 2]
    full_sign = np.where(full_target > 0, 1, -1)
    T = full_target.shape[0]
    first = np.full((T, 2), n, dtype=int)                                      # steps needed
    seen = np.zeros(n, dtype=bool)
    order = (list(range(n)) if policy == "chronological" else
             coarse_to_fine_order(n) if policy == "coarse_to_fine" else None)
    violations = 0
    heur = {"stop_step": None, "decision": None}
    history = []
    for step in range(1, n + 1):
        if policy == "greedy":
            active = (first == n).T                                            # [2, T]
            nxt = int(np.argmax(cert.width_scores(seen, active, target)))
        else:
            nxt = order[step - 1]
        seen[nxt] = True
        lo, hi = cert.bounds(seen, target)
        lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
        c = certified(lo, hi)
        newly = (c != 0) & (first == n)
        if np.any(c[newly] != full_sign[newly]):
            violations += int((c[newly] != full_sign[newly]).sum())
        first[newly] = step
        if target == "song" and policy != "greedy":
            part = cert.partial_logits(seen).mean(0)
            sign = np.where(part > 0, 1, -1)
            history.append(sign)
            if heur["stop_step"] is None and step >= stable_steps and \
                    all(np.array_equal(h, sign) for h in history[-stable_steps:]):
                heur = {"stop_step": step, "decision": sign}
        if (first < n).all() and (target != "song" or policy == "greedy" or heur["stop_step"]):
            break
    first[first == n] = n
    return dict(first_step=first, full_sign=full_sign, violations=violations, n=n, heuristic=heur)


def summarise(records: List[Dict], budgets: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)) -> Dict:
    """records: run_policy outputs for one (model, policy, target) over many songs."""
    out = {}
    for d, name in enumerate(("valence", "arousal")):
        frac = np.concatenate([r["first_step"][:, d] / r["n"] for r in records])
        song_frac = np.array([r["first_step"][:, d].max() / r["n"] for r in records])
        res = dict(mean_fraction_encoded=float(frac.mean()), median_fraction_encoded=float(np.median(frac)),
                   p90_fraction_encoded=float(np.percentile(frac, 90)),
                   share_certified_at_budget={str(b): float((frac <= b + 1e-12).mean()) for b in budgets},
                   all_targets_of_song_mean_fraction=float(song_frac.mean()),
                   violations=int(sum(r["violations"] for r in records)))
        h = [r["heuristic"] for r in records if r["heuristic"]["stop_step"] is not None]
        if h:
            agree = [float(r["heuristic"]["decision"][d] == r["full_sign"][0, d]) for r in records
                     if r["heuristic"]["stop_step"] is not None]
            res["heuristic_stable_sign"] = dict(
                mean_fraction_encoded=float(np.mean([r["heuristic"]["stop_step"] / r["n"] for r in records
                                                     if r["heuristic"]["stop_step"] is not None])),
                agreement_with_full_decision=float(np.mean(agree)), songs=len(h))
        out[name] = res
    return out


# ---------------------------------------------------------------------------
# Differentiable song-level bounds for certification-aware training (v1.2)
# ---------------------------------------------------------------------------
def _activation_range_torch(model: GFDPModel):
    """Differentiable [K] lower/upper activation bounds (same as activation_range)."""
    c = model.config
    dims = model.prototypes.proto_dim
    if c.activation == "differential":
        lam = model.lam()
        return -lam[dims], torch.ones_like(lam[dims])
    if c.activation == "cosine_kernel":
        tau = model.log_tau[:, 0].clamp(math.log(0.02), math.log(5.0)).exp()
        return torch.exp(-4.0 / (2 * tau[dims])), torch.ones_like(tau[dims])
    if c.activation == "protopnet_log":
        k = model.prototypes.size
        dev = model.w_loc.device
        dt = model.w_loc.dtype
        return (torch.full((k,), math.log(5.0 / 4.0001), device=dev, dtype=dt),
                torch.full((k,), math.log(1.0 / 1e-4), device=dev, dtype=dt))
    raise ValueError("unknown activation")


def song_bounds_torch(model: GFDPModel, out: Dict, song_index: int, sl: slice, seen: torch.Tensor,
                      detach_range: bool = False):
    """Lower/upper bound [2] of the full-song mean logit given heard segments
    `seen` (bool [n]), from a full forward `out = model(..., decompose=True)`.
    Mirrors SongCertifier.bounds(target='song') and is differentiable in the
    readout, temperatures and the heard activations."""
    a = out["activation"][sl]                                   # [n, K]
    kern = out["kernels"][song_index]                           # [H, n, n] full-song normalised
    mask = model.readout_mask
    w_loc = model.w_loc * mask[:, None, :]                      # [2, H, K]
    w_glb = model.w_glb * mask if model.config.glance else torch.zeros_like(model.w_glb)
    cbar = torch.einsum("dhk,hs->dsk", w_loc, kern.mean(1))     # [2, n, K]
    allowed = out["allowed"][sl][0]                             # [K] (same for every row of a song)
    L, U = _activation_range_torch(model)
    if detach_range:
        # Training: the range is a property of the activation family, not something
        # the certification loss may shrink (raising tau flattens every kernel and
        # trivially "certifies" an uninformative model - observed in smoke runs).
        L, U = L.detach(), U.detach()
    L = torch.where(allowed, L, torch.zeros_like(L)); U = torch.where(allowed, U, torch.zeros_like(U))
    exact = torch.einsum("dsk,sk->d", cbar[:, seen], a[seen])
    cu = cbar[:, ~seen]                                         # [2, u, K]
    if model.config.activation == "differential":
        lam = model.lam().detach() if detach_range else model.lam()
        up, dn = [], []
        for d in range(2):
            idx = torch.nonzero((model.prototypes.proto_dim == d) & allowed, as_tuple=True)[0]
            if idx.numel() == 0 or cu.shape[1] == 0:
                up.append(exact.new_zeros(())); dn.append(exact.new_zeros(())); continue
            cd = cu[d][:, idx]
            up.append((cd.max(-1).values - lam[d] * cd.min(-1).values).sum())
            dn.append((cd.min(-1).values - lam[d] * cd.max(-1).values).sum())
        up, dn = torch.stack(up), torch.stack(dn)
    else:
        up = torch.maximum(cu * U, cu * L).sum((1, 2))
        dn = torch.minimum(cu * U, cu * L).sum((1, 2))
    # glance: G_k = softmax-weighted mean over all segments
    n_u = int((~seen).sum())
    tau = model.config.glance_tau
    if n_u == 0:
        al = torch.softmax(a / tau, 0)
        g_lo = g_hi = (al * a).sum(0)
    else:
        sv = a[seen]
        g_lo = torch.minimum(L, sv.min(0).values) if sv.shape[0] else L
        ref = torch.maximum(U, sv.max(0).values if sv.shape[0] else U).detach()
        es = torch.exp((sv - ref) / tau) if sv.shape[0] else a.new_zeros(0, a.shape[1])
        num_s, den_s = (es * sv).sum(0), es.sum(0)
        m = torch.arange(n_u + 1, device=a.device, dtype=a.dtype)[:, None]
        eU, eL = torch.exp((U - ref) / tau), torch.exp((L - ref) / tau)
        num = num_s + m * eU * U + (n_u - m) * eL * L
        den = den_s + m * eU + (n_u - m) * eL
        g_hi = (num / den).max(0).values
    gw_hi = torch.maximum(w_glb * g_hi, w_glb * g_lo).sum(1)
    gw_lo = torch.minimum(w_glb * g_hi, w_glb * g_lo).sum(1)
    lo = model.bias + exact + dn + gw_lo
    hi = model.bias + exact + up + gw_hi
    return lo, hi


def sample_heard(n: int, frac_min: float, frac_max: float, generator: torch.Generator = None) -> torch.Tensor:
    """Half chronological prefixes, half random subsets, of a random heard fraction."""
    frac = frac_min + (frac_max - frac_min) * float(torch.rand((), generator=generator))
    m = max(1, min(n, int(round(frac * n))))
    seen = torch.zeros(n, dtype=torch.bool)
    if float(torch.rand((), generator=generator)) < 0.5:
        seen[:m] = True
    else:
        seen[torch.randperm(n, generator=generator)[:m]] = True
    return seen
