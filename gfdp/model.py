"""GFDP-MER: Glance-Focus Differential Prototype network for music emotion.

Prediction for row t of song S and dimension d in {valence, arousal}:

    logit_d(t) = b_d
               + sum_h sum_k  W_loc[d,h,k] * A_h[k](t)      (focus, local evidence)
               + sum_k        W_glb[d,k]   * G[k](S)        (glance, song evidence)
               + rho_d(t)                                    (optional residual, arm R only)

    a[k](s)   = differential prototype activation of segment s (Differential
                Transformer idea: two competitive maps, the second subtracted
                with a learned lambda to cancel common-mode similarity)
    A_h[k](t) = sum_s K_h(t,s) a[k](s)   public Gaussian time kernels, within song
    G[k](S)   = sum_s alpha_k(s) a[k](s), alpha_k = softmax_s(a[k](s)/tau_g)
                (Agent-Attention idea: each prototype is an agent token that
                attends over the whole song; alpha_k is its temporal "glance")

For task="regression" (v1.4) the same additive output is the predicted V/A value
itself (no link function), so the decomposition is over the prediction.

Every term except rho is linear in a[k](s), so the logit decomposes exactly
over (prototype, segment, time-scale) -- this is the ante-hoc explanation.
After `push`, each prototype is literally E(x_src) for a stored real TRAIN
segment x_src, re-encoded at every forward pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoder import DualViewEncoder, count_parameters

DIMENSIONS = ("valence", "arousal")


@dataclass
class ModelConfig:
    mel_bins: int = 128
    second_bins: int = 84
    frames: int = 44
    channels: int = 128
    dim: int = 128
    proj_dim: int = 64
    dropout: float = 0.1
    drop_path: float = 0.1
    envelope: bool = True                  # re-inject per-bin level lost by LayerNorm(frames)
    prototypes_per_class: int = 8
    activation: str = "differential"       # differential | protopnet_log | cosine_kernel
    tau_initial: float = 0.1
    lambda_initial: float = 0.5
    time_scales: Sequence[float] = (0.0, 4.0, 16.0)
    glance: bool = True
    glance_tau: float = 0.5
    residual: bool = False                 # arm R: capped residual explained post hoc
    residual_cap: float = 1.0
    readout_init: float = 0.1              # small: initial logits ~0, no incentive to collapse
    views: str = "both"                    # v1.4: both | mel | second (single-view fusion ablations)
    task: str = "classification"           # v1.4: classification (logit, y>0) | regression (value in [-1,1])

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def gaussian_time_kernel(times: Tensor, seconds: float) -> Tensor:
    """Row-normalised within-song kernel [n, n]; seconds == 0 is identity."""
    n = times.shape[0]
    if seconds == 0:
        return torch.eye(n, dtype=times.dtype, device=times.device)
    delta = (times[:, None] - times[None, :]).abs()
    k = torch.exp(-0.5 * (delta / seconds) ** 2) * (delta <= 3 * seconds)
    return k / k.sum(-1, keepdim=True).clamp_min(1e-12)


class PrototypeBank(nn.Module):
    """Class-labelled prototypes for each dimension.

    Layout: index k in [0, K) with K = 2 dims * 2 classes * m.
    proto_dim[k] in {0,1}, proto_class[k] in {0 (low), 1 (high)}.
    Mode 'free': vectors are parameters. Mode 'anchored': vectors are
    E(source_x[k]) recomputed with the live encoder (real TRAIN segments).
    """

    def __init__(self, dim: int, m: int, input_shape: Sequence[int]) -> None:
        super().__init__()
        k = 2 * 2 * m
        self.m = m
        self.vectors = nn.Parameter(F.normalize(torch.randn(k, dim), dim=-1))
        dims = torch.arange(k) // (2 * m)
        classes = (torch.arange(k) // m) % 2
        self.register_buffer("proto_dim", dims)
        self.register_buffer("proto_class", classes)
        self.register_buffer("anchored", torch.tensor(False))
        self.register_buffer("source_x", torch.zeros(k, *input_shape))
        self.register_buffer("source_song", torch.full((k,), -1, dtype=torch.long))
        self.register_buffer("source_row", torch.full((k,), -1, dtype=torch.long))
        self.register_buffer("source_time", torch.full((k,), float("nan")))

    @property
    def size(self) -> int:
        return self.vectors.shape[0]

    def indices(self, d: int) -> Tensor:
        return torch.nonzero(self.proto_dim == d, as_tuple=True)[0]

    def embed(self, encoder: nn.Module) -> Tensor:
        if bool(self.anchored):
            return encoder(self.source_x)
        return F.normalize(self.vectors, dim=-1)

    @torch.no_grad()
    def anchor(self, source_x: Tensor, song: Tensor, row: Tensor, time: Tensor) -> None:
        self.source_x.copy_(source_x)
        self.source_song.copy_(song)
        self.source_row.copy_(row)
        self.source_time.copy_(time)
        self.anchored.fill_(True)
        self.vectors.requires_grad_(False)

    def metadata(self) -> List[Dict]:
        out = []
        for k in range(self.size):
            out.append(dict(index=k, dimension=DIMENSIONS[int(self.proto_dim[k])],
                            cls="high" if int(self.proto_class[k]) else "low",
                            anchored=bool(self.anchored), source_song=int(self.source_song[k]),
                            source_row=int(self.source_row[k]),
                            source_time=float(self.source_time[k])))
        return out


class GFDPModel(nn.Module):
    FORMAT = "gfdp_mer_v1"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        c = config
        self.config = c
        if c.activation not in ("differential", "protopnet_log", "cosine_kernel"):
            raise ValueError("unknown activation")
        if c.task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")
        self.encoder = DualViewEncoder(c.mel_bins, c.second_bins, c.frames, c.channels,
                                       c.dim, c.dropout, c.drop_path, c.envelope, c.views)
        self.prototypes = PrototypeBank(c.dim, c.prototypes_per_class,
                                        (c.mel_bins + c.second_bins, c.frames))
        # Two similarity "heads" for the differential activation (bias-free so
        # that identical embeddings are always at distance zero in both heads).
        self.head1 = nn.Linear(c.dim, c.proj_dim, bias=False)
        self.head2 = nn.Linear(c.dim, c.proj_dim, bias=False)
        self.log_tau = nn.Parameter(torch.full((2, 2), math.log(c.tau_initial)))  # [dim, head]
        lam = min(max(c.lambda_initial, 1e-3), 1 - 1e-3)
        self.lambda_logit = nn.Parameter(torch.full((2,), math.log(lam / (1 - lam))))
        k = self.prototypes.size
        h = len(c.time_scales)
        self.w_loc = nn.Parameter(torch.zeros(2, h, k))
        self.w_glb = nn.Parameter(torch.zeros(2, k))
        self.bias = nn.Parameter(torch.zeros(2))
        self.register_buffer("readout_mask", self._readout_mask())
        if c.residual:
            self.residual_net = nn.Sequential(nn.Linear(c.dim, 64), nn.GELU(), nn.Linear(64, 2))
            self.residual_scale_logit = nn.Parameter(torch.full((2,), -3.0))
        self.reset_readout()

    # ------------------------------------------------------------------ setup
    def _readout_mask(self) -> Tensor:
        """Dimension d may only read its own prototypes (clean explanations)."""
        mask = torch.zeros(2, self.prototypes.size)
        for d in range(2):
            mask[d, self.prototypes.indices(d)] = 1.0
        return mask

    @torch.no_grad()
    def reset_readout(self) -> None:
        """Class-connection initialisation: 'high' prototypes push the logit up,
        'low' prototypes down, scaled by `readout_init` so that the initial loss
        is close to log 2 (a large random-sign start drove the encoder into a
        collapsed constant embedding in the smoke tests)."""
        sign = (self.prototypes.proto_class.float() * 2 - 1)          # high:+1, low:-1
        base = sign * self.config.readout_init * self.prototypes.m * 2
        h = self.w_loc.shape[1]
        n_terms = h + (1 if self.config.glance else 0)
        for d in range(2):
            self.w_loc[d] = base[None, :] * self.readout_mask[d][None, :] / n_terms
            self.w_glb[d] = base * self.readout_mask[d] / n_terms
        if not self.config.glance:
            self.w_glb.zero_()
        self.bias.zero_()

    def readout_parameters(self) -> List[nn.Parameter]:
        return [self.w_loc, self.w_glb, self.bias]

    def parameter_count(self) -> Dict[str, int]:
        return dict(total=count_parameters(self), encoder=count_parameters(self.encoder))

    # ---------------------------------------------------------- activations
    def distances(self, z: Tensor, p: Tensor) -> Dict[str, Tensor]:
        """Squared distances in the raw unit-sphere space and both heads."""
        raw = (2 - 2 * z @ p.t()).clamp_min(0)
        # unit-norm heads keep both distance maps in [0, 4] so the temperatures
        # are meaningful (unnormalised random heads gave near-uniform softmaxes)
        u1, q1 = F.normalize(self.head1(z), dim=-1), F.normalize(self.head1(p), dim=-1)
        u2, q2 = F.normalize(self.head2(z), dim=-1), F.normalize(self.head2(p), dim=-1)
        d1 = (2 - 2 * u1 @ q1.t()).clamp_min(0)
        d2 = (2 - 2 * u2 @ q2.t()).clamp_min(0)
        return {"raw": raw, "h1": d1, "h2": d2}

    def lam(self) -> Tensor:
        return torch.sigmoid(self.lambda_logit)

    def activations(self, dist: Dict[str, Tensor], allowed: Tensor) -> Tensor:
        """a: [rows, K]. `allowed` [rows, K] excludes same-song prototypes."""
        c = self.config
        rows, k = dist["raw"].shape
        a = dist["raw"].new_zeros(rows, k)
        for d in range(2):
            idx = self.prototypes.indices(d)
            ok = allowed[:, idx]
            if c.activation == "differential":
                tau = self.log_tau[d].clamp(math.log(0.02), math.log(5.0)).exp()
                l1 = (-dist["h1"][:, idx] / tau[0]).masked_fill(~ok, -torch.inf)
                l2 = (-dist["h2"][:, idx] / tau[1]).masked_fill(~ok, -torch.inf)
                s1 = torch.softmax(l1, -1).nan_to_num(0.0)
                s2 = torch.softmax(l2, -1).nan_to_num(0.0)
                a[:, idx] = s1 - self.lam()[d] * s2
            elif c.activation == "protopnet_log":
                r = dist["raw"][:, idx]
                a[:, idx] = torch.log((r + 1) / (r + 1e-4)) * ok
            else:  # cosine_kernel: closest to the previous TRACE-MER source response
                tau = self.log_tau[d, 0].clamp(math.log(0.02), math.log(5.0)).exp()
                a[:, idx] = torch.exp(-dist["raw"][:, idx] / (2 * tau)) * ok
        return a

    # -------------------------------------------------------------- forward
    def forward(self, x: Tensor, times: Tensor, songs: Tensor, song_slices: Sequence[slice],
                select: Optional[Tensor] = None, decompose: bool = False,
                proto_embedding: Optional[Tensor] = None, return_embedding: bool = False):
        """x [N, bins, frames]; times [N] seconds; songs [N] song ids;
        song_slices: contiguous slices covering N, one per song.
        select [N] bool (optional focus mask): unselected rows are not encoded and
        contribute zero evidence; kernels renormalise over selected rows."""
        c = self.config
        n = x.shape[0]
        if select is None:
            select = torch.ones(n, dtype=torch.bool, device=x.device)
        z = x.new_zeros(n, c.dim)
        if bool(select.any()):
            z[select] = self.encoder(x[select])
        p = self.prototypes.embed(self.encoder) if proto_embedding is None else proto_embedding
        dist = self.distances(z, p)
        allowed = songs[:, None] != self.prototypes.source_song[None, :]
        a = self.activations(dist, allowed) * select[:, None]

        h = len(c.time_scales)
        k = self.prototypes.size
        local = a.new_zeros(n, h, k)
        glance = a.new_zeros(n, k)
        alpha = a.new_zeros(n, k)
        kernels = []
        for sl in song_slices:
            t = times[sl]
            sel = select[sl].to(a.dtype)
            ks = []
            for j, s in enumerate(c.time_scales):
                kern = gaussian_time_kernel(t, float(s)) * sel[None, :]
                kern = kern / kern.sum(-1, keepdim=True).clamp_min(1e-12)
                local[sl, j] = kern @ a[sl]
                ks.append(kern)
            kernels.append(torch.stack(ks))
            if c.glance:
                logits = (a[sl] / c.glance_tau).masked_fill(~select[sl, None], -torch.inf)
                al = torch.softmax(logits, 0).nan_to_num(0.0)      # agent attends over time
                alpha[sl] = al
                glance[sl] = (al * a[sl]).sum(0, keepdim=True).expand(sl.stop - sl.start, -1)

        w_loc = self.w_loc * self.readout_mask[:, None, :]
        w_glb = self.w_glb * self.readout_mask
        loc_terms = local[:, None] * w_loc[None]                    # [n, 2, h, k]
        glb_terms = glance[:, None] * w_glb[None]                   # [n, 2, k]
        logit = self.bias + loc_terms.sum((2, 3)) + glb_terms.sum(-1)
        residual = logit.new_zeros(n, 2)
        if c.residual:
            cap = c.residual_cap * torch.sigmoid(self.residual_scale_logit)
            residual = cap * torch.tanh(self.residual_net(z)) * select[:, None]
            logit = logit + residual
        if not decompose and not return_embedding:
            return logit
        out = dict(logit=logit, bias=self.bias, local_terms=loc_terms, glance_terms=glb_terms,
                   residual=residual, activation=a, local_features=local, glance_features=glance,
                   glance_attention=alpha, kernels=kernels, embedding=z, distances=dist,
                   allowed=allowed, select=select)
        return out

    def features(self, out: Dict[str, Tensor], d: int) -> Tensor:
        """Readout input f_d(t) (for ISDA): concatenated local and glance features."""
        idx = self.prototypes.indices(d)
        parts = [out["local_features"][:, :, idx].flatten(1)]
        if self.config.glance:
            parts.append(out["glance_features"][:, idx])
        return torch.cat(parts, -1)

    def readout_vector(self, d: int) -> Tensor:
        idx = self.prototypes.indices(d)
        parts = [self.w_loc[d][:, idx].flatten()]
        if self.config.glance:
            parts.append(self.w_glb[d, idx])
        return torch.cat(parts)

    def checkpoint(self, extra: Optional[Dict] = None) -> Dict:
        return dict(extra or {}, format=self.FORMAT, config=self.config.__dict__,
                    state={k: v.detach().cpu() for k, v in self.state_dict().items()})


def restore(payload: Dict, device: str = "cpu") -> GFDPModel:
    if payload.get("format") != GFDPModel.FORMAT:
        raise ValueError("not a GFDP-MER checkpoint")
    model = GFDPModel(ModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["state"], strict=True)
    if bool(model.prototypes.anchored):
        model.prototypes.vectors.requires_grad_(False)
    return model.to(device)
