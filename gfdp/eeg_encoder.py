"""v1.6.1: EEG window encoder for GFDP (and its black-box twin).

v1.6 used the music DualViewEncoder unchanged on EEG rows. It convolves along the row axis,
which for EEG is the electrode index (not a spatial neighbourhood), and its per-bin temporal
LayerNorm removes each electrode's band level. Both the GFDP model and its twin stayed near
chance on DEAP. This encoder treats a row as a set of electrodes:

  x [rows, 2C, F]  (view 1 = rows 0..C-1, view 2 = rows C..2C-1, F features per electrode/view)
    -> fixed per-feature standardisation with FIT-row statistics (buffers set once before training;
       a constant affine map, so every window is still encoded on its own)
    -> per electrode: concat(view1, view2) [2F] -> shared MLP -> hidden, + learned electrode embedding
    -> electrode dropout (training only; zero = the fit mean after standardisation)
    -> L blocks over electrodes, either
         "graph": Chebyshev graph convolution (order K) on one learnable non-negative symmetric
                  adjacency (DGCNN-style), pre-LayerNorm, residual
         "attn":  pre-norm transformer encoder layer (multi-head self-attention over electrodes)
    -> concat(mean, max) over electrodes -> Linear -> LayerNorm -> embedding e [rows, dim]
    -> same centring and unit normalisation as DualViewEncoder (z = normalize(e - centre))

No cross-row interaction: the embedding of a window depends on that window only, as required by
prototype matching and by the certificates.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

FAMILIES = ("graph", "attn")


class ChebBlock(nn.Module):
    def __init__(self, hidden: int, K: int, dropout: float) -> None:
        super().__init__()
        self.K = K
        self.norm = nn.LayerNorm(hidden)
        self.theta = nn.Parameter(torch.randn(K, hidden, hidden) / math.sqrt(hidden * K))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, T: list) -> Tensor:
        u = self.norm(h)
        out = sum(torch.einsum("ij,bjf,fh->bih", T[k], u, self.theta[k]) for k in range(self.K))
        return h + self.dropout(F.gelu(out))


class EEGEncoder(nn.Module):
    def __init__(self, electrodes: int, features: int, *, family: str = "graph", hidden: int = 32,
                 layers: int = 1, dim: int = 64, dropout: float = 0.25, electrode_dropout: float = 0.1,
                 cheb_k: int = 3, heads: int = 4, pooling: str = "meanmax") -> None:
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        self.C, self.F, self.family = electrodes, features, family
        if pooling not in ("meanmax", "flatten"): raise ValueError("unknown electrode pooling")
        self.pooling = pooling
        self.electrode_dropout = float(electrode_dropout)
        self.register_buffer("feat_mean", torch.zeros(2 * electrodes, features))
        self.register_buffer("feat_std", torch.ones(2 * electrodes, features))
        self.register_buffer("standardised", torch.tensor(False))
        self.pre = nn.Sequential(nn.Linear(2 * features, hidden), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.randn(electrodes, hidden) * 0.02)
        if family == "graph":
            self.adj = nn.Parameter(torch.rand(electrodes, electrodes) * 0.1 + torch.eye(electrodes))
            self.blocks = nn.ModuleList(ChebBlock(hidden, cheb_k, dropout) for _ in range(layers))
            self.cheb_k = cheb_k
        else:
            if hidden % heads:
                raise ValueError("hidden must be divisible by heads")
            self.blocks = nn.ModuleList(
                nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout, activation="gelu",
                                           batch_first=True, norm_first=True) for _ in range(layers))
        self.out_norm = nn.LayerNorm(hidden)
        self.projection = nn.Sequential(nn.Linear((electrodes if pooling == "flatten" else 2) * hidden, dim), nn.LayerNorm(dim))
        self.dim = dim
        self.register_buffer("centre", torch.zeros(dim))
        self.centre_momentum = 0.05

    # ------------------------------------------------------------ fit statistics
    @torch.no_grad()
    def init_standardizer(self, batches) -> dict:
        """Per-feature mean/std over all FIT rows (called once, before training)."""
        s = s2 = None
        n = 0
        for x in batches:
            x = x.double()
            s = x.sum(0) if s is None else s + x.sum(0)
            s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
            n += x.shape[0]
        mean = s / n
        std = (s2 / n - mean * mean).clamp_min(0).sqrt().clamp_min(1e-3)
        self.feat_mean.copy_(mean.to(self.feat_mean.dtype))
        self.feat_std.copy_(std.to(self.feat_std.dtype))
        self.standardised.fill_(True)
        return dict(rows=n, mean_abs_mean=float(mean.abs().mean()), mean_std=float(std.mean()))

    def _laplacian_powers(self, dtype, device):
        A = F.relu(self.adj)
        A = (A + A.t()) / 2
        d = A.sum(1).clamp_min(1e-6).pow(-0.5)
        L = torch.eye(self.C, device=device, dtype=dtype) - d[:, None] * A * d[None, :]
        T = [torch.eye(self.C, device=device, dtype=dtype), L]
        for _ in range(2, self.cheb_k):
            T.append(2 * L @ T[-1] - T[-2])
        return T[:self.cheb_k]

    def _fuse(self, x: Tensor):
        if x.shape[1:] != (2 * self.C, self.F):
            raise ValueError(f"expected [*, {2 * self.C}, {self.F}], got {tuple(x.shape)}")
        x = (x - self.feat_mean) / self.feat_std
        u = torch.cat([x[:, :self.C], x[:, self.C:]], -1)              # [rows, C, 2F]
        if self.training and self.electrode_dropout > 0:
            keep = torch.rand(u.shape[0], self.C, 1, device=u.device) >= self.electrode_dropout
            u = u * keep.to(u.dtype)
        h = self.pre(u) + self.pos
        if self.family == "graph":
            T = self._laplacian_powers(h.dtype, h.device)
            for block in self.blocks:
                h = block(h, T)
        else:
            for block in self.blocks:
                h = block(h)
        h = self.out_norm(h)
        pooled = h.flatten(1) if self.pooling == "flatten" else torch.cat([h.mean(1), h.amax(1)], -1)
        e = self.projection(pooled)
        return e, {}

    # ------------------------------------------------- same interface as DualViewEncoder
    @torch.no_grad()
    def init_centre(self, batches) -> float:
        self.centre.zero_()
        total, count, last = 0.0, 0, None
        for x in batches:
            e, _ = self._fuse(x)
            total = total + e.sum(0); count += e.shape[0]; last = x
        self.centre.copy_(total / max(count, 1))
        z = self.forward(last)
        return float((z @ z.t()).mean())

    def forward(self, x: Tensor, return_views: bool = False):
        e, parts = self._fuse(x)
        if self.training and self.centre_momentum > 0:
            with torch.no_grad():
                self.centre.mul_(1 - self.centre_momentum).add_(self.centre_momentum * e.mean(0))
        z = F.normalize(e - self.centre, dim=-1, eps=1e-8)
        if return_views:
            return z, parts
        return z
