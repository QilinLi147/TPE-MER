"""Losses: soft-label BCE, ISDA upper bound, prototype geometry, sparsity, KD.

ISDA (Wang, Pan, Song, Zhang, Wu, Huang; NeurIPS 2019 / TPAMI 2021) adapted to a
binary logistic readout over prototype-evidence features f:
    augment f ~ N(f, lambda * Sigma_c), c = hard class of the row;
    E[log(1+exp(-s(l + w.delta)))] <= log(1 + exp(-s l + lambda/2 w^T Sigma_c w))
(Jensen; log(1+u) is concave in u). The bound is >= plain BCE and adds no
inference cost; the deployed model is unchanged.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def soft_targets(y_continuous: Tensor, sigma: float) -> Tensor:
    """Hard label = y > 0 (project rule). sigma > 0 softens near the threshold."""
    if sigma <= 0:
        return (y_continuous > 0).to(y_continuous.dtype)
    return torch.sigmoid(y_continuous / sigma)


class ClassCovariance:
    """Running class-conditional covariance of readout features (per dimension)."""

    def __init__(self, feature_dim: int, device=None, dtype=torch.float32) -> None:
        self.count = torch.zeros(2, device=device, dtype=dtype)
        self.mean = torch.zeros(2, feature_dim, device=device, dtype=dtype)
        self.cov = torch.zeros(2, feature_dim, feature_dim, device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, f: Tensor, hard: Tensor) -> None:
        f = f.detach().to(self.mean.dtype)
        for c in (0, 1):
            fc = f[hard == c]
            n = fc.shape[0]
            if n == 0:
                continue
            m = fc.mean(0)
            centred = fc - m
            cov = centred.t() @ centred / n
            total = self.count[c] + n
            w_old, w_new = self.count[c] / total, n / total
            delta = m - self.mean[c]
            self.cov[c] = w_old * self.cov[c] + w_new * cov + w_old * w_new * torch.outer(delta, delta)
            self.mean[c] = w_old * self.mean[c] + w_new * m
            self.count[c] = total


def isda_bce(logit: Tensor, target: Tensor, hard: Tensor, w: Tensor,
             cov: Optional[ClassCovariance], strength: float) -> Tensor:
    """Soft-label BCE with the ISDA quadratic margin. strength=0 -> plain BCE."""
    if cov is None or strength <= 0:
        return F.binary_cross_entropy_with_logits(logit, target)
    q = 0.5 * strength * torch.einsum("i,cij,j->c", w, cov.cov.to(w.dtype), w)   # [2]
    qc = q[hard.long()]
    # positive part sees logit - q (harder), negative part sees logit + q (harder)
    pos = F.softplus(-(logit - qc))
    neg = F.softplus(logit + qc)
    return (target * pos + (1 - target) * neg).mean()


def prototype_geometry(raw_dist: Tensor, proto_dim: Tensor, proto_class: Tensor,
                       hard: Tensor, allowed: Tensor) -> Dict[str, Tensor]:
    """ProtoPNet cluster / separation on unit-sphere squared distances.
    raw_dist [n, K]; hard [n, 2] (0/1 per dimension)."""
    clst, sep = [], []
    big = raw_dist.new_tensor(10.0)
    for d in range(2):
        in_dim = (proto_dim == d)[None, :] & allowed
        same = in_dim & (proto_class[None, :] == hard[:, d:d + 1])
        other = in_dim & (proto_class[None, :] != hard[:, d:d + 1])
        clst.append(torch.where(same, raw_dist, big).min(-1).values.mean())
        sep.append(torch.where(other, raw_dist, big).min(-1).values.clamp(max=4.0).mean())
    return {"cluster": torch.stack(clst).mean(), "separation": -torch.stack(sep).mean()}


def prototype_diversity(p: Tensor, proto_dim: Tensor, margin: float = 0.5) -> Tensor:
    """Penalise near-duplicate prototypes inside a dimension (cosine > margin)."""
    loss = p.new_zeros(())
    for d in range(2):
        q = p[proto_dim == d]
        sim = q @ q.t()
        off = sim - torch.eye(q.shape[0], device=q.device, dtype=q.dtype) * 2
        loss = loss + F.relu(off - margin).square().mean()
    return loss / 2


def wrong_class_l1(model) -> Tensor:
    """ProtoPNet last-layer sparsity: a 'high' prototype should not push toward
    'low' and vice versa (sign mismatch with its class)."""
    sign = (model.prototypes.proto_class.float() * 2 - 1)
    total = model.w_loc.new_zeros(())
    for d in range(2):
        mask = model.readout_mask[d]
        wl = model.w_loc[d] * mask[None]
        wg = model.w_glb[d] * mask
        total = total + F.relu(-wl * sign[None]).sum() + F.relu(-wg * sign).sum()
    return total


def distillation(logit: Tensor, teacher_prob: Optional[Tensor]) -> Tensor:
    if teacher_prob is None:
        return logit.new_zeros(())
    return F.binary_cross_entropy_with_logits(logit, teacher_prob)


def embedding_variance(z: Tensor, gamma: float = 0.5) -> Tensor:
    """VICReg-style variance hinge (training only): keeps every embedding
    coordinate's batch std above gamma / sqrt(dim) and prevents the collapsed
    constant-embedding solution. Not used at inference."""
    std = torch.sqrt(z.var(0) + 1e-6)
    return F.relu(gamma / z.shape[1] ** 0.5 - std).mean() * z.shape[1] ** 0.5
