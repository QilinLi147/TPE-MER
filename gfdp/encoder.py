"""Shared dual-view local encoder (slimmed UDDABLSv2 matched-v1 family).

Kept from the original UDDABLSv2 matched-v1 encoder (source SHA ee9d6837...):
  * frequency bins are the convolution sequence, frames are per-bin features;
  * depthwise dilated residual blocks (dilations 1/2/4) with per-segment LayerNorm;
  * DenseNet-style concatenation of every block depth, squeeze-excitation and
    deterministic hierarchical pooling (1/4/8 bins);
  * the learned element-wise view gate between the mel and second view.

Changed on purpose (see DESIGN_ZH.md section 4.1):
  * ONE encoder shared by valence and arousal (old releases trained two
    28.5M-parameter encoders, one per dimension, on 416 songs);
  * native 44 frames are used directly (the old adapter averaged 44 -> 87,
    i.e. duplicated frames, which only inflated the projection width);
  * conv width 512 -> 128 and embedding 256 -> 128 (about 1.5M parameters);
  * stochastic depth / drop-path on every residual block (Huang et al., ECCV 2016);
  * spectral-envelope re-injection: the original `pre = LayerNorm(frames)` normalises
    every frequency bin over time, which deletes each bin's mean level and spread
    (the spectral envelope / band energy).  We keep that temporal-shape path and
    append the per-bin mean and log-std as two extra features per bin, so band
    energy is visible again.  (Found by the synthetic smoke test: a pure
    band-energy signal was invisible to the original encoder.)

No BatchNorm, no cross-song state: the embedding of a segment depends only on
that segment, which is what makes prototype matching auditable.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def deterministic_adaptive_avg_pool1d(inputs: Tensor, output_size: int) -> Tensor:
    """Matmul form of adaptive average pooling (deterministic CUDA backward)."""
    length = inputs.shape[-1]
    if length == output_size:
        return inputs
    weights = inputs.new_zeros(output_size, length)
    for i in range(output_size):
        start = (i * length) // output_size
        end = ((i + 1) * length + output_size - 1) // output_size
        weights[i, start:end] = 1.0 / float(end - start)
    return torch.matmul(inputs, weights.t())


class DropPath(nn.Module):
    """Stochastic depth: drop a whole residual branch per sample (train only)."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop-path probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(self, residual: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return residual
        keep = 1.0 - self.probability
        shape = (residual.shape[0],) + (1,) * (residual.ndim - 1)
        mask = residual.new_empty(shape).bernoulli_(keep)
        return residual * mask / keep


class ResidualFrequencyBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float, drop_path: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 3, padding=dilation,
                                   dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        r = self.pointwise(self.depthwise(x))
        r = F.gelu(self.norm(r.transpose(1, 2)).transpose(1, 2))
        return x + self.drop_path(self.dropout(r))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(),
                                nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class ViewEncoder(nn.Module):
    """[rows, bins, frames] -> [rows, dim]."""

    def __init__(self, bins: int, frames: int, channels: int, dim: int, *,
                 dilations=(1, 2, 4), pool_bins=(1, 4, 8), dropout: float = 0.1,
                 drop_path: float = 0.1, envelope: bool = True) -> None:
        super().__init__()
        self.bins, self.frames, self.envelope = bins, frames, envelope
        self.shape_norm = nn.LayerNorm(frames)
        self.pre = nn.Sequential(nn.Linear(frames + (2 if envelope else 0), channels),
                                 nn.GELU(), nn.Dropout(dropout))
        # linearly increasing drop-path rate with depth, as in the original paper
        rates = [drop_path * (i + 1) / len(dilations) for i in range(len(dilations))]
        self.blocks = nn.ModuleList(ResidualFrequencyBlock(channels, d, dropout, p)
                                    for d, p in zip(dilations, rates))
        stacked = channels * len(dilations)
        self.se = SqueezeExcite(stacked)
        self.pool_bins = tuple(pool_bins)
        # The original projection ended in GELU; its mostly-positive outputs made all
        # unit-normalised embeddings nearly parallel (mean cosine ~0.99 at init), a
        # common-mode that swamps prototype similarity. We end at LayerNorm instead.
        self.projection = nn.Sequential(nn.Linear(sum(pool_bins) * stacked, dim),
                                        nn.LayerNorm(dim))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1:] != (self.bins, self.frames):
            raise ValueError(f"expected [*, {self.bins}, {self.frames}], got {tuple(x.shape)}")
        shape = self.shape_norm(x)                    # original per-bin temporal shape
        if self.envelope:
            level = x.mean(-1, keepdim=True)
            spread = torch.log(x.var(-1, unbiased=False, keepdim=True) + 1e-4) * 0.5
            shape = torch.cat([shape, level, spread], -1)
        h = self.pre(shape).transpose(1, 2)           # [rows, channels, bins]
        depths = []
        for block in self.blocks:
            h = block(h)
            depths.append(h)
        h = self.se(torch.cat(depths, 1))             # dense multi-depth reuse
        pooled = [deterministic_adaptive_avg_pool1d(h, b).flatten(1) for b in self.pool_bins]
        return self.projection(torch.cat(pooled, -1))


class DualViewEncoder(nn.Module):
    """x: [rows, mel_bins + second_bins, frames] -> unit-norm z: [rows, dim]."""

    VIEWS = ("both", "mel", "second")

    def __init__(self, mel_bins: int = 128, second_bins: int = 84, frames: int = 44,
                 channels: int = 128, dim: int = 128, dropout: float = 0.1,
                 drop_path: float = 0.1, envelope: bool = True, views: str = "both") -> None:
        super().__init__()
        if views not in self.VIEWS:
            raise ValueError(f"views must be one of {self.VIEWS}")
        # v1.4: single-view arms ("mel" / "second") build only that view's encoder and
        # no gate; the input keeps all 212 channels and the unused view is ignored.
        # views="both" constructs exactly the v1.1-v1.3 modules (checkpoints unchanged).
        self.mel_bins, self.second_bins, self.views = mel_bins, second_bins, views
        if views in ("both", "mel"):
            self.mel = ViewEncoder(mel_bins, frames, channels, dim, dropout=dropout,
                                   drop_path=drop_path, envelope=envelope)
        if views in ("both", "second"):
            self.second = ViewEncoder(second_bins, frames, channels, dim, dropout=dropout,
                                      drop_path=drop_path, envelope=envelope)
        if views == "both":
            self.view_gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(),
                                           nn.Linear(dim, dim), nn.Sigmoid())
        self.dim = dim
        self.register_buffer("centre", torch.zeros(dim))
        self.centre_momentum = 0.05

    @torch.no_grad()
    def init_centre(self, batches) -> float:
        """Set the centre to the mean un-normalised TRAIN embedding; returns the
        mean pairwise cosine after centring on the last batch (diagnostic)."""
        self.centre.zero_()
        total, count, last = 0.0, 0, None
        for x in batches:
            e, _ = self._fuse(x)
            total = total + e.sum(0); count += e.shape[0]; last = x
        self.centre.copy_(total / max(count, 1))
        z = self.forward(last)
        return float((z @ z.t()).mean())

    def _fuse(self, x: Tensor):
        """Un-normalised fused embedding e and the per-view parts."""
        parts = {}
        if self.views in ("both", "mel"):
            parts["mel"] = self.mel(x[:, :self.mel_bins])
        if self.views in ("both", "second"):
            parts["second"] = self.second(x[:, self.mel_bins:self.mel_bins + self.second_bins])
        if self.views == "both":
            gate = self.view_gate(torch.cat([parts["mel"], parts["second"]], -1))
            parts["gate"] = gate
            return gate * parts["mel"] + (1 - gate) * parts["second"], parts
        return parts[self.views], parts

    def forward(self, x: Tensor, return_views: bool = False):
        e, parts = self._fuse(x)
        # Common-mode removal: subtract an input-independent centre before
        # normalising. It is initialised from the mean TRAIN embedding
        # (`init_centre`) and tracked by a detached EMA of TRAIN batch means while
        # training (like BatchNorm running statistics, but the forward pass always
        # uses the stored buffer, never the current batch). At inference it is a
        # frozen constant: no cross-song or cross-batch dependence.
        if self.training and self.centre_momentum > 0:
            with torch.no_grad():
                self.centre.mul_(1 - self.centre_momentum).add_(self.centre_momentum * e.mean(0))
        z = F.normalize(e - self.centre, dim=-1, eps=1e-8)
        if return_views:
            return z, parts
        return z


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
