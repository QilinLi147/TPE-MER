"""Leakage-safe joint-VA regression adapter for the historical DAMER model.

The server DAMER model projected every frequency node's fixed-width temporal
vector, exchanged evidence with bidirectional cross-view attention, pooled
each view, and predicted from their concatenation.  This adapter preserves
that parameterized path while changing only the experiment contract:

* inputs are complete-track batches from ``collate_track_sequences``;
* valence and arousal are predicted jointly for every segment;
* padding, invalid frames, and wholly missing views are explicitly masked;
* a fixed diagonal log-variance makes the output consumable by the common
  probabilistic regression runner without giving DAMER an extra uncertainty
  head.

The mounted corpora use either 44 or 87 frames per segment.  A parameter-free,
mask-aware adaptive pooling step maps the observed temporal samples to one
configurable width before the original per-node linear projection.  With the
default width of 87 and complete 87-frame inputs, this operation is the
identity and the default trainable parameter count is exactly that of the
server DAMER classifier (the two class logits are replaced one-for-one by
valence/arousal means).

Track-level split enforcement remains the responsibility of
``TrackSequenceDataset`` and the frozen manifests.  This model never reads
targets, split labels, track identifiers, or samples from another batch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DAMERRegressionPrediction:
    """Runner-compatible joint valence/arousal prediction and diagnostics."""

    mean: Tensor
    log_variance: Tensor
    mask: Tensor
    mel_mean: Tensor
    cochleagram_mean: Tensor
    mel_embedding: Tensor
    cochleagram_embedding: Tensor
    fusion_embedding: Tensor
    view_presence: Tensor

    @property
    def variance(self) -> Tensor:
        return self.log_variance.exp()


@dataclass(frozen=True)
class DAMERParameterCount:
    """Auditable parameter accounting for a DAMER adapter instance."""

    total: int
    trainable: int

    @property
    def frozen(self) -> int:
        return self.total - self.trainable


def count_damer_parameters(model: nn.Module) -> DAMERParameterCount:
    """Return total/trainable counts without treating buffers as parameters."""

    parameters = tuple(model.parameters())
    return DAMERParameterCount(
        total=sum(parameter.numel() for parameter in parameters),
        trainable=sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    )


class _MaskedTemporalResampler(nn.Module):
    """Resample only observed frames of every view/segment.

    The mask is shared by all frequency nodes of a segment.  Selecting valid
    frames before adaptive pooling keeps NaN/Inf padding outside every learned
    operation and makes predictions invariant to collate-time frame padding.
    """

    def __init__(self, frequency_bins: int, output_frames: int) -> None:
        super().__init__()
        if frequency_bins <= 0 or output_frames <= 0:
            raise ValueError("frequency_bins and output_frames must be positive")
        self.frequency_bins = int(frequency_bins)
        self.output_frames = int(output_frames)

    def forward(
        self,
        features: Tensor,
        frame_mask: Tensor,
        segment_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        self._validate_inputs(features, frame_mask, segment_mask)
        effective_mask = frame_mask & segment_mask.unsqueeze(-1)
        present = effective_mask.any(dim=-1)
        batch, time, _, _ = features.shape

        # Complete-frame segments share one pooling layout. Batch their independent
        # channel reductions; parameters, reduction width, order and RNG are unchanged.
        # Absent segments are excluded before finite checks and pooling, as before.
        if torch.equal(effective_mask, present.unsqueeze(-1).expand_as(effective_mask)):
            observed = features[present]
            if not torch.isfinite(observed).all():
                raise ValueError("observed feature frames must be finite")
            resampled = features.new_zeros(
                batch, time, self.frequency_bins, self.output_frames
            )
            if observed.shape[0]:
                resampled[present] = F.adaptive_avg_pool1d(observed, self.output_frames)
            return resampled, present

        # CopySlice is differentiable: gradients flow to selected real frames,
        # while absent views and padded frames receive exactly zero gradient.
        resampled = features.new_zeros(
            batch, time, self.frequency_bins, self.output_frames
        )
        for batch_index, time_index in present.nonzero(as_tuple=False).tolist():
            valid_frames = effective_mask[batch_index, time_index]
            observed = features[batch_index, time_index, :, valid_frames]
            if not torch.isfinite(observed).all():
                raise ValueError("observed feature frames must be finite")
            resampled[batch_index, time_index] = F.adaptive_avg_pool1d(
                observed, self.output_frames
            )
        return resampled, present

    def _validate_inputs(
        self,
        features: Tensor,
        frame_mask: Tensor,
        segment_mask: Tensor,
    ) -> None:
        if features.ndim != 4:
            raise ValueError("view features must have shape [batch,time,bins,frames]")
        if features.shape[2] != self.frequency_bins:
            raise ValueError(
                f"expected {self.frequency_bins} frequency bins, "
                f"got {features.shape[2]}"
            )
        if not features.is_floating_point():
            raise ValueError("view features must use a floating-point dtype")
        expected_frame_mask = (
            features.shape[0],
            features.shape[1],
            features.shape[3],
        )
        if frame_mask.shape != expected_frame_mask or frame_mask.dtype != torch.bool:
            raise ValueError("frame masks must be boolean [batch,time,frames]")
        if segment_mask.shape != features.shape[:2] or segment_mask.dtype != torch.bool:
            raise ValueError("segment mask must be boolean [batch,time]")
        if any(
            value.device != features.device for value in (frame_mask, segment_mask)
        ):
            raise ValueError("features and masks must be on the same device")


class DAMERCrossViewLayer(nn.Module):
    """The bidirectional cross-view Transformer block used by server DAMER."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.cross_attn_mel = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_cochleagram = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_mel_1 = nn.LayerNorm(embed_dim)
        self.norm_cochleagram_1 = nn.LayerNorm(embed_dim)
        self.ffn_mel = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ffn_cochleagram = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm_mel_2 = nn.LayerNorm(embed_dim)
        self.norm_cochleagram_2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mel: Tensor, cochleagram: Tensor) -> Tuple[Tensor, Tensor]:
        if mel.ndim != 3 or cochleagram.ndim != 3:
            raise ValueError("cross-view tokens must have shape [segments,nodes,dim]")
        if mel.shape[0] != cochleagram.shape[0] or mel.shape[-1] != cochleagram.shape[-1]:
            raise ValueError("cross-view token batches and embedding dimensions must match")

        mel_cross, _ = self.cross_attn_mel(
            mel, cochleagram, cochleagram, need_weights=False
        )
        cochleagram_cross, _ = self.cross_attn_cochleagram(
            cochleagram, mel, mel, need_weights=False
        )
        mel = self.norm_mel_1(mel + self.dropout(mel_cross))
        cochleagram = self.norm_cochleagram_1(
            cochleagram + self.dropout(cochleagram_cross)
        )
        mel = self.norm_mel_2(mel + self.dropout(self.ffn_mel(mel)))
        cochleagram = self.norm_cochleagram_2(
            cochleagram + self.dropout(self.ffn_cochleagram(cochleagram))
        )
        return mel, cochleagram


class DAMERJointVARegressor(nn.Module):
    """DAMER's dual-view mixer adapted to joint dynamic VA regression.

    The forward signature intentionally accepts the common runner's sequence,
    Broad, and routing options as no-ops.  This permits programmatic injection
    into ``run_training(..., model=baseline)`` without letting DAMER consume
    capabilities that it does not implement.
    """

    view_names = ("mel", "cochleagram")
    output_names = ("valence", "arousal")

    def __init__(
        self,
        mel_bins: int,
        cochleagram_bins: int,
        temporal_bins: int = 87,
        embed_dim: int = 128,
        fusion_dim: int = 256,
        num_heads: int = 4,
        cross_view_layers: int = 2,
        dropout: float = 0.1,
        fixed_log_variance: float = 0.0,
    ) -> None:
        super().__init__()
        if min(mel_bins, cochleagram_bins, temporal_bins, embed_dim, fusion_dim) <= 0:
            raise ValueError("all DAMER dimensions must be positive")
        if (
            isinstance(cross_view_layers, bool)
            or not isinstance(cross_view_layers, int)
            or cross_view_layers < 0
        ):
            raise ValueError("cross_view_layers must be a non-negative integer")
        if (
            isinstance(num_heads, bool)
            or not isinstance(num_heads, int)
            or num_heads <= 0
        ):
            raise ValueError("num_heads must be a positive integer")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(fixed_log_variance):
            raise ValueError("fixed_log_variance must be finite")

        self.mel_bins = int(mel_bins)
        self.cochleagram_bins = int(cochleagram_bins)
        self.temporal_bins = int(temporal_bins)
        self.embed_dim = int(embed_dim)
        self.fusion_dim = int(fusion_dim)

        self.temporal_resamplers = nn.ModuleDict(
            {
                "mel": _MaskedTemporalResampler(mel_bins, temporal_bins),
                "cochleagram": _MaskedTemporalResampler(
                    cochleagram_bins, temporal_bins
                ),
            }
        )
        # These are the original DAMER per-frequency-node projections.
        self.mel_projection = nn.Linear(temporal_bins, embed_dim)
        self.cochleagram_projection = nn.Linear(temporal_bins, embed_dim)
        self.cross_view_blocks = nn.ModuleList(
            DAMERCrossViewLayer(embed_dim, num_heads, dropout=dropout)
            for _ in range(cross_view_layers)
        )
        self.mel_pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, fusion_dim),
            nn.GELU(),
        )
        self.cochleagram_pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, fusion_dim),
            nn.GELU(),
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )

        # The original three 2-logit classification heads become three joint
        # [valence, arousal] mean heads, preserving the exact parameter count.
        self.mel_head = nn.Linear(fusion_dim, len(self.output_names))
        self.cochleagram_head = nn.Linear(fusion_dim, len(self.output_names))
        self.fusion_head = nn.Linear(fusion_dim, len(self.output_names))
        self.register_buffer(
            "fixed_log_variance",
            torch.full((len(self.output_names),), float(fixed_log_variance)),
        )

    def forward(
        self,
        batch: Mapping[str, object],
        *,
        solver: str = "auto",
        budget_fraction: Optional[float] = None,
        leave_one_out: bool = True,
        routing_straight_through: Optional[bool] = None,
        routing_mass_compensation: Optional[bool] = None,
        routing_policy: str = "learned",
        disconnect_fusion_broad: bool = False,
        disconnect_broad_router: bool = False,
        disconnect_router_feedback: bool = False,
    ) -> DAMERRegressionPrediction:
        # Explicitly ignored compatibility arguments.  Keeping them named
        # avoids silently accepting misspelled experiment controls.
        del (
            solver,
            budget_fraction,
            leave_one_out,
            routing_straight_through,
            routing_mass_compensation,
            routing_policy,
            disconnect_fusion_broad,
            disconnect_broad_router,
            disconnect_router_feedback,
        )

        features = _require_mapping(batch, "features")
        feature_masks = _require_mapping(batch, "feature_masks")
        segment_mask = batch.get("mask")
        if not isinstance(segment_mask, Tensor) or segment_mask.dtype != torch.bool:
            raise ValueError("batch['mask'] must be a boolean tensor [batch,time]")
        if segment_mask.ndim != 2:
            raise ValueError("batch['mask'] must have shape [batch,time]")
        if torch.any(segment_mask.sum(dim=1) <= 0):
            raise ValueError("every batch item must contain at least one real segment")

        temporal_features = {}
        presence = {}
        for name in self.view_names:
            view_features = features.get(name)
            view_mask = feature_masks.get(name)
            if not isinstance(view_features, Tensor) or not isinstance(view_mask, Tensor):
                raise ValueError(f"batch is missing tensor data for view {name!r}")
            temporal_features[name], presence[name] = self.temporal_resamplers[name](
                view_features, view_mask, segment_mask
            )

        view_presence = torch.stack(
            (presence["mel"], presence["cochleagram"]), dim=-1
        )
        if torch.any(segment_mask & ~view_presence.any(dim=-1)):
            raise ValueError("every real segment needs at least one observed view")

        mel_tokens = self.mel_projection(temporal_features["mel"])
        cochleagram_tokens = self.cochleagram_projection(
            temporal_features["cochleagram"]
        )
        mel_tokens = torch.where(
            presence["mel"].unsqueeze(-1).unsqueeze(-1),
            mel_tokens,
            torch.zeros_like(mel_tokens),
        )
        cochleagram_tokens = torch.where(
            presence["cochleagram"].unsqueeze(-1).unsqueeze(-1),
            cochleagram_tokens,
            torch.zeros_like(cochleagram_tokens),
        )

        # MHA cannot safely receive a segment whose entire key view is masked.
        # Run the historical cross-view path only where both views are present;
        # a single available view retains its projected tokens as a fallback.
        both_present = presence["mel"] & presence["cochleagram"]
        if torch.any(both_present):
            mel_both = mel_tokens[both_present]
            cochleagram_both = cochleagram_tokens[both_present]
            for block in self.cross_view_blocks:
                mel_both, cochleagram_both = block(mel_both, cochleagram_both)
            mel_tokens = mel_tokens.masked_scatter(
                both_present.unsqueeze(-1).unsqueeze(-1).expand_as(mel_tokens),
                mel_both.to(dtype=mel_tokens.dtype).reshape(-1),
            )
            cochleagram_tokens = cochleagram_tokens.masked_scatter(
                both_present.unsqueeze(-1).unsqueeze(-1).expand_as(
                    cochleagram_tokens
                ),
                cochleagram_both.to(dtype=cochleagram_tokens.dtype).reshape(-1),
            )

        mel_embedding = self.mel_pool(mel_tokens.mean(dim=-2))
        cochleagram_embedding = self.cochleagram_pool(
            cochleagram_tokens.mean(dim=-2)
        )
        mel_embedding = torch.where(
            presence["mel"].unsqueeze(-1),
            mel_embedding,
            torch.zeros_like(mel_embedding),
        )
        cochleagram_embedding = torch.where(
            presence["cochleagram"].unsqueeze(-1),
            cochleagram_embedding,
            torch.zeros_like(cochleagram_embedding),
        )
        fusion_embedding = self.fusion_projection(
            torch.cat((mel_embedding, cochleagram_embedding), dim=-1)
        )
        fusion_embedding = torch.where(
            segment_mask.unsqueeze(-1),
            fusion_embedding,
            torch.zeros_like(fusion_embedding),
        )

        mel_mean = self.mel_head(mel_embedding)
        cochleagram_mean = self.cochleagram_head(cochleagram_embedding)
        mean = self.fusion_head(fusion_embedding)
        mel_mean = torch.where(
            presence["mel"].unsqueeze(-1), mel_mean, torch.zeros_like(mel_mean)
        )
        cochleagram_mean = torch.where(
            presence["cochleagram"].unsqueeze(-1),
            cochleagram_mean,
            torch.zeros_like(cochleagram_mean),
        )
        mean = torch.where(
            segment_mask.unsqueeze(-1), mean, torch.zeros_like(mean)
        )
        fixed_log_variance = self.fixed_log_variance.to(
            device=mean.device, dtype=mean.dtype
        ).view(1, 1, -1)
        log_variance = torch.where(
            segment_mask.unsqueeze(-1),
            fixed_log_variance.expand_as(mean),
            torch.zeros_like(mean),
        )
        return DAMERRegressionPrediction(
            mean=mean,
            log_variance=log_variance,
            mask=segment_mask,
            mel_mean=mel_mean,
            cochleagram_mean=cochleagram_mean,
            mel_embedding=mel_embedding,
            cochleagram_embedding=cochleagram_embedding,
            fusion_embedding=fusion_embedding,
            view_presence=view_presence,
        )


def _require_mapping(
    batch: Mapping[str, object], key: str
) -> Mapping[str, object]:
    value = batch.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"batch[{key!r}] must be a modality mapping")
    return value


__all__ = [
    "DAMERCrossViewLayer",
    "DAMERJointVARegressor",
    "DAMERParameterCount",
    "DAMERRegressionPrediction",
    "count_damer_parameters",
]
