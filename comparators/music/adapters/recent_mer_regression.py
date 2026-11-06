"""Leakage-safe, common-protocol adapters for published DMER architectures.

The mounted corpora contain pre-computed Mel spectrograms and cochleagrams,
not raw audio.  Consequently these models are *architecture-matched adapters*,
not claims of bit-for-bit reproduction of each paper's feature pipeline.  Adapters use a shared song-split and source-validation protocol.

Implemented families:

``crnn``
    A compact Mel CNN followed by a bidirectional GRU (Malik et al., 2017).
``mcrnn``
    Independent Mel/cochleagram CNN views followed by recurrent fusion (He
    and Ferguson, 2020).
``damff``
    Temporal-frequency multi-scale Mel convolutions, channel/spatial dual
    attention, and a bidirectional LSTM (Zhang et al., ISMIR 2023).
``dsaml``
    Dual-view feature fusion plus shared local/global temporal attention and a
    bidirectional LSTM, matching the non-meta DMER path of DSAML (AAAI 2025).
``cnn_transformer``
    A deliberately generic spectral-CNN/temporal-Transformer hybrid used as a
    2026-era architecture control.  It must not be labelled an exact PVAN
    reproduction because the required CQT and theory-guidance features are not
    present on the server.

No adapter reads targets, track identifiers, split labels, or another batch.
Mask-aware per-segment standardization makes invalid/padded values inert and
avoids cross-track batch statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


BASELINE_NAMES = ("crnn", "mcrnn", "damff", "dsaml", "cnn_transformer")


@dataclass(frozen=True)
class MERBaselinePrediction:
    """Minimal prediction contract consumed by the common regression runner."""

    mean: Tensor
    log_variance: Tensor
    mask: Tensor
    fusion_embedding: Tensor
    view_presence: Tensor

    @property
    def variance(self) -> Tensor:
        return self.log_variance.exp()


def _require_tensor_mapping(
    batch: Mapping[str, object], key: str
) -> Mapping[str, Tensor]:
    value = batch.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"batch[{key!r}] must be a modality mapping")
    if not all(isinstance(item, Tensor) for item in value.values()):
        raise ValueError(f"batch[{key!r}] values must be tensors")
    return value  # type: ignore[return-value]


def _batch_inputs(
    batch: Mapping[str, object],
) -> Tuple[Mapping[str, Tensor], Mapping[str, Tensor], Tensor]:
    features = _require_tensor_mapping(batch, "features")
    feature_masks = _require_tensor_mapping(batch, "feature_masks")
    segment_mask = batch.get("mask")
    if not isinstance(segment_mask, Tensor):
        raise ValueError("batch['mask'] must be a tensor")
    if segment_mask.ndim != 2 or segment_mask.dtype != torch.bool:
        raise ValueError("batch['mask'] must be boolean [batch,time]")
    if torch.any(segment_mask.sum(dim=1) <= 0):
        raise ValueError("every batch item must contain at least one segment")
    for name in ("mel", "cochleagram"):
        if name not in features or name not in feature_masks:
            raise ValueError(f"batch is missing the {name!r} view")
        view = features[name]
        mask = feature_masks[name]
        if view.ndim != 4:
            raise ValueError(f"{name} features must be [batch,time,bins,frames]")
        if mask.shape != (view.shape[0], view.shape[1], view.shape[3]):
            raise ValueError(f"{name} frame mask shape is invalid")
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} frame mask must be boolean")
        if view.shape[:2] != segment_mask.shape:
            raise ValueError(f"{name} and segment masks disagree")
    return features, feature_masks, segment_mask


def _standardize_view(
    features: Tensor, frame_mask: Tensor, segment_mask: Tensor
) -> Tuple[Tensor, Tensor]:
    """Return finite, segment-local standardized features and view presence."""

    valid_frames = frame_mask & segment_mask.unsqueeze(-1)
    presence = valid_frames.any(dim=-1)
    valid = valid_frames.unsqueeze(-2).expand_as(features)
    safe = torch.where(valid, features, torch.zeros_like(features)).float()
    count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1).float()
    mean = safe.sum(dim=(-2, -1), keepdim=True) / count
    centered = torch.where(valid, safe - mean, torch.zeros_like(safe))
    variance = centered.square().sum(dim=(-2, -1), keepdim=True) / count
    standardized = centered * torch.rsqrt(variance.clamp_min(1e-5))
    standardized = standardized.clamp(-8.0, 8.0)
    standardized = torch.where(valid, standardized, torch.zeros_like(standardized))
    if not torch.isfinite(standardized).all():
        raise ValueError("observed view features must be finite")
    return standardized, presence


class _SegmentCNN(nn.Module):
    """Track-independent 2-D segment encoder without BatchNorm leakage."""

    def __init__(self, output_dim: int, *, width: int = 24, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, width * 2),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 2 * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, view: Tensor, presence: Tensor) -> Tensor:
        batch, time = view.shape[:2]
        features = self.network(view.reshape(batch * time, 1, *view.shape[2:]))
        encoded = self.projection(_deterministic_quadrant_pool(features))
        encoded = encoded.reshape(batch, time, -1)
        return torch.where(presence.unsqueeze(-1), encoded, torch.zeros_like(encoded))


def _deterministic_quadrant_pool(inputs: Tensor) -> Tensor:
    """Pool a feature map to 2x2 without CUDA adaptive-pool atomics."""

    if inputs.ndim != 4 or min(inputs.shape[-2:]) < 2:
        raise ValueError("quadrant pooling needs [batch,channels,height,width] >= 2x2")
    height_cut = inputs.shape[-2] // 2
    width_cut = inputs.shape[-1] // 2
    rows = ((0, height_cut), (height_cut, inputs.shape[-2]))
    columns = ((0, width_cut), (width_cut, inputs.shape[-1]))
    quadrants = [
        inputs[:, :, row_start:row_end, column_start:column_end].mean(
            dim=(-2, -1)
        )
        for row_start, row_end in rows
        for column_start, column_end in columns
    ]
    return torch.stack(quadrants, dim=-1).reshape(inputs.shape[0], inputs.shape[1], 2, 2)


def _packed_recurrent(module: nn.Module, inputs: Tensor, mask: Tensor) -> Tensor:
    lengths = mask.sum(dim=1).to(dtype=torch.long).cpu()
    packed = nn.utils.rnn.pack_padded_sequence(
        inputs, lengths, batch_first=True, enforce_sorted=False
    )
    packed_output, _ = module(packed)
    output, _ = nn.utils.rnn.pad_packed_sequence(
        packed_output, batch_first=True, total_length=inputs.shape[1]
    )
    return torch.where(mask.unsqueeze(-1), output, torch.zeros_like(output))


class _ProbabilisticHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean = nn.Linear(input_dim, 2)
        self.log_variance = nn.Linear(input_dim, 2)

    def forward(self, inputs: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = self.shared(inputs)
        active = mask.unsqueeze(-1)
        return (
            torch.where(active, self.mean(hidden), torch.zeros_like(inputs[..., :2])),
            torch.where(
                active,
                self.log_variance(hidden),
                torch.zeros_like(inputs[..., :2]),
            ),
        )


class _RunnerCompatibleRegressor(nn.Module):
    """Named-argument compatibility layer for the shared runner."""

    view_names = ("mel", "cochleagram")
    output_names = ("valence", "arousal")

    def _predict(self, batch: Mapping[str, object]) -> MERBaselinePrediction:
        raise NotImplementedError

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
    ) -> MERBaselinePrediction:
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
        return self._predict(batch)

    @staticmethod
    def _inputs(batch: Mapping[str, object]):
        features, masks, segment_mask = _batch_inputs(batch)
        mel, mel_presence = _standardize_view(
            features["mel"], masks["mel"], segment_mask
        )
        coch, coch_presence = _standardize_view(
            features["cochleagram"], masks["cochleagram"], segment_mask
        )
        presence = torch.stack((mel_presence, coch_presence), dim=-1)
        if torch.any(segment_mask & ~presence.any(dim=-1)):
            raise ValueError("every real segment needs at least one observed view")
        return mel, coch, mel_presence, coch_presence, presence, segment_mask


class CRNNJointVARegressor(_RunnerCompatibleRegressor):
    """Matched log-Mel CRNN baseline."""

    def __init__(self, model_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        hidden = max(32, model_dim // 2)
        self.mel_encoder = _SegmentCNN(model_dim, dropout=dropout)
        self.recurrent = nn.GRU(
            model_dim, hidden, batch_first=True, bidirectional=True
        )
        self.head = _ProbabilisticHead(2 * hidden, dropout)

    def _predict(self, batch: Mapping[str, object]) -> MERBaselinePrediction:
        mel, _, mel_p, coch_p, presence, mask = self._inputs(batch)
        embedding = self.mel_encoder(mel, mel_p)
        temporal = _packed_recurrent(self.recurrent, embedding, mask)
        mean, log_variance = self.head(temporal, mask)
        return MERBaselinePrediction(mean, log_variance, mask, temporal, presence)


class MCRNNJointVARegressor(_RunnerCompatibleRegressor):
    """Matched dual-view MCRNN baseline."""

    def __init__(self, model_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        view_dim = max(48, model_dim // 2)
        hidden = max(32, model_dim // 2)
        self.mel_encoder = _SegmentCNN(view_dim, dropout=dropout)
        self.coch_encoder = _SegmentCNN(view_dim, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(2 * view_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.recurrent = nn.LSTM(
            model_dim, hidden, batch_first=True, bidirectional=True
        )
        self.head = _ProbabilisticHead(2 * hidden, dropout)

    def _predict(self, batch: Mapping[str, object]) -> MERBaselinePrediction:
        mel, coch, mel_p, coch_p, presence, mask = self._inputs(batch)
        embedding = self.fusion(
            torch.cat(
                (self.mel_encoder(mel, mel_p), self.coch_encoder(coch, coch_p)),
                dim=-1,
            )
        )
        embedding = torch.where(mask.unsqueeze(-1), embedding, torch.zeros_like(embedding))
        temporal = _packed_recurrent(self.recurrent, embedding, mask)
        mean, log_variance = self.head(temporal, mask)
        return MERBaselinePrediction(mean, log_variance, mask, temporal, presence)


class _DAMFFSegmentEncoder(nn.Module):
    """Multi-scale convolution plus channel/spatial attention."""

    def __init__(self, output_dim: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        kernels = ((3, 1), (1, 3), (3, 3))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(16, 16, kernel_size=k, padding=(k[0] // 2, k[1] // 2)),
                    nn.GroupNorm(4, 16),
                    nn.GELU(),
                )
                for k in kernels
            ]
        )
        channels = 48
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, 12, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(12, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid()
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, view: Tensor, presence: Tensor) -> Tensor:
        batch, time = view.shape[:2]
        stem = self.stem(view.reshape(batch * time, 1, *view.shape[2:]))
        multi_scale = torch.cat(tuple(branch(stem) for branch in self.branches), dim=1)
        channel_descriptor = multi_scale.mean(dim=(-2, -1), keepdim=True)
        channel = multi_scale * self.channel_attention(channel_descriptor)
        spatial_descriptor = torch.cat(
            (channel.mean(dim=1, keepdim=True), channel.amax(dim=1, keepdim=True)),
            dim=1,
        )
        attended = channel * self.spatial_attention(spatial_descriptor)
        embedding = self.projection(_deterministic_quadrant_pool(attended)).reshape(
            batch, time, -1
        )
        return torch.where(presence.unsqueeze(-1), embedding, torch.zeros_like(embedding))


class DAMFFJointVARegressor(_RunnerCompatibleRegressor):
    """Matched DAMFF baseline on the mounted log-Mel representation."""

    def __init__(self, model_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        hidden = max(32, model_dim // 2)
        self.encoder = _DAMFFSegmentEncoder(model_dim, dropout)
        self.recurrent = nn.LSTM(
            model_dim, hidden, batch_first=True, bidirectional=True
        )
        self.head = _ProbabilisticHead(2 * hidden, dropout)

    def _predict(self, batch: Mapping[str, object]) -> MERBaselinePrediction:
        mel, _, mel_p, coch_p, presence, mask = self._inputs(batch)
        embedding = self.encoder(mel, mel_p)
        temporal = _packed_recurrent(self.recurrent, embedding, mask)
        mean, log_variance = self.head(temporal, mask)
        return MERBaselinePrediction(mean, log_variance, mask, temporal, presence)


class DSAMLJointVARegressor(_RunnerCompatibleRegressor):
    """Matched non-meta DSAML path using the two available acoustic views."""

    def __init__(self, model_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        if model_dim % 4:
            raise ValueError("DSAML model_dim must be divisible by four")
        self.local_encoder = _SegmentCNN(model_dim, dropout=dropout)
        self.global_encoder = _SegmentCNN(model_dim, dropout=dropout)
        self.feature_fusion = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # The official implementation shares its encoder between local/global
        # attention calls.  Preserve that weight-sharing here.
        self.shared_attention = nn.TransformerEncoder(layer, num_layers=2)
        hidden = max(32, model_dim // 4)
        self.recurrent = nn.LSTM(
            model_dim, hidden, batch_first=True, bidirectional=True
        )
        self.head = _ProbabilisticHead(2 * hidden, dropout)

    @staticmethod
    def _context_mask(length: int, context: int, device: torch.device) -> Tensor:
        positions = torch.arange(length, device=device)
        return (positions[:, None] - positions[None, :]).abs() > context

    def _safe_context_mask(self, valid: Tensor, context: int) -> Tensor:
        # Padded queries outside the context otherwise have no visible key.
        # A fully masked softmax produces NaNs in eval attention; these can
        # contaminate valid queries in the next shared-attention layer.
        # Valid queries keep exactly the original context and valid-key set.
        # Padded queries may attend to the first valid key and are discarded.
        batch, length = valid.shape
        blocked = self._context_mask(length, context, valid.device)[None].expand(batch, -1, -1).clone()
        blocked |= ~valid[:, None, :]
        first = valid.to(torch.int64).argmax(dim=1)
        for i in range(batch):
            padded = ~valid[i]
            blocked[i, padded, first[i]] = False
        heads = self.shared_attention.layers[0].self_attn.num_heads
        return blocked.repeat_interleave(heads, dim=0)

    def _predict(self, batch: Mapping[str, object]) -> MERBaselinePrediction:
        mel, coch, mel_p, coch_p, presence, mask = self._inputs(batch)
        fused = self.feature_fusion(
            torch.cat(
                (self.local_encoder(mel, mel_p), self.global_encoder(coch, coch_p)),
                dim=-1,
            )
        )
        fused = torch.where(mask.unsqueeze(-1), fused, torch.zeros_like(fused))
        local = self.shared_attention(
            fused,
            mask=self._safe_context_mask(mask, 3),
        )
        global_context = min(30, max(1, fused.shape[1] - 1))
        global_output = self.shared_attention(
            fused,
            mask=self._safe_context_mask(mask, global_context),
        )
        attended = torch.sigmoid(local + global_output)
        attended = torch.where(mask.unsqueeze(-1), attended, torch.zeros_like(attended))
        temporal = _packed_recurrent(self.recurrent, attended, mask)
        mean, log_variance = self.head(temporal, mask)
        return MERBaselinePrediction(mean, log_variance, mask, temporal, presence)


class CNNTransformerJointVARegressor(_RunnerCompatibleRegressor):
    """Generic dual-path spectral/temporal architecture control."""

    def __init__(self, model_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        if model_dim % 4:
            raise ValueError("model_dim must be divisible by four")
        self.spectral = _SegmentCNN(model_dim, dropout=dropout)
        self.auxiliary = _SegmentCNN(model_dim, dropout=dropout)
        self.input_projection = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=3)
        self.local_temporal = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1, groups=4),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=1),
        )
        self.gate = nn.Linear(2 * model_dim, model_dim)
        self.head = _ProbabilisticHead(model_dim, dropout)

    def _predict(self, batch: Mapping[str, object]) -> MERBaselinePrediction:
        mel, coch, mel_p, coch_p, presence, mask = self._inputs(batch)
        inputs = self.input_projection(
            torch.cat((self.spectral(mel, mel_p), self.auxiliary(coch, coch_p)), dim=-1)
        )
        inputs = torch.where(mask.unsqueeze(-1), inputs, torch.zeros_like(inputs))
        long_range = self.temporal(inputs, src_key_padding_mask=~mask)
        local = self.local_temporal(inputs.transpose(1, 2)).transpose(1, 2)
        gate = torch.sigmoid(self.gate(torch.cat((local, long_range), dim=-1)))
        fused = gate * local + (1.0 - gate) * long_range
        fused = torch.where(mask.unsqueeze(-1), fused, torch.zeros_like(fused))
        mean, log_variance = self.head(fused, mask)
        return MERBaselinePrediction(mean, log_variance, mask, fused, presence)


def build_recent_mer_baseline(
    name: str, *, model_dim: int = 128, dropout: float = 0.1
) -> _RunnerCompatibleRegressor:
    """Construct one audited architecture adapter by stable CLI name."""

    constructors = {
        "crnn": CRNNJointVARegressor,
        "mcrnn": MCRNNJointVARegressor,
        "damff": DAMFFJointVARegressor,
        "dsaml": DSAMLJointVARegressor,
        "cnn_transformer": CNNTransformerJointVARegressor,
    }
    try:
        constructor = constructors[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown recent MER baseline {name!r}; choose from {BASELINE_NAMES}"
        ) from exc
    return constructor(model_dim=model_dim, dropout=dropout)


__all__ = [
    "BASELINE_NAMES",
    "CNNTransformerJointVARegressor",
    "CRNNJointVARegressor",
    "DAMFFJointVARegressor",
    "DSAMLJointVARegressor",
    "MCRNNJointVARegressor",
    "MERBaselinePrediction",
    "build_recent_mer_baseline",
]
