"""Auditable MASA-TCN adapter for the Chapter-4 EEG feature caches.

This module follows the author implementation at
https://github.com/yi-ding-cs/MASA-TCN (commit
8789f7392505ca9a649aca7b5a0f80c8e9091b9f).  The network equations and
initialisation mirror ``model.py`` and ``temporal_convolutional_layers.py``;
the only model-side adaptations are the dataset channel count, five DE bands,
and the number of classes.  Input is accepted in the common cache layout
``[batch, sequence, channel, band]`` and permuted to the author layout
``[batch, 1, channel * band, sequence]`` without changing values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils import weight_norm


PAPER_DOI = "10.1109/JBHI.2024.3392564"
PAPER_URL = "https://ieeexplore.ieee.org/document/10506986"
AUTHOR_REPOSITORY = "https://github.com/yi-ding-cs/MASA-TCN"
AUTHOR_COMMIT = "8789f7392505ca9a649aca7b5a0f80c8e9091b9f"
AUTHOR_MODEL_SHA256 = "5ae1bb4fa2c1fd1db2432b300a1b8084fdfc378c04a74fda880983e01c01f1de"
AUTHOR_TEMPORAL_SHA256 = "5983870da9359dea953705d583fbf9c4b4f0d6f37c99535f3825fdb2c1950fa8"


class Chomp2d(nn.Module):
    """Remove the right padding used by the author's causal convolutions."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        if chomp_size < 1:
            raise ValueError("MASA-TCN chomp size must be positive")
        self.chomp_size = chomp_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            weight_norm(nn.Conv2d(
                inputs, outputs, (1, kernel_size), stride=1,
                padding=(0, padding), dilation=(1, dilation),
            )),
            Chomp2d(padding),
            nn.PReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv2d(
                outputs, outputs, (1, kernel_size), stride=1,
                padding=(0, padding), dilation=(1, dilation),
            )),
            Chomp2d(padding),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv2d(inputs, outputs, 1) if inputs != outputs else None
        )
        self.activation = nn.PReLU()
        self._initialise()

    def _initialise(self) -> None:
        self.network[0].weight.data.normal_(0, 0.01)
        self.network[4].weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        return self.activation(self.network(inputs) + residual)


class SpaceAwareTemporalBlock(nn.Module):
    def __init__(
        self,
        output_channels: int,
        eeg_channels: int,
        bands: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * 2
        # Keep the author's construction order: the spatial fusion module is
        # initialised before the temporal sequential block.  This preserves
        # exact same-seed parameter draws, not merely architectural equality.
        self.spatial_fusion = weight_norm(nn.Conv2d(
            output_channels, output_channels, (eeg_channels, 1), stride=1,
        ))
        self.temporal = weight_norm(nn.Conv2d(
            1, output_channels, (bands, kernel_size),
            stride=(bands, 1), dilation=(1, 2), padding=(0, padding),
        ))
        self.chomp = Chomp2d(padding)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.temporal.weight.data.normal_(0, 0.01)
        self.spatial_fusion.weight.data.normal_(0, 0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.dropout(self.activation(self.chomp(self.temporal(inputs))))
        return self.spatial_fusion(value)


class MultiAnchorSpaceAwareTCN(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        kernels: tuple[int, int, int],
        dropout: float,
        eeg_channels: int,
        bands: int,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("MASA-TCN requires at least two channel levels")
        self.anchors = nn.ModuleList([
            SpaceAwareTemporalBlock(
                channels[0], eeg_channels, bands, kernel, dropout
            )
            for kernel in kernels
        ])
        # The author creates the dilated temporal stack before the 1x1 anchor
        # fusion.  Retain that order for exact deterministic equivalence.
        self.temporal = nn.Sequential(*[
            TemporalBlock(
                channels[level], channels[level + 1], kernels[1],
                dilation=2 ** (level + 2), dropout=dropout,
            )
            for level in range(len(channels) - 1)
        ])
        self.anchor_fusion = weight_norm(nn.Conv2d(
            3 * channels[0], channels[0], (1, 1), stride=1,
        ))
        self.anchor_fusion.weight.data.normal_(0, 0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        anchors = torch.cat([branch(inputs) for branch in self.anchors], dim=1)
        return self.temporal(self.anchor_fusion(anchors))


@dataclass(frozen=True)
class MASATCNConfig:
    eeg_channels: int
    bands: int
    classes: int
    hidden_channels: tuple[int, ...] = (64, 64)
    anchor_kernels: tuple[int, int, int] = (3, 5, 15)
    dropout: float = 0.15
    early_fusion: bool = True


class MASATCNAdapter(nn.Module):
    """Official classification path with a value-preserving cache adapter."""

    def __init__(self, config: MASATCNConfig) -> None:
        super().__init__()
        if not config.early_fusion:
            raise ValueError("the frozen formal setting requires early fusion")
        self.config = config
        self.temporal = MultiAnchorSpaceAwareTCN(
            config.hidden_channels,
            config.anchor_kernels,
            config.dropout,
            config.eeg_channels,
            config.bands,
        )
        self.classifier = nn.Linear(config.hidden_channels[-1], config.classes)
        self.resolved_config = {
            **asdict(config),
            "architecture": "official MASA_TCN classification path",
            "input_adapter": (
                "[B,S,C,F] -> [B,1,C*F,S] value-preserving permutation"
            ),
            "temporal_pooling": "author mean over sequence logits",
            "author_repository": AUTHOR_REPOSITORY,
            "author_commit": AUTHOR_COMMIT,
        }

    def author_layout(self, inputs: torch.Tensor) -> torch.Tensor:
        expected = (
            inputs.ndim == 4
            and inputs.shape[2] == self.config.eeg_channels
            and inputs.shape[3] == self.config.bands
        )
        if not expected:
            raise ValueError(
                "MASA-TCN expects [batch,sequence,"
                f"{self.config.eeg_channels},{self.config.bands}], got "
                f"{tuple(inputs.shape)}"
            )
        batch, sequence, channels, bands = inputs.shape
        return (
            inputs.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch, 1, channels * bands, sequence)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.temporal(self.author_layout(inputs))
        value = value.transpose(1, 3).contiguous().squeeze(-2)
        return self.classifier(value).mean(dim=1)


def build_masatcn(channels: int, bands: int, classes: int) -> MASATCNAdapter:
    if bands != 5:
        raise ValueError(f"formal MASA-TCN adapter expects five DE bands, got {bands}")
    return MASATCNAdapter(MASATCNConfig(
        eeg_channels=channels, bands=bands, classes=classes
    ))
