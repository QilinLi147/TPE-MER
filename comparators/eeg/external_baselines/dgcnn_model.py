"""Faithful compact PyTorch implementation of the DGCNN EEG baseline.

Architecture and defaults follow Song et al., IEEE TAFFC 2020, and the
adapted implementation: a learned non-negative normalized
adjacency, four graph-filter powers, 64 graph features, a 64-unit classifier,
and dropout 0.5.  Dataset-specific changes are limited to electrode count and
class count.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def normalized_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    value = F.relu(adjacency)
    degree = value.sum(dim=1).add(1e-10).rsqrt()
    normalized = degree[:, None] * value * degree[None, :]
    return torch.eye(len(value), device=value.device, dtype=value.dtype) - normalized


class DGCNNBaseline(nn.Module):
    input_view = "static_mean"

    def __init__(
        self,
        channels: int,
        bands: int,
        num_classes: int,
        order: int = 4,
        graph_features: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if bands != 5:
            raise ValueError("DGCNN baseline requires the five DE bands")
        if order < 1:
            raise ValueError("graph-filter order must be positive")
        self.channels = channels
        self.bands = bands
        self.order = order
        self.batch_normalization = nn.BatchNorm1d(bands, eps=0.001)
        self.adjacency = nn.Parameter(torch.empty(channels, channels))
        nn.init.xavier_normal_(self.adjacency)
        self.graph_weights = nn.ParameterList([
            nn.Parameter(torch.empty(bands, graph_features))
            for _ in range(order)
        ])
        for weight in self.graph_weights:
            nn.init.xavier_normal_(weight)
        self.fc1 = nn.Linear(channels * graph_features, 64)
        self.fc2 = nn.Linear(64, num_classes)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.fc2.bias)
        self.dropout = nn.Dropout(dropout)
        self.resolved_config = {
            "variant": "DGCNN paper-derived server implementation",
            "num_electrodes": channels,
            "num_features": bands,
            "num_classes": num_classes,
            "graph_filter_order": order,
            "graph_features": graph_features,
            "classifier_hidden": 64,
            "dropout": dropout,
            "input_reduction": "mean_over_sequence",
            "adjacency": "learned non-negative normalized graph",
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected = (self.channels, self.bands)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                f"DGCNN expects [B,{self.channels},{self.bands}], "
                f"got {tuple(inputs.shape)}"
            )
        value = self.batch_normalization(inputs.transpose(1, 2)).transpose(1, 2)
        graph = normalized_laplacian(self.adjacency)
        support = torch.eye(
            self.channels, device=inputs.device, dtype=inputs.dtype
        )
        output = None
        for weight in self.graph_weights:
            current = support @ value @ weight
            output = current if output is None else output + current
            support = support @ graph
        feature = F.relu(output).reshape(len(inputs), -1)
        return self.fc2(self.dropout(F.relu(self.fc1(feature))))


class ResidualGraphCNN(nn.Module):
    """Residual convolutional branch used by the released RGCB reproduction."""

    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(64, 64, 7, padding="same"),
            nn.BatchNorm1d(64, eps=0.001), nn.ReLU(),
            nn.Conv1d(64, 64, 7, padding="same"),
            nn.BatchNorm1d(64, eps=0.001), nn.ReLU(),
            nn.Conv1d(64, 32, 7, padding="same"),
            nn.BatchNorm1d(32, eps=0.001),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 32, 7, padding="same"),
            nn.BatchNorm1d(32, eps=0.001), nn.ReLU(),
            nn.Conv1d(32, 32, 7, padding="same"),
            nn.BatchNorm1d(32, eps=0.001), nn.ReLU(),
            nn.Conv1d(32, 16, 7, padding="same"),
            nn.BatchNorm1d(16),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(16, 16, 7, padding="same"),
            nn.BatchNorm1d(16, eps=0.001), nn.ReLU(),
            nn.Conv1d(16, 16, 7, padding="same"),
            nn.BatchNorm1d(16, eps=0.001), nn.ReLU(),
            nn.Conv1d(16, 8, 7, padding="same"),
            nn.BatchNorm1d(8, eps=0.001),
        )
        self.down1 = nn.Conv1d(64, 32, 7, padding="same")
        self.down2 = nn.Conv1d(32, 16, 7, padding="same")
        self.down3 = nn.Conv1d(16, 8, 7, padding="same")
        self.pool = nn.MaxPool1d(2, 2)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = F.relu(self.block1(inputs) + self.down1(inputs))
        first = self.pool(first)
        second = F.relu(self.block2(first) + self.down2(first))
        third = F.relu(self.block3(second) + self.down3(second))
        return torch.cat([
            first.reshape(len(inputs), -1),
            second.reshape(len(inputs), -1),
            third.reshape(len(inputs), -1),
        ], dim=1)


class RGCBBaseline(DGCNNBaseline):
    """Residual GCB-Net using the same learned graph as DGCNN."""

    def __init__(
        self, channels: int, bands: int, num_classes: int,
        order: int = 4, graph_features: int = 64, dropout: float = 0.5,
    ) -> None:
        super().__init__(
            channels, bands, num_classes, order, graph_features, dropout
        )
        self.batch_normalization = nn.BatchNorm1d(bands, eps=0.01)
        self.residual_cnn = ResidualGraphCNN()
        self.fc1 = nn.Linear(channels * 92, 64)
        self.fc2 = nn.Linear(64, num_classes)
        for linear in (self.fc1, self.fc2):
            nn.init.normal_(linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(linear.bias)
        self.resolved_config.update({
            "variant": "RGCB paper-derived server implementation",
            "graph_features": graph_features,
            "residual_cnn_features_per_electrode": 28,
            "classifier_input_features_per_electrode": 92,
        })

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected = (self.channels, self.bands)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                f"RGCB expects [B,{self.channels},{self.bands}], "
                f"got {tuple(inputs.shape)}"
            )
        value = self.batch_normalization(inputs.transpose(1, 2)).transpose(1, 2)
        graph = normalized_laplacian(self.adjacency)
        support = torch.eye(
            self.channels, device=inputs.device, dtype=inputs.dtype
        )
        output = None
        for weight in self.graph_weights:
            current = support @ value @ weight
            output = current if output is None else output + current
            support = support @ graph
        graph_feature = F.relu(output)
        residual_feature = self.residual_cnn(graph_feature.transpose(1, 2))
        feature = torch.cat([
            graph_feature.reshape(len(inputs), -1), residual_feature
        ], dim=1)
        return self.fc2(self.dropout(F.relu(self.fc1(feature))))
