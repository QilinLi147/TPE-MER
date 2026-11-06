"""Runnable DGC-Link model for the frozen cross-subject evaluation protocol.

This module recovers the working architecture archived as Figshare v1 of
Li et al.'s DGC-Link release.  The later v3 ``model.py`` removed the shared
pre-dense convolution and the first transition pooling operation while
retaining a 14,136-wide classifier, which makes both its 62- and 14-channel
forward passes dimensionally inconsistent.  The v1 data path is therefore
the auditable architecture source; the released v3 ``train.py`` and its
supplementary document remain the source of training defaults.

Only the input electrode count, the five-band input width, and the class
count are represented dynamically here.  Dataset splitting, sequence-mean
reduction, train-only normalisation, development-only epoch selection, and
fresh U2/T33 refitting belong to the external runner rather than this model.
No test-set input or label is consumed while constructing the graph.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


PAPER_DOI = "10.1109/TAFFC.2025.3602823"
SOFTWARE_DOI_V1 = "10.6084/m9.figshare.29519597.v1"
SOFTWARE_DOI_V3 = "10.6084/m9.figshare.29519597.v3"


def normalized_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Match the author's non-negative row-degree graph normalisation."""

    value = F.relu(adjacency)
    inverse_sqrt_degree = value.sum(dim=1).add(1e-10).rsqrt()
    normalized = (
        inverse_sqrt_degree[:, None]
        * value
        * inverse_sqrt_degree[None, :]
    )
    identity = torch.eye(
        value.shape[0], device=value.device, dtype=value.dtype
    )
    return identity - normalized


def linkage_supports(laplacian: torch.Tensor, order: int) -> list[torch.Tensor]:
    """Recover v1's ``I, L, L^2, ...`` linkage-support construction."""

    if order < 1:
        raise ValueError("DGC-Link order must be positive")
    identity = torch.eye(
        laplacian.shape[0], device=laplacian.device, dtype=laplacian.dtype
    )
    supports = [identity]
    for index in range(1, order):
        supports.append(
            laplacian if index == 1 else supports[-1] @ laplacian
        )
    return supports


class DualGateChebyshev(nn.Module):
    """One released DGC-Link order branch, including its matrix-valued gates.

    The author implementation uses matrix multiplication in the reset/update
    expressions.  It is intentionally retained rather than silently replaced
    by element-wise multiplication, even though the latter may look more like
    a conventional gated recurrent unit.
    """

    def __init__(
        self,
        input_features: int,
        graph_features: int,
        channels: int,
        branch_order: int,
    ) -> None:
        super().__init__()
        self.branch_order = branch_order
        self.input_weight = nn.Parameter(torch.empty(input_features, channels))
        self.previous_input_weight = nn.Parameter(
            torch.empty(input_features, channels)
        )
        self.reset_weight = nn.Parameter(torch.empty(channels, channels))
        self.previous_reset_weight = nn.Parameter(
            torch.empty(channels, channels)
        )
        self.update_weight = nn.Parameter(torch.empty(channels, channels))
        self.previous_update_weight = nn.Parameter(
            torch.empty(channels, channels)
        )
        self.candidate_weight = nn.Parameter(torch.empty(channels, channels))
        self.previous_candidate_weight = nn.Parameter(
            torch.empty(channels, channels)
        )
        self.output_weight = nn.Parameter(torch.empty(channels, graph_features))
        for parameter in self.parameters():
            nn.init.xavier_normal_(parameter)

    def forward(
        self, inputs: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        if self.branch_order == 0:
            hidden = inputs @ self.input_weight
        elif self.branch_order == 1:
            hidden = (adjacency @ inputs) @ self.input_weight
        else:
            chebyshev = [inputs, adjacency @ inputs]
            hidden = None
            for index in range(2, self.branch_order + 1):
                current = (
                    2 * adjacency @ chebyshev[index - 1]
                    - chebyshev[index - 2]
                    + chebyshev[index - 1]
                )
                chebyshev.append(current)
                current_gate_input = current @ self.input_weight
                previous_gate_input = (
                    chebyshev[index - 1] @ self.previous_input_weight
                )
                reset = torch.sigmoid(
                    current_gate_input @ self.reset_weight
                    + previous_gate_input @ self.previous_reset_weight
                )
                update = torch.sigmoid(
                    current_gate_input @ self.update_weight
                    + previous_gate_input @ self.previous_update_weight
                )
                candidate = torch.tanh(
                    current_gate_input @ self.candidate_weight
                    + reset
                    @ (previous_gate_input @ self.previous_candidate_weight)
                )
                # These are matrix multiplications in the released v1 code.
                hidden = (1 - update) @ candidate + update @ previous_gate_input
            if hidden is None:  # defensive guard for future order refactors
                raise AssertionError("higher-order DGC-Link branch produced no state")
        return F.elu(hidden @ self.output_weight)


class DualGateLinkage(nn.Module):
    """Four-order linkage layer from the working Figshare v1 release."""

    def __init__(
        self,
        input_features: int,
        graph_features: int,
        channels: int,
        order: int,
    ) -> None:
        super().__init__()
        self.order = order
        self.branches = nn.ModuleList([
            DualGateChebyshev(
                input_features, graph_features, channels, branch_order
            )
            for branch_order in range(order)
        ])

    def forward(
        self, inputs: torch.Tensor, laplacian: torch.Tensor
    ) -> torch.Tensor:
        supports = linkage_supports(laplacian, self.order)
        result = None
        for branch, support in zip(self.branches, supports):
            current = branch(inputs, support)
            result = current if result is None else result + current
        if result is None:
            raise AssertionError("DGC-Link requires at least one graph branch")
        return F.relu(result)


def _dense_layer(input_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(input_channels, 128, 1, padding="same", bias=True),
        nn.BatchNorm1d(128, eps=0.001),
        nn.ELU(),
        nn.Conv1d(128, 32, 3, padding="same", bias=True),
        nn.BatchNorm1d(32, eps=0.001),
        nn.ELU(),
    )


class RecoveredDenseNetwork(nn.Module):
    """Working v1 dense path, including shared convolution and one pooling.

    ``shared_conv`` is one parameterised layer called three times, exactly as
    in Figshare v1.  Creating three independent convolutions would change both
    the parameter count and the released computation.
    """

    output_channels_after_pool = 96 + 112 + 120

    def __init__(self, graph_features: int = 64) -> None:
        super().__init__()
        if graph_features != 64:
            raise ValueError(
                "the recovered DGC-Link dense path requires 64 graph features"
            )
        self.shared_conv = nn.Conv1d(
            graph_features, graph_features, 7, padding="same", bias=True
        )
        self.block1 = nn.ModuleList([
            _dense_layer(input_channels)
            for input_channels in (64, 96, 128, 160)
        ])
        self.transition1 = nn.Conv1d(192, 96, 1, bias=True)
        self.pool = nn.AvgPool1d(2, 2)
        self.block2 = nn.ModuleList([
            _dense_layer(input_channels)
            for input_channels in (96, 128, 160, 192)
        ])
        self.transition2 = nn.Conv1d(224, 112, 1, bias=True)
        self.block3 = nn.ModuleList([
            _dense_layer(input_channels)
            for input_channels in (112, 144, 176, 208)
        ])
        self.transition3 = nn.Conv1d(240, 120, 1, bias=True)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _dense_block(
        inputs: torch.Tensor, layers: nn.ModuleList
    ) -> torch.Tensor:
        value = inputs
        for layer in layers:
            value = torch.cat((value, layer(value)), dim=1)
        return value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs
        for _ in range(3):
            value = self.shared_conv(value)
        value = self._dense_block(value, self.block1)
        value = self.pool(self.transition1(value))
        first = value.reshape(len(inputs), -1)
        value = self._dense_block(value, self.block2)
        value = self.transition2(value)
        second = value.reshape(len(inputs), -1)
        value = self._dense_block(value, self.block3)
        value = self.transition3(value)
        third = value.reshape(len(inputs), -1)
        return torch.cat((first, second, third), dim=1)


class DGCLinkBaseline(nn.Module):
    """Figshare-v1 DGC-Link recovery for U2/T33 five-band DE inputs."""

    input_view = "static_mean"

    def __init__(
        self,
        channels: int,
        bands: int,
        num_classes: int,
        order: int = 4,
        graph_features: int = 64,
        classifier_hidden: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if channels not in (14, 62):
            raise ValueError(
                "DGC-Link U2 adaptation supports 14- or 62-channel montages"
            )
        if bands != 5:
            raise ValueError(
                "DGC-Link U2 adaptation requires the unified five DE bands"
            )
        if num_classes < 2:
            raise ValueError("DGC-Link requires at least two classes")
        if order != 4:
            raise ValueError("the audited DGC-Link release uses order K=4")
        if graph_features != 64 or classifier_hidden != 64:
            raise ValueError(
                "the audited DGC-Link release uses 64 graph/hidden features"
            )
        self.channels = channels
        self.bands = bands
        self.num_classes = num_classes
        self.order = order
        self.graph_features = graph_features
        self.batch_normalization = nn.BatchNorm1d(bands, eps=0.01)
        self.adjacency = nn.Parameter(torch.empty(channels, channels))
        nn.init.xavier_normal_(self.adjacency)
        self.linkage = DualGateLinkage(
            bands, graph_features, channels, order
        )
        self.dense_network = RecoveredDenseNetwork(graph_features)
        pooled_channels = channels // 2
        self.classifier_input_features = (
            channels * graph_features
            + pooled_channels
            * RecoveredDenseNetwork.output_channels_after_pool
        )
        self.fc1 = nn.Linear(self.classifier_input_features, classifier_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(classifier_hidden, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.resolved_config = {
            "variant": "DGC-Link official Figshare v1 architecture recovery",
            "execution_label": (
                "official Figshare-v1 architecture recovery + v3 defaults / "
                "U2 adaptation"
            ),
            "paper_doi": PAPER_DOI,
            "architecture_source": SOFTWARE_DOI_V1,
            "training_defaults_source": SOFTWARE_DOI_V3,
            "architecture_recovery_reason": (
                "Figshare v3 removed v1 shared-convolution/pooling operations "
                "but retained the fixed 14136-wide classifier"
            ),
            "num_electrodes": channels,
            "num_features": bands,
            "num_classes": num_classes,
            "graph_filter_order": order,
            "graph_features": graph_features,
            "classifier_input_features": self.classifier_input_features,
            "classifier_hidden": classifier_hidden,
            "dropout": dropout,
            "learned_adjacency": "non-negative normalized Laplacian",
            "dense_shared_conv_calls": 3,
            "dense_first_transition_pool": "AvgPool1d(kernel=2,stride=2)",
            "input_reduction": "mean_over_sequence in U2 data adapter",
            "protocol_adaptation": (
                "dynamic 62/14-channel and 3/2-class dimensions; unified "
                "five-band DE; train-only normalization; outer-test-independent "
                "selection and fresh refit are runner responsibilities"
            ),
        }

    def _features(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.channels, self.bands)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                f"DGC-Link expects [B,{self.channels},{self.bands}], "
                f"got {tuple(inputs.shape)}"
            )
        value = self.batch_normalization(
            inputs.transpose(1, 2)
        ).transpose(1, 2)
        laplacian = normalized_laplacian(self.adjacency)
        graph_feature = self.linkage(value, laplacian)
        flattened_graph = graph_feature.reshape(len(inputs), -1)
        dense_feature = self.dense_network(graph_feature.transpose(1, 2))
        feature = torch.cat((flattened_graph, dense_feature), dim=1)
        if feature.shape[1] != self.classifier_input_features:
            raise AssertionError(
                "recovered DGC-Link feature width drifted: "
                f"expected {self.classifier_input_features}, got {feature.shape[1]}"
            )
        return feature, graph_feature

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feature, _ = self._features(inputs)
        return self.fc2(self.dropout(F.relu(self.fc1(feature))))

    def forward_with_aux(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose released dense features and raw A for planned visualisation."""

        feature, _ = self._features(inputs)
        logits = self.fc2(self.dropout(F.relu(self.fc1(feature))))
        return logits, feature, self.adjacency
