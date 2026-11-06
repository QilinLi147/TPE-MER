"""Optimisers and loss functions for the EEG comparisons."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
import torch
from torch import nn
from torch.nn import functional as F

def pgcn_label_smoothing(
    logits: torch.Tensor, labels: torch.Tensor, epsilon: float
) -> torch.Tensor:
    classes = logits.shape[-1]
    weight = logits.new_full(logits.shape, epsilon / (classes - 1))
    weight.scatter_(-1, labels.unsqueeze(-1), 1.0 - epsilon)
    return (-weight * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

def loss_function(method: str, config: dict[str, Any]):
    if method == "emt":
        return lambda logits, labels: F.cross_entropy(
            logits, labels, label_smoothing=config["label_smoothing"]
        )
    if method == "pgcn":
        return lambda logits, labels: pgcn_label_smoothing(
            logits, labels, config["label_smoothing"]
        )
    if method == "hdgnet":
        # The released model applies softmax in forward, and the released
        # training script passes those probabilities to CrossEntropyLoss.
        return lambda probabilities, labels: F.cross_entropy(
            probabilities, labels
        )
    if method == "bf_gcn_de":
        return lambda logits, labels: F.cross_entropy(logits, labels)
    if method in {"dgcnn", "rgcb", "dgclink"}:
        return lambda logits, labels: F.cross_entropy(logits, labels)
    raise ValueError(f"unsupported loss for {method}")

class UntunedLinearWarmup:
    """Behavioral equivalent of the author's missing pytorch_warmup helper."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.periods = [
            int(2.0 / (1.0 - group["betas"][1]))
            for group in optimizer.param_groups
        ]
        self.last_step = -1
        self.lrs = [group["lr"] for group in optimizer.param_groups]
        self.dampen()

    def dampen(self) -> None:
        self.last_step += 1
        for group, period in zip(self.optimizer.param_groups, self.periods):
            group["lr"] *= min(1.0, (self.last_step + 1) / period)

    @contextmanager
    def dampening(self) -> Iterator[None]:
        for group, value in zip(self.optimizer.param_groups, self.lrs):
            group["lr"] = value
        yield
        self.lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.dampen()

@dataclass
class Optimisation:
    optimizer: torch.optim.Optimizer
    scheduler: Any | None
    warmup: UntunedLinearWarmup | None

def build_optimisation(
    method: str, model: nn.Module, config: dict[str, Any]
) -> Optimisation:
    if method == "emt":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        return Optimisation(optimizer, None, None)
    if method == "hdgnet":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        return Optimisation(optimizer, None, None)
    if method == "bf_gcn_de":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        return Optimisation(optimizer, None, None)
    if method in {"dgcnn", "rgcb", "dgclink"}:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["max_epochs"]
        )
        return Optimisation(optimizer, scheduler, None)

    laplacian, local, remaining = [], [], []
    for name, parameter in model.named_parameters():
        if name == "author_model.adj":
            laplacian.append(parameter)
        elif "local" in name:
            local.append(parameter)
        else:
            remaining.append(parameter)
    if not laplacian or not local or not remaining:
        raise AssertionError("PGCN author parameter groups were not resolved")
    optimizer = torch.optim.AdamW(
        [
            {"params": laplacian, "lr": config["laplacian_learning_rate"]},
            {"params": local, "lr": config["learning_rate"]},
            {"params": remaining, "lr": config["learning_rate"]},
        ],
        betas=tuple(config["betas"]),
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[config["max_epochs"] // 3],
        gamma=0.1,
    )
    warmup = UntunedLinearWarmup(optimizer)
    # The author resets this public field directly after construction.
    warmup.last_step = -1
    return Optimisation(optimizer, scheduler, warmup)
