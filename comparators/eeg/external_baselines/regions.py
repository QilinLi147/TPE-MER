"""Dataset channel orders and a conservative anatomical region prior."""

from __future__ import annotations

import numpy as np


SEED_CHANNELS = (
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7",
    "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2",
)

DREAMER_CHANNELS = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)

REGIONS = (
    "frontal_left", "frontal_right", "temporal_left", "temporal_right",
    "central", "parietal", "occipital",
)


def dataset_channels(dataset: str) -> tuple[str, ...]:
    """Return the channel order used by a frozen dataset cache."""

    dataset = dataset.lower()
    if dataset in {"seed", "seediv", "seedv", "mped"}:
        return SEED_CHANNELS
    if dataset == "dreamer":
        return DREAMER_CHANNELS
    raise ValueError(f"unknown EEG dataset: {dataset}")


def _side(name: str) -> str:
    if name.endswith("z"):
        return "mid"
    digits = "".join(character for character in name if character.isdigit())
    if not digits:
        return "mid"
    return "left" if int(digits) % 2 else "right"


def anatomical_membership(channel_names: tuple[str, ...]) -> np.ndarray:
    """Return a 7 x C soft membership matrix.

    Midline electrodes contribute equally to left and right frontal regions.
    This prior is deliberately coarse; the model may only make a low-rank
    residual correction downstream.
    """

    membership = np.zeros((len(REGIONS), len(channel_names)), dtype=np.float32)
    for channel, name in enumerate(channel_names):
        side = _side(name)
        if name.startswith(("Fp", "AF")) or (
            name.startswith("F") and not name.startswith(("FC", "FT"))
        ):
            if side == "left":
                membership[0, channel] = 1.0
            elif side == "right":
                membership[1, channel] = 1.0
            else:
                membership[0:2, channel] = 0.5
        elif name.startswith(("FT", "T", "TP")):
            membership[2 if side == "left" else 3, channel] = 1.0
        elif name.startswith(("FC", "C", "CP")):
            membership[4, channel] = 1.0
        elif name.startswith(("P", "PO")):
            membership[5, channel] = 1.0
        elif name.startswith(("O", "CB")):
            membership[6, channel] = 1.0
        else:
            raise ValueError(f"unmapped EEG channel: {name}")
    if not np.allclose(membership.sum(axis=0), 1.0):
        raise ValueError("each EEG channel must have unit anatomical membership")
    return membership


def region_graph() -> np.ndarray:
    """Small symmetric region graph used by the single low-order graph block."""

    edges = (
        (0, 1), (0, 2), (1, 3), (0, 4), (1, 4),
        (2, 4), (3, 4), (2, 5), (3, 5), (4, 5), (5, 6),
    )
    adjacency = np.eye(len(REGIONS), dtype=np.float32)
    for left, right in edges:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    return adjacency


def dataset_regions(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    return anatomical_membership(dataset_channels(dataset)), region_graph()


def controlled_membership(
    dataset: str,
    control: str = "anatomical",
    seed: int = 8635,
) -> np.ndarray:
    """Return a capacity-matched region-membership control.

    ``random`` permutes complete channel-membership columns.  It therefore
    preserves the exact number and total membership weight of every region,
    including the two 0.5 frontal memberships assigned to midline channels.
    ``left_right`` swaps the labels of the paired frontal and temporal rows.
    This latter control is intentionally a label-permutation null when the
    downstream connector is fully permutation equivariant; callers must
    record that fact instead of presenting a null difference as anatomical
    evidence.
    """

    membership, _ = dataset_regions(dataset)
    if control == "anatomical":
        return membership
    if control == "random":
        permutation = np.random.default_rng(seed).permutation(
            membership.shape[1]
        )
        return membership[:, permutation].copy()
    if control == "left_right":
        swapped = membership.copy()
        swapped[[0, 1]] = swapped[[1, 0]]
        swapped[[2, 3]] = swapped[[3, 2]]
        return swapped
    raise ValueError(f"unknown membership control: {control}")
