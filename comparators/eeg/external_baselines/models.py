"""Thin adapters that execute author-released EmT, PGCN, and HDGNet code."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
import importlib
import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator
import sys

import numpy as np
import torch
from torch import nn

from .dgcnn_model import DGCNNBaseline, RGCBBaseline
from .dgclink_model import DGCLinkBaseline
from .regions import DREAMER_CHANNELS, SEED_CHANNELS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = PROJECT_ROOT / "third_party" / "recent_baselines"


# Author-recommended/default values.  Only dataset dimensions and class count
# are replaced by the adapter; these values are intentionally not searched.
METHOD_DEFAULTS: dict[str, dict[str, Any]] = {
    "emt": {
        "layers_graph": [1, 2],
        "layers_transformer": 8,
        "num_adj": 2,
        "hidden_graph": 32,
        "K": 4,
        "num_head": 16,
        "dim_head": 32,
        "dropout": 0.25,
        "graph2token": "Linear",
        "encoder_type": "Cheby",
        "alpha": 0.25,
    },
    "pgcn": {
        "lr": 0.01,
        "beta": 5e-5,
        "weight_decay": 5e-4,
        "dropout": 0.4,
        "epsilon": 0.05,
        "batch_size": 32,
        "epochs": 300,
        "patience": 20,
        "distance_delta": 9,
    },
    "hdgnet": {
        "graph_definition": "GENERAL",
        "seed_hid_channels": 5,
        "seed_out_channels": 10,
        "dreamer_hid_channels": 3,
        "dreamer_out_channels": 3,
        "dropout": 0.5,
        "l_fc": 1,
        "g_fc": 1,
        "get_adj": False,
    },
    "bf_gcn_de": {
        # Architecture values are the exact public Simple_Demo.py values.
        "kadj": 2,
        "num_out": 16,
        "att_hidden": 16,
        "classifier_hidden": 32,
        "avgpool": 2,
        "dropout": 0.0,
        "classification_branch": "subject-dependent fully-connected branch",
        "functional_graph": "training-only absolute DE correlation",
    },
    "dgcnn": {
        "order": 4,
        "graph_features": 64,
        "dropout": 0.5,
    },
    "rgcb": {
        "order": 4,
        "graph_features": 64,
        "dropout": 0.5,
    },
    "dgclink": {
        "order": 4,
        "graph_features": 64,
        "classifier_hidden": 64,
        "dropout": 0.5,
    },
}


def compatibility(method: str, channels: int, bands: int, view: str) -> tuple[bool, str]:
    method = method.lower()
    if bands != 5:
        return False, f"{method} adapter requires the five DE bands, got {bands}"
    if method == "emt":
        if view != "sequence":
            return False, "EmT requires the ordered sequence view"
        return True, "native temporal-graph input; channel/class dimensions adapted"
    if method == "pgcn":
        if channels not in (14, 62):
            return False, "PGCN supports the author 62-channel graph or frozen PGCN-14 adaptation"
        if view != "static_mean":
            return False, "PGCN expects one [channel,5] graph per sample"
        if channels == 14:
            return True, (
                "PGCN-14 adaptation with frozen DREAMER montage, seven regions, "
                "and two hemispheres; sequence reduced by time mean"
            )
        return True, "62-channel author graph; sequence reduced by time mean at input"
    if method == "hdgnet":
        if channels not in (14, 62):
            return False, "HDGNet has released GENERAL graphs for 14 or 62 channels"
        if view != "static_mean":
            return False, "HDGNet expects one [channel,band] graph per sample"
        return True, (
            "author-defined GENERAL graph fixed before training; sequence reduced "
            "by time mean at input"
        )
    if method == "bf_gcn_de":
        if channels not in (14, 62):
            return False, "BF-GCN-DE supports the DREAMER or SEED/MPED montage"
        if view != "static_mean":
            return False, "BF-GCN-DE expects one [channel,5] DE graph per sample"
        return True, (
            "official BF-GCN model with disclosed training-only DE-correlation "
            "functional graph replacing unavailable raw-signal PLV"
        )
    if method in {"dgcnn", "rgcb"}:
        if channels not in (14, 62):
            return False, f"{method} supports the DREAMER or SEED/MPED montage"
        if view != "static_mean":
            return False, f"{method} expects one [channel,5] DE graph per sample"
        return True, "learned graph; channel and class dimensions adapted only"
    if method == "dgclink":
        if channels not in (14, 62):
            return False, "DGC-Link supports the DREAMER or SEED/MPED montage"
        if view != "static_mean":
            return False, "DGC-Link expects one [channel,5] DE graph per sample"
        return True, (
            "official Figshare-v1 architecture recovery with v3 defaults; "
            "channel and class dimensions adapted only"
        )
    return False, f"unsupported external method: {method}"


def _load_file_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load author module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_named_file_module(name: str, path: Path) -> ModuleType:
    """Load a file under an explicit package name for relative imports."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load author module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_hdgnet_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Load only HDGNet's model and graph constants under an isolated namespace."""

    author_root = THIRD_PARTY / "HDGNet" / "Affectiv_AI" / "torcheeg"
    constants = author_root / "datasets" / "constants"
    namespace = "_external_author_hdgnet_constants"
    packages = {
        namespace: constants,
        f"{namespace}.emotion_recognition": constants / "emotion_recognition",
    }
    for name, directory in packages.items():
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = [str(directory)]  # type: ignore[attr-defined]
            sys.modules[name] = package
    _load_named_file_module(
        f"{namespace}.region_1020", constants / "region_1020.py"
    )
    _load_named_file_module(f"{namespace}.utils", constants / "utils.py")
    seed = _load_named_file_module(
        f"{namespace}.emotion_recognition.seed",
        constants / "emotion_recognition" / "seed.py",
    )
    dreamer = _load_named_file_module(
        f"{namespace}.emotion_recognition.dreamer",
        constants / "emotion_recognition" / "dreamer.py",
    )
    model = _load_file_module(
        "_external_author_hdgnet_model", author_root / "models" / "pyg" / "hdgnet.py"
    )
    return model, seed, dreamer


def _module_is_within(module: ModuleType, directory: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        Path(module_file).resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _load_pgcn_modules(author_dir: Path) -> tuple[ModuleType, ModuleType]:
    """Load PGCN's non-package author modules without silent name collision."""

    expected = ("graphpool", "layer_PGCN", "utils", "node_location", "model_PGCN")
    for name in expected:
        existing = sys.modules.get(name)
        if existing is not None and not _module_is_within(existing, author_dir):
            raise RuntimeError(
                f"cannot safely load PGCN: sys.modules[{name!r}] comes from "
                f"{getattr(existing, '__file__', None)}, not {author_dir}"
            )
    with _prepend_import_path(author_dir):
        for name in expected:
            importlib.import_module(name)
    for name in expected:
        if not _module_is_within(sys.modules[name], author_dir):
            raise RuntimeError(f"PGCN dependency {name} was not loaded from author checkout")
    return sys.modules["model_PGCN"], sys.modules["node_location"]


@lru_cache(maxsize=1)
def _load_bfgcn_model() -> ModuleType:
    """Load the two-file author demo while guarding its flat imports."""

    author_dir = THIRD_PARTY / "BF-GCN"
    expected = ("functions", "model")
    for name in expected:
        existing = sys.modules.get(name)
        if existing is not None and not _module_is_within(existing, author_dir):
            raise RuntimeError(
                f"cannot safely load BF-GCN: sys.modules[{name!r}] comes from "
                f"{getattr(existing, '__file__', None)}, not {author_dir}"
            )
    with _prepend_import_path(author_dir):
        for name in expected:
            importlib.import_module(name)
    for name in expected:
        if not _module_is_within(sys.modules[name], author_dir):
            raise RuntimeError(
                f"BF-GCN dependency {name} was not loaded from author checkout"
            )
    return sys.modules["model"]


@contextmanager
def _prepend_import_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == value:
            sys.path.pop(0)
        elif value in sys.path:
            sys.path.remove(value)


class EmTAdapter(nn.Module):
    """Execute the author ``EmT_Cls.EmT`` with dimension-only adaptation."""

    input_view = "sequence"

    def __init__(self, channels: int, bands: int, num_classes: int) -> None:
        super().__init__()
        valid, reason = compatibility("emt", channels, bands, self.input_view)
        if not valid:
            raise ValueError(reason)
        self.author_module = _load_file_module(
            "_external_author_emt_cls",
            THIRD_PARTY / "EmT" / "EmT_Cls.py",
        )
        config = deepcopy(METHOD_DEFAULTS["emt"])
        self.resolved_config = {
            **config,
            "num_chan": channels,
            "num_feature": bands,
            "num_class": num_classes,
        }
        self.author_model = self.author_module.EmT(**self.resolved_config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"EmT expects [B,S,C,5], got {tuple(inputs.shape)}")
        # The author module stores this as a module-global string rather than a
        # registered buffer.  Updating it is a device fix, not a model change.
        self.author_module.DEVICE = str(inputs.device)
        return self.author_model(inputs)


def _dreamer_pgcn_geometry(
    location_module: ModuleType, distance_delta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze DREAMER coordinates as the author-coordinate 10-20 subset."""

    seed_index = {name.upper(): index for index, name in enumerate(SEED_CHANNELS)}
    indices = [seed_index[name.upper()] for name in DREAMER_CHANNELS]
    coordinate = np.asarray(location_module.return_coordinates(), dtype=np.float32)[
        indices
    ]
    difference = coordinate[:, None, :] - coordinate[None, :, :]
    distance = np.sqrt(np.square(difference).sum(axis=-1))
    adjacency = np.eye(len(coordinate), dtype=np.float32) + distance
    adjacency = distance_delta / np.square(adjacency)
    adjacency = np.where(adjacency > 1.0, 1.0, adjacency)
    adjacency = np.where(adjacency < 0.1, 0.0, adjacency)
    return coordinate, adjacency.astype(np.float32)


class FrozenMontageMesoLayer(nn.Module):
    """Author-equivalent PGCN meso pooling with frozen montage index groups."""

    def __init__(
        self, groups: tuple[tuple[int, ...], ...], coordinate: torch.Tensor
    ) -> None:
        super().__init__()
        flattened = [index for group in groups for index in group]
        expected = list(range(len(coordinate)))
        if sorted(flattened) != expected:
            raise ValueError("frozen PGCN montage groups must cover each channel once")
        self.groups = groups
        self.coordinate = coordinate
        self.trainable_vec = nn.Parameter(torch.empty(len(flattened)))
        self.weight = nn.Parameter(torch.empty(30, 10))
        standard = 1.0 / math.sqrt(len(flattened))
        self.trainable_vec.data.uniform_(-standard, standard)
        self.weight.data.uniform_(-standard, standard)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coarsened_features = []
        coarsened_coordinates = []
        for group in self.groups:
            index = torch.as_tensor(group, device=features.device)
            sub_feature = features.index_select(1, index)
            sub_coordinate = self.coordinate.index_select(0, index)
            projected = torch.einsum("b j g, g h -> b j h", sub_feature, self.weight)
            attention_matrix = torch.einsum(
                "b j h, b h i -> b j i", projected, projected.transpose(1, 2)
            )
            attention = torch.softmax(attention_matrix.sum(dim=2), dim=1)
            pooled_feature = torch.einsum(
                "b j, b j g -> b g", attention, sub_feature
            )
            pooled_coordinate = torch.einsum(
                "b j, j g -> b g", attention, sub_coordinate
            ).mean(dim=0)
            coarsened_features.append(pooled_feature.unsqueeze(1))
            coarsened_coordinates.append(pooled_coordinate.unsqueeze(0))
        return torch.cat(coarsened_features, dim=1), torch.cat(
            coarsened_coordinates, dim=0
        )


class PGCNAdapter(nn.Module):
    """Execute the author PGCN with its fixed 62-electrode spatial pyramid."""

    input_view = "static_mean"

    def __init__(self, channels: int, bands: int, num_classes: int) -> None:
        super().__init__()
        valid, reason = compatibility("pgcn", channels, bands, self.input_view)
        if not valid:
            raise ValueError(reason)
        author_dir = THIRD_PARTY / "PGCN" / "PGCN"
        model_module, location_module = _load_pgcn_modules(author_dir)

        defaults = deepcopy(METHOD_DEFAULTS["pgcn"])
        self.resolved_config = {
            **defaults,
            "in_feature": bands,
            "n_class": num_classes,
            "channels": channels,
            "input_reduction": "mean_over_sequence",
        }
        args = SimpleNamespace(
            in_feature=bands,
            n_class=num_classes,
            dropout=defaults["dropout"],
            lr=defaults["lr"],
            module="",
        )
        if channels == 62:
            adjacency = location_module.convert_dis_m(
                location_module.get_ini_dis_m(), defaults["distance_delta"]
            )
            coordinate = np.asarray(location_module.return_coordinates())
            montage_adaptation = "none_author_62_channel"
        else:
            coordinate, adjacency = _dreamer_pgcn_geometry(
                location_module, defaults["distance_delta"]
            )
            montage_adaptation = (
                "PGCN-14 adaptation: frozen DREAMER 14-channel coordinates, "
                "GENERAL seven-region pooling, and two hemispheres"
            )
        adjacency = nn.Parameter(torch.as_tensor(adjacency, dtype=torch.float32))
        coordinate = torch.as_tensor(
            coordinate, dtype=torch.float32
        )
        self.author_model = model_module.PGCN(args, adjacency, coordinate)
        if channels == 14:
            regional_groups = (
                (0, 13), (1, 2, 11, 12), (3, 10), (4, 9),
                (5, 8), (6,), (7,),
            )
            hemisphere_groups = (
                (0, 1, 2, 3, 4, 5, 6),
                (13, 12, 11, 10, 9, 8, 7),
            )
            self.author_model.meso_layer_1 = FrozenMontageMesoLayer(
                regional_groups, coordinate
            )
            self.author_model.meso_layer_2 = FrozenMontageMesoLayer(
                hemisphere_groups, coordinate
            )
            self.author_model.mlp0 = nn.Linear((14 + 7 + 2) * 70, 2048)
        self.channels = channels
        self.resolved_config["montage_adaptation"] = montage_adaptation
        self.resolved_config["variant"] = (
            "PGCN-14 adaptation" if channels == 14 else "PGCN author 62-channel"
        )

    def _move_author_coordinates(self, device: torch.device) -> None:
        # Coordinates are plain author-code attributes, not registered buffers.
        coordinate = self.author_model.coordinate.to(device)
        self.author_model.coordinate = coordinate
        self.author_model.meso_layer_1.coordinate = coordinate
        self.author_model.meso_layer_2.coordinate = coordinate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (self.channels, 5):
            raise ValueError(
                f"PGCN expects [B,{self.channels},5], got {tuple(inputs.shape)}"
            )
        self._move_author_coordinates(inputs.device)
        logits, _, _ = self.author_model(inputs)
        return logits


class HDGNetAdapter(nn.Module):
    """Execute author HDGNet with its predefined GENERAL local-global graph."""

    input_view = "static_mean"
    outputs_probabilities = True

    def __init__(self, channels: int, bands: int, num_classes: int) -> None:
        super().__init__()
        valid, reason = compatibility("hdgnet", channels, bands, self.input_view)
        if not valid:
            raise ValueError(reason)
        model_module, seed_constants, dreamer_constants = _load_hdgnet_modules()
        defaults = deepcopy(METHOD_DEFAULTS["hdgnet"])
        if channels == 62:
            region_mask = seed_constants.SEED_GENERAL_REGION_MASK_MATRIX
            region_list = seed_constants.SEED_GENERAL_REGION_LIST
            graph_source = "SEED_GENERAL"
            hid_channels = defaults["seed_hid_channels"]
            out_channels = defaults["seed_out_channels"]
        else:
            region_mask = dreamer_constants.DREAMER_GENERAL_REGION_MASK_MATRIX
            region_list = dreamer_constants.DREAMER_GENERAL_REGION_LIST
            graph_source = "DREAMER_GENERAL"
            hid_channels = defaults["dreamer_hid_channels"]
            out_channels = defaults["dreamer_out_channels"]

        self.resolved_config = {
            **defaults,
            "num_electrodes": channels,
            "in_channels": bands,
            "num_classes": num_classes,
            "hid_channels": hid_channels,
            "out_channels": out_channels,
            "graph_source": graph_source,
            "input_reduction": "mean_over_sequence",
            "author_forward_output": "softmax_probability",
        }
        self.channels = channels
        self.bands = bands
        self.author_model = model_module.HDGNet(
            region_mask=region_mask,
            region_list=region_list,
            num_electrodes=channels,
            in_channels=bands,
            hid_channels=hid_channels,
            out_channels=out_channels,
            num_classes=num_classes,
            dropout=defaults["dropout"],
            l_fc=defaults["l_fc"],
            g_fc=defaults["g_fc"],
            get_adj=defaults["get_adj"],
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected = (self.channels, self.bands)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                f"HDGNet expects [B,{self.channels},{self.bands}], "
                f"got {tuple(inputs.shape)}"
            )
        return self.author_model(inputs)


class BFGCNDEAdapter(nn.Module):
    """Author BF-GCN with a frozen train-only DE-correlation graph.

    The public implementation requires a band-wise PLV tensor, but the frozen
    feature caches contain DE only.  This adapter changes that one unavailable
    input to a band-wise absolute DE-correlation proxy and keeps the released
    graph-fusion/model code and Simple_Demo architecture intact.
    """

    input_view = "static_mean"

    def __init__(
        self,
        channels: int,
        bands: int,
        num_classes: int,
        functional_adjacency: np.ndarray,
    ) -> None:
        super().__init__()
        valid, reason = compatibility(
            "bf_gcn_de", channels, bands, self.input_view
        )
        if not valid:
            raise ValueError(reason)
        adjacency = np.asarray(functional_adjacency, dtype=np.float32)
        expected = (channels, channels, bands)
        if adjacency.shape != expected:
            raise ValueError(
                f"BF-GCN-DE graph must have shape {expected}, got {adjacency.shape}"
            )
        if not np.all(np.isfinite(adjacency)):
            raise ValueError("BF-GCN-DE graph contains non-finite values")
        if adjacency.min() < 0 or adjacency.max() > 1:
            raise ValueError("BF-GCN-DE graph must be bounded in [0,1]")
        self.channels = channels
        self.bands = bands
        self.register_buffer(
            "functional_adjacency", torch.from_numpy(adjacency.copy())
        )
        defaults = deepcopy(METHOD_DEFAULTS["bf_gcn_de"])
        self.resolved_config = {
            **defaults,
            "num_electrodes": channels,
            "in_channels": bands,
            "num_classes": num_classes,
            "att_plv_hidden": channels,
            "input_reduction": "mean_over_sequence",
            "functional_graph_bands": bands,
            "official_model_variant": "BFGCN subject-dependent FC branch",
        }
        # The released model allocates graph parameters with .cuda() in its
        # constructors.  The formal runner sets the requested current device
        # immediately before reaching this point.
        author_module = _load_bfgcn_model()
        self.author_model = author_module.BFGCN(
            nclass=num_classes,
            xdim=[channels, bands],
            kadj=defaults["kadj"],
            num_out=defaults["num_out"],
            att_hidden=defaults["att_hidden"],
            att_plv_hidden=channels,
            classifier_hidden=defaults["classifier_hidden"],
            avgpool=defaults["avgpool"],
            dropout=defaults["dropout"],
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected = (self.channels, self.bands)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                f"BF-GCN-DE expects [B,{self.channels},{self.bands}], "
                f"got {tuple(inputs.shape)}"
            )
        functional = self.functional_adjacency.unsqueeze(0).expand(
            len(inputs), -1, -1, -1
        )
        outputs = self.author_model(inputs, functional, alpha=0.0)
        return outputs[0]


def build_model(
    method: str,
    *,
    channels: int,
    bands: int,
    num_classes: int,
    functional_adjacency: np.ndarray | None = None,
) -> nn.Module:
    method = method.lower()
    if method == "emt":
        return EmTAdapter(channels, bands, num_classes)
    if method == "pgcn":
        return PGCNAdapter(channels, bands, num_classes)
    if method == "hdgnet":
        return HDGNetAdapter(channels, bands, num_classes)
    if method == "bf_gcn_de":
        if functional_adjacency is None:
            raise ValueError("BF-GCN-DE requires a frozen functional adjacency")
        return BFGCNDEAdapter(
            channels, bands, num_classes, functional_adjacency
        )
    if method == "dgcnn":
        defaults = METHOD_DEFAULTS["dgcnn"]
        return DGCNNBaseline(
            channels, bands, num_classes,
            order=defaults["order"],
            graph_features=defaults["graph_features"],
            dropout=defaults["dropout"],
        )
    if method == "rgcb":
        defaults = METHOD_DEFAULTS["rgcb"]
        return RGCBBaseline(
            channels, bands, num_classes,
            order=defaults["order"],
            graph_features=defaults["graph_features"],
            dropout=defaults["dropout"],
        )
    if method == "dgclink":
        defaults = METHOD_DEFAULTS["dgclink"]
        return DGCLinkBaseline(
            channels, bands, num_classes,
            order=defaults["order"],
            graph_features=defaults["graph_features"],
            classifier_hidden=defaults["classifier_hidden"],
            dropout=defaults["dropout"],
        )
    raise ValueError(f"unsupported external method: {method}")
