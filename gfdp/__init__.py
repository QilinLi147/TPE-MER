"""GFDP-MER: Glance-Focus Differential Prototype network (ante-hoc + post-hoc)."""
__version__ = "1.5"

from .model import GFDPModel, ModelConfig, restore
from .train import TrainConfig, Trainer

__all__ = ["__version__", "GFDPModel", "ModelConfig", "TrainConfig", "Trainer", "restore"]
