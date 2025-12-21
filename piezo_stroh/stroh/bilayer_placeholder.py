from __future__ import annotations
from dataclasses import dataclass
from ..material import TensorMaterial

@dataclass(frozen=True)
class BilayerConfig:
    top: TensorMaterial
    bottom: TensorMaterial
    thickness_top_m: float  # placeholder

class BilayerStrohSolver:
    """
    Placeholder for future 2-layer Stroh implementation.
    """
    def __init__(self, cfg: BilayerConfig):
        self.cfg = cfg

    def solve(self, *args, **kwargs):
        raise NotImplementedError(
            "Bilayer Stroh solver is not implemented yet. "
            "This class is a placeholder to keep the architecture stable."
        )
