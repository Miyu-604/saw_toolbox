from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .voigt import voigtC_to_tensor4, voigt_e_to_tensor3, tensor4_to_voigtC, tensor3_to_voigt_e
from .rotation import rotate_rank2, rotate_rank3, rotate_rank4, assert_rotation_matrix


@dataclass(frozen=True)
class PiezoMaterial:
    """
    Minimal piezoelectric material container.

    Conventions:
      - C6: stiffness in Voigt (shear convention set by `shear`), units: Pa
      - e36: piezo tensor in Voigt (shear convention set by `shear`), units: C/m^2
      - eps: permittivity tensor (rank-2), units: F/m
      - rho: density, units: kg/m^3
    """
    name: str
    rho: float
    C6: np.ndarray          # (6,6) Pa
    e36: np.ndarray         # (3,6) C/m^2
    eps: np.ndarray         # (3,3) F/m
    shear: str = "tensorial"

    def __post_init__(self):
        C6 = np.asarray(self.C6, float)
        e36 = np.asarray(self.e36, float)
        eps = np.asarray(self.eps, float)
        if C6.shape != (6, 6):
            raise ValueError("C6 must be (6,6).")
        if e36.shape != (3, 6):
            raise ValueError("e36 must be (3,6).")
        if eps.shape != (3, 3):
            raise ValueError("eps must be (3,3).")

        shear = _normalize_shear(self.shear)

        # enforce symmetry on stiffness for sanity
        C6s = 0.5 * (C6 + C6.T)
        object.__setattr__(self, "C6", C6s)
        object.__setattr__(self, "e36", e36)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "shear", shear)


def _normalize_shear(shear: str | None) -> str:
    if shear is None:
        return "tensorial"
    key = str(shear).strip().lower().replace("-", "").replace("_", "")
    if key in ("tensorial", "tensor", "true", "epsilon", "strain"):
        return "tensorial"
    if key in ("engineering", "engineer", "eng", "gamma"):
        return "engineering"
    raise ValueError(f"Unknown shear convention: {shear}")


def _strain_scale_from_shear(shear: str) -> np.ndarray:
    if shear == "engineering":
        return np.array([1, 1, 1, 2, 2, 2], dtype=float)
    return np.array([1, 1, 1, 1, 1, 1], dtype=float)

    # --- tensor views (true strain tensors) ---
    @property
    def C4(self) -> np.ndarray:
        return voigtC_to_tensor4(self.C6, strain_scale=_strain_scale_from_shear(self.shear))

    @property
    def e3(self) -> np.ndarray:
        return voigt_e_to_tensor3(self.e36, strain_scale=_strain_scale_from_shear(self.shear))

    # --- rotation ---
    def rotated(self, R: np.ndarray, *, name_suffix: str | None = None) -> "PiezoMaterial":
        """
        Rotate material tensors:
          C'ijkl = R_ip R_jq R_kr R_ls C_pqrs
          e'kij  = R_kp R_iq R_jr e_pqr
          eps'   = R eps R^T
        Return rotated material in the same Voigt convention (`shear`).
        """
        assert_rotation_matrix(R)

        C4r = rotate_rank4(self.C4, R)
        e3r = rotate_rank3(self.e3, R)
        epsr = rotate_rank2(self.eps, R)

        scale = _strain_scale_from_shear(self.shear)
        C6r = tensor4_to_voigtC(C4r, strain_scale=scale)
        e36r = tensor3_to_voigt_e(e3r, strain_scale=scale)

        new_name = self.name if name_suffix is None else f"{self.name}_{name_suffix}"
        return PiezoMaterial(new_name, self.rho, C6r, e36r, epsr, shear=self.shear)
