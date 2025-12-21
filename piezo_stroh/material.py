from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .voigt import voigtC_to_tensor4, voigt_e_to_tensor3, tensor4_to_voigtC, tensor3_to_voigt_e
from .rotation import rotate_rank2, rotate_rank3, rotate_rank4, assert_rotation_matrix


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


@dataclass(frozen=True)
class VoigtMaterial:
    """
    Voigt-based material container (data layer).

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

    def to_tensor(self) -> "TensorMaterial":
        scale = _strain_scale_from_shear(self.shear)
        C4 = voigtC_to_tensor4(self.C6, strain_scale=scale)
        e3 = voigt_e_to_tensor3(self.e36, strain_scale=scale)
        return TensorMaterial(self.name, self.rho, C4, e3, self.eps)


@dataclass(frozen=True)
class TensorMaterial:
    """
    Tensor-based material container (physics layer).

    Conventions:
      - C4: stiffness tensor for true strain ε
      - e3: piezo tensor e_kij with ij symmetric
      - eps: permittivity tensor (rank-2), units: F/m
      - rho: density, units: kg/m^3
    """
    name: str
    rho: float
    C4: np.ndarray          # (3,3,3,3)
    e3: np.ndarray          # (3,3,3)
    eps: np.ndarray         # (3,3)

    def __post_init__(self):
        C4 = np.asarray(self.C4, float)
        e3 = np.asarray(self.e3, float)
        eps = np.asarray(self.eps, float)
        if C4.shape != (3, 3, 3, 3):
            raise ValueError("C4 must be (3,3,3,3).")
        if e3.shape != (3, 3, 3):
            raise ValueError("e3 must be (3,3,3).")
        if eps.shape != (3, 3):
            raise ValueError("eps must be (3,3).")

        object.__setattr__(self, "C4", C4)
        object.__setattr__(self, "e3", e3)
        object.__setattr__(self, "eps", eps)

    def rotated(self, R: np.ndarray, *, name_suffix: str | None = None) -> "TensorMaterial":
        """
        Rotate material tensors:
          C'ijkl = R_ip R_jq R_kr R_ls C_pqrs
          e'kij  = R_kp R_iq R_jr e_pqr
          eps'   = R eps R^T
        """
        assert_rotation_matrix(R)

        C4r = rotate_rank4(self.C4, R)
        e3r = rotate_rank3(self.e3, R)
        epsr = rotate_rank2(self.eps, R)

        new_name = self.name if name_suffix is None else f"{self.name}_{name_suffix}"
        return TensorMaterial(new_name, self.rho, C4r, e3r, epsr)

    def to_voigt(self, *, shear: str = "tensorial") -> VoigtMaterial:
        shear = _normalize_shear(shear)
        scale = _strain_scale_from_shear(shear)
        C6 = tensor4_to_voigtC(self.C4, strain_scale=scale)
        e36 = tensor3_to_voigt_e(self.e3, strain_scale=scale)
        return VoigtMaterial(self.name, self.rho, C6, e36, self.eps, shear=shear)
