from __future__ import annotations
import numpy as np

# 0..5 <-> (xx, yy, zz, yz, xz, xy)
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

# Strain-side scaling for Voigt shear convention.
# This project uses the *tensorial* shear convention by default:
#   E6 = [εxx, εyy, εzz, εyz, εxz, εxy]
# hence no extra scaling is needed (all ones).
# If your input matrices are written with *engineering* shear γ=2ε (common in some texts),
# you can adapt by setting:
#   VOIGT_STRAIN_SCALE = [1, 1, 1, 2, 2, 2]
# so that tensor forms (rank-3/4) are consistent with true strain ε.
VOIGT_STRAIN_SCALE = np.array([1, 1, 1, 1, 1, 1], dtype=float)


def _get_strain_scale(strain_scale: np.ndarray | None) -> np.ndarray:
    if strain_scale is None:
        return VOIGT_STRAIN_SCALE
    s = np.asarray(strain_scale, dtype=float)
    if s.shape != (6,):
        raise ValueError("strain_scale must be shape (6,).")
    return s


def voigtC_to_tensor4(C6: np.ndarray, *, strain_scale: np.ndarray | None = None) -> np.ndarray:
    """
    6x6 stiffness (Voigt) -> 3x3x3x3 stiffness tensor for true strain:
        sigma_ij = C_ijkl * epsilon_kl

    This module returns tensor forms consistent with *true strain* ε.
    If your Voigt matrices use engineering shear γ=2ε on the strain-like 6-index,
    set `VOIGT_STRAIN_SCALE = [1,1,1,2,2,2]` so that shear components are scaled on the
    strain side when mapping C6 -> C4.

    (Current project default: tensorial shear, so `VOIGT_STRAIN_SCALE` is all ones.)
    """
    C6 = np.asarray(C6, dtype=float)
    if C6.shape != (6, 6):
        raise ValueError("C6 must be (6,6).")

    s = _get_strain_scale(strain_scale)
    C4 = np.zeros((3, 3, 3, 3), dtype=float)

    for I, (i, j) in enumerate(VOIGT_PAIRS):
        for J, (k, l) in enumerate(VOIGT_PAIRS):
            val = C6[I, J] * s[J]
            C4[i, j, k, l] = val
            C4[j, i, k, l] = val
            C4[i, j, l, k] = val
            C4[j, i, l, k] = val

    # major symmetry cleanup
    C4 = 0.5 * (C4 + C4.swapaxes(0, 2).swapaxes(1, 3))
    return C4


def tensor4_to_voigtC(C4: np.ndarray, *, strain_scale: np.ndarray | None = None) -> np.ndarray:
    """Inverse of voigtC_to_tensor4 under same convention: C6[I,J] = C4 / scale[J]."""
    C4 = np.asarray(C4, dtype=float)
    if C4.shape != (3, 3, 3, 3):
        raise ValueError("C4 must be (3,3,3,3).")

    s = _get_strain_scale(strain_scale)
    C6 = np.zeros((6, 6), dtype=float)

    for I, (i, j) in enumerate(VOIGT_PAIRS):
        for J, (k, l) in enumerate(VOIGT_PAIRS):
            C6[I, J] = C4[i, j, k, l] / s[J]

    return 0.5 * (C6 + C6.T)


def voigt_e_to_tensor3(e36: np.ndarray, *, strain_scale: np.ndarray | None = None) -> np.ndarray:
    """
    3x6 piezo tensor e (Voigt, engineering shear on 6-index) -> 3x3x3 tensor e_kij.

    Assumption (consistent with many COMSOL-style inputs):
        T_ij = ... - e_kij E_k  (sign is handled in the solver equations)
    with (ij) symmetric and the 6-index being strain-like.
    `VOIGT_STRAIN_SCALE[J]` is applied on the strain side to adapt shear conventions when needed:
        e_kij = e36[k,J] * scale[J]
    """
    e36 = np.asarray(e36, dtype=float)
    if e36.shape != (3, 6):
        raise ValueError("e36 must be (3,6).")

    s = _get_strain_scale(strain_scale)
    e3 = np.zeros((3, 3, 3), dtype=float)  # [k,i,j]

    for k in range(3):
        for J, (i, j) in enumerate(VOIGT_PAIRS):
            val = e36[k, J] * s[J]
            e3[k, i, j] = val
            e3[k, j, i] = val

    return e3


def tensor3_to_voigt_e(e3: np.ndarray, *, strain_scale: np.ndarray | None = None) -> np.ndarray:
    """Inverse of voigt_e_to_tensor3: e36[k,J] = e3 / scale[J]."""
    e3 = np.asarray(e3, dtype=float)
    if e3.shape != (3, 3, 3):
        raise ValueError("e3 must be (3,3,3).")

    s = _get_strain_scale(strain_scale)
    e36 = np.zeros((3, 6), dtype=float)

    for k in range(3):
        for J, (i, j) in enumerate(VOIGT_PAIRS):
            e36[k, J] = e3[k, i, j] / s[J]

    return e36
