from __future__ import annotations

import numpy as np

from ..material import VoigtMaterial


def _flatten_col_major(A: np.ndarray) -> np.ndarray:
    """Return A flattened in COMSOL-friendly order: (i varies fastest) -> column-major."""
    return np.asarray(A, dtype=float).reshape(-1, order="F")


def _var_list(prefix: str, shape: tuple[int, int]) -> str:
    """Build {prefix11, prefix21, ..., prefixm1, prefix12, ...} in column-major."""
    m, n = shape
    names = [f"{prefix}{i + 1}{j + 1}" for j in range(n) for i in range(m)]
    return "{" + ", ".join(names) + "}"


def _value_list(
    A: np.ndarray,
    *,
    unit: str | None,
    fmtstr: str,
) -> str:
    """Build {v11, v21, ...} matching _var_list order."""
    vals = _flatten_col_major(A)
    items = [fmtstr.format(x) for x in vals]
    return "{" + ", ".join(items) + "}"


def _block(
    *,
    prefix: str,
    A: np.ndarray,
    unit: str | None,
    fmtstr: str,
) -> str:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be rank-2 for COMSOL matrix export.")
    return "\n".join(
        [
            "variables:",
            _var_list(prefix, (A.shape[0], A.shape[1])),
            "value:",
            _value_list(A, unit=unit, fmtstr=fmtstr),
        ]
    )


def to_comsol_text(
    mat: VoigtMaterial,
    *,
    eps_as_relative: bool = True,
    eps0: float = 8.854187817e-12,
    c_prefix: str = "cE",
    c_unit: str = "Pa",
    c_scale: float = 1.0,
    e_prefix: str = "eES",
    e_unit: str = "C/m^2",
    e_scale: float = 1.0,
    eps_prefix: str = "epsilonrS",
    eps_unit: str | None = "",
    eps_scale: float = 1.0,
    fmt: str = "{: .6e}",
) -> str:
    """Export COMSOL-ready assignment blocks.

    This prints `variables:` / `value:` blocks that can be pasted into COMSOL.

    Ordering:
      - Variables and values are listed in column-major order:
        (11,21,31,...,m1, 12,22,32,...)

    Defaults:
      - C6 is exported in Pa (mat.C6 is Pa).
      - e36 is exported with prefix `eES` in C/m^2 (mat.e36 is C/m^2).
      - eps is exported as relative permittivity (dimensionless) by default.

    Output formatting:
      - Values are numeric only; unit parameters are retained for scaling/reference
        but are not printed.

    Scaling:
      - You can change units by scaling values yourself, e.g. export GPa via
        `c_unit="GPa", c_scale=1e-9` (since mat.C6 is Pa).
    """
    if mat.shear != "engineering":
        mat = mat.to_tensor().to_voigt(shear="engineering")

    # --- C (6x6) ---
    C6 = np.asarray(mat.C6, float) * c_scale

    # --- e (3x6) ---
    e36 = np.asarray(mat.e36, float) * e_scale

    # --- eps (3x3) ---
    if eps_as_relative:
        eps = (np.asarray(mat.eps, float) / eps0) * eps_scale
        # dimensionless: default to no unit suffix
        eps_unit_eff = "" if eps_unit is None else eps_unit
    else:
        eps = np.asarray(mat.eps, float) * eps_scale
        eps_unit_eff = "F/m" if (eps_unit is None or eps_unit == "") else eps_unit

    out: list[str] = []
    out.append(f"# Material: {mat.name}")
    out.append(f"# rho (kg/m^3): {mat.rho:.6g}")

    out.append("# C (6x6) (Voigt)")
    out.append(_block(prefix=c_prefix, A=C6, unit=c_unit, fmtstr=fmt))

    out.append("# e (3x6) (Voigt)")
    out.append(_block(prefix=e_prefix, A=e36, unit=e_unit, fmtstr=fmt))

    out.append(f"# eps (3x3) ({'relative' if eps_as_relative else 'absolute'})")
    out.append(_block(prefix=eps_prefix, A=eps, unit=eps_unit_eff, fmtstr=fmt))

    return "\n".join(out)
