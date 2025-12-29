from __future__ import annotations

import numpy as np

from ..material import VoigtMaterial

EPS0 = 8.854187817e-12  # F/m


def _format_matrix(
    A: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    fmt: str,
) -> str:
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if len(row_labels) != m or len(col_labels) != n:
        raise ValueError("row_labels/col_labels must match matrix shape.")

    data = [[fmt.format(x) for x in row] for row in A]

    col_widths = []
    for j in range(n):
        width = max(len(col_labels[j]), max(len(data[i][j]) for i in range(m)))
        col_widths.append(width)

    row_label_width = max(len(lbl) for lbl in row_labels)

    header = " " * row_label_width + " | " + " ".join(
        col_labels[j].rjust(col_widths[j]) for j in range(n)
    )
    sep = "-" * row_label_width + "-+-" + "-".join("-" * w for w in col_widths)

    lines = [header, sep]
    for i in range(m):
        line = row_labels[i].rjust(row_label_width) + " | " + " ".join(
            data[i][j].rjust(col_widths[j]) for j in range(n)
        )
        lines.append(line)

    return "\n".join(lines)


def to_visual_text(
    mat: VoigtMaterial,
    *,
    eps_as_relative: bool = True,
    eps0: float = EPS0,
    c_unit: str = "GPa",
    c_scale: float = 1e-9,
    e_unit: str = "C/m^2",
    e_scale: float = 1.0,
    eps_unit: str | None = None,
    eps_scale: float = 1.0,
    fmt: str = "{: .6g}",
) -> str:
    """
    Render a human-readable table for Voigt material constants.

    Defaults:
      - C6 is shown in GPa.
      - eps is shown as relative permittivity (eps/eps0).
    """
    voigt_labels = ["xx", "yy", "zz", "yz", "xz", "xy"]

    C6 = np.asarray(mat.C6, float) * c_scale
    e36 = np.asarray(mat.e36, float) * e_scale

    if eps_as_relative:
        eps = (np.asarray(mat.eps, float) / eps0) * eps_scale
        eps_unit_eff = "eps_r" if eps_unit is None else eps_unit
    else:
        eps = np.asarray(mat.eps, float) * eps_scale
        eps_unit_eff = "F/m" if (eps_unit is None or eps_unit == "") else eps_unit

    out: list[str] = []
    out.append(f"# Material: {mat.name}")
    out.append(f"# rho (kg/m^3): {mat.rho:.6g}")
    out.append(f"# shear: {mat.shear}")
    out.append("")

    out.append(f"C6 (6x6) [{c_unit}]")
    out.append(
        _format_matrix(C6, row_labels=voigt_labels, col_labels=voigt_labels, fmt=fmt)
    )
    out.append("")

    out.append(f"e (3x6) [{e_unit}]")
    out.append(
        _format_matrix(
            e36, row_labels=["x", "y", "z"], col_labels=voigt_labels, fmt=fmt
        )
    )
    out.append("")

    out.append(f"eps (3x3) [{eps_unit_eff}]")
    out.append(
        _format_matrix(
            eps, row_labels=["x", "y", "z"], col_labels=["x", "y", "z"], fmt=fmt
        )
    )

    return "\n".join(out)
