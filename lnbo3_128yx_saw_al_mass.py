from __future__ import annotations

from piezo_stroh.io import MaterialDB
from piezo_stroh.rotation import R_yxcut_theta_xprop
from piezo_stroh.stroh.piezo_saw import PiezoAlSAWSolver

import numpy as np


def main():
    # --- Load LiNbO3 (crystal axes) from YAML DB ---
    db = MaterialDB()
    ln_voigt = db.get("linbo3", "bulk_ogi2002")
    ln = ln_voigt.to_tensor()

    # --- Rotate to 128YX (x propagation) ---
    R = R_yxcut_theta_xprop(127.86)
    ln128 = ln.rotated(R, name_suffix="128YX_xprop")

    # --- Al mass loading settings ---
    wavelength = 1.0e-6  # [m], ratio controls the mass-load term

    solver = PiezoAlSAWSolver(ln128)

    t_over_lambda = np.concatenate(([0.0], np.logspace(-4, -1, 61)))
    velocities = []
    errors = []

    for i, ratio in enumerate(t_over_lambda):
        thickness = ratio * wavelength
        if i == 0:
            v_guess = 4000.0
            search_range = 500.0
        else:
            v_guess = velocities[-1]
            search_range = 100.0
        v_short = solver.find_velocity_al(
            thickness,
            wavelength,
            electric_bc="short",
            v_guess=v_guess,
            search_range=search_range,
        )
        velocities.append(v_short)
        B = solver.boundary_matrix_with_al(
            v_short,
            thickness,
            wavelength,
            electric_bc="short",
        )
        errors.append(float(np.linalg.svd(B, compute_uv=False)[-1]))

    print(f"[{ln128.name}]")
    print(f"t/lambda sweep: {t_over_lambda[0]:.4g} to {t_over_lambda[-1]:.4g}")
    print(f"wavelength = {wavelength:.3e} m")

    comsol_t = None
    comsol_v = None
    try:
        comsol = np.genfromtxt(
            "comsol_data/ln_al_mass.csv",
            delimiter=",",
            skip_header=1,
        )
        if comsol.ndim == 2 and comsol.shape[1] >= 2:
            comsol_t = comsol[:, 0]
            comsol_v = comsol[:, 1]
    except OSError:
        pass

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.rcParams["font.family"] = "Meiryo"
    plt.figure(figsize=(7, 5))
    plt.plot(t_over_lambda, velocities, label="Stroh")
    if comsol_t is not None:
        plt.scatter(
            comsol_t,
            comsol_v,
            label="FEM",
            facecolors="none",
            edgecolors="tab:orange",
            s=20,
        )
    plt.xlabel("t/λ")
    plt.ylabel("SAW velocity (m/s)")
    plt.title("LiNbO3 128YX with Al mass loading (short)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(t_over_lambda, errors, s=20)
    plt.xlabel("t/λ")
    plt.ylabel("min singular value (err)")
    plt.title("Boundary condition error (short)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
