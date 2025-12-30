from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from piezo_stroh.io import MaterialDB
from piezo_stroh.rotation import R_yxcut_theta_xprop, Rz
from piezo_stroh.stroh.piezo_saw import PiezoSAWSolver


def main():
    # --- Load LiNbO3 (crystal axes) from YAML DB ---
    db = MaterialDB()
    ln_voigt = db.get("linbo3", "bulk_ogi2002")
    ln = ln_voigt.to_tensor()

    # --- Base cut orientation (128 YX, x propagation) ---
    R_cut = R_yxcut_theta_xprop(127.86)

    # --- Sweep in-plane propagation direction (about +z) ---
    angles_deg = np.arange(0.0, 180.0 + 1e-9, 1.0)
    v_short = []
    v_open = []
    k2 = []
    err_short = []
    err_open = []

    for ang in angles_deg:
        R_inplane = Rz(ang)
        R = R_inplane @ R_cut
        mat = ln.rotated(R, name_suffix=f"128YX_xprop_Rz{ang:.0f}")

        solver = PiezoSAWSolver(mat)

        if len(v_short) == 0:
            vmin_s, vmax_s = 3760.0, 3960.0
            vmin_o, vmax_o = 3870.0, 4070.0
        else:
            vmin_s, vmax_s = v_short[-1] - 100.0, v_short[-1] + 100.0
            vmin_o, vmax_o = v_open[-1] - 100.0, v_open[-1] + 100.0

        v_s, err_s = solver.find_velocity(electric_bc="short", vmin=vmin_s, vmax=vmax_s)
        v_o, err_o = solver.find_velocity(electric_bc="open", vmin=vmin_o, vmax=vmax_o)

        v_short.append(v_s)
        v_open.append(v_o)
        k2.append(2 * (v_o - v_s) / v_o)
        err_short.append(err_s)
        err_open.append(err_o)

        if ang % 10 == 0:
            print(
                f"[Rz={ang:5.1f} deg] short={v_s:8.2f} (err={err_s:.2e}) "
                f"open={v_o:8.2f} (err={err_o:.2e}) k2={k2[-1]*100:7.3f} %"
            )

    angles_deg = np.asarray(angles_deg)
    v_short = np.asarray(v_short)
    v_open = np.asarray(v_open)
    k2 = np.asarray(k2)
    err_short = np.asarray(err_short)
    err_open = np.asarray(err_open)

    # --- Plot results ---
    plt.rcParams["lines.markersize"] = 2
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(8, 9), nrows=3, sharex=True)
    ax1.scatter(angles_deg, v_short, label="Short")
    ax1.scatter(angles_deg, v_open, label="Open")
    ax1.set_ylabel("SAW velocity (m/s)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2.scatter(angles_deg, k2 * 100.0, color="tab:red", label="K^2")
    ax2.set_xlabel("Rz angle (deg)")
    ax2.set_ylabel("K^2 (%)")
    ax2.set_ylim(0, 6)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    ax3.scatter(angles_deg, err_short, label="Short err")
    ax3.scatter(angles_deg, err_open, label="Open err")
    ax3.set_xlabel("Rz angle (deg)")
    ax3.set_ylabel("Solver err")
    ax3.set_yscale("log")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
