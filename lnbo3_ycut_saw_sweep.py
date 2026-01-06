from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from piezo_stroh.io import MaterialDB
from piezo_stroh.rotation import R_y2z, Rz
from piezo_stroh.stroh.piezo_saw import PiezoSAWSolver


def main():
    # --- Load LiNbO3 (crystal axes) from YAML DB ---
    db = MaterialDB()
    ln_voigt = db.get("linbo3", "bulk_ogi2002")
    ln = ln_voigt.to_tensor()

    # --- Base cut orientation (Y-cut, x propagation) ---
    R_cut = R_y2z()

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
        mat = ln.rotated(R, name_suffix=f"Ycut_xprop_Rz{ang:.0f}")

        solver = PiezoSAWSolver(mat)

        if len(v_short) == 0:
            vmin_s, vmax_s = 3000.0, 4500.0
            vmin_o, vmax_o = 3000.0, 4500.0
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
    plt.rcParams["font.family"] = "Meiryo"
    plt.rcParams["lines.markersize"] = 2
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 6.5), nrows=2, sharex=True)
    ax1.plot(angles_deg, v_short, label="Stroh - 短絡条件")
    ax1.plot(angles_deg, v_open, label="Stroh - 開放条件")
    ax1.set_ylabel("SAW velocity (m/s)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper center")

    ax2.plot(angles_deg, k2 * 100.0, color="tab:red", label="Stroh - K^2")
    ax2.set_xlabel("Angle (deg)")
    ax2.set_ylabel("K^2 (%)")
    ax2.set_ylim(0, 6)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper center")

    plt.tight_layout()
    plt.show()

    fig_err, ax_err = plt.subplots(figsize=(8, 3.8))
    ax_err.plot(angles_deg, err_short, label="Short err")
    ax_err.plot(angles_deg, err_open, label="Open err")
    ax_err.set_xlabel("Angle (deg)")
    ax_err.set_ylabel("Solver err")
    ax_err.set_yscale("log")
    ax_err.grid(True, linestyle=":", alpha=0.6)
    ax_err.legend(loc="upper center")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
