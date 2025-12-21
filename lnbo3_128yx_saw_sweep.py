from __future__ import annotations

import numpy as np

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

        if ang % 10 == 0:
            print(
                f"[Rz={ang:5.1f} deg] short={v_s:8.2f} (err={err_s:.2e}) "
                f"open={v_o:8.2f} (err={err_o:.2e}) k2={k2[-1]*100:7.3f} %"
            )

    angles_deg = np.asarray(angles_deg)
    v_short = np.asarray(v_short)
    v_open = np.asarray(v_open)
    k2 = np.asarray(k2)

    # --- Plot results ---
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(angles_deg, v_short, "o-", label="Short")
    ax1.plot(angles_deg, v_open, "s-", label="Open")
    ax1.set_xlabel("Rz angle (deg)")
    ax1.set_ylabel("SAW velocity (m/s)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(angles_deg, k2 * 100.0, "d-", color="tab:red", label="K^2")
    ax2.set_ylabel("K^2 (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
