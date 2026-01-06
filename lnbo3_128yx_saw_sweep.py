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

    comsol_angles = None
    comsol_open = None
    comsol_short = None
    comsol_k2_pct = None
    try:
        comsol = np.genfromtxt(
            "comsol_data/ln_rotation.csv",
            delimiter=",",
            skip_header=1,
        )
        if comsol.ndim == 2 and comsol.shape[1] >= 4:
            comsol_angles = comsol[:, 0]
            comsol_open = comsol[:, 1] * 10.0
            comsol_short = comsol[:, 2] * 10.0
            comsol_k2_pct = comsol[:, 3] * 100.0

            # mirror around 90 deg to fill 90..180
            mask_lt90 = comsol_angles < 90.0
            angles_mirror = 180.0 - comsol_angles[mask_lt90]
            comsol_angles = np.concatenate([comsol_angles, angles_mirror])
            comsol_open = np.concatenate([comsol_open, comsol_open[mask_lt90]])
            comsol_short = np.concatenate([comsol_short, comsol_short[mask_lt90]])
            comsol_k2_pct = np.concatenate([comsol_k2_pct, comsol_k2_pct[mask_lt90]])
    except OSError:
        pass

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

    def rel_span_pct(values: np.ndarray, ref_value: float) -> float:
        span = float(np.max(values) - np.min(values))
        if ref_value == 0.0:
            return float("nan")
        return 100.0 * span / ref_value

    stroh_ref_short = float(v_short[0])
    stroh_ref_open = float(v_open[0])
    print(
        f"Stroh short span: {rel_span_pct(v_short, stroh_ref_short):.3f}% "
        f"(min={np.min(v_short):.2f}, max={np.max(v_short):.2f}, ref v0={stroh_ref_short:.2f})"
    )
    print(
        f"Stroh open  span: {rel_span_pct(v_open, stroh_ref_open):.3f}% "
        f"(min={np.min(v_open):.2f}, max={np.max(v_open):.2f}, ref v0={stroh_ref_open:.2f})"
    )

    if comsol_angles is not None:
        idx0 = np.where(np.isclose(comsol_angles, 0.0))[0]
        if idx0.size > 0:
            comsol_ref_short = float(comsol_short[idx0[0]])
            comsol_ref_open = float(comsol_open[idx0[0]])
            print(
                f"FEM short span: {rel_span_pct(comsol_short, comsol_ref_short):.3f}% "
                f"(min={np.min(comsol_short):.2f}, max={np.max(comsol_short):.2f}, ref v0={comsol_ref_short:.2f})"
            )
            print(
                f"FEM open  span: {rel_span_pct(comsol_open, comsol_ref_open):.3f}% "
                f"(min={np.min(comsol_open):.2f}, max={np.max(comsol_open):.2f}, ref v0={comsol_ref_open:.2f})"
            )

    # --- Plot results ---
    plt.rcParams["font.family"] = "Meiryo"
    plt.rcParams["lines.markersize"] = 2
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 6.5), nrows=2, sharex=True)
    ax1.plot(angles_deg, v_short, label="Stroh - 短絡条件")
    ax1.plot(angles_deg, v_open, label="Stroh - 開放条件")
    if comsol_angles is not None:
        ax1.scatter(
            comsol_angles,
            comsol_short,
            label="FEM - 短絡条件",
            facecolors="none",
            edgecolors="tab:blue",
        )
        ax1.scatter(
            comsol_angles,
            comsol_open,
            label="FEM - 開放条件",
            facecolors="none",
            edgecolors="tab:orange",
        )
    ax1.set_ylabel("SAW velocity (m/s)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper center")

    ax2.plot(angles_deg, k2 * 100.0, color="tab:red", label="Stroh - $K^2$")
    if comsol_angles is not None:
        ax2.scatter(
            comsol_angles,
            comsol_k2_pct,
            color="tab:red",
            label="FEM - $K^2$",
            facecolors="none",
            edgecolors="tab:red",
        )
    ax2.set_xlabel("Angle (deg)")
    ax2.set_ylabel("$K^2$ (%)")
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
