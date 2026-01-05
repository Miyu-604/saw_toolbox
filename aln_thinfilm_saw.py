from __future__ import annotations
import numpy as np

from piezo_stroh.io import MaterialDB
from piezo_stroh.material import VoigtMaterial
from piezo_stroh.stroh.piezo_saw import PiezoSAWSolver
from piezo_stroh.rotation import Rz, R_I

epsilon0 = 8.854187817e-12  # F/m


def main():
    # --- Load single-crystal AlN from YAML DB ---
    db = MaterialDB()  # finds ./materials_db or env PIEZO_MATERIAL_DB
    aln_voigt: VoigtMaterial = db.get("aln", "thinfilm_tsubouchi1981")

    # --- Orientation ---
    # For c-plane AlN with x-propagation & z-depth, identity is fine.
    aln_oriented = aln_voigt.to_tensor()
    R = R_I()
    aln_prop = aln_oriented.rotated(R, name_suffix="rotated 60deg about z")

    # --- Stroh solver ---
    solver = PiezoSAWSolver(aln_prop)

    # AlN is fast; search around the expected ~7 km/s range
    v_short, err_short = solver.find_velocity(electric_bc="short", vmin=5000, vmax=6000)
    v_open, err_open = solver.find_velocity(electric_bc="open", vmin=5000, vmax=6000)

    print(f"[{aln_oriented.name}]")
    print(f"Free (Open) Velocity:        {v_open:.2f} m/s (err={err_open:.2e})")
    print(f"Metallized (Short) Velocity: {v_short:.2f} m/s (err={err_short:.2e})")
    k2 = 2 * (v_open - v_short) / v_open
    print(f"Calculated K^2: {k2*100:.4f} %")

    # --- Mode profile plotting (optional) ---
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    z, prof_open = solver.mode_profile(v_open, electric_bc="open")
    z, prof_short = solver.mode_profile(v_short, electric_bc="short")

    def plot_profile(z, prof, title):
        u1, u2, u3, phi = prof[:, 0], prof[:, 1], prof[:, 2], prof[:, 3]
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(z, np.abs(u1), label="|u_x|")
        ax1.plot(z, np.abs(u2), label="|u_y|")
        ax1.plot(z, np.abs(u3), label="|u_z|")
        ax1.set_xlabel("Depth (z/λ)")
        ax1.set_ylabel("Normalized displacement")
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2 = ax1.twinx()
        ax2.plot(z, np.abs(phi), label="|phi|", linewidth=2)
        ax2.set_ylabel("Potential (arb.)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.set_title(title)
        plt.tight_layout()
        plt.show()

    plot_profile(z, prof_short, f"AlN short surface (v={v_short:.2f} m/s)")
    plot_profile(z, prof_open,  f"AlN open surface (v={v_open:.2f} m/s)")



if __name__ == "__main__":
    main()
