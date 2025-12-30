from __future__ import annotations
import numpy as np

from piezo_stroh.io import MaterialDB
from piezo_stroh.material import VoigtMaterial
from piezo_stroh.stroh.piezo_saw import PiezoSAWSolver
from piezo_stroh.rotation import R_yxcut_theta_xprop, Rz


def main():
    # --- Load sapphire (z-cut, no rotation) from YAML DB ---
    db = MaterialDB()
    mat_voigt: VoigtMaterial = db.get("LiNbO3", "bulk_ogi2002")
    mat = mat_voigt.to_tensor()
    R = R_yxcut_theta_xprop(127.86)
    mat_orientation = mat.rotated(R, name_suffix="128YX_xprop")

    # --- Stroh solver ---
    solver = PiezoSAWSolver(mat_orientation)

    # Sapphire SAW is relatively fast; scan a broad range
    vmin, vmax = 3000, 5000

    v_scan = np.linspace(vmin, vmax, 2001)
    err_short_scan = np.array([solver.objective(v, electric_bc="short") for v in v_scan])
    err_open_scan = np.array([solver.objective(v, electric_bc="open") for v in v_scan])

    i_short = int(np.argmin(err_short_scan))
    i_open = int(np.argmin(err_open_scan))
    v_short, err_short = float(v_scan[i_short]), float(err_short_scan[i_short])
    v_open, err_open = float(v_scan[i_open]), float(err_open_scan[i_open])

    print(f"[{mat_orientation.name}]")
    print(f"Scan min (Short): v={v_short:.2f} m/s (err={err_short:.2e})")
    print(f"Scan min (Open):  v={v_open:.2f} m/s (err={err_open:.2e})")

    # --- Simple error sweep plot (optional) ---
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(v_scan, err_short_scan, label="short")
    ax.semilogy(v_scan, err_open_scan, label="open")
    ax.axvline(v_short, color="C0", linestyle=":", alpha=0.6)
    ax.axvline(v_open, color="C1", linestyle=":", alpha=0.6)
    ax.set_xlabel("Phase velocity (m/s)")
    ax.set_ylabel("min singular value (objective)")
    ax.set_title("Sapphire z-cut SAW objective sweep")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- Mode profile plotting (optional) ---
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

    plot_profile(z, prof_open, f"Sapphire open surface (v={v_open:.2f} m/s)")
    plot_profile(z, prof_short, f"Sapphire short surface (v={v_short:.2f} m/s)")


if __name__ == "__main__":
    main()
