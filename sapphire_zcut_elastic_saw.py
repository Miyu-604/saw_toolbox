from __future__ import annotations
import numpy as np

from piezo_stroh.io import MaterialDB
from piezo_stroh.material import VoigtMaterial
from piezo_stroh.stroh.elastic_saw import ElasticSAWSolver


def main():
    # --- Load sapphire (z-cut, no rotation) from YAML DB ---
    db = MaterialDB()
    sap_voigt: VoigtMaterial = db.get("sapphire", "singlecrystal_gladden2004")
    sap = sap_voigt.to_tensor()

    # --- Elastic SAW solver ---
    solver = ElasticSAWSolver(sap)

    vmin, vmax = 4000, 6000

    v_scan = np.linspace(vmin, vmax, 601)
    err_scan = np.array([solver.objective(v) for v in v_scan])

    i_best = int(np.argmin(err_scan))
    v_best, err_best = float(v_scan[i_best]), float(err_scan[i_best])

    print(f"[{sap.name}]")
    print(f"Scan min: v={v_best:.2f} m/s (err={err_best:.2e})")

    # --- Simple error sweep plot (optional) ---
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(v_scan, err_scan, label="objective")
    ax.axvline(v_best, color="C0", linestyle=":", alpha=0.6)
    ax.set_xlabel("Phase velocity (m/s)")
    ax.set_ylabel("min singular value (objective)")
    ax.set_title("Sapphire z-cut elastic SAW objective sweep")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- Mode profile plotting (optional) ---
    z, prof = solver.mode_profile(v_best, z_over_lambda=np.linspace(0, 3.0, 240))

    def plot_profile(z, prof, title):
        u1, u2, u3 = prof[:, 0], prof[:, 1], prof[:, 2]
        umag = np.sqrt(np.abs(u1) ** 2 + np.abs(u2) ** 2 + np.abs(u3) ** 2)
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 8), sharex=True)
        ax1.plot(z, np.abs(u1), label="|u_x|")
        ax1.plot(z, np.abs(u2), label="|u_y|")
        ax1.plot(z, np.abs(u3), label="|u_z|")
        ax1.plot(z, umag, label="|u|", linewidth=2)
        ax1.set_ylabel("Normalized displacement")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")
        ax1.set_title(title)

        ax2.semilogy(z, umag, label="|u|", linewidth=2)
        ax2.set_xlabel("Depth (z/λ)")
        ax2.set_ylabel("Abs. (log)")
        ax2.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.show()

    plot_profile(z, prof, f"{sap.name} surface (v={v_best:.2f} m/s)")


if __name__ == "__main__":
    main()
