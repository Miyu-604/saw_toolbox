from __future__ import annotations
import numpy as np

from piezo_stroh.io import MaterialDB
from piezo_stroh.material import VoigtMaterial
from piezo_stroh.stroh.piezo_saw_bilayer import PiezoSAWBilayerSolver


def main():
    # --- Materials: AlN film on sapphire substrate ---
    db = MaterialDB()
    aln_voigt: VoigtMaterial = db.get("linbo3", "bulk_ogi2002")
    sap_voigt: VoigtMaterial = db.get("sapphire", "singlecrystal_gladden2004")

    aln = aln_voigt.to_tensor()
    sapphire = sap_voigt.to_tensor()

    # --- Geometry: H = 1 um, lambda = 10 um ---
    h_over_lambda = 1.0 / 10.0

    # --- Bilayer SAW solver ---
    solver = PiezoSAWBilayerSolver(aln, sapphire)

    vmin, vmax = 3000, 9000
    v_scan = np.linspace(vmin, vmax, 601)
    err_short_scan = np.array(
        [solver.objective(v, h_over_lambda=h_over_lambda, electric_bc="short") for v in v_scan]
    )
    err_open_scan = np.array(
        [solver.objective(v, h_over_lambda=h_over_lambda, electric_bc="open") for v in v_scan]
    )

    i_short = int(np.argmin(err_short_scan))
    i_open = int(np.argmin(err_open_scan))
    v_short, err_short = float(v_scan[i_short]), float(err_short_scan[i_short])
    v_open, err_open = float(v_scan[i_open]), float(err_open_scan[i_open])

    print("[AlN(1um) / Sapphire(half-space)]")
    print(f"h_over_lambda = {h_over_lambda:.3f}")
    print(f"Scan min (Short): v={v_short:.2f} m/s (err={err_short:.2e})")
    print(f"Scan min (Open):  v={v_open:.2f} m/s (err={err_open:.2e})")

    # --- Simple objective sweep plot ---
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
    ax.set_title("Bilayer SAW objective sweep (AlN/Sapphire)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
