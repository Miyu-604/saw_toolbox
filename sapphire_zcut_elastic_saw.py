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

    vmin, vmax = 5000, 8000
    v_saw, err = solver.find_velocity(vmin=vmin, vmax=vmax)

    print(f"[{sap.name}]")
    print(f"Elastic SAW Velocity: {v_saw:.2f} m/s (err={err:.2e})")

    # --- Simple error sweep plot (optional) ---
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    v_scan = np.linspace(vmin, vmax, 601)
    err_scan = np.array([solver.objective(v) for v in v_scan])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(v_scan, err_scan, label="elastic")
    ax.axvline(v_saw, color="C0", linestyle=":", alpha=0.6)
    ax.set_xlabel("Phase velocity (m/s)")
    ax.set_ylabel("min singular value (objective)")
    ax.set_title("Sapphire z-cut elastic SAW objective sweep")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
