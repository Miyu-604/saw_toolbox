from __future__ import annotations

from piezo_stroh.io import MaterialDB
from piezo_stroh.io.comsol import to_comsol_text

MATERIAL_KEY = "aln"
DATASET_KEY = "singlecrystal_sotnikov2010"

def main():
    # --- Load AlN (crystal axes) from YAML DB ---
    db = MaterialDB()
    if DATASET_KEY is None:
        aln = db.get(MATERIAL_KEY)  # uses YAML default_dataset if implemented
    else:
        aln = db.get(MATERIAL_KEY, DATASET_KEY)

    # Quick sanity print
    print(f"[Loaded] {aln.name} (rho={aln.rho:.6g} kg/m^3)")

    # orientation: not rotated

    # export to COMSOL-like text
    print(to_comsol_text(aln, eps_as_relative=True))


if __name__ == "__main__":
    main()
