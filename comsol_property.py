from __future__ import annotations

from piezo_stroh.io import MaterialDB
from piezo_stroh import rotation as rot
from piezo_stroh.io.comsol import to_comsol_text


# --- DB keys (edit these to match your YAML) ---
MATERIAL_KEY = "AlN"
DATASET_KEY = "singlecrystal_sotnikov2010"  # set to None to use YAML default_dataset if your DB supports it


def main():
    # --- Load crystal from YAML DB ---
    db = MaterialDB()
    if DATASET_KEY is None:
        mat = db.get(MATERIAL_KEY)  # uses YAML default_dataset if implemented
    else:
        mat = db.get(MATERIAL_KEY, DATASET_KEY)

    mat_tensor = mat.to_tensor()

    # Quick sanity print
    print(f"[Loaded] {mat.name} (rho={mat.rho:.6g} kg/m^3)")

    # rotate for orientation (optional)
    R = rot.R_I()
    mat_orientation = mat_tensor.rotated(R, name_suffix="Identity")

    # export to COMSOL-like text
    mat_voigt = mat_orientation.to_voigt(shear=mat.shear)
    print(to_comsol_text(mat_voigt, eps_as_relative=True))

if __name__ == "__main__":
    main()
