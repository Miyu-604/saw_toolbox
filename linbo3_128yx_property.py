from __future__ import annotations

from piezo_stroh.io import MaterialDB
from piezo_stroh.rotation import R_yxcut_theta_xprop
from piezo_stroh.io.comsol import to_comsol_text


# --- DB keys (edit these to match your YAML) ---
# Example expected files:
#   materials_db/linbo3.yml
# with a dataset name such as `bulk_ogi2002`.
MATERIAL_KEY = "linbo3"
DATASET_KEY = "bulk_ogi2002"  # set to None to use YAML default_dataset if your DB supports it


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

    # rotate for orientation
    R = R_yxcut_theta_xprop(127.86)
    mat_orientation = mat_tensor.rotated(R, name_suffix="128YX_xprop")

    # export to COMSOL-like text
    mat128_voigt = mat_orientation.to_voigt(shear=mat.shear)
    print(to_comsol_text(mat128_voigt, eps_as_relative=True))

if __name__ == "__main__":
    main()
