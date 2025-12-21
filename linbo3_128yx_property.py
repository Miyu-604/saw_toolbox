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
    # --- Load LiNbO3 (crystal axes) from YAML DB ---
    db = MaterialDB()
    if DATASET_KEY is None:
        ln = db.get(MATERIAL_KEY)  # uses YAML default_dataset if implemented
    else:
        ln = db.get(MATERIAL_KEY, DATASET_KEY)

    ln_tensor = ln.to_tensor()

    # Quick sanity print
    print(f"[Loaded] {ln.name} (rho={ln.rho:.6g} kg/m^3)")

    # rotate to 128YX (x propagation)
    R = R_yxcut_theta_xprop(127.86)
    ln128 = ln_tensor.rotated(R, name_suffix="128YX_xprop")

    # export to COMSOL-like text
    ln128_voigt = ln128.to_voigt(shear=ln.shear)
    print(to_comsol_text(ln128_voigt, eps_as_relative=True))

if __name__ == "__main__":
    main()
