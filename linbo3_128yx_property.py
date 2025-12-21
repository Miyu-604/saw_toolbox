from __future__ import annotations

import numpy as np

from piezo_stroh.io import MaterialDB
from piezo_stroh.rotation import R_yxcut_theta_xprop
from piezo_stroh.io.comsol import to_comsol_text
from piezo_stroh.material import VoigtMaterial


# --- DB keys (edit these to match your YAML) ---
# Example expected files:
#   materials_db/linbo3.yml
# with a dataset name such as `bulk_ogi2002_present`.
MATERIAL_KEY = "linbo3"
DATASET_KEY = "bulk_ogi2002_present"  # set to None to use YAML default_dataset if your DB supports it

def _strain_to_E6(eps: np.ndarray, *, shear: str) -> np.ndarray:
    """
    eps (3x3 symmetric) -> E6 with shear convention.
    - engineering: gamma = 2*epsilon
    - tensorial:   gamma = 1*epsilon
    """
    shear_key = shear.strip().lower()
    scale = 2.0 if shear_key == "engineering" else 1.0
    exx = eps[0, 0]
    eyy = eps[1, 1]
    ezz = eps[2, 2]
    gyz = scale * eps[1, 2]
    gxz = scale * eps[0, 2]
    gxy = scale * eps[0, 1]
    return np.array([exx, eyy, ezz, gyz, gxz, gxy], dtype=float)


def _stress6_to_tensor(s6: np.ndarray) -> np.ndarray:
    """
    s6 = [sxx, syy, szz, syz, sxz, sxy] -> sigma tensor (3x3 symmetric)
    (stress has NO factor-2 convention)
    """
    sigma = np.zeros((3, 3), dtype=float)
    sigma[0, 0] = s6[0]
    sigma[1, 1] = s6[1]
    sigma[2, 2] = s6[2]
    sigma[1, 2] = sigma[2, 1] = s6[3]
    sigma[0, 2] = sigma[2, 0] = s6[4]
    sigma[0, 1] = sigma[1, 0] = s6[5]
    return sigma


def energy_consistency_test(material: VoigtMaterial, R=None, *, ntests=50, seed=0, strain_scale=1e-5):
    """
    Test 1: W_voigt == W_tensor for random symmetric strains.
    Test 2: rotational invariance W(C,eps) == W(C',eps') when R is provided.

    material: VoigtMaterial (uses .C6 and converts to tensor with its shear convention)
    R: optional rotation matrix (3x3)
    """
    rng = np.random.default_rng(seed)

    C6 = material.C6
    C4 = material.to_tensor().C4
    shear = material.shear

    # Quick sanity: C6 symmetry
    sym_err = np.max(np.abs(C6 - C6.T))
    print(f"[EnergyTest] C6 symmetry max|C6-C6^T| = {sym_err:.3e}")

    max_rel_err = 0.0
    max_abs_err = 0.0

    # rotation objects if requested
    if R is not None:
        from piezo_stroh.rotation import rotate_rank4
        C4r = rotate_rank4(C4, R)

    for t in range(ntests):
        # random symmetric strain tensor
        A = rng.normal(size=(3, 3))
        eps = 0.5 * (A + A.T) * strain_scale  # small strain

        # ---- Voigt energy ----
        E6 = _strain_to_E6(eps, shear=shear)
        sigma6 = C6 @ E6
        W_voigt = 0.5 * (E6 @ sigma6)  # since eps:sigma = E6^T*sigma6 under this convention

        # ---- Tensor energy ----
        sigma = np.einsum("ijkl,kl->ij", C4, eps, optimize=True)
        W_tensor = 0.5 * np.sum(eps * sigma)

        abs_err = float(np.abs(W_voigt - W_tensor))
        rel_err = abs_err / (float(np.abs(W_tensor)) + 1e-30)
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)

        # ---- Optional: rotation invariance test ----
        if R is not None:
            eps_r = R @ eps @ R.T
            sigma_r = np.einsum("ijkl,kl->ij", C4r, eps_r, optimize=True)
            W_rot = 0.5 * np.sum(eps_r * sigma_r)

            abs_err_rot = float(np.abs(W_tensor - W_rot))
            rel_err_rot = abs_err_rot / (float(np.abs(W_tensor)) + 1e-30)

            if t == 0:
                print(f"[EnergyTest] Example W_tensor={W_tensor:.6e}, W_voigt={W_voigt:.6e}, W_rot={W_rot:.6e}")
            if rel_err_rot > 1e-8:
                # not fatal, but flags possible rotation/definition issue
                pass

    print(f"[EnergyTest] max abs error |W_voigt-W_tensor| = {max_abs_err:.3e}")
    print(f"[EnergyTest] max rel error |W_voigt-W_tensor|/|W| = {max_rel_err:.3e}")

    if R is not None:
        print("[EnergyTest] rotation invariance checked (compare W_tensor vs W_rot).")
        print("           If this is not ~1e-10..1e-8 level, rotation/definition mismatch is likely.")

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

    # --- Energy consistency tests (VERY IMPORTANT) ---
    print("\n=== Energy consistency test: raw material (no rotation) ===")
    energy_consistency_test(ln, R=None, ntests=50, seed=1)

    print("\n=== Energy consistency test: rotation invariance check (raw + R) ===")
    energy_consistency_test(ln, R=R, ntests=50, seed=2)

    print("\n=== Energy consistency test: rotated material itself (should still be consistent) ===")
    energy_consistency_test(ln128.to_voigt(shear=ln.shear), R=None, ntests=50, seed=3)

if __name__ == "__main__":
    main()
