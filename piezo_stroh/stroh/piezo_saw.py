from __future__ import annotations
import numpy as np

try:
    from scipy import linalg
    from scipy.optimize import minimize_scalar
except Exception as e:
    raise ImportError("piezo_stroh.stroh.piezo_saw requires scipy (scipy.linalg, scipy.optimize).") from e

from ..material import TensorMaterial


class PiezoSAWSolver:
    """
    Piezoelectric SAW solver (single half-space) using a polynomial eigenvalue -> companion matrix approach.

    Coordinates assumption:
      - x1: propagation direction
      - x3: depth into substrate (x3 > 0)
      - Ansatz: exp(i k (x1 + beta x3) - i w t)
        Decay as x3 -> +inf requires Im(beta) > 0.
    """

    def __init__(self, mat: TensorMaterial, *, epsilon0: float = 8.854187817e-12):
        self.mat = mat
        self.rho = float(mat.rho)
        self.C = mat.C4   # true-strain stiffness tensor
        self.e = mat.e3   # e[k,i,j]
        self.eps = mat.eps
        self.eps0 = float(epsilon0)

    def solve_beta(self, v: float) -> tuple[np.ndarray, np.ndarray]:
        """
        For a trial phase velocity v, solve for beta roots and eigenvectors alpha.
        Returns:
          betas: (4,) selected decaying roots (Im(beta) > 0)
          alphas: (4,4) corresponding eigenvectors for [u1,u2,u3,phi]
        """
        P = np.zeros((4, 4), dtype=complex)
        Q = np.zeros((4, 4), dtype=complex)
        R = np.zeros((4, 4), dtype=complex)

        # d1 = 1, d3 = beta, d2 = 0
        for i in range(3):
            for k in range(3):
                P[i, k] = -self.C[i, 2, k, 2]
                Q[i, k] = -(self.C[i, 0, k, 2] + self.C[i, 2, k, 0])
                R[i, k] = -self.C[i, 0, k, 0]
                if i == k:
                    R[i, k] += self.rho * v**2

        # Coupling (mechanical eq) column for phi
        for i in range(3):
            P[i, 3] = -self.e[2, i, 2]
            Q[i, 3] = -(self.e[0, i, 2] + self.e[2, i, 0])
            R[i, 3] = -self.e[0, i, 0]

        # Coupling (electrical eq) row for u
        for k in range(3):
            P[3, k] = -self.e[2, k, 2]
            Q[3, k] = -(self.e[0, k, 2] + self.e[2, k, 0])
            R[3, k] = -self.e[0, k, 0]

        # Electric potential part (eps_ik d_i d_k)
        P[3, 3] = self.eps[2, 2]
        Q[3, 3] = self.eps[0, 2] + self.eps[2, 0]
        R[3, 3] = self.eps[0, 0]

        P_inv = linalg.inv(P)

        M = np.zeros((8, 8), dtype=complex)
        M[0:4, 4:8] = np.eye(4)
        M[4:8, 0:4] = -(P_inv @ R)
        M[4:8, 4:8] = -(P_inv @ Q)

        betas, eigvecs = linalg.eig(M)

        # select decaying roots for x3>0 depth: Im(beta) > 0
        idx = np.where(np.imag(betas) > 1e-8)[0]
        if len(idx) != 4:
            # fallback: take 4 with largest imag parts
            idx = np.argsort(np.imag(betas))[-4:]

        betas_sel = betas[idx]
        alphas_sel = eigvecs[0:4, idx]  # [u1,u2,u3,phi] rows

        return betas_sel, alphas_sel

    def boundary_matrix(self, v: float, *, electric_bc: str = "short") -> np.ndarray:
        """
        Build boundary condition matrix B (4x4).
        Rows: T_31, T_32, T_33, electrical BC
        """
        betas, alphas = self.solve_beta(v)
        B = np.zeros((4, 4), dtype=complex)

        for n in range(4):
            beta = betas[n]
            alpha = alphas[:, n]  # [u1,u2,u3,phi]

            # Mechanical traction T_3j at x3=0
            for j in range(3):
                stress = 0j
                for k in range(3):
                    stress += self.C[2, j, k, 0] * alpha[k]         # l=1
                    stress += self.C[2, j, k, 2] * beta * alpha[k]  # l=3
                stress += self.e[0, 2, j] * alpha[3]               # l=1
                stress += self.e[2, 2, j] * beta * alpha[3]        # l=3
                B[j, n] = stress

            if electric_bc == "short":
                # metallized: phi = 0
                B[3, n] = alpha[3]

            elif electric_bc == "open":
                # continuity with vacuum approx: D3 + eps0 * phi = 0 (k factor removed)
                D3 = 0j
                for k in range(3):
                    D3 += self.e[2, k, 0] * alpha[k]
                    D3 += self.e[2, k, 2] * beta * alpha[k]
                D3 -= self.eps[2, 0] * alpha[3]
                D3 -= self.eps[2, 2] * beta * alpha[3]
                B[3, n] = D3 + self.eps0 * alpha[3]
            else:
                raise ValueError("electric_bc must be 'short' or 'open'.")

        return B

    def objective(self, v: float, *, electric_bc: str = "short") -> float:
            try:
                B = self.boundary_matrix(v, electric_bc=electric_bc)
                
                # --- スケーリング処理を追加 ---
                # 応力の行 (0,1,2) は ~10^11 なので、
                # 電気の行 (3) を 10^11 倍程度にしてバランスを取る
                B_scaled = np.copy(B)
                if electric_bc == "short":
                    # phi (V) を 1e11 倍して応力 (Pa) に合わせる
                    B_scaled[3, :] *= 1e11 
                elif electric_bc == "open":
                    # D3 (C/m^2) は非常に小さい (~10^-10) ので、1e21 倍程度必要
                    B_scaled[3, :] *= 1e21
                    
                _, s, _ = linalg.svd(B_scaled)
                return float(s[-1])
            except np.linalg.LinAlgError:
                return 1e30

    def find_velocity(self, *, electric_bc: str, vmin: float, vmax: float) -> tuple[float, float]:
        func = lambda vv: self.objective(vv, electric_bc=electric_bc)
        res = minimize_scalar(func, bounds=(vmin, vmax), method="bounded")
        return float(res.x), float(res.fun)

    def mode_profile(
        self,
        v: float,
        *,
        electric_bc: str = "short",
        z_over_lambda: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return depth profiles for [u1,u2,u3,phi] as complex values.
        z_over_lambda is dimensionless depth (0..N), where exp(i 2pi beta z_over_lambda).
        """
        if z_over_lambda is None:
            z_over_lambda = np.linspace(0, 2.5, 200)

        betas, alphas = self.solve_beta(v)
        B = self.boundary_matrix(v, electric_bc=electric_bc)

        # null vector via SVD
        U, S, Vh = linalg.svd(B)
        coeff = Vh[-1]  # (4,)

        prof = np.zeros((len(z_over_lambda), 4), dtype=complex)
        for i, z in enumerate(z_over_lambda):
            for n in range(4):
                prof[i, :] += alphas[:, n] * coeff[n] * np.exp(1j * 2 * np.pi * betas[n] * z)

        # normalize by surface displacement magnitude
        u0 = np.sqrt(np.sum(np.abs(prof[0, :3]) ** 2))
        if u0 > 0:
            prof /= u0

        return z_over_lambda, prof


class PiezoAlSAWSolver(PiezoSAWSolver):
    """
    SAW solver with Al electrode mass loading modeled as a surface mass sheet.
    """

    RHO_AL = 2700.0  # Al density [kg/m^3]

    def boundary_matrix_with_al(
        self,
        v: float,
        thickness: float,
        wavelength: float,
        *,
        electric_bc: str = "short",
    ) -> np.ndarray:
        """
        Boundary condition matrix with Al film mass loading.
        thickness: Al film thickness [m]
        wavelength: SAW wavelength [m]
        """
        m_s = self.RHO_AL * thickness
        k = 2.0 * np.pi / wavelength

        betas, alphas = self.solve_beta(v)
        B = np.zeros((4, 4), dtype=complex)

        mass_load_coeff = 1j * m_s * k * (v**2)

        for n in range(4):
            beta = betas[n]
            alpha = alphas[:, n]

            for j in range(3):
                stress_term = 0j
                for l in range(3):
                    stress_term += self.C[2, j, l, 0] * alpha[l]
                    stress_term += self.C[2, j, l, 2] * beta * alpha[l]
                stress_term += self.e[0, 2, j] * alpha[3]
                stress_term += self.e[2, 2, j] * beta * alpha[3]
                B[j, n] = stress_term - mass_load_coeff * alpha[j]

            if electric_bc == "short":
                B[3, n] = alpha[3]
            elif electric_bc == "open":
                D3 = 0j
                for k_comp in range(3):
                    D3 += self.e[2, k_comp, 0] * alpha[k_comp]
                    D3 += self.e[2, k_comp, 2] * beta * alpha[k_comp]
                D3 -= self.eps[2, 0] * alpha[3]
                D3 -= self.eps[2, 2] * beta * alpha[3]
                B[3, n] = D3 + self.eps0 * alpha[3]
            else:
                raise ValueError("electric_bc must be 'short' or 'open'.")

        return B

    def find_velocity_al(
        self,
        thickness: float,
        wavelength: float,
        *,
        electric_bc: str = "short",
        v_guess: float = 4000.0,
        search_range: float = 500.0,
    ) -> float:
        """
        Search SAW velocity with Al mass loading for given thickness.
        """
        def obj(v):
            try:
                B = self.boundary_matrix_with_al(
                    v,
                    thickness,
                    wavelength,
                    electric_bc=electric_bc,
                )
                return float(linalg.svd(B, compute_uv=False)[-1])
            except Exception:
                return 1e30

        res = minimize_scalar(
            obj,
            bounds=(v_guess - search_range, v_guess + search_range),
            method="bounded",
        )
        return float(res.x)
