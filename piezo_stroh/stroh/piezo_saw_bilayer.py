from __future__ import annotations
import numpy as np

try:
    from scipy import linalg
    from scipy.optimize import minimize_scalar
except Exception as e:
    raise ImportError("piezo_stroh.stroh.piezo_saw_bilayer requires scipy (scipy.linalg, scipy.optimize).") from e

from ..material import TensorMaterial


class PiezoSAWBilayerSolver:
    """
    Bilayer SAW: piezo thin film (0<z<H) + semi-infinite substrate (z>H)

    Ansatz:
      exp(i k (x1 + beta z) - i w t)
    We use dimensionless thickness h_over_lambda = H/lambda, so exp(i 2π beta h_over_lambda).
    """

    def __init__(
        self,
        film: TensorMaterial,
        sub: TensorMaterial,
        *,
        epsilon0: float = 8.854187817e-12,
    ):
        self.film = film
        self.sub = sub
        self.eps0 = float(epsilon0)

    # ---------- core: build companion eigensystem for a given material ----------
    def _solve_beta_all(self, mat: TensorMaterial, v: float):
        rho = float(mat.rho)
        C = mat.C4
        e = mat.e3
        eps = mat.eps

        P = np.zeros((4, 4), dtype=complex)
        Q = np.zeros((4, 4), dtype=complex)
        R = np.zeros((4, 4), dtype=complex)

        # d1=1, d3=beta, d2=0 (same as your current code)
        for i in range(3):
            for k in range(3):
                P[i, k] = -C[i, 2, k, 2]
                Q[i, k] = -(C[i, 0, k, 2] + C[i, 2, k, 0])
                R[i, k] = -C[i, 0, k, 0]
                if i == k:
                    R[i, k] += rho * v**2

        for i in range(3):
            P[i, 3] = -e[2, i, 2]
            Q[i, 3] = -(e[0, i, 2] + e[2, i, 0])
            R[i, 3] = -e[0, i, 0]

        for k in range(3):
            P[3, k] = -e[2, k, 2]
            Q[3, k] = -(e[0, k, 2] + e[2, k, 0])
            R[3, k] = -e[0, k, 0]

        P[3, 3] = eps[2, 2]
        Q[3, 3] = eps[0, 2] + eps[2, 0]
        R[3, 3] = eps[0, 0]

        P_inv = linalg.inv(P)

        M = np.zeros((8, 8), dtype=complex)
        M[0:4, 4:8] = np.eye(4)
        M[4:8, 0:4] = -(P_inv @ R)
        M[4:8, 4:8] = -(P_inv @ Q)

        betas, eigvecs = linalg.eig(M)      # (8,), (8,8)
        alphas = eigvecs[0:4, :]            # (4,8)  [u1,u2,u3,phi]
        return betas, alphas

    def _select_decaying(self, betas: np.ndarray, alphas: np.ndarray):
        idx = np.where(np.imag(betas) > 1e-8)[0]
        if len(idx) != 4:
            idx = np.argsort(np.imag(betas))[-4:]
        return betas[idx], alphas[:, idx]   # (4,), (4,4)

    # ---------- traction & D3 for one partial wave ----------
    def _t_and_D3(self, mat: TensorMaterial, beta: complex, alpha: np.ndarray):
        C = mat.C4
        e = mat.e3
        eps = mat.eps

        # traction T_3j  (j=0..2)
        t = np.zeros(3, dtype=complex)
        for j in range(3):
            s = 0j
            for k in range(3):
                s += C[2, j, k, 0] * alpha[k]         # l=1
                s += C[2, j, k, 2] * beta * alpha[k]  # l=3
            s += e[0, 2, j] * alpha[3]
            s += e[2, 2, j] * beta * alpha[3]
            t[j] = s

        # D3 (no vacuum term here; add eps0*phi only for free surface "open")
        D3 = 0j
        for k in range(3):
            D3 += e[2, k, 0] * alpha[k]
            D3 += e[2, k, 2] * beta * alpha[k]
        D3 -= eps[2, 0] * alpha[3]
        D3 -= eps[2, 2] * beta * alpha[3]

        return t, D3

    # ---------- build 12x12 boundary matrix ----------
    def boundary_matrix(
        self,
        v: float,
        *,
        h_over_lambda: float,
        electric_bc: str = "short",
    ) -> np.ndarray:
        # film: use all 8
        bet_f, alp_f = self._solve_beta_all(self.film, v)          # (8,), (4,8)

        # substrate: use decaying 4, and shift origin to interface z=H (so phase factor=1 at interface)
        bet_s_all, alp_s_all = self._solve_beta_all(self.sub, v)
        bet_s, alp_s = self._select_decaying(bet_s_all, alp_s_all) # (4,), (4,4)

        B = np.zeros((12, 12), dtype=complex)

        # ---- Surface z=0 : traction free + electric BC, film only ----
        for m in range(8):
            beta = bet_f[m]
            alpha = alp_f[:, m]
            t, D3 = self._t_and_D3(self.film, beta, alpha)

            # T31,T32,T33
            B[0:3, m] = t

            # electrical BC
            if electric_bc == "short":
                B[3, m] = alpha[3]                       # phi=0
            elif electric_bc == "open":
                B[3, m] = D3 + self.eps0 * alpha[3]      # D3 + eps0*phi = 0 (vacuum approx)
            else:
                raise ValueError("electric_bc must be 'short' or 'open'.")

        # ---- Interface z=H : continuity of u,phi, traction, D3 ----
        phase_f = np.exp(1j * 2 * np.pi * bet_f * h_over_lambda)   # (8,)

        row0 = 4

        # (a) u1,u2,u3,phi continuity
        for comp in range(4):
            r = row0 + comp
            # film side: + alpha * exp(i2πβH/λ)
            for m in range(8):
                B[r, m] = alp_f[comp, m] * phase_f[m]
            # substrate side: - alpha (at interface origin)
            for n in range(4):
                B[r, 8 + n] = -alp_s[comp, n]

        # (b) T31,T32,T33 continuity
        for j in range(3):
            r = row0 + 4 + j
            for m in range(8):
                t_f, _ = self._t_and_D3(self.film, bet_f[m], alp_f[:, m])
                B[r, m] = t_f[j] * phase_f[m]
            for n in range(4):
                t_s, _ = self._t_and_D3(self.sub, bet_s[n], alp_s[:, n])
                B[r, 8 + n] = -t_s[j]

        # (c) D3 continuity
        r = row0 + 7
        for m in range(8):
            _, D3_f = self._t_and_D3(self.film, bet_f[m], alp_f[:, m])
            B[r, m] = D3_f * phase_f[m]
        for n in range(4):
            _, D3_s = self._t_and_D3(self.sub, bet_s[n], alp_s[:, n])
            B[r, 8 + n] = -D3_s

        return B

    def objective(self, v: float, *, h_over_lambda: float, electric_bc: str) -> float:
        try:
            B = self.boundary_matrix(v, h_over_lambda=h_over_lambda, electric_bc=electric_bc)
            _, s, _ = linalg.svd(B)
            return float(s[-1])
        except np.linalg.LinAlgError:
            return 1e30

    def find_velocity(self, *, h_over_lambda: float, electric_bc: str, vmin: float, vmax: float):
        f = lambda vv: self.objective(vv, h_over_lambda=h_over_lambda, electric_bc=electric_bc)
        res = minimize_scalar(f, bounds=(vmin, vmax), method="bounded")
        return float(res.x), float(res.fun)
