from __future__ import annotations
import numpy as np

try:
    from scipy import linalg
    from scipy.optimize import minimize_scalar
except Exception as e:
    raise ImportError("piezo_stroh.stroh.elastic_saw requires scipy (scipy.linalg, scipy.optimize).") from e

from ..material import TensorMaterial


class ElasticSAWSolver:
    """
    Elastic (non-piezoelectric) SAW solver for a single half-space.

    Coordinates:
      - x1: propagation direction
      - x3: depth into substrate (x3 > 0)
      - Ansatz: exp(i k (x1 + beta x3) - i w t)
        Decay as x3 -> +inf requires Im(beta) > 0.
    """

    def __init__(self, mat: TensorMaterial):
        self.mat = mat
        self.rho = float(mat.rho)
        self.C = mat.C4

    def solve_beta(self, v: float) -> tuple[np.ndarray, np.ndarray]:
        """
        For a trial phase velocity v, solve for beta roots and eigenvectors alpha.
        Returns:
          betas: (3,) selected decaying roots (Im(beta) > 0)
          alphas: (3,3) corresponding eigenvectors for [u1,u2,u3]
        """
        P = np.zeros((3, 3), dtype=complex)
        Q = np.zeros((3, 3), dtype=complex)
        R = np.zeros((3, 3), dtype=complex)

        # d1 = 1, d3 = beta, d2 = 0
        for i in range(3):
            for k in range(3):
                P[i, k] = -self.C[i, 2, k, 2]
                Q[i, k] = -(self.C[i, 0, k, 2] + self.C[i, 2, k, 0])
                R[i, k] = -self.C[i, 0, k, 0]
                if i == k:
                    R[i, k] += self.rho * v**2

        P_inv = linalg.inv(P)

        M = np.zeros((6, 6), dtype=complex)
        M[0:3, 3:6] = np.eye(3)
        M[3:6, 0:3] = -(P_inv @ R)
        M[3:6, 3:6] = -(P_inv @ Q)

        betas, eigvecs = linalg.eig(M)

        idx = np.where(np.imag(betas) > 1e-8)[0]
        if len(idx) != 3:
            idx = np.argsort(np.imag(betas))[-3:]

        betas_sel = betas[idx]
        alphas_sel = eigvecs[0:3, idx]

        return betas_sel, alphas_sel

    def boundary_matrix(self, v: float) -> np.ndarray:
        """
        Build traction-free boundary matrix B (3x3).
        Rows: T_31, T_32, T_33
        """
        betas, alphas = self.solve_beta(v)
        B = np.zeros((3, 3), dtype=complex)

        for n in range(3):
            beta = betas[n]
            alpha = alphas[:, n]

            for j in range(3):
                stress = 0j
                for k in range(3):
                    stress += self.C[2, j, k, 0] * alpha[k]
                    stress += self.C[2, j, k, 2] * beta * alpha[k]
                B[j, n] = stress

        return B

    def objective(self, v: float) -> float:
        """
        Minimize smallest singular value of boundary matrix.
        """
        try:
            B = self.boundary_matrix(v)
            _, s, _ = linalg.svd(B)
            return float(s[-1])
        except np.linalg.LinAlgError:
            return 1e30

    def find_velocity(self, *, vmin: float, vmax: float) -> tuple[float, float]:
        func = lambda vv: self.objective(vv)
        res = minimize_scalar(func, bounds=(vmin, vmax), method="bounded")
        return float(res.x), float(res.fun)
