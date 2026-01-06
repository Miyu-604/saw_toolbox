from __future__ import annotations
import numpy as np


def assert_rotation_matrix(R: np.ndarray, atol: float = 1e-12) -> None:
    R = np.asarray(R, float)
    if R.shape != (3, 3):
        raise ValueError("R must be (3,3).")
    if not np.allclose(R @ R.T, np.eye(3), atol=atol):
        raise ValueError("R is not orthonormal.")
    if not np.isclose(np.linalg.det(R), 1.0, atol=atol):
        raise ValueError("det(R) is not +1 (right-handed).")

# ---- テンソルの回転 ----
def rotate_rank2(T2: np.ndarray, R: np.ndarray) -> np.ndarray:
    """T' = R T R^T"""
    return R @ T2 @ R.T


def rotate_rank3(T3: np.ndarray, R: np.ndarray) -> np.ndarray:
    """T'ijk = R_ip R_jq R_kr T_pqr"""
    return np.einsum("ip,jq,kr,pqr->ijk", R, R, R, T3, optimize=True)


def rotate_rank4(T4: np.ndarray, R: np.ndarray) -> np.ndarray:
    """T'ijkl = R_ip R_jq R_kr R_ls T_pqrs"""
    return np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, T4, optimize=True)


# ---- common orientation helpers ----
def R_I() -> np.ndarray:
    """
    Identity rotation
    """
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=float)
    assert_rotation_matrix(R)
    return R

def R_ycut() -> np.ndarray:
    """
    Y-cut, Z propagation (example mapping used earlier):
      x' = Z, y' = X, z' = Y
    """
    R = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]], dtype=float)
    assert_rotation_matrix(R)
    return R

def R_y2z() -> np.ndarray:
    """
    Map crystal Y axis to new Z while keeping X as propagation axis:
      new_x = old_x, new_y = -old_z, new_z = old_y
    """
    R = np.array([[1.0, 0.0,  0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0,  0.0]], dtype=float)
    assert_rotation_matrix(R)
    return R


def R_128yxcut_xprop() -> np.ndarray:
    """
    Build a right-handed rotation for 128° Y-X cut with X propagation.
    This matches your earlier construction:
      new_x = X
      new_z = Rx(128deg about X) * Y
      new_y = new_z x new_x
    """
    theta = np.deg2rad(128.0)

    # Xc軸を中心に theta 回転させる
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]], dtype=float)
    # z軸 <- Yc軸をXc軸中心に128回転させたベクトル
    new_z = Rx @ np.array([0.0, 1.0, 0.0])
    # x軸 <- Xc軸(Xc軸伝播)
    new_x = np.array([1.0, 0.0, 0.0])
    # y軸 <- 直交するベクトル
    new_y = np.cross(new_z, new_x)
    new_y /= np.linalg.norm(new_y)

    R = np.stack([new_x, new_y, new_z], axis=0)
    assert_rotation_matrix(R)
    return R

def R_yxcut_theta_xprop(theta) -> np.ndarray:
    """
    Build a right-handed rotation for 128° Y-X cut with X propagation.
    This matches your earlier construction:
      new_x = X
      new_z = Rx(128deg about X) * Y
      new_y = new_z x new_x
    """
    theta_rad = np.deg2rad(theta)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_rad), -np.sin(theta_rad)],
                   [0, np.sin(theta_rad),  np.cos(theta_rad)]], dtype=float)

    new_z = Rx @ np.array([0.0, 1.0, 0.0])
    new_x = np.array([1.0, 0.0, 0.0])
    new_y = np.cross(new_z, new_x)
    new_y /= np.linalg.norm(new_y)

    R = np.stack([new_x, new_y, new_z], axis=0)
    assert_rotation_matrix(R)
    return R

def Rz(theta_deg: float) -> np.ndarray:
    """Rotate about +z by theta_deg (right-hand rule).

    This is the standard *active* rotation acting on vector components:
      v' = Rz(theta) @ v

    Matrix form:
      [ cos -sin  0 ]
      [ sin  cos  0 ]
      [  0    0   1 ]
    """
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    assert_rotation_matrix(R)
    return R


def R_z2y() -> np.ndarray:
    """
    Requested mapping (old -> new):
    old x -> new x
    old y -> new -z
    old z -> new y
    """
    R = np.array([[1.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0],
                  [0.0, -1.0, 0.0]], dtype=float)
    assert_rotation_matrix(R)
    return R
