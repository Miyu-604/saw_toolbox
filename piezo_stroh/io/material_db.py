from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..material import PiezoMaterial

EPS0 = 8.854187817e-12  # F/m


def _import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception as e:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _find_db_dir(explicit: Optional[str | Path] = None) -> Path:
    """
    Mac/Windows両対応のmaterials_db探索。
    優先順位:
      1) 明示パス
      2) 環境変数 PIEZO_MATERIAL_DB
      3) CWDから親へ辿って ./materials_db を探す
    """
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"material db dir not found: {p}")
        return p

    env = os.getenv("PIEZO_MATERIAL_DB")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"PIEZO_MATERIAL_DB points to missing dir: {p}")

    start = Path.cwd().resolve()
    for p in [start, *start.parents]:
        cand = p / "materials_db"
        if cand.is_dir():
            return cand

    raise FileNotFoundError(
        "materials_db directory not found. "
        "Create ./materials_db in your project root, or set PIEZO_MATERIAL_DB."
    )


def _as_array(x: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must be shape {shape}, got {arr.shape}.")
    return arr


def _read_matrix(d: dict, key: str, shape: tuple[int, ...], *, optional: bool = False) -> np.ndarray:
    if key not in d:
        if optional:
            return np.zeros(shape, dtype=float)
        raise KeyError(f"Missing required key: {key}")
    return _as_array(d[key], shape, key)


def _convert_units(C6: np.ndarray, eps: np.ndarray, units: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Supported unit strings (case/space/underscore insensitive):
      - C6: "Pa" or "GPa"
      - eps: "F/m" or "eps_r"
    """
    uC = _norm(units.get("C6", "Pa"))
    if uC == "gpa":
        C6 = C6 * 1e9
    elif uC == "pa":
        pass
    else:
        raise ValueError(f"Unsupported units.C6: {units.get('C6')}")

    ueps = _norm(units.get("eps", "f/m"))
    if ueps in ("f/m", "fpm", "fperm", "fpermeter", "fpermetre"):
        pass
    elif ueps in ("epsr", "eps_r", "epsrel", "eps_rel", "relative", "epsrelative"):
        eps = eps * EPS0
    else:
        raise ValueError(f"Unsupported units.eps: {units.get('eps')}")

    return C6, eps


@dataclass
class MaterialDB:
    """
    YAML material database (Voigt matrices are the source of truth).

    - get(): returns PiezoMaterial using ONLY rho, C6, e36, eps
      (symmetry_note / independent_constants / source は計算に使わない)
    - get_meta(): returns metadata dict for notes/citation
    """
    db_dir: Path
    strict_voigt_order: bool = True

    def __init__(self, db_dir: Optional[str | Path] = None, *, strict_voigt_order: bool = True):
        self.db_dir = _find_db_dir(db_dir)
        self.strict_voigt_order = strict_voigt_order
        self._cache: Dict[str, dict] = {}

    def list_materials(self) -> list[str]:
        stems = [p.stem for p in self.db_dir.glob("*.yml")] + [p.stem for p in self.db_dir.glob("*.yaml")]
        return sorted(set(stems))

    def _load(self, material_file: str) -> dict:
        key = _norm(material_file)
        if key in self._cache:
            return self._cache[key]

        p = Path(material_file)
        if p.suffix.lower() in (".yml", ".yaml"):
            path = (self.db_dir / p.name).resolve()
        else:
            path = (self.db_dir / f"{material_file}.yml")
            if not path.exists():
                path = (self.db_dir / f"{material_file}.yaml")

        if not path.exists():
            # fallback: partial match by stem
            matches = []
            for f in list(self.db_dir.glob("*.yml")) + list(self.db_dir.glob("*.yaml")):
                if _norm(material_file) in _norm(f.stem):
                    matches.append(f)
            if len(matches) == 1:
                path = matches[0]
            else:
                raise FileNotFoundError(f"Material file not found: {material_file} in {self.db_dir}")

        yaml = _import_yaml()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML structure: {path.name} (must be a mapping/dict).")

        self._cache[key] = data
        return data

    def list_datasets(self, material_file: str) -> list[str]:
        data = self._load(material_file)
        ds = data.get("datasets", {})
        if not isinstance(ds, dict):
            return []
        return sorted(ds.keys())

    def get_meta(self, material_file: str, dataset: str) -> dict:
        data = self._load(material_file)
        ds = data.get("datasets", {})
        if dataset not in ds:
            raise KeyError(f"Dataset '{dataset}' not found. Available: {sorted(ds.keys())}")
        d = ds[dataset]

        keys = [
            "kind", "temp_K", "axes", "voigt_order", "shear",
            "independent_constants", "symmetry_note", "source", "units",
        ]
        return {k: d.get(k) for k in keys if k in d}

    def get(self, material_file: str, dataset: str) -> PiezoMaterial:
        data = self._load(material_file)
        name = data.get("name", material_file)

        ds = data.get("datasets", {})
        if dataset not in ds:
            raise KeyError(f"Dataset '{dataset}' not found. Available: {sorted(ds.keys())}")
        d = ds[dataset]

        # 事故りやすいので voigt_order を軽くチェック（任意）
        vo = d.get("voigt_order", None)
        if self.strict_voigt_order and vo is not None:
            expected = ["xx", "yy", "zz", "yz", "xz", "xy"]
            got = [str(x).lower() for x in vo]
            if got != expected:
                raise ValueError(f"voigt_order mismatch. expected {expected}, got {got}")

        units = d.get("units", {})

        rho = float(d["rho"])
        C6 = _read_matrix(d, "C6", (6, 6))
        e36 = _read_matrix(d, "e36", (3, 6), optional=True)  # 非圧電なら省略OK → 0埋め
        eps = _read_matrix(d, "eps", (3, 3))

        C6, eps = _convert_units(C6, eps, units)

        mat_name = f"{name}:{dataset}"
        return PiezoMaterial(name=mat_name, rho=rho, C6=C6, e36=e36, eps=eps)