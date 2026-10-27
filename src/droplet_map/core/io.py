"""Reading particle CSVs and guessing what each column is for."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import re

import numpy as np
import pandas as pd

ID_COLUMNS = {"group", "image", "image_id", "source", "particle", "source_file"}
POSITION_COLUMNS = {"x", "y", "x_um", "y_um", "x_px", "y_px"}
SHAPE_HINTS = ("circularity", "solidity", "aspect_ratio", "roundness", "convexity", "eccentricity")
LOG_HINTS = ("diameter", "area", "perimeter", "feret", "major", "minor", "nnd", "_mean", "stddev", "partition", "aspect_ratio")
# Measurements that repeat information already carried by a default feature.
REDUNDANT = {
    "area_px", "area_um2", "shape_area_um2", "perimeter_um", "feret_um", "min_feret_um",
    "major_um", "minor_um", "roundness", "timepoint",
}

ROLES = ("feature", "measure", "id", "meta", "exclude")
ROLE_LABELS = {
    "feature": "분석에 사용",
    "measure": "측정값 (사용 안 함)",
    "id": "ID",
    "meta": "설정 기록",
    "exclude": "제외",
}


@dataclass
class ColumnSpec:
    name: str
    role: str = "exclude"
    log: bool = False
    shape: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class ColumnRoles:
    columns: list[ColumnSpec] = field(default_factory=list)
    size_column: str | None = None

    def features(self) -> list[str]:
        return [c.name for c in self.columns if c.role == "feature"]

    def spec(self, name: str) -> ColumnSpec:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def to_dict(self):
        return {"columns": [c.to_dict() for c in self.columns], "size_column": self.size_column}

    @classmethod
    def from_dict(cls, d):
        return cls([ColumnSpec(**c) for c in d["columns"]], d.get("size_column"))


def read_particle_csv(path: str | Path) -> pd.DataFrame:
    """Read one CSV exported by the image analysis step (UTF-8 with or without BOM)."""
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    if "group" not in df.columns:
        # One file = one group when the file has no group column.
        df.insert(0, "group", path.stem)
    if "image" not in df.columns:
        df.insert(1, "image", df["group"])
    df["group"] = df["group"].astype(str).str.strip()
    df["image"] = df["image"].astype(str).str.strip()
    df["source_file"] = path.name
    return df


def load_particles(paths) -> pd.DataFrame:
    frames = [read_particle_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True, sort=False)
    return df.reset_index(drop=True)


def _is_angle(name: str) -> bool:
    return "angle" in name.lower()


def guess_roles(df: pd.DataFrame) -> ColumnRoles:
    specs = []
    size_col = None
    for cand in ("eq_diameter_um", "diameter_um", "eq_diameter"):
        if cand in df.columns:
            size_col = cand
            break
    has_partition = {re.sub(r"_partition$", "", c) for c in df.columns if c.endswith("_partition")}
    for c in df.columns:
        low = c.lower()
        spec = ColumnSpec(c)
        numeric = pd.api.types.is_numeric_dtype(df[c])
        if low in ID_COLUMNS or re.search(r"(^|_)(id|index|label)$", low):
            spec.role = "id"
        elif not numeric or df[c].nunique(dropna=True) <= 1:
            spec.role = "meta"
        elif low in POSITION_COLUMNS or low.startswith(("centroid", "bbox")) or _is_angle(low):
            spec.role = "exclude"
        else:
            spec.role = "measure"
            spec.log = any(h in low for h in LOG_HINTS)
            spec.shape = any(h in low for h in SHAPE_HINTS)
            redundant = low in REDUNDANT
            # A channel's mean intensity mostly repeats its partition coefficient.
            m = re.match(r"(.+)_mean$", c, flags=re.I)
            if m and m.group(1) in has_partition:
                redundant = True
            if low.endswith("_stddev"):
                redundant = True
            if not redundant:
                spec.role = "feature"
        specs.append(spec)
    roles = ColumnRoles(specs, size_col)
    if size_col is None and "area_um2" in df.columns:
        roles.size_column = "__eq_diameter_from_area"
    return roles


def size_values(df: pd.DataFrame, roles: ColumnRoles) -> np.ndarray:
    """Equivalent diameter in micrometres, used for the shape reliability weight."""
    if roles.size_column and roles.size_column in df.columns:
        return df[roles.size_column].to_numpy(float)
    if "area_um2" in df.columns:
        return 2 * np.sqrt(df["area_um2"].to_numpy(float) / np.pi)
    raise ValueError("입자 크기(지름 또는 면적) 열을 찾을 수 없습니다.")
