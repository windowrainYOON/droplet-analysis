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
    # group / image names stay text: "01" must not turn into 1
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"group": str, "image": str}, keep_default_na=True)
    df.columns = [str(c).strip() for c in df.columns]
    if "group" not in df.columns:
        # One file = one group when the file has no group column.
        df.insert(0, "group", path.stem)
    if "image" not in df.columns:
        df.insert(1, "image", df["group"])
    df["group"] = df["group"].astype(str).str.strip()
    df["image"] = df["image"].astype(str).str.strip()
    df["source_file"] = path.name
    df.attrs["path"] = str(path)
    return df


def load_particles(paths, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    return combine_frames([read_particle_csv(p) for p in paths], existing)


def combine_frames(frames, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stack particle tables from several files.
    - Each file gets a unique source label: its name, or 'folder/name' when another file with
      the same name (e.g. every experiment's particle_shape.csv) is already there.
    - When two files use the same group and image names (e.g. both call their images 'img1'),
      the image name gets the file label in front so the images stay separate."""
    labels = set(existing["source_file"].astype(str)) if existing is not None and "source_file" in existing else set()
    seen = {}
    if existing is not None and len(existing):
        for (g, im), src in existing.groupby(["group", "image"])["source_file"].first().items():
            seen[(g, im)] = src
    for f in frames:
        if not len(f):
            continue
        label = _unique_label(f, labels)
        labels.add(label)
        f["source_file"] = label
        keys = set(zip(f["group"], f["image"]))
        clash = {k for k in keys if k in seen and seen[k] != label}
        if clash:
            stem = label.rsplit(".", 1)[0]
            m = [(g, im) in clash for g, im in zip(f["group"], f["image"])]
            f.loc[m, "image"] = stem + ":" + f.loc[m, "image"]
        for k in set(zip(f["group"], f["image"])):
            seen.setdefault(k, label)
    frames = [f for f in frames if len(f)]
    if not frames:
        raise ValueError("불러올 입자가 없습니다.")
    return pd.concat(frames, ignore_index=True, sort=False).reset_index(drop=True)


def _unique_label(f, taken) -> str:
    name = str(f["source_file"].iloc[0])
    path = Path(str(f.attrs.get("path", name)))
    if name not in taken:
        return name
    cand = f"{path.parent.name}/{name}" if path.parent.name else name
    i = 2
    base = cand
    while cand in taken:
        cand = f"{base} ({i})"; i += 1
    return cand


def content_digest(df: pd.DataFrame) -> str:
    """Fingerprint of a file's measurements (numeric columns only, so it does not depend on the
    file name, the source label or renamed images). Equal digest = same data loaded twice."""
    import hashlib
    cols = [c for c in df.columns if c != "source_file" and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    num = df[sorted(cols)].astype("float64").reset_index(drop=True)   # same digest after concat / save
    h = hashlib.sha1(",".join(num.columns).encode())
    h.update(pd.util.hash_pandas_object(num, index=False).to_numpy().tobytes())
    return h.hexdigest()


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


def classify_file(path: str | Path) -> str:
    """Guess what a file dropped on the app is, from its content rather than its name:
    'project' (.dmap), 'template' (condition-list JSON), 'conditions' (one row per group),
    'delivery' (cell delivery summary: group, region, parameter, mean) or 'particles'."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".dmap":
        return "project"
    if suf == ".json":
        import json
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return "unknown"
        return "template" if isinstance(data, dict) and "schema" in data else "unknown"
    if suf not in (".csv", ".txt", ".tsv"):
        return "unknown"
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", nrows=20000, sep=None, engine="python")
    except Exception:
        return "unknown"
    if df.empty:
        return "unknown"
    cols = [str(c).strip().lower() for c in df.columns]
    if all(c in cols for c in ("group", "region", "parameter", "mean")):
        return "delivery"
    first = df.iloc[:, 0].astype(str)
    numeric = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    if "image" in cols or (first.duplicated().any() and numeric >= 3):
        return "particles"
    # ImageJ-style results: first column is a running 1, 2, 3 ... index
    f0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if numeric >= 3 and len(df) >= 3 and f0.notna().all() and (f0.diff().dropna() == 1).all():
        return "particles"
    if cols[0] in ("group", "그룹") or not first.duplicated().any():
        return "conditions" if len(df) <= 5000 else "particles"
    return "particles" if numeric >= 3 else "unknown"
