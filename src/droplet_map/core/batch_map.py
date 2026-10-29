"""Batch map: every purification batch is one point, placed by everything that describes it.

A batch is described by items ("descriptors") in categories (purification, in vitro observation,
cell delivery, and any the user adds):
- values typed into the batch table (one value per batch, e.g. zeta potential);
- values derived from the batch's experiments (particle formation groups, cell delivery groups):
    * 경향 (trend): within one batch, how a measurement changes per unit of a condition
      (e.g. circularity per pH unit), from a linear fit over that batch's groups with the other
      varying conditions held in the fit. Works even when batches ran different conditions.
    * 같은 조건 (matched): the measurement at a condition set that two or more batches ran.
    * 대표값 (level): the median over all of the batch's groups.

Distance between two batches is Gower-like: each item contributes |a - b| / range over batches
(categorical: 0 same / 1 different), averaged within a part (source x kind), parts averaged
within a category, categories averaged with the user's weights. Only items both batches have
count, so blanks never act as zeros. The map is a PCoA (classical MDS) of that distance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from .batch_compare import condition_key
from .batches import UNASSIGNED, Batches
from .conditions import ConditionTable

CAT_PURIFICATION = "정제 조건"
CAT_INVITRO = "in vitro 관측"
CAT_DELIVERY = "세포 전달"
DEFAULT_CATEGORIES = (CAT_PURIFICATION, CAT_INVITRO, CAT_DELIVERY)

KIND_DIRECT = "직접 입력"
KIND_TREND = "경향"
KIND_MATCHED = "같은 조건"
KIND_LEVEL = "대표값"

SRC_BATCH = "배치 표"
SRC_PARTICLE = "입자"
SRC_DELIVERY = "세포 전달"

# which derived kinds are on by default, per source
DEFAULT_SOURCES = {
    f"{SRC_PARTICLE}:{KIND_TREND}": True,
    f"{SRC_PARTICLE}:{KIND_MATCHED}": True,
    f"{SRC_PARTICLE}:{KIND_LEVEL}": False,     # depends on which conditions a batch happened to run
    f"{SRC_DELIVERY}:{KIND_TREND}": True,
    f"{SRC_DELIVERY}:{KIND_MATCHED}": True,
    f"{SRC_DELIVERY}:{KIND_LEVEL}": True,
}
SOURCE_CATEGORY = {SRC_PARTICLE: CAT_INVITRO, SRC_DELIVERY: CAT_DELIVERY}
META_COLUMNS = ["category", "source", "kind", "part", "measure", "condition", "unit", "type", "scale"]
Z_SCALE = 2.0     # robust-z units that count as a full difference (median z, or z change over a condition's span)
REL_FLOOR = 0.5   # other numbers: a difference counts in full only when it reaches half the typical size


@dataclass
class Descriptors:
    values: pd.DataFrame      # batch x item; numbers as float, categorical items as text
    meta: pd.DataFrame        # item -> category, source, kind, part, measure, condition, unit, type

    def items(self, category=None) -> list[str]:
        m = self.meta if category is None else self.meta[self.meta["category"] == category]
        return list(m.index)

    def categories(self) -> list[str]:
        return list(pd.unique(self.meta["category"])) if len(self.meta) else []


# ------------------------------------------------------------------ deriving items
def within_batch_trends(measures: pd.DataFrame, cond: ConditionTable, groups) -> dict:
    """{(measure, condition): slope} over `groups` (one batch). Conditions = numeric conditions
    with two or more values among these groups. A multiple linear fit when there are enough
    groups (n >= conditions + 2), else one fit per condition (n >= 3)."""
    groups = [g for g in groups if g in measures.index and g in cond.values.index]
    if len(groups) < 2:
        return {}
    num = {}
    for c in cond.schema:
        if c.kind != "numeric":
            continue
        v = cond.numeric(c.name).reindex(groups)
        if v.dropna().nunique() >= 2:
            num[c.name] = v
    if not num:
        return {}
    X = pd.DataFrame(num)
    out = {}
    for m in measures.columns:
        y = pd.to_numeric(measures.loc[groups, m], errors="coerce")
        ok = y.notna() & X.notna().all(axis=1)
        cols = [c for c in X.columns if X.loc[ok, c].nunique() >= 2]
        if cols and ok.sum() >= len(cols) + 2:
            A = np.column_stack([np.ones(ok.sum())] + [X.loc[ok, c].to_numpy(float) for c in cols])
            coef, *_ = np.linalg.lstsq(A, y[ok].to_numpy(float), rcond=None)
            for c, b in zip(cols, coef[1:]):
                out[(m, c)] = float(b)
            continue
        for c in X.columns:                      # too few groups for one fit: one condition at a time
            ok1 = y.notna() & X[c].notna()
            if ok1.sum() >= 2 and X.loc[ok1, c].nunique() >= 2:
                out[(m, c)] = float(np.polyfit(X.loc[ok1, c].to_numpy(float), y[ok1].to_numpy(float), 1)[0])
    return out


def _derive(source: str, measures: pd.DataFrame, cond: ConditionTable, batches: Batches, groups,
            kinds: dict, unit):
    """Items from one experiment kind. measures: group x measure (numbers); unit: text or
    {measure: text}."""
    unit_of = (lambda m: unit.get(m, "")) if isinstance(unit, dict) else (lambda m: unit)
    cat = SOURCE_CATEGORY[source]
    groups = [g for g in groups if g in measures.index and batches.batch_of(g) != UNASSIGNED]
    by_batch: dict[str, list] = {}
    for g in groups:
        by_batch.setdefault(batches.batch_of(g), []).append(g)
    vals: dict[str, dict] = {}
    meta: dict[str, dict] = {}

    def put(item, b, v, kind, measure, condition, u, scale=np.nan):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return
        vals.setdefault(item, {})[b] = v
        meta[item] = dict(category=cat, source=source, kind=kind, part=f"{source} · {kind}", measure=measure,
                          condition=condition, unit=u, type="numeric", scale=scale)

    def z_scale(m):
        return Z_SCALE if unit_of(m) == "z" else np.nan

    if kinds.get(f"{source}:{KIND_TREND}", True):
        units = {c.name: c.unit for c in cond.schema}
        span = {c.name: float(np.ptp(v.dropna())) if (v := cond.numeric(c.name).reindex(groups)).notna().any() else np.nan
                for c in cond.schema if c.kind == "numeric"}
        for b, gs in by_batch.items():
            for (m, c), slope in within_batch_trends(measures, cond, gs).items():
                cu = units.get(c, "")
                mu = unit_of(m) or "값"
                # a z slope counts by the change it makes over the tested range of the condition
                sc = Z_SCALE / span[c] if unit_of(m) == "z" and span.get(c, 0) > 0 else np.nan
                put(f"{m} / {c}", b, slope, KIND_TREND, m, c, f"{mu}/{cu}" if cu else f"{mu}/단위", sc)
    if kinds.get(f"{source}:{KIND_MATCHED}", True):
        keys = {}
        for g in groups:
            k = condition_key(cond, g)
            if k is not None:
                keys.setdefault(k, {}).setdefault(batches.batch_of(g), g)
        for k, per_b in keys.items():
            if len(per_b) < 2:
                continue
            for b, g in per_b.items():
                for m in measures.columns:
                    v = pd.to_numeric(pd.Series([measures.at[g, m]]), errors="coerce").iloc[0]
                    put(f"{m} @ {k}", b, float(v) if pd.notna(v) else None, KIND_MATCHED, m, k, unit_of(m), z_scale(m))
    if kinds.get(f"{source}:{KIND_LEVEL}", True):
        for b, gs in by_batch.items():
            med = measures.loc[gs].apply(pd.to_numeric, errors="coerce").median()
            for m, v in med.items():
                put(f"{m} (배치 중앙값)", b, float(v) if pd.notna(v) else None, KIND_LEVEL, m, "", unit_of(m), z_scale(m))
    return vals, meta


def batch_descriptors(batches: Batches, particle_measures: pd.DataFrame | None = None,
                      formation: ConditionTable | None = None, delivery_measures: pd.DataFrame | None = None,
                      delivery_cond: ConditionTable | None = None, kinds: dict | None = None,
                      particle_unit="z") -> Descriptors:
    """All items for all real batches (the unassigned pseudo batch is left out)."""
    kinds = {**DEFAULT_SOURCES, **(kinds or {})}
    names = [b for b in batches.names() if b != UNASSIGNED]
    vals: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    t = batches.table
    for c in t.schema:
        cat = batches.category(c.name)
        col = t.numeric(c.name) if c.kind == "numeric" else t.categorical(c.name)
        kind, unit = c.kind, c.unit
        if kind != "numeric" and (days := _as_days(col)) is not None:      # dates count as days
            col, kind, unit = days, "numeric", "일"
        vals[c.name] = {b: col.get(b) for b in names if b in col.index and pd.notna(col.get(b))}
        meta[c.name] = dict(category=cat, source=SRC_BATCH, kind=KIND_DIRECT, part=f"{cat} · {KIND_DIRECT}",
                            measure=c.name, condition="", unit=unit, type=kind, scale=np.nan)
    for src, measures, cond, unit in ((SRC_PARTICLE, particle_measures, formation, particle_unit),
                                      (SRC_DELIVERY, delivery_measures, delivery_cond, "")):
        if measures is None or cond is None or measures.empty:
            continue
        v, m = _derive(src, measures, cond, batches, list(measures.index), kinds, unit)
        vals.update(v)
        meta.update(m)
    values = pd.DataFrame({k: pd.Series(v, dtype=object) for k, v in vals.items()}, index=names, dtype=object)
    values = values.reindex(columns=list(meta))
    for k, mm in meta.items():
        if mm["type"] == "numeric":
            values[k] = pd.to_numeric(values[k], errors="coerce")
    return Descriptors(values, pd.DataFrame.from_dict(meta, orient="index").reindex(columns=META_COLUMNS))


def _as_days(col: pd.Series) -> pd.Series | None:
    """Text that is all dates (2026-03-02, 2026.3.2, 2026/03/02) -> days since the earliest."""
    v = col.dropna().astype(str).str.strip()
    if len(v) < 2 or not v.str.match(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$").all():
        return None
    d = pd.to_datetime(col.astype(object).where(col.notna()).map(lambda x: None if pd.isna(x) else str(x).replace(".", "-").replace("/", "-")),
                       errors="coerce")
    if d.dropna().empty:
        return None
    return (d - d.min()).dt.days.astype(float)


# ------------------------------------------------------------------ distance
@dataclass
class BatchMap:
    batches: list
    distance: pd.DataFrame            # combined, 0..1
    coverage: pd.DataFrame            # number of items both batches have
    by_category: dict                 # category -> distance frame
    coords: pd.DataFrame | None       # batch x [축 1, 축 2] (PCoA), None below 3 batches
    explained: list                   # % of variation on each axis
    axis_items: dict                  # axis -> [(item, spearman rho)], strongest first
    used: list                        # items that entered the distance
    weights: dict                     # category -> weight used
    linkage: np.ndarray | None = None


def _scaled(desc: Descriptors, items) -> dict:
    """item -> (kind, array over batches): numbers scaled to 0..1 by their range over batches."""
    out = {}
    for it in items:
        col = desc.values[it]
        if desc.meta.at[it, "type"] == "numeric":
            x = pd.to_numeric(col, errors="coerce").to_numpy(float)
            ok = np.isfinite(x)
            if ok.sum() < 2:
                continue
            rng = np.nanmax(x) - np.nanmin(x)
            if not rng > 0:
                continue
            scale = desc.meta.at[it, "scale"] if "scale" in desc.meta else np.nan
            if pd.notna(scale) and scale > 0:     # fixed scale (z units): small noise stays small
                out[it] = ("num", (x - np.nanmin(x)) / scale)
            else:                                 # range over batches, but never below REL_FLOOR x typical size
                out[it] = ("num", (x - np.nanmin(x)) / max(rng, REL_FLOOR * float(np.nanmedian(np.abs(x)))))
        else:
            s = col.map(lambda v: None if pd.isna(v) else str(v))
            if s.dropna().nunique() < 2:
                continue
            out[it] = ("cat", s.to_numpy(object))
    return out


def _pair_diff(kind, x, i, j):
    if kind == "num":
        a, b = x[i], x[j]
        return min(abs(a - b), 1.0) if np.isfinite(a) and np.isfinite(b) else None
    a, b = x[i], x[j]
    return None if a is None or b is None else float(a != b)


def batch_distance_map(desc: Descriptors, weights: dict | None = None, skip=()) -> BatchMap:
    names = list(desc.values.index)
    n = len(names)
    skip = set(skip)
    weights = dict(weights or {})
    items = [it for it in desc.meta.index if it not in skip and desc.meta.at[it, "part"] not in skip
             and desc.meta.at[it, "category"] not in skip and weights.get(desc.meta.at[it, "category"], 1) > 0]
    sc = _scaled(desc, items)
    used = list(sc)
    meta = desc.meta.loc[used] if used else desc.meta.iloc[:0]
    cats = list(pd.unique(meta["category"])) if len(meta) else []
    D = np.full((n, n), np.nan)
    N = np.zeros((n, n), int)
    per_cat = {c: np.full((n, n), np.nan) for c in cats}
    for i in range(n):
        D[i, i] = 0.0
        for c in cats:
            per_cat[c][i, i] = 0.0
        for j in range(i + 1, n):
            cat_d, cat_w = [], []
            for c in cats:
                parts = []
                for _p, sub in meta[meta["category"] == c].groupby("part", sort=False):
                    ds = [d for it in sub.index if (d := _pair_diff(*sc[it], i, j)) is not None]
                    N[i, j] += len(ds)
                    if ds:
                        parts.append(float(np.mean(ds)))
                if parts:
                    dc = float(np.mean(parts))
                    per_cat[c][i, j] = per_cat[c][j, i] = dc
                    cat_d.append(dc)
                    cat_w.append(float(weights.get(c, 1)))
            if cat_d:
                D[i, j] = D[j, i] = float(np.average(cat_d, weights=cat_w))
            N[j, i] = N[i, j]
    dist = pd.DataFrame(D, index=names, columns=names)
    cov = pd.DataFrame(N, index=names, columns=names)
    coords, explained, axis_items, link = None, [], {}, None
    if n >= 3 and np.isfinite(D).any():
        Df = np.where(np.isfinite(D), D, np.nanmax(D) if np.isfinite(D).any() else 1.0)
        coords, explained = pcoa(Df, names)
        axis_items = _axis_items(desc, sc, coords)
        link = linkage(squareform(Df, checks=False), method="average")
    return BatchMap(names, dist, cov, {c: pd.DataFrame(v, index=names, columns=names) for c, v in per_cat.items()},
                    coords, explained, axis_items, used, {c: float(weights.get(c, 1)) for c in cats}, link)


def pcoa(D: np.ndarray, names, k=2):
    """Classical MDS. Returns (batch x axis frame, % of positive eigenvalue sum per axis)."""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    pos = w[w > 1e-12]
    total = pos.sum() if len(pos) else 1.0
    cols = {}
    explained = []
    for a in range(k):
        if a < len(w) and w[a] > 1e-12:
            cols[f"축 {a + 1}"] = v[:, a] * np.sqrt(w[a])
            explained.append(float(100 * w[a] / total))
        else:
            cols[f"축 {a + 1}"] = np.zeros(n)
            explained.append(0.0)
    return pd.DataFrame(cols, index=names), explained


def _axis_items(desc: Descriptors, sc: dict, coords: pd.DataFrame, top=3) -> dict:
    """Per axis: the items whose order follows it best (Spearman), ties broken by how much the
    item differs between batches."""
    out = {}
    for ax in coords.columns:
        rows = []
        for it, (kind, xs) in sc.items():
            if kind != "num":
                continue
            x = pd.to_numeric(desc.values[it], errors="coerce")
            ok = x.notna()
            if ok.sum() < 3 or x[ok].nunique() < 2 or coords.loc[ok, ax].nunique() < 2:
                continue
            r = spearmanr(x[ok], coords.loc[ok, ax]).statistic
            if np.isfinite(r):
                rows.append((it, float(r), float(np.nanmax(xs) - np.nanmin(xs))))
        rows.sort(key=lambda t: (-round(abs(t[1]), 2), -t[2]))
        out[ax] = [(it, r) for it, r, _s in rows[:top]]
    return out


def pair_differences(desc: Descriptors, a: str, b: str, used=None) -> pd.DataFrame:
    """Per item: value in a, value in b, signed difference as a share of the range over batches."""
    items = list(used) if used is not None else list(desc.meta.index)
    sc = _scaled(desc, items)
    names = list(desc.values.index)
    i, j = names.index(a), names.index(b)
    rows = []
    for it, (kind, x) in sc.items():
        va, vb = desc.values.at[a, it], desc.values.at[b, it]
        if kind == "num":
            if not (np.isfinite(x[i]) and np.isfinite(x[j])):
                continue
            d = float(np.clip(x[j] - x[i], -1, 1))
        else:
            if x[i] is None or x[j] is None:
                continue
            d = float(x[i] != x[j])
        m = desc.meta.loc[it]
        rows.append({"항목": it, "범주": m["category"], "종류": m["kind"], a: va, b: vb, "차이 (범위 대비)": d})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.reindex(out["차이 (범위 대비)"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return out


def batch_scores(bm: BatchMap, ref: str) -> pd.Series:
    """0-100: 100 = the reference batch, 0 = as far as two batches can be."""
    if ref not in bm.distance.index:
        raise ValueError(f"'{ref}' 배치가 지도에 없습니다.")
    return (100 * (1 - bm.distance[ref])).rename("score")
