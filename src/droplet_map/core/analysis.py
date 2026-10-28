"""The analysis pipeline: weights -> scaled features -> particle map -> group profiles ->
similarity, reference score and condition association. Pure numpy/pandas, no GUI."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, spearmanr, kruskal

from .io import ColumnRoles, size_values
from .conditions import ConditionTable

MAX_PER_GROUP_FOR_DISTANCE = 3000


@dataclass
class Settings:
    d50: float = 1.2           # um; diameter at which a shape value counts half
    k: float = 4.0             # steepness of the reliability curve
    min_px: int = 3            # QC: drop particles smaller than this many pixels (if area_px exists)
    n_neighbors: int = 15
    resolution: float = 0.5
    score_scale: float = 0.5   # z units beyond the reference spread that cost a factor of e
    min_groups_assoc: int = 8
    random_state: int = 0
    feature_weights: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def reliability_weight(d, d50=1.2, k=4.0):
    d = np.asarray(d, float)
    with np.errstate(divide="ignore", over="ignore"):
        return 1.0 / (1.0 + (d50 / np.clip(d, 1e-9, None)) ** k)


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        sd = np.nanstd(x)
        mad = sd if sd > 0 else 1.0
    return (x - med) / mad


def weighted_quantile(x, w, q):
    x = np.asarray(x, float); w = np.asarray(w, float)
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return np.nan
    x, w = x[ok], w[ok]
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, q * cw[-1])])


@dataclass
class Prepared:
    data: pd.DataFrame          # QC-passed particles, original columns + weight
    Z: pd.DataFrame             # transformed + robust-z features (shape features NOT yet shrunk)
    w: np.ndarray               # shape reliability per particle
    features: list
    shape_features: list
    n_dropped: int


def prepare(df: pd.DataFrame, roles: ColumnRoles, s: Settings) -> Prepared:
    feats = roles.features()
    if not feats:
        raise ValueError("분석에 사용할 측정값을 하나 이상 고르세요.")
    data = df.copy()
    keep = np.ones(len(data), bool)
    if "area_px" in data.columns and s.min_px > 0:
        keep &= data["area_px"].to_numpy() >= s.min_px
    for f in feats:
        v = pd.to_numeric(data[f], errors="coerce").to_numpy(float)
        keep &= np.isfinite(v)
        if roles.spec(f).log:
            keep &= v > 0
    n_dropped = int((~keep).sum())
    data = data.loc[keep].reset_index(drop=True)
    w = reliability_weight(size_values(data, roles), s.d50, s.k)
    data["shape_weight"] = w
    Z = pd.DataFrame(index=data.index)
    for f in feats:
        v = data[f].to_numpy(float)
        if roles.spec(f).log:
            v = np.log10(v)
        Z[f] = robust_z(v)
    shape = [f for f in feats if roles.spec(f).shape]
    return Prepared(data, Z, w, feats, shape, n_dropped)


def map_matrix(p: Prepared, s: Settings) -> np.ndarray:
    """Feature matrix for the particle map: shape features pulled toward 0 for small particles."""
    X = p.Z.to_numpy(float).copy()
    for j, f in enumerate(p.features):
        if f in p.shape_features:
            X[:, j] *= p.w
        X[:, j] *= float(s.feature_weights.get(f, 1.0))
    return X


@dataclass
class ParticleMap:
    umap: np.ndarray
    clusters: np.ndarray


def particle_map(p: Prepared, s: Settings, progress=None, idx=None) -> ParticleMap:
    """UMAP + Leiden of all particles, or only of the rows in `idx` (a sub-map for drilling into
    one cluster or selection: neighbours and scaling are then found among those particles only)."""
    import umap  # heavy import (numba); keep it out of module import time
    import igraph as ig
    import leidenalg

    X = map_matrix(p, s)
    if idx is not None:
        X = X[np.asarray(idx)]
        if len(X) < 10:
            raise ValueError("하위 지도를 그리기에는 입자가 너무 적습니다 (10개 이상 필요).")
    if progress:
        progress("이웃 그래프와 UMAP 계산 중…")
    reducer = umap.UMAP(n_neighbors=min(s.n_neighbors, max(2, len(X) - 1)), min_dist=0.3,
                        random_state=s.random_state, n_jobs=1)
    emb = reducer.fit_transform(X)
    G = reducer.graph_.tocoo()
    if progress:
        progress("Leiden 군집화 중…")
    mask = G.row < G.col
    g = ig.Graph(n=X.shape[0], edges=list(zip(G.row[mask].tolist(), G.col[mask].tolist())), directed=False)
    g.es["weight"] = G.data[mask].tolist()
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=s.resolution, seed=s.random_state)
    cl = np.asarray(part.membership)
    # Number clusters by size, largest = 0.
    order = {c: i for i, c in enumerate(pd.Series(cl).value_counts().index)}
    cl = np.array([order[c] for c in cl])
    return ParticleMap(emb, cl)


def cohesion(emb: np.ndarray, labels, max_n=4000, seed=0) -> float | None:
    """Mean silhouette of `labels` on the map (-1..1): higher = each label sits in its own region.
    Labels that are None / NaN are left out; None when fewer than two labels remain."""
    from sklearn.metrics import silhouette_score

    lab = pd.Series(labels, dtype=object)
    ok = lab.notna().to_numpy()
    if lab[ok].nunique() < 2 or ok.sum() < 10:
        return None
    X, y = emb[ok], lab[ok].astype(str).to_numpy()
    if len(X) > max_n:
        sel = np.random.default_rng(seed).choice(len(X), max_n, replace=False)
        X, y = X[sel], y[sel]
        if len(set(y)) < 2:
            return None
    return float(silhouette_score(X, y))


# ---------------------------------------------------------------- group level
@dataclass
class GroupProfiles:
    groups: list
    median_z: pd.DataFrame          # group x feature, weighted for shape features
    median_raw: pd.DataFrame        # group x feature in original units
    n_particles: pd.Series
    n_images: pd.Series
    n_eff_shape: pd.Series          # effective number of particles behind shape values
    mean_weight: pd.Series
    image_stats: pd.DataFrame       # group x [per-image particle count mean/sd, total area mean]
    composition: pd.DataFrame | None = None   # group x cluster fraction


def group_profiles(p: Prepared, pmap: ParticleMap | None = None) -> GroupProfiles:
    d = p.data
    groups = list(pd.unique(d["group"]))
    mz, mr = {}, {}
    for g in groups:
        idx = (d["group"] == g).to_numpy()
        w = p.w[idx]
        rz, rr = {}, {}
        for f in p.features:
            z = p.Z.loc[idx, f].to_numpy()
            raw = d.loc[idx, f].to_numpy(float)
            if f in p.shape_features:
                rz[f] = weighted_quantile(z, w, .5)
                rr[f] = weighted_quantile(raw, w, .5)
            else:
                rz[f] = float(np.median(z))
                rr[f] = float(np.median(raw))
        mz[g], mr[g] = rz, rr
    gb = d.groupby("group", sort=False)
    n_eff = gb["shape_weight"].apply(lambda w: (w.sum() ** 2) / max((w ** 2).sum(), 1e-12))
    per_img = d.groupby(["group", "image"], sort=False)
    area_col = "area_um2" if "area_um2" in d.columns else None
    img = pd.DataFrame({"count": per_img.size()})
    if area_col:
        img["total_area"] = per_img[area_col].sum()
    img_stats = pd.DataFrame({
        "이미지당 입자 수 (평균)": img.groupby(level=0)["count"].mean(),
        "이미지당 입자 수 (SD)": img.groupby(level=0)["count"].std(),
    })
    if area_col:
        img_stats["이미지당 총 입자 면적 µm² (평균)"] = img.groupby(level=0)["total_area"].mean()
    comp = None
    if pmap is not None:
        comp = pd.crosstab(d["group"], pmap.clusters, normalize="index").reindex(groups)
    return GroupProfiles(groups, pd.DataFrame(mz).T.reindex(groups), pd.DataFrame(mr).T.reindex(groups),
                         gb.size().reindex(groups), gb["image"].nunique().reindex(groups),
                         n_eff.reindex(groups), gb["shape_weight"].mean().reindex(groups),
                         img_stats.reindex(groups), comp)


def _group_samples(p: Prepared, rng) -> dict:
    out = {}
    d = p.data
    for g in pd.unique(d["group"]):
        idx = np.flatnonzero((d["group"] == g).to_numpy())
        if len(idx) > MAX_PER_GROUP_FOR_DISTANCE:
            idx = np.sort(rng.choice(idx, MAX_PER_GROUP_FOR_DISTANCE, replace=False))
        out[g] = idx
    return out


def _w1(p: Prepared, a_idx, b_idx, f) -> float:
    a = p.Z[f].to_numpy()[a_idx]
    b = p.Z[f].to_numpy()[b_idx]
    if f in p.shape_features:
        wa, wb = p.w[a_idx], p.w[b_idx]
        if wa.sum() < 1e-6 or wb.sum() < 1e-6:
            return np.nan
        return wasserstein_distance(a, b, wa, wb)
    return wasserstein_distance(a, b)


def group_distance(p: Prepared, s: Settings) -> pd.DataFrame:
    """Mean over features of the 1-D Wasserstein distance between group distributions (z units)."""
    rng = np.random.default_rng(s.random_state)
    samp = _group_samples(p, rng)
    groups = list(samp)
    n = len(groups)
    D = np.zeros((n, n))
    wts = {f: float(s.feature_weights.get(f, 1.0)) for f in p.features}
    for i in range(n):
        for j in range(i + 1, n):
            num = den = 0.0
            for f in p.features:
                v = _w1(p, samp[groups[i]], samp[groups[j]], f)
                if np.isfinite(v):
                    c = wts[f]
                    if f in p.shape_features:
                        c *= min(p.w[samp[groups[i]]].mean(), p.w[samp[groups[j]]].mean())
                    num += c * v; den += c
            D[i, j] = D[j, i] = num / den if den > 0 else np.nan
    return pd.DataFrame(D, index=groups, columns=groups)


def reference_score(p: Prepared, s: Settings, references) -> pd.DataFrame:
    """Score 0-100: how close each group's distributions are to the pooled reference groups.
    Differences no larger than the spread among the references themselves cost nothing."""
    refs = [r for r in references if r in set(p.data["group"])]
    if not refs:
        raise ValueError("기준 그룹을 하나 이상 지정하세요.")
    rng = np.random.default_rng(s.random_state)
    samp = _group_samples(p, rng)
    R = np.concatenate([samp[r] for r in refs])
    tol = {}
    for f in p.features:
        tol[f] = float(np.nanmean([_w1(p, samp[r], R, f) for r in refs])) if len(refs) > 1 else 0.0
    rows = {}
    zref = {f: (weighted_quantile(p.Z[f].to_numpy()[R], p.w[R], .5) if f in p.shape_features
                else float(np.median(p.Z[f].to_numpy()[R]))) for f in p.features}
    for g, idx in samp.items():
        num = den = 0.0
        row = {}
        for f in p.features:
            v = _w1(p, idx, R, f)
            c = float(s.feature_weights.get(f, 1.0))
            if f in p.shape_features:
                c *= float(p.w[idx].mean())
            gz = (weighted_quantile(p.Z[f].to_numpy()[idx], p.w[idx], .5) if f in p.shape_features
                  else float(np.median(p.Z[f].to_numpy()[idx])))
            row[f"Δ {f}"] = gz - zref[f]
            if np.isfinite(v) and c > 0:
                d = max(0.0, v - tol[f]) / max(tol[f], s.score_scale)
                num += c * d; den += c
        row["score"] = 100.0 * np.exp(-num / den) if den > 0 else np.nan
        row["기준"] = "★" if g in refs else ""
        rows[g] = row
    out = pd.DataFrame(rows).T
    cols = ["score", "기준"] + [c for c in out.columns if c.startswith("Δ ")]
    out = out[cols]
    out["score"] = out["score"].astype(float)
    return out.sort_values("score", ascending=False)


@dataclass
class Association:
    strength: pd.DataFrame    # condition x target, 0..1 (|rho| or epsilon^2)
    sign: pd.DataFrame        # +1 / -1 for numeric, 0 for categorical
    n: pd.Series              # groups with that condition recorded
    kind: pd.Series
    hidden: pd.Series         # True when too few groups or no variation
    confounded: list          # (cond_a, cond_b, strength)


def association(cond: ConditionTable, targets: pd.DataFrame, min_groups=8) -> Association:
    """Condition x target association at group level. targets: group x column (e.g. medians, score)."""
    names = cond.names()
    S = pd.DataFrame(np.nan, index=names, columns=targets.columns)
    sign = pd.DataFrame(0.0, index=names, columns=targets.columns)
    n = pd.Series(0, index=names, dtype=int)
    hidden = pd.Series(False, index=names)
    kinds = pd.Series({c.name: c.kind for c in cond.schema})
    for c in cond.schema:
        if c.kind == "numeric":
            x = cond.numeric(c.name)
        else:
            x = cond.categorical(c.name)
        x = x.reindex(targets.index)
        ok = x.notna()
        n[c.name] = int(ok.sum())
        levels = x[ok].nunique()
        if n[c.name] < min_groups or levels < 2:
            hidden[c.name] = True
            continue
        for t in targets.columns:
            y = targets.loc[ok, t].astype(float)
            good = y.notna()
            if good.sum() < 3:
                continue
            if c.kind == "numeric":
                r = spearmanr(x[ok][good].astype(float), y[good]).statistic
                if np.isfinite(r):
                    S.at[c.name, t] = abs(r); sign.at[c.name, t] = np.sign(r)
            else:
                lab = x[ok][good]
                grp = [y[good][lab == v].to_numpy() for v in pd.unique(lab)]
                grp = [g for g in grp if len(g) >= 1]
                if len(grp) >= 2 and len(y[good]) > len(grp):
                    try:
                        H = kruskal(*grp).statistic
                        S.at[c.name, t] = float(np.clip(H / (len(y[good]) - 1), 0, 1))
                    except ValueError:
                        pass
    conf = []
    for i, a in enumerate(cond.schema):
        for b in cond.schema[i + 1:]:
            v = _condition_link(cond, a, b)
            if v is not None and v > 0.7:
                conf.append((a.name, b.name, v))
    return Association(S, sign, n, kinds, hidden, conf)


def _condition_link(cond: ConditionTable, a, b):
    xa = cond.numeric(a.name) if a.kind == "numeric" else cond.categorical(a.name)
    xb = cond.numeric(b.name) if b.kind == "numeric" else cond.categorical(b.name)
    ok = xa.notna() & xb.notna()
    if ok.sum() < 4 or xa[ok].nunique() < 2 or xb[ok].nunique() < 2:
        return None
    if a.kind == "numeric" and b.kind == "numeric":
        return abs(spearmanr(xa[ok], xb[ok]).statistic)
    tab = pd.crosstab(xa[ok].astype(str), xb[ok].astype(str)).to_numpy(float)
    nt = tab.sum()
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / nt
    chi2 = ((tab - exp) ** 2 / np.where(exp > 0, exp, 1)).sum()
    k = min(tab.shape) - 1
    return float(np.sqrt(chi2 / (nt * k))) if k > 0 else None


# ---------------------------------------------------------------- one-shot run
@dataclass
class Result:
    prepared: Prepared
    pmap: ParticleMap | None
    profiles: GroupProfiles
    distance: pd.DataFrame
    score: pd.DataFrame | None
    assoc: Association | None


def run_all(df, roles, cond: ConditionTable, s: Settings, progress=None, with_map=True) -> Result:
    if progress:
        progress("측정값 변환 중…")
    p = prepare(df, roles, s)
    pm = particle_map(p, s, progress) if with_map else None
    if progress:
        progress("그룹 프로필과 유사도 계산 중…")
    prof = group_profiles(p, pm)
    dist = group_distance(p, s)
    score = None
    refs = [r for r in cond.references if r in prof.groups]
    if refs:
        score = reference_score(p, s, refs)
    assoc = None
    if cond.schema:
        targets = prof.median_raw.copy()
        targets = targets.join(prof.image_stats[[c for c in prof.image_stats.columns if "(SD)" not in c]])
        if score is not None:
            targets["기준 score"] = score["score"].reindex(targets.index)
        assoc = association(cond, targets, s.min_groups_assoc)
    return Result(p, pm, prof, dist, score, assoc)
