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
    # particle map only (the score and comparison still use everything):
    map_skip_features: list = field(default_factory=list)   # measurements left out of the map
    map_skip_values: dict = field(default_factory=dict)     # condition -> values whose groups are left out ("" = not recorded)
    map_skip_groups: list = field(default_factory=list)     # groups left out by name
    # batch map:
    bm_kinds: dict = field(default_factory=dict)      # "입자:경향" -> on/off (defaults in batch_map.DEFAULT_SOURCES)
    bm_weights: dict = field(default_factory=dict)    # category -> weight (default 1; 0 = off)
    bm_skip: list = field(default_factory=list)       # items, parts or categories left out of the distance
    bm_color: str = ""                                # item or batch column that colours the batch map

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


def map_matrix(p: Prepared, s: Settings, idx=None, restandardize=False) -> np.ndarray:
    """Feature matrix for the particle map: shape features pulled toward 0 for small particles,
    then each feature scaled by its user weight (weight 0 drops it, 2 counts it double).
    With idx, only those particles; restandardize re-centres each feature within the subset so
    a sub-map spreads out differences that were small on the full map."""
    Z = p.Z if idx is None else p.Z.iloc[idx]
    w = p.w if idx is None else p.w[idx]
    X = Z.to_numpy(float).copy()
    if restandardize:
        for j in range(X.shape[1]):
            X[:, j] = robust_z(X[:, j])
    for j, f in enumerate(p.features):
        if f in p.shape_features:
            X[:, j] *= w
        X[:, j] *= 0.0 if f in s.map_skip_features else float(s.feature_weights.get(f, 1.0))
    return X


def map_skipped_groups(cond: ConditionTable | None, s: Settings) -> set:
    """Groups the user chose to leave out of the particle map (by name or by condition value)."""
    out = set(s.map_skip_groups)
    if cond is not None:
        for name, values in s.map_skip_values.items():
            if name not in cond.values.columns or not values:
                continue
            v = cond.categorical(name)
            skip = set(map(str, values))
            out |= {g for g, x in v.items() if ("" if pd.isna(x) else str(x)) in skip}
    return out


def map_rows(p: Prepared, cond: ConditionTable | None, s: Settings, idx=None):
    """Rows of p.data that go into the map: idx (or all) minus the skipped groups. None = all rows."""
    skip = map_skipped_groups(cond, s)
    if not skip:
        return idx
    keep = ~p.data["group"].isin(skip).to_numpy()
    rows = np.flatnonzero(keep) if idx is None else np.asarray(idx)[keep[np.asarray(idx)]]
    return rows


@dataclass
class ParticleMap:
    umap: np.ndarray
    clusters: np.ndarray
    idx: np.ndarray | None = None     # rows of Prepared.data on this map (None = all)
    name: str = "전체"


def particle_map(p: Prepared, s: Settings, progress=None, idx=None, name="전체", restandardize=None) -> ParticleMap:
    """UMAP + Leiden on all particles, or (hierarchical map) on the subset idx.
    Sub-maps re-centre the measurements within the subset unless restandardize=False."""
    import umap  # heavy import (numba); keep it out of module import time
    import igraph as ig
    import leidenalg

    X = map_matrix(p, s, idx, restandardize=(idx is not None) if restandardize is None else restandardize)
    keep = np.abs(X).sum(axis=0) > 0          # features with weight 0 contribute nothing
    if not keep.any():
        raise ValueError("지도에 넣을 측정값이 없습니다. '지도에 넣을 값'에서 측정값을 하나 이상 켜세요.")
    X = X[:, keep]
    if len(X) < 10:
        raise ValueError("지도를 그리려면 입자가 10개 이상 필요합니다. 지도에서 뺀 그룹이 너무 많은지 확인하세요.")
    if progress:
        progress(f"이웃 그래프와 UMAP 계산 중… ({len(X):,}개 입자)")
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
    return ParticleMap(emb, cl, None if idx is None else np.asarray(idx), name)


def cohesion(emb: np.ndarray, labels, k=15, max_n=5000, seed=0) -> float | None:
    """How well the map separates the label categories: share of each particle's k nearest map
    neighbours with the same label, rescaled so random mixing = 0 and full separation = 1.
    Blank labels (None/NaN) are left out; None when fewer than two labels remain."""
    from sklearn.neighbors import NearestNeighbors

    lab = pd.Series(labels, dtype=object)
    ok = lab.notna().to_numpy()
    if ok.sum() < 30 or lab[ok].nunique() < 2:
        return None
    X, y = np.asarray(emb)[ok], lab[ok].astype(str).to_numpy()
    if len(X) > max_n:
        sel = np.random.default_rng(seed).choice(len(X), max_n, replace=False)
        X, y = X[sel], y[sel]
    k = min(k, len(X) - 1)
    nb = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(return_distance=False)[:, 1:]
    purity = float((y[nb] == y[:, None]).mean())
    p = pd.Series(y).value_counts(normalize=True).to_numpy()
    expect = float((p ** 2).sum())
    return (purity - expect) / (1 - expect) if expect < 1 else None


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
        g_on_map = d["group"] if pmap.idx is None else d["group"].iloc[pmap.idx]
        comp = pd.crosstab(g_on_map.to_numpy(), pmap.clusters, normalize="index").reindex(groups)
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
    n: pd.Series              # groups (or batches, for batch-level conditions) behind each row
    kind: pd.Series
    hidden: pd.Series         # True when too few groups or no variation
    confounded: list          # (cond_a, cond_b, strength)
    level: pd.Series = None   # "group": compared within batches · "batch": compared between batches


def _centered_ranks(v: pd.Series, strata: pd.Series) -> pd.Series:
    """Ranks within each stratum minus that stratum's mean rank: removes batch offsets."""
    r = v.groupby(strata).rank()
    return r - r.groupby(strata).transform("mean")


def association(cond: ConditionTable, targets: pd.DataFrame, min_groups=8, strata: pd.Series | None = None,
                min_batches=3) -> Association:
    """Condition x target association at group level. targets: group x column (e.g. medians, score).

    With strata (group -> batch), a condition that varies inside batches is compared within each
    batch (ranks centred per batch, so a batch difference cannot pose as a condition effect), and a
    condition that is constant inside every batch is compared between batches on batch medians."""
    names = cond.names()
    S = pd.DataFrame(np.nan, index=names, columns=targets.columns)
    sign = pd.DataFrame(0.0, index=names, columns=targets.columns)
    n = pd.Series(0, index=names, dtype=int)
    hidden = pd.Series(False, index=names)
    level = pd.Series("group", index=names, dtype=object)
    kinds = pd.Series({c.name: c.kind for c in cond.schema})
    st = None if strata is None else strata.reindex(targets.index).fillna("—").astype(str)
    multi = st is not None and st.nunique() > 1
    for c in cond.schema:
        x = (cond.numeric(c.name) if c.kind == "numeric" else cond.categorical(c.name)).reindex(targets.index)
        ok = x.notna()
        between = False
        if multi:
            per = x[ok].groupby(st[ok]).nunique()
            between = bool(len(per)) and (per <= 1).all()
        if between:                    # batch-level condition: one value per batch
            level[c.name] = "batch"
            xb = x[ok].groupby(st[ok]).first()
            n[c.name] = len(xb)
            if len(xb) < min_batches or xb.nunique() < 2:
                hidden[c.name] = True
                continue
            for t in targets.columns:
                yb = targets.loc[ok, t].astype(float).groupby(st[ok]).median().reindex(xb.index)
                good = yb.notna()
                if good.sum() < 3:
                    continue
                if c.kind == "numeric":
                    r = spearmanr(xb[good].astype(float), yb[good]).statistic
                    if np.isfinite(r):
                        S.at[c.name, t] = abs(r); sign.at[c.name, t] = np.sign(r)
                else:
                    S.at[c.name, t] = _epsilon2(yb[good], xb[good])
            continue
        if multi:                      # only batches where the condition actually varies
            per = x[ok].groupby(st[ok]).transform("nunique")
            ok = ok & (per.reindex(x.index).fillna(0) >= 2)
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
            xs, ys = x[ok][good], y[good]
            if c.kind == "numeric":
                xs = xs.astype(float)
                if multi:
                    xs, ys = _centered_ranks(xs, st[xs.index]), _centered_ranks(ys, st[ys.index])
                r = spearmanr(xs, ys).statistic
                if np.isfinite(r):
                    S.at[c.name, t] = abs(r); sign.at[c.name, t] = np.sign(r)
            else:
                if multi:
                    ys = _centered_ranks(ys, st[ys.index])
                S.at[c.name, t] = _epsilon2(ys, xs)
    conf = []
    for i, a in enumerate(cond.schema):
        for b in cond.schema[i + 1:]:
            v = _condition_link(cond, a, b)
            if v is not None and v > 0.7:
                conf.append((a.name, b.name, v))
    return Association(S, sign, n, kinds, hidden, conf, level)


def _epsilon2(y: pd.Series, lab: pd.Series) -> float:
    grp = [y[lab == v].to_numpy() for v in pd.unique(lab)]
    grp = [g for g in grp if len(g) >= 1]
    if len(grp) >= 2 and len(y) > len(grp):
        try:
            H = kruskal(*grp).statistic
            return float(np.clip(H / (len(y) - 1), 0, 1))
        except ValueError:
            return np.nan
    return np.nan


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


def run_all(df, roles, cond: ConditionTable, s: Settings, progress=None, with_map=True, strata=None) -> Result:
    """strata: group -> batch, so condition effects are judged within batches."""
    if progress:
        progress("측정값 변환 중…")
    p = prepare(df, roles, s)
    pm = None
    if with_map:
        rows = map_rows(p, cond, s)
        pm = particle_map(p, s, progress, idx=rows, name="전체" if rows is None else "전체 (일부 그룹 제외)",
                          restandardize=False)
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
        assoc = association(cond, targets, s.min_groups_assoc, strata)
    return Result(p, pm, prof, dist, score, assoc)
