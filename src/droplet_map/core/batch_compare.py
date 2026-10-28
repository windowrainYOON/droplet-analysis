"""Batch comparison with the formation conditions held fixed.

Groups from different batches are paired when their own (group-level) conditions are identical,
e.g. pH 7 · 150 mM NaCl · 10 % PEG in batch A and in batch B. Differences between such pairs are
batch differences, not condition differences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import Prepared, Settings, reference_score, weighted_quantile
from .batches import UNASSIGNED, Batches
from .conditions import ConditionTable


def condition_key(cond: ConditionTable, group: str) -> str | None:
    """'pH=7 · NaCl (mM)=150 · PEG (%)=10' from the group's own conditions; None when all blank."""
    parts, any_value = [], False
    for c in cond.schema:
        v = cond.categorical(c.name).get(group)
        if pd.isna(v):
            parts.append(f"{c.label}=—")
        else:
            any_value = True
            parts.append(f"{c.label}={v}")
    return " · ".join(parts) if any_value else None


def matched_groups(groups, cond: ConditionTable, batches: Batches) -> pd.DataFrame:
    """Rows = formation-condition keys, columns = batches, cells = group name (or NaN).
    Only keys found in two or more batches. When a batch has two groups with the same key, the
    first one is used."""
    rows = {}
    for g in groups:
        b = batches.batch_of(g)
        k = condition_key(cond, g)
        if k is None or b == UNASSIGNED:
            continue
        rows.setdefault(k, {}).setdefault(b, g)
    t = pd.DataFrame(rows).T
    if t.empty:
        return t
    order = [b for b in batches.names() if b in t.columns]
    t = t[order]
    return t[t.notna().sum(axis=1) >= 2]


def batch_distance(distance: pd.DataFrame, matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean group distance between batches over the condition keys both have. Returns (D, n pairs)."""
    bs = list(matched.columns)
    D = pd.DataFrame(np.nan, index=bs, columns=bs)
    N = pd.DataFrame(0, index=bs, columns=bs)
    for i, a in enumerate(bs):
        D.at[a, a] = 0.0
        for b in bs[i + 1:]:
            both = matched[[a, b]].dropna()
            vals = [distance.at[ga, gb] for ga, gb in both.itertuples(index=False)
                    if ga in distance.index and gb in distance.index]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                D.at[a, b] = D.at[b, a] = float(np.mean(vals))
            N.at[a, b] = N.at[b, a] = len(vals)
    return D, N


def _group_median_z(p: Prepared, g: str, f: str) -> float:
    idx = np.flatnonzero((p.data["group"] == g).to_numpy())
    if not len(idx):
        return np.nan
    z = p.Z[f].to_numpy()[idx]
    if f in p.shape_features:
        return weighted_quantile(z, p.w[idx], .5)
    return float(np.median(z))


def batch_feature_delta(p: Prepared, matched: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Per condition key and measurement: median z of batch b's group minus batch a's group.
    The last row ('평균') is the mean over keys."""
    both = matched[[a, b]].dropna()
    out = {}
    for key, (ga, gb) in both.iterrows():
        out[key] = {f: _group_median_z(p, gb, f) - _group_median_z(p, ga, f) for f in p.features}
    t = pd.DataFrame(out).T
    if not t.empty:
        t.loc["평균"] = t.mean()
    return t


def batch_reference_score(p: Prepared, s: Settings, groups, cond: ConditionTable, batches: Batches,
                          ref_batch: str) -> pd.DataFrame:
    """Score each group against the reference batch's group with the same formation conditions.
    Groups whose conditions the reference batch lacks get no score."""
    present = set(p.data["group"])
    ref_of_key = {}
    for g in groups:
        if batches.batch_of(g) == ref_batch and g in present:
            k = condition_key(cond, g)
            if k is not None:
                ref_of_key.setdefault(k, g)
    if not ref_of_key:
        raise ValueError(f"기준 배치 '{ref_batch}'에 조건이 입력된 그룹이 없습니다.")
    rows = {}
    cache = {}
    for g in groups:
        if g not in present:
            continue
        k = condition_key(cond, g)
        ref = ref_of_key.get(k)
        row = {"배치": batches.batch_of(g), "같은 조건의 기준 그룹": ref or "—", "score": np.nan}
        if ref is not None:
            if ref not in cache:
                cache[ref] = reference_score(p, s, [ref])
            row["score"] = float(cache[ref].at[g, "score"]) if g in cache[ref].index else np.nan
        rows[g] = row
    out = pd.DataFrame(rows).T
    out["score"] = out["score"].astype(float)
    return out.sort_values(["배치", "score"], ascending=[True, False])


def batch_summary(score: pd.DataFrame) -> pd.DataFrame:
    """Per batch: mean score over its matched groups and how many groups had a match."""
    g = score.dropna(subset=["score"]).groupby("배치")["score"]
    return pd.DataFrame({"평균 score": g.mean(), "비교한 그룹 수": g.size()}).sort_values("평균 score", ascending=False)
