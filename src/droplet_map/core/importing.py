"""Bringing image-analysis CSVs into a batch file.

Particle CSVs (one row per particle, with a group column) go to the per-particle sheet; cell
delivery summaries (group x region x parameter) go to a per-condition sheet as one row per group.
Each CSV group is put under a formation condition (조건 ID) chosen by the user. Condition values
are read from group names only when the user asks (suggest_conditions / conditions_from_names).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .batchset import SOURCE_COLUMN
from .batchfile import (COND_ID, ROLE_CONDITION, ROLE_SUB, BatchFile, Column, Sheet, blank, fmt, to_number)
from .conditions import parse_group_name
from .delivery import Delivery, parse_delivery_name, read_delivery_csv
from .io import content_digest, read_particle_csv

GROUP_COLUMN = "원본 그룹"


def read_particle_files(paths) -> tuple[pd.DataFrame, list[str]]:
    """Particle CSVs stacked: group, image, source_file + measurements. Repeated files are skipped."""
    frames, notes, seen = [], [], set()
    for p in paths:
        f = read_particle_csv(p)
        d = content_digest(f.drop(columns=["source_file"]))
        if d in seen:
            notes.append(f"{Path(p).name}: 같은 내용의 파일이 이미 있어 건너뛰었습니다.")
            continue
        seen.add(d)
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["group", "image", "source_file"]), notes
    return pd.concat(frames, ignore_index=True, sort=False), notes


def _sheet_digests(s: Sheet) -> set:
    if SOURCE_COLUMN not in s.names() or not len(s.data):
        return set()
    out = set()
    for _src, g in s.data.groupby(s.data[SOURCE_COLUMN].astype(str), sort=False):
        num = g.drop(columns=[c for c in (COND_ID, SOURCE_COLUMN, GROUP_COLUMN) if c in g]).apply(
            lambda c: pd.to_numeric(c.map(to_number), errors="coerce"))
        out.add(content_digest(num))
    return out


def add_particles(bf: BatchFile, frame: pd.DataFrame, mapping: dict) -> list[str]:
    """Put particle rows under formation conditions. mapping: CSV group -> 조건 ID (groups mapped to
    nothing are skipped). A condition ID not in the condition table gets a new row there.
    Returns notes."""
    s = bf.particles
    if s is None:
        raise ValueError("입자별 측정 시트가 없습니다.")
    notes = []
    have = {c for c in s.by_role("측정값")}
    new_meas = [c for c in frame.columns if c not in ("group", "image", "source_file")
                and pd.api.types.is_numeric_dtype(frame[c])]
    if len(have) >= 3 and new_meas and len(set(new_meas) & have) < 0.5 * len(have):
        raise ValueError("이 파일의 측정값 열이 시트의 열과 대부분 다릅니다 (다른 형식의 파일인지 확인하세요). "
                         f"시트: {', '.join(sorted(have)[:4])} … / 파일: {', '.join(new_meas[:4])} …")
    old = _sheet_digests(s)
    rows = []
    for src, f in frame.groupby("source_file", sort=False):
        num = f.drop(columns=["group", "image", "source_file"]).apply(pd.to_numeric, errors="coerce")
        if content_digest(num) in old:
            notes.append(f"{src}: 이 시트에 이미 있는 데이터라 건너뛰었습니다.")
            continue
        f = f[f["group"].map(lambda g: not blank(mapping.get(g)))]
        if not len(f):
            continue
        body = f.drop(columns=["group", "image", "source_file"]).copy()
        body.insert(0, COND_ID, f["group"].map(mapping).to_numpy())
        body.insert(1, "이미지", f["image"].to_numpy())
        body[GROUP_COLUMN] = f["group"].to_numpy()
        body[SOURCE_COLUMN] = src
        rows.append(body)
    if rows:
        _ensure_conditions(bf, {c for r in rows for c in r[COND_ID]})
        s.append_frame(pd.concat(rows, ignore_index=True, sort=False))
        for c in (GROUP_COLUMN, SOURCE_COLUMN):
            s.column(c).role = "메모"
        n = sum(len(r) for r in rows)
        notes.insert(0, f"입자 {n:,}개를 '{s.name}' 시트에 넣었습니다.")
    return notes


def read_delivery_files(paths) -> tuple[Delivery, list[str]]:
    d, notes = Delivery(), []
    for p in paths:
        notes += d.add(read_delivery_csv(p))
    return d, notes


def add_delivery(bf: BatchFile, sheet: str, delivery: Delivery, mapping: dict, sub_from_names=False) -> list[str]:
    """Cell delivery summary -> one row per delivery group: per channel '세포질 X', '입자 안 X',
    '세포질/입자 X'. mapping: delivery group -> 조건 ID of the particles used (blank = keep the row,
    ID left empty). With sub_from_names, amounts and times in the group names fill the sheet's
    sub-condition columns (blank cells only)."""
    from .project import delivery_wide
    s = bf.sheet(sheet)
    wide = delivery_wide(delivery)
    if wide.empty:
        return ["세포 전달 요약에서 강도 값을 찾지 못했습니다."]
    body = pd.DataFrame(index=wide.index)
    body[COND_ID] = [mapping.get(g, np.nan) if not blank(mapping.get(g)) else np.nan for g in wide.index]
    rep = s.by_role("반복")
    body[rep[0] if rep else GROUP_COLUMN] = list(wide.index)
    if sub_from_names:
        for g in wide.index:
            for name, (v, unit) in parse_delivery_name(g).items():
                if name not in s.names():
                    s.add_column(Column(name, unit, ROLE_SUB))
                body.loc[g, name] = v
    for c in wide.columns:
        body[c] = wide[c].to_numpy()
    s.append_frame(body.reset_index(drop=True))
    _ensure_conditions(bf, {c for c in body[COND_ID] if not blank(c)})
    bf.fill_ratios()
    missing = int(body[COND_ID].isna().sum())
    out = [f"세포 전달 {len(body)}행을 '{s.name}' 시트에 넣었습니다."]
    if missing:
        out.append(f"{missing}행은 어떤 형성 조건의 입자를 썼는지 정하지 않아 조건 ID가 비어 있습니다 (분석에서 빠짐).")
    return out


def _ensure_conditions(bf: BatchFile, ids):
    cs = bf.conditions
    have = set(bf.condition_ids())
    for i in sorted(i for i in ids if not blank(i) and str(i) not in have):
        cs.add_rows(1)
        cs.set_value(len(cs.data) - 1, COND_ID, str(i))


# ---------------------------------------------------------------- names -> conditions (on request)
def suggest_conditions(bf: BatchFile, groups) -> dict:
    """CSV group -> existing 조건 ID whose pH / NaCl / PEG values match the ones written in the
    group name (None when no single row matches)."""
    cs = bf.conditions
    out = {}
    for g in groups:
        parsed = parse_group_name(g)
        if not parsed or cs is None:
            out[g] = None
            continue
        hits = []
        for r in range(len(cs.data)):
            ok = True
            for name, v in parsed.items():
                if not cs.has(name):
                    ok = False
                    break
                cell = cs.data.at[r, name]
                if blank(cell) or to_number(cell) != to_number(v):
                    ok = False
                    break
            if ok and not blank(cs.data.at[r, COND_ID]):
                hits.append(fmt(cs.data.at[r, COND_ID]))
        out[g] = hits[0] if len(hits) == 1 else None
    return out


def conditions_from_names(bf: BatchFile, groups) -> dict:
    """New condition rows (C.., values from the names) for groups that match no row. Returns
    group -> 조건 ID for those groups."""
    cs = bf.conditions
    out = {}
    taken = set(bf.condition_ids())
    k = 1
    for g in groups:
        parsed = parse_group_name(g)
        if not parsed:
            continue
        while f"C{k:02d}" in taken:
            k += 1
        cid = f"C{k:02d}"
        taken.add(cid)
        cs.add_rows(1)
        r = len(cs.data) - 1
        cs.set_value(r, COND_ID, cid)
        for name, v in parsed.items():
            if not cs.has(name):
                unit = {"NaCl": "mM", "PEG": "%"}.get(name, "")
                cs.add_column(Column(name, unit, ROLE_CONDITION), pos=len(cs.columns))
            cs.set_value(r, name, v)
        if cs.has("메모"):
            cs.set_value(r, "메모", g)
        out[g] = cid
    return out
