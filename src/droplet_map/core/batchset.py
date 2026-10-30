"""A folder of batch files read as one analysis set.

Every batch file is one purification batch. Its formation conditions become particle formation
groups named '<batch> · <condition id>'; its per-particle sheet becomes the particle table; every
per-condition sheet becomes a measurement set whose rows are summarised per condition (and per
sub-condition such as time) for the batch map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .batches import Batches
from .batchfile import (CAT_PURIFICATION, COND_ID, KIND_MEASURE, KIND_PARTICLES, ROLE_CONDITION, ROLE_REPLICATE, ROLE_SUB, ROLE_VALUE, BatchFile,
                        batch_paths, blank, fmt, read_batch_file, to_number)
from .conditions import ConditionDef, ConditionTable

SOURCE_COLUMN = "원본 파일"     # importing.py writes it: the CSV a particle row came from

SEP = " · "
NO_IMAGE = "이미지 없음"


def group_name(batch: str, cid: str) -> str:
    return f"{batch}{SEP}{cid}"


@dataclass
class MeasureSet:
    """One per-condition sheet over all batches."""
    name: str
    category: str
    rows: pd.DataFrame          # raw rows: batch, 조건 ID, 그룹, sub-conditions, measurements
    measures: pd.DataFrame      # summary group x measurement (median over replicates)
    cond: ConditionTable        # summary group -> formation conditions + sub-conditions
    batch_of: dict              # summary group -> batch
    formation_of: dict          # summary group -> formation group
    units: dict                 # measurement -> unit
    values: list                # measurement columns
    subs: list                  # sub-condition columns


@dataclass
class BatchSet:
    folder: Path | None = None
    files: dict = field(default_factory=dict)    # batch name -> BatchFile
    notes: list = field(default_factory=list)

    def names(self) -> list[str]:
        return list(self.files)

    def file_of(self, batch: str) -> BatchFile:
        return self.files[batch]


_CACHE: dict = {}       # path -> (mtime, size, BatchFile, notes); a re-read only parses changed files


def load_folder(folder) -> BatchSet:
    folder = Path(folder)
    bs = BatchSet(folder)
    seen = set()
    for p in batch_paths(folder):
        st = p.stat()
        hit = _CACHE.get(str(p))
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            bf, notes = hit[2], hit[3]
        else:
            try:
                bf, notes = read_batch_file(p)
            except Exception as e:      # a file Excel is still writing, or not a batch file
                bs.notes.append(f"{p.name}: 읽지 못했습니다 ({e}).")
                continue
            _CACHE[str(p)] = (st.st_mtime, st.st_size, bf, notes)
        name = bf.name
        if name in seen:
            bs.notes.append(f"{p.name}: 배치 이름 '{name}'이(가) 다른 파일과 같아 파일 이름을 배치 이름으로 씁니다.")
            name = p.stem
        seen.add(name)
        bs.files[name] = bf
        bs.notes += [f"{p.name}: {n}" for n in notes]
    return bs


def forget_cache(path=None):
    if path is None:
        _CACHE.clear()
    else:
        _CACHE.pop(str(path), None)


# ------------------------------------------------------------------ building the analysis tables
def _numeric_frame(df: pd.DataFrame, keep_text=()) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in keep_text:
            out[c] = out[c].map(lambda v: np.nan if blank(v) else str(v))
            continue
        num = pd.to_numeric(out[c].map(to_number), errors="coerce")
        filled = out[c].map(lambda v: not blank(v))
        if filled.sum() and num.notna().sum() >= 0.9 * filled.sum():
            out[c] = num
    return out


def batches_table(bs: BatchSet) -> Batches:
    """Batch table from the batch-value sheets: one column per item, its category from 구분."""
    b = Batches()
    b.categories = []
    for name in bs.names():
        b.add(name)
    kinds, units, cats, order = {}, {}, {}, []
    vals: dict = {}
    for name, bf in bs.files.items():
        for it in bf.items():
            n = it["name"]
            if n not in order:
                order.append(n)
                units[n] = it["unit"]
                cats[n] = it["category"]
            if it["value"] is not None:
                vals.setdefault(n, {})[name] = it["value"]
    for n in order:
        v = [x for x in vals.get(n, {}).values()]
        kind = "numeric" if v and all(to_number(x) is not None for x in v) else "categorical"
        b.table.add_condition(ConditionDef(n, kind, units.get(n, "")))
        b.set_category(n, cats[n])
        for bn, x in vals.get(n, {}).items():
            b.table.set_value(bn, n, fmt(x) if kind == "categorical" else x)
    for bf in bs.files.values():                # sheet categories follow, in sheet order
        for s in bf.sheets:
            if s.kind in (KIND_MEASURE, KIND_PARTICLES) and s.category not in b.categories:
                b.categories.append(s.category)
    if not b.categories:
        b.categories = [CAT_PURIFICATION]
    return b


def formation_tables(bs: BatchSet):
    """(particles frame, formation ConditionTable, group -> batch, notes)."""
    notes = []
    schema: list[ConditionDef] = []
    rows: dict = {}
    of = {}
    frames = []
    for bname, bf in bs.files.items():
        cs = bf.conditions
        ids = set()
        if cs is not None and cs.has(COND_ID):
            cid = cs.text(COND_ID)
            for col in cs.columns:
                if col.role != ROLE_CONDITION:
                    continue
                if col.name not in [c.name for c in schema]:
                    schema.append(ConditionDef(col.name, "numeric", col.unit))
            for r, i in cid.items():
                if not isinstance(i, str):
                    continue
                g = group_name(bname, i)
                ids.add(i)
                of[g] = bname
                rows[g] = {col.name: cs.data.at[r, col.name] for col in cs.columns if col.role == ROLE_CONDITION}
        ps = bf.particles
        if ps is None or not len(ps.data) or not ps.has(COND_ID):
            continue
        d = ps.data
        cid = d[COND_ID].map(lambda v: np.nan if blank(v) else fmt(v))
        ok = cid.isin(ids)
        if (~ok).any():
            notes.append(f"{bname}: '{ps.name}' 시트의 {int((~ok).sum())}행은 조건표에 없는 조건 ID라 뺐습니다.")
        d = d[ok].copy()
        rep = ps.by_role(ROLE_REPLICATE)
        image = d[rep[0]].map(lambda v: NO_IMAGE if blank(v) else str(v)) if rep else pd.Series(NO_IMAGE, index=d.index)
        d = d.drop(columns=[COND_ID] + rep[:1])
        d.insert(0, "group", [group_name(bname, i) for i in cid[ok]])
        d.insert(1, "image", image.to_numpy())
        here = bf.path.name if bf.path else bname
        if SOURCE_COLUMN in d:          # the image-analysis file each particle came from
            d["source_file"] = d[SOURCE_COLUMN].map(lambda v: here if blank(v) else f"{here}: {v}")
        else:
            d["source_file"] = here
        frames.append(d)
    used = {k for r in rows.values() for k, v in r.items() if not blank(v)}
    schema = [c for c in schema if c.name in used]          # a column nobody filled in says nothing
    cond = ConditionTable(list(rows), schema)
    for g, r in rows.items():
        for k, v in r.items():
            if not blank(v) and k in used:
                cond.set_value(g, k, fmt(v))
    for c in cond.schema:                       # a condition with any text value is categorical
        vals = [v for v in cond.values[c.name] if not blank(v)]
        if vals and not all(to_number(v) is not None for v in vals):
            c.kind = "categorical"
    if frames:
        parts = pd.concat(frames, ignore_index=True, sort=False)
        parts = _numeric_frame(parts, keep_text=("group", "image", "source_file"))
    else:
        parts = pd.DataFrame(columns=["group", "image", "source_file"])
    return parts, cond, of, notes


def measure_sets(bs: BatchSet, formation: ConditionTable) -> tuple[list[MeasureSet], list[str]]:
    notes = []
    names, cat_of = [], {}
    for bf in bs.files.values():
        for s in bf.measure_sheets():
            if s.name not in names:
                names.append(s.name)
                cat_of[s.name] = s.category
    out = []
    for sname in names:
        frames, units, values, subs = [], {}, [], []
        for bname, bf in bs.files.items():
            try:
                s = bf.sheet(sname)
            except KeyError:
                continue
            if s.kind != KIND_MEASURE or not s.has(COND_ID):
                continue
            ids = set(bf.condition_ids())
            d = s.data.copy()
            cid = d[COND_ID].map(lambda v: np.nan if blank(v) else fmt(v))
            ok = cid.isin(ids)
            n_bad = int((~ok & d.apply(lambda r: any(not blank(x) for x in r), axis=1)).sum())
            if n_bad:
                notes.append(f"{bname}: '{sname}' 시트의 {n_bad}행은 조건 ID가 비었거나 조건표에 없어 뺐습니다.")
            d = d[ok]
            if not len(d):
                continue
            for c in s.columns:
                if c.role == ROLE_VALUE and c.name not in values:
                    values.append(c.name)
                    units[c.name] = c.unit
                if c.role == ROLE_SUB and c.name not in subs:
                    subs.append(c.name)
                    units[c.name] = c.unit
            keep = [COND_ID] + [c.name for c in s.columns if c.role in (ROLE_SUB, ROLE_VALUE, ROLE_REPLICATE)]
            d = d[keep].copy()
            d[COND_ID] = cid[ok]
            d.insert(0, "배치", bname)
            frames.append(d)
        if not frames or not values:
            continue
        rows = pd.concat(frames, ignore_index=True, sort=False)
        for c in values + subs:
            if c in rows:
                rows[c] = pd.to_numeric(rows[c].map(to_number), errors="coerce")
        rows.insert(2, "그룹", [_summary_group(r, subs, units) for _, r in rows.iterrows()])
        meas = rows.groupby("그룹", sort=False)[[v for v in values if v in rows]].median()
        first = rows.drop_duplicates("그룹").set_index("그룹")
        schema = [ConditionDef(c.name, c.kind, c.unit) for c in formation.schema]
        schema += [ConditionDef(s_, "numeric", units.get(s_, "")) for s_ in subs if s_ not in [c.name for c in schema]]
        cond = ConditionTable(list(meas.index), schema)
        batch_of, formation_of = {}, {}
        for g in meas.index:
            b, cid = first.at[g, "배치"], first.at[g, COND_ID]
            fg = group_name(b, cid)
            batch_of[g], formation_of[g] = b, fg
            for c in formation.schema:
                v = formation.value(fg, c.name) if fg in formation.values.index else None
                if v is not None:
                    cond.set_value(g, c.name, v)
            for s_ in subs:
                v = first.at[g, s_] if s_ in first.columns else np.nan
                if pd.notna(v):
                    cond.set_value(g, s_, fmt(v))
        out.append(MeasureSet(sname, cat_of[sname], rows, meas, cond, batch_of, formation_of, units,
                              [v for v in values if v in meas.columns], subs))
    return out, notes


def _summary_group(r, subs, units) -> str:
    g = group_name(r["배치"], r[COND_ID])
    for s in subs:
        v = r.get(s)
        if pd.notna(v):
            u = units.get(s, "")
            g += f"{SEP}{s} {fmt(v)}{(' ' + u) if u else ''}"
    return g
