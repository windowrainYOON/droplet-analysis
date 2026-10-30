"""Analysis project.

The data live in a folder of batch files (one Excel workbook per purification batch, see
batchfile.py); Droplet Map only reads them. The project file (.dmap) keeps what belongs to the
analysis: the folder it reads, column roles, settings, references, UI state and the work log.

Projects from before batch files (format 1-3, data inside the .dmap) open by converting their
data to batch files in memory; the user then writes them to a folder.
"""
from __future__ import annotations

import io
import os
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .io import ColumnRoles, guess_roles
from .conditions import ConditionTable, ConditionDef
from .analysis import Settings, group_profiles, prepare
from .batch_map import Descriptors, Source, batch_descriptors
from .batches import UNASSIGNED, Batches, batches_from_conditions, merged_conditions
from .batchfile import (COND_ID, INSIDE_PREFIX, CYTO_PREFIX, ITEM_NAME, KIND_PARTICLES, RATIO_PREFIX, ROLE_CONDITION,
                        ROLE_NOTE, ROLE_REPLICATE, ROLE_SUB, ROLE_VALUE, SHEET_DELIVERY, TEMPLATE_NAME, BatchFile, Column,
                        safe_file_name, split_header, template, write_batch_file)
from .batchset import BatchSet, MeasureSet, batches_table, formation_tables, load_folder, measure_sets
from .delivery import Delivery, region_label

FORMAT_VERSION = 4


@dataclass
class Project:
    particles: pd.DataFrame
    roles: ColumnRoles
    conditions: ConditionTable
    settings: Settings = field(default_factory=Settings)
    path: Path | None = None
    ui: dict = field(default_factory=dict)      # open tabs, view choices, widget values
    log: list = field(default_factory=list)     # work log rows: [time, message]
    notes: list = field(default_factory=list)   # messages for the user after loading
    batches: Batches = field(default_factory=Batches)
    delivery: Delivery = field(default_factory=Delivery)    # legacy only; batch files use measure sets
    measures: list = field(default_factory=list)            # MeasureSet per per-condition sheet
    source: BatchSet = field(default_factory=BatchSet)

    def __post_init__(self):
        self.batches.sync_groups(self.groups)

    # ---- data -------------------------------------------------------------
    @property
    def folder(self) -> Path | None:
        return self.source.folder

    @property
    def groups(self) -> list[str]:
        """Particle formation groups ('<batch> · <condition id>'), whether or not they have particles."""
        return list(self.conditions.values.index)

    def particle_groups(self) -> list[str]:
        return list(pd.unique(self.particles["group"])) if len(self.particles) else []

    def all_groups(self) -> list[str]:
        return self.groups

    @property
    def view(self) -> ConditionTable:
        """Formation groups' conditions with their batch and the batch's values (read-only)."""
        return merged_conditions(self.groups, self.conditions, self.batches)

    @property
    def delivery_view(self) -> ConditionTable:
        return merged_conditions(self.delivery.groups(), self.delivery.conditions, self.batches)

    def batch_of(self, group) -> str:
        return self.batches.batch_of(group)

    def particle_sheet(self) -> tuple[str, str]:
        """(sheet name, category) of the per-particle sheet."""
        for bf in self.source.files.values():
            if bf.particles is not None:
                return bf.particles.name, bf.particles.category
        return "입자 형태", "입자 형태"

    def measure_set(self, name) -> MeasureSet | None:
        return next((m for m in self.measures if m.name == name), None)

    # ---- batch description --------------------------------------------------
    def particle_measures(self, prepared=None) -> tuple[pd.DataFrame, dict]:
        """Formation group x measurement (median robust z; shape values size-weighted) plus
        particles per image, with units."""
        p = prepared if prepared is not None else prepare(self.particles, self.roles, self.settings)
        prof = group_profiles(p)
        m = prof.median_z.copy()
        units = {c: "z" for c in m.columns}
        col = "이미지당 입자 수 (평균)"
        if col in prof.image_stats:
            m["이미지당 입자 수"] = prof.image_stats[col]
            units["이미지당 입자 수"] = "개"
        return m, units

    def sources(self, prepared=None) -> list[Source]:
        out = []
        if len(self.particles) and self.roles.features():
            meas, units = self.particle_measures(prepared)
            name, cat = self.particle_sheet()
            out.append(Source(name, cat, meas, self.conditions, {g: self.batch_of(g) for g in meas.index},
                              units, particle=True))
        for m in self.measures:
            out.append(Source(m.name, m.category, m.measures, m.cond, m.batch_of, m.units))
        return out

    def batch_descriptors(self, prepared=None) -> Descriptors:
        return batch_descriptors(self.batches, self.sources(prepared), kinds=self.settings.bm_kinds)

    # ---- building from batch files -------------------------------------------
    @classmethod
    def from_batchset(cls, bs: BatchSet, roles: ColumnRoles | None = None, settings: Settings | None = None,
                      references=()) -> "Project":
        parts, cond, of, notes = formation_tables(bs)
        batches = batches_table(bs)
        for g, b in of.items():
            batches.of[g] = b
        ms, n2 = measure_sets(bs, cond)
        pr = cls(parts, _merge_roles(guess_roles(parts), roles), cond, settings or Settings(), batches=batches,
                 measures=ms, source=bs)
        pr.batches.references = {r for r in references if r in batches.names()}
        pr.notes = list(bs.notes) + notes + n2 + pr.harmonize_features()
        return pr

    @classmethod
    def open_folder(cls, folder) -> "Project":
        return cls.from_batchset(load_folder(folder))

    def reload(self) -> list[str]:
        """Read the folder again (only changed files are parsed). Settings, roles, references and
        UI state stay. Returns notes."""
        if self.folder is None:
            return []
        new = Project.from_batchset(load_folder(self.folder), self.roles, self.settings, self.batches.references)
        for k in ("particles", "roles", "conditions", "batches", "measures", "source"):
            setattr(self, k, getattr(new, k))
        self.delivery = Delivery()
        return new.notes

    def export_batch_files(self, folder) -> list[Path]:
        """Write every batch (and a template) as batch files into `folder`, then read from there."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out, used = [], set()
        for name, bf in self.source.files.items():
            fname = safe_file_name(name)
            while fname.lower() in used:
                fname += "_2"
            used.add(fname.lower())
            out.append(write_batch_file(bf, folder / (fname + ".xlsx")))
        tpl = folder / TEMPLATE_NAME
        if not tpl.exists() and self.source.files:
            write_batch_file(next(iter(self.source.files.values())).layout_only(), tpl)
        self.source.folder = folder
        return out

    def harmonize_features(self) -> list[str]:
        """Measurements missing from some batches' particle sheets would make every particle of
        those batches fail QC: such a measurement is switched to 'measure' (reported)."""
        if "source_file" not in self.particles or not len(self.particles):
            return []
        src = self.particles["source_file"].astype(str)
        feats = self.roles.features()
        if not feats or src.nunique() < 2:
            return []
        missing = pd.DataFrame({f: self.particles[f].isna().groupby(src).all() for f in feats})
        lines = []
        for f in feats:
            files = list(missing.index[missing[f]])
            if files:
                self.roles.spec(f).role = "measure"
                lines.append(f"• '{f}' 측정값을 분석에서 뺐습니다 (없는 파일: {', '.join(files)})")
        if not lines:
            return []
        return ["배치마다 입자 측정값이 다릅니다.\n" + "\n".join(lines) + "\n\n데이터 탭의 '열 역할'에서 바꿀 수 있습니다."]

    # ---- project file ---------------------------------------------------------
    def save(self, path):
        if self.folder is None:
            raise ValueError("배치 파일이 아직 폴더에 없습니다. 먼저 '배치 파일로 꺼내기'로 폴더에 쓰세요.")
        path = Path(path)
        try:
            rel = os.path.relpath(self.folder, path.parent)
        except ValueError:
            rel = None
        tmp = path.with_name(path.name + ".saving")
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("source.json", json.dumps({"folder": str(self.folder), "relative": rel,
                                                      "references": sorted(self.batches.references),
                                                      "group_references": sorted(self.conditions.references)},
                                                     ensure_ascii=False, indent=2))
                z.writestr("roles.json", json.dumps(self.roles.to_dict(), ensure_ascii=False, indent=2))
                z.writestr("settings.json", json.dumps(self.settings.to_dict(), ensure_ascii=False, indent=2, default=_plain))
                z.writestr("ui.json", json.dumps(self.ui, ensure_ascii=False, indent=2, default=_plain))
                z.writestr("log.json", json.dumps(self.log, ensure_ascii=False, indent=1, default=_plain))
                z.writestr("meta.json", json.dumps({"format": FORMAT_VERSION, "app": "Droplet Map"}))
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        self.path = path

    @staticmethod
    def linked_folder(path) -> Path | None:
        """The batch folder a format-4 project reads (None for older projects)."""
        with zipfile.ZipFile(path) as z:
            if "source.json" not in z.namelist():
                return None
            src = json.loads(z.read("source.json"))
        rel = src.get("relative")
        cands = [Path(path).parent / rel] if rel else []
        cands.append(Path(src["folder"]))
        return next((c for c in cands if c.is_dir()), cands[-1])

    @classmethod
    def load(cls, path, folder=None) -> "Project":
        """Open a project file. `folder` overrides the linked batch folder (when it moved)."""
        path = Path(path)
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if "source.json" not in names:
                return _load_legacy(z, path)
            src = json.loads(z.read("source.json"))
            roles = ColumnRoles.from_dict(json.loads(z.read("roles.json")))
            settings = Settings.from_dict(json.loads(z.read("settings.json")))
            ui = json.loads(z.read("ui.json")) if "ui.json" in names else {}
            log = json.loads(z.read("log.json")) if "log.json" in names else []
        folder = Path(folder) if folder else cls.linked_folder(path)
        if not folder.is_dir():
            raise FileNotFoundError(str(folder))
        pr = cls.from_batchset(load_folder(folder), roles, settings, src.get("references", []))
        pr.conditions.references = {g for g in src.get("group_references", []) if g in pr.groups}
        pr.path, pr.ui, pr.log = path, ui, log
        return pr


def _merge_roles(guessed: ColumnRoles, saved: ColumnRoles | None) -> ColumnRoles:
    """Keep the user's role choices for columns that are still there; guess the new ones."""
    if saved is None:
        return guessed
    have = {c.name: c for c in saved.columns}
    cols = [have.get(c.name, c) for c in guessed.columns]
    size = saved.size_column if saved.size_column and (saved.size_column in have or saved.size_column.startswith("__")) \
        else guessed.size_column
    return ColumnRoles(cols, size)


def _plain(o):
    """json fallback for numpy scalars, sets and paths kept in UI state."""
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


# ------------------------------------------------------------------ projects before batch files
def _load_legacy(z: zipfile.ZipFile, path: Path) -> Project:
    names = set(z.namelist())
    df = pd.read_csv(io.BytesIO(z.read("particles.csv")), low_memory=False, float_precision="round_trip",
                     dtype={"group": str, "image": str, "source_file": str})
    for c in ("group", "image"):
        df[c] = df[c].astype(str)
    roles = ColumnRoles.from_dict(json.loads(z.read("roles.json")))
    cond = ConditionTable.from_dict(json.loads(z.read("conditions.json")), pd.unique(df["group"]))
    settings = Settings.from_dict(json.loads(z.read("settings.json")))
    ui = json.loads(z.read("ui.json")) if "ui.json" in names else {}
    log = json.loads(z.read("log.json")) if "log.json" in names else []
    delivery = Delivery.from_parts(z.read("delivery.csv").decode("utf-8") if "delivery.csv" in names else None,
                                   json.loads(z.read("delivery.json")) if "delivery.json" in names else None)
    if "batches.json" in names:
        batches = Batches.from_dict(json.loads(z.read("batches.json")))
    else:                       # format 1: the batch was a group condition
        batches, _moved = batches_from_conditions(cond)
    bs = batchset_from_legacy(df, cond, batches, delivery)
    pr = Project.from_batchset(bs, roles, settings, batches.references)
    pr.ui, pr.log = ui, log
    pr.notes.insert(0, "이전 형식의 프로젝트입니다. 데이터를 배치 파일(배치마다 엑셀 파일 하나)로 바꿔 열었습니다. "
                       "'파일 › 배치 파일로 꺼내기'로 폴더에 써 두면, 그다음부터는 그 폴더를 엽니다.")
    return pr


def batchset_from_legacy(particles: pd.DataFrame, cond: ConditionTable, batches: Batches,
                         delivery: Delivery | None = None) -> BatchSet:
    """Old project data -> one batch file per batch. Formation groups become conditions C01, C02, ...
    (the old group name goes to 메모); cell delivery groups become rows of the 세포 전달 sheet, with
    the formation condition they used when that link was recorded."""
    bs = BatchSet()
    real = [b for b in batches.names() if b != UNASSIGNED]
    groups_all = list(cond.values.index)
    dgroups = delivery.groups() if delivery is not None and not delivery.empty() else []
    if any(batches.batch_of(g) == UNASSIGNED for g in groups_all + dgroups):
        real.append(UNASSIGNED)
    wide = delivery_wide(delivery) if dgroups else None
    for b in real:
        bf = template()
        bf.set_item(ITEM_NAME, b)
        # batch values
        if b in batches.table.values.index:
            for c in batches.table.schema:
                v = batches.table.value(b, c.name)
                if v is not None:
                    bf.set_item(c.name, v, c.unit, batches.category(c.name))
                    _set_unit(bf, c.name, c.unit, batches.category(c.name))
        # conditions
        cs = bf.conditions
        for c in list(cs.columns):
            if c.role == ROLE_CONDITION:
                cs.remove_column(c.name)
        names = {}
        for c in cond.schema:           # old names often carried the unit ('NaCl (mM)')
            n, u = split_header(c.label)
            names[c.name] = n
            cs.add_column(Column(n, u, ROLE_CONDITION), pos=len(cs.columns) - 1)
        groups = [g for g in groups_all if batches.batch_of(g) == b]
        cid = {}
        for i, g in enumerate(groups):
            cid[g] = f"C{i + 1:02d}"
            cs.add_rows(1)
            r = len(cs.data) - 1
            cs.set_value(r, COND_ID, cid[g])
            cs.set_value(r, "메모", g)
            for c in cond.schema:
                v = cond.value(g, c.name)
                if v is not None:
                    cs.set_value(r, names[c.name], v)
        # particles
        d = particles[particles["group"].isin(groups)]
        if len(d):
            ps = bf.particles
            body = d.drop(columns=["group", "image"]).copy()
            body.insert(0, COND_ID, d["group"].map(cid).to_numpy())
            body.insert(1, "이미지", d["image"].to_numpy())
            ps.append_frame(body.rename(columns={"source_file": "원본 파일"}))
        # delivery
        if wide is not None:
            rows = wide[[batches.batch_of(g) == b for g in wide.index]]
            if len(rows):
                ds = bf.sheet(SHEET_DELIVERY)
                for c in list(ds.columns):
                    if c.name not in (COND_ID, "반복", "이미지 수"):
                        ds.remove_column(c.name)
                subs = [c for c in delivery.conditions.schema]
                for c in subs:
                    ds.add_column(Column(c.name, c.unit, ROLE_SUB))
                body = pd.DataFrame(index=rows.index)
                used = {g: delivery.uses.get(g) for g in rows.index}
                body[COND_ID] = [cid.get(used[g]) if used[g] else np.nan for g in rows.index]
                body["반복"] = list(rows.index)          # the old delivery group name
                for c in subs:
                    body[c.name] = [delivery.conditions.value(g, c.name) for g in rows.index]
                for col in rows.columns:
                    body[col] = rows[col].to_numpy()
                ds.append_frame(body.reset_index(drop=True))
                if body[COND_ID].isna().any():
                    bs.notes.append(f"{b}: 세포 전달 행에 어떤 형성 조건의 입자를 썼는지 기록이 없어 조건 ID가 비어 있습니다. "
                                    "배치 노트의 '세포 전달' 시트에서 조건 ID를 넣으면 분석에 들어갑니다.")
        bf.fill_ratios()
        bs.files[b] = bf
    return bs


def _set_unit(bf: BatchFile, item, unit, cat):
    s = bf.batch_sheet
    rows = s.data.index[s.data["항목"].astype(str) == item]
    if len(rows):
        s.set_value(int(rows[0]), "단위", unit or np.nan)
        s.set_value(int(rows[0]), "구분", cat)


def delivery_wide(delivery: Delivery) -> pd.DataFrame:
    """Delivery summary (group x region x parameter) -> one row per group: '세포질 CH2_Mean',
    '입자 안 CH2_Mean', '세포질/입자 CH2_Mean', '이미지 수'."""
    m = delivery.value("mean")
    if m.empty:
        return pd.DataFrame()
    chans = delivery.channels()
    out = pd.DataFrame(index=m.index)
    if "n_images" in delivery.summary:
        out["이미지 수"] = delivery.summary.groupby("group")["n_images"].max().reindex(m.index)
    regions = delivery.regions()
    inside = "particle" if "particle" in regions else None
    cyto = next((r for r in regions if "cyto" in r), None)
    for ch in chans:
        if cyto and (cyto, ch) in m.columns:
            out[CYTO_PREFIX + ch] = m[(cyto, ch)]
        if inside and (inside, ch) in m.columns:
            out[INSIDE_PREFIX + ch] = m[(inside, ch)]
        if cyto and inside and (cyto, ch) in m.columns and (inside, ch) in m.columns:
            out[RATIO_PREFIX + ch] = m[(cyto, ch)] / m[(inside, ch)].replace(0, np.nan)
        for r in regions:               # other regions keep their own label
            if r not in (cyto, inside) and (r, ch) in m.columns:
                out[f"{region_label(r)} {ch}"] = m[(r, ch)]
    return out
