"""Project file (.dmap): a zip holding the particle table, column roles, conditions and settings."""
from __future__ import annotations

import io
import os
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .io import ColumnRoles, combine_frames, content_digest, guess_roles, read_particle_csv
from .conditions import ConditionTable
from .analysis import Settings

FORMAT_VERSION = 1


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

    @property
    def groups(self) -> list[str]:
        return list(pd.unique(self.particles["group"]))

    @classmethod
    def from_csv(cls, paths) -> "Project":
        frames, notes = _read_unique(paths, set())
        df = combine_frames(frames)
        roles = guess_roles(df)
        cond = ConditionTable(pd.unique(df["group"]))
        pr = cls(df, roles, cond)
        pr.notes = notes + pr.harmonize_features()
        return pr

    def add_csv(self, paths) -> list[str]:
        """Append CSVs. A file whose measurements are already in the project (same content,
        whatever its name) is skipped. Returns notes for the user."""
        have = {content_digest(g.drop(columns=["source_file"])) for _s, g in self.particles.groupby("source_file", sort=False)} \
            if "source_file" in self.particles else set()
        frames, notes = _read_unique(paths, have)
        if frames:
            new = combine_frames(frames, self.particles)
            self.particles = pd.concat([self.particles, new], ignore_index=True, sort=False)
            known = {c.name for c in self.roles.columns}
            for spec in guess_roles(self.particles).columns:
                if spec.name not in known:
                    self.roles.columns.append(spec)
            self.conditions.set_groups(self.groups)
            notes += self.harmonize_features()
        return notes

    def harmonize_features(self) -> list[str]:
        """Measurements missing from some files would make every particle of those files fail QC.
        - A file that lacks most of the analysed measurements is a different format: the
          measurements stay on and that file's particles are left out of the analysis (reported).
        - Otherwise the missing measurement is switched to 'measure' for everyone (reported).
        Returns at most one message."""
        if "source_file" not in self.particles:
            return []
        src = self.particles["source_file"].astype(str)
        feats = self.roles.features()
        if not feats:
            return []
        missing = pd.DataFrame({f: self.particles[f].isna().groupby(src).all() for f in feats})
        # judge "different format" against the measurements most particles have, so the columns an
        # odd file brings along (e.g. ImageJ's Area, Circ.) do not make the normal files look odd
        size = src.value_counts().reindex(missing.index)
        share = (~missing).mul(size, axis=0).sum() / size.sum()
        core = [f for f in feats if share[f] >= 0.5] or feats
        odd = missing.index[missing[core].mean(axis=1) >= 0.5]
        normal = [x for x in missing.index if x not in odd]
        lines = []
        for x in odd:
            cols = [f for f in core if missing.at[x, f]]
            lines.append(f"• {x}: 표준 측정값 {len(cols)}개({', '.join(cols[:4])}{'…' if len(cols) > 4 else ''})가 없어 "
                         "이 파일의 입자는 분석에서 제외했습니다. 열 이름이 다른 형식의 파일인지 확인하세요.")
        only_odd = [f for f in feats if normal and missing.loc[normal, f].all()]
        for f in only_odd:
            self.roles.spec(f).role = "measure"
        if only_odd and len(odd):
            lines.append(f"• 위 파일에만 있는 열 {len(only_odd)}개({', '.join(only_odd[:4])}{'…' if len(only_odd) > 4 else ''})는 "
                         "분석에 넣지 않았습니다.")
        for f in feats:
            if f in only_odd:
                continue
            files = [x for x in normal if missing.at[x, f]]
            if files:
                self.roles.spec(f).role = "measure"
                lines.append(f"• '{f}' 측정값을 분석에서 뺐습니다 (없는 파일: {', '.join(files)})")
        if not lines:
            return []
        return ["파일마다 있는 측정값이 다릅니다.\n" + "\n".join(lines) +
                "\n\n데이터 탭의 '열 역할'에서 바꿀 수 있습니다."]

    def save(self, path):
        path = Path(path)
        buf = io.StringIO()
        self.particles.to_csv(buf, index=False)
        # write next to the target first so a failed save never leaves a broken project file
        tmp = path.with_name(path.name + ".saving")
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("particles.csv", buf.getvalue())
                z.writestr("roles.json", json.dumps(self.roles.to_dict(), ensure_ascii=False, indent=2))
                z.writestr("conditions.json", self.conditions.to_json())
                z.writestr("settings.json", json.dumps(self.settings.to_dict(), ensure_ascii=False, indent=2, default=_plain))
                z.writestr("ui.json", json.dumps(self.ui, ensure_ascii=False, indent=2, default=_plain))
                z.writestr("log.json", json.dumps(self.log, ensure_ascii=False, indent=1, default=_plain))
                z.writestr("meta.json", json.dumps({"format": FORMAT_VERSION, "app": "Droplet Map"}))
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        self.path = path

    @classmethod
    def load(cls, path) -> "Project":
        path = Path(path)
        with zipfile.ZipFile(path) as z:
            df = pd.read_csv(io.BytesIO(z.read("particles.csv")), low_memory=False,
                             dtype={"group": str, "image": str, "source_file": str})
            for c in ("group", "image"):
                df[c] = df[c].astype(str)
            roles = ColumnRoles.from_dict(json.loads(z.read("roles.json")))
            cond = ConditionTable.from_dict(json.loads(z.read("conditions.json")), pd.unique(df["group"]))
            settings = Settings.from_dict(json.loads(z.read("settings.json")))
            names = set(z.namelist())
            ui = json.loads(z.read("ui.json")) if "ui.json" in names else {}
            log = json.loads(z.read("log.json")) if "log.json" in names else []
        return cls(df, roles, cond, settings, path, ui, log)


def _plain(o):
    """json fallback for numpy scalars, sets and paths kept in UI state."""
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def _read_unique(paths, have: set):
    """Read files, skipping any whose measurements are already loaded (or repeated in this batch)."""
    frames, notes = [], []
    for p in paths:
        f = read_particle_csv(p)
        d = content_digest(f.drop(columns=["source_file"]))
        if d in have:
            notes.append(f"{Path(p).name}: 같은 내용의 데이터가 이미 있어 건너뛰었습니다 ({p}).")
            continue
        have.add(d)
        frames.append(f)
    return frames, notes
