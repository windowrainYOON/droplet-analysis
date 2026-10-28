"""Project file (.dmap): a zip holding the particle table, column roles, conditions and settings."""
from __future__ import annotations

import io
import os
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .io import ColumnRoles, guess_roles, load_particles
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
        df = load_particles(paths)
        roles = guess_roles(df)
        cond = ConditionTable(pd.unique(df["group"]))
        pr = cls(df, roles, cond)
        pr.notes = pr.harmonize_features()
        return pr

    def add_csv(self, paths) -> list[str]:
        """Append CSVs. Files already in the project (same file name) are skipped.
        Returns notes for the user (skipped files, measurements turned off)."""
        loaded = set(self.particles["source_file"].astype(str)) if "source_file" in self.particles else set()
        notes = [f"{Path(p).name}: 이미 불러온 파일이라 건너뛰었습니다." for p in paths if Path(p).name in loaded]
        paths = [p for p in paths if Path(p).name not in loaded]
        if paths:
            new = load_particles(paths, self.particles)
            self.particles = pd.concat([self.particles, new], ignore_index=True, sort=False)
            known = {c.name for c in self.roles.columns}
            for spec in guess_roles(self.particles).columns:
                if spec.name not in known:
                    self.roles.columns.append(spec)
            self.conditions.set_groups(self.groups)
            notes += self.harmonize_features()
        return notes

    def harmonize_features(self) -> list[str]:
        """A measurement that some files do not have would make every particle of those files
        fail QC. Such measurements are switched from analysis to 'measure' and reported."""
        notes = []
        if "source_file" not in self.particles:
            return notes
        src = self.particles["source_file"].astype(str)
        for f in self.roles.features():
            missing = self.particles[f].isna().groupby(src).all()
            files = sorted(missing[missing].index)
            if files:
                self.roles.spec(f).role = "measure"
                notes.append(f"'{f}' 값이 없는 파일이 있어 분석에서 뺐습니다 ({', '.join(files)}). "
                             "데이터 탭에서 다시 켜면 그 파일의 입자는 모두 제외됩니다.")
        return notes

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
