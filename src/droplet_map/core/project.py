"""Project file (.dmap): a zip holding the particle table, column roles, conditions and settings."""
from __future__ import annotations

import io
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

    @property
    def groups(self) -> list[str]:
        return list(pd.unique(self.particles["group"]))

    @classmethod
    def from_csv(cls, paths) -> "Project":
        df = load_particles(paths)
        roles = guess_roles(df)
        cond = ConditionTable(pd.unique(df["group"]))
        return cls(df, roles, cond)

    def add_csv(self, paths):
        new = load_particles(paths)
        self.particles = pd.concat([self.particles, new], ignore_index=True, sort=False)
        known = {c.name for c in self.roles.columns}
        for spec in guess_roles(self.particles).columns:
            if spec.name not in known:
                self.roles.columns.append(spec)
        self.conditions.set_groups(self.groups)

    def save(self, path):
        path = Path(path)
        buf = io.StringIO()
        self.particles.to_csv(buf, index=False)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("particles.csv", buf.getvalue())
            z.writestr("roles.json", json.dumps(self.roles.to_dict(), ensure_ascii=False, indent=2))
            z.writestr("conditions.json", self.conditions.to_json())
            z.writestr("settings.json", json.dumps(self.settings.to_dict(), ensure_ascii=False, indent=2))
            z.writestr("ui.json", json.dumps(self.ui, ensure_ascii=False, indent=2))
            z.writestr("log.json", json.dumps(self.log, ensure_ascii=False, indent=1))
            z.writestr("meta.json", json.dumps({"format": FORMAT_VERSION, "app": "Droplet Map"}))
        self.path = path

    @classmethod
    def load(cls, path) -> "Project":
        path = Path(path)
        with zipfile.ZipFile(path) as z:
            df = pd.read_csv(io.BytesIO(z.read("particles.csv")), low_memory=False)
            for c in ("group", "image"):
                df[c] = df[c].astype(str)
            roles = ColumnRoles.from_dict(json.loads(z.read("roles.json")))
            cond = ConditionTable.from_dict(json.loads(z.read("conditions.json")), pd.unique(df["group"]))
            settings = Settings.from_dict(json.loads(z.read("settings.json")))
            names = set(z.namelist())
            ui = json.loads(z.read("ui.json")) if "ui.json" in names else {}
            log = json.loads(z.read("log.json")) if "log.json" in names else []
        return cls(df, roles, cond, settings, path, ui, log)
