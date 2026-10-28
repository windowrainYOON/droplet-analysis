"""Cell delivery results: per-group intensity summaries (inside particles / cytoplasm).

Input is the long table written by the image analysis, one row per group x region x parameter:
group, region, n_images, parameter, n, mean, sd, min, max. Delivery groups have their own
conditions (dose, time, ...), belong to a batch like every group, and may name the particle
formation group whose particles they used.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .conditions import ConditionDef, ConditionTable

REQUIRED = ("group", "region", "parameter", "mean")
STATS = ("n_images", "n", "mean", "sd", "min", "max")
REGION_LABELS = {"particle": "입자 안", "cyto - particle": "세포질 (입자 제외)", "cyto": "세포질"}
META_PARAMETERS = ("particle_threshold",)    # analysis settings, not results
RATIO_LABEL = "세포질/입자"


def region_label(r: str) -> str:
    return REGION_LABELS.get(str(r).strip(), str(r))


def is_delivery_table(df: pd.DataFrame) -> bool:
    cols = {str(c).strip().lower() for c in df.columns}
    return all(c in cols for c in REQUIRED)


def read_delivery_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"group": str, "region": str, "parameter": str})
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    df = df.rename(columns={low[c]: c for c in list(REQUIRED) + list(STATS) if c in low})
    if not is_delivery_table(df):
        raise ValueError(f"{Path(path).name}: group, region, parameter, mean 열이 있는 세포 전달 요약 파일이 아닙니다.")
    for c in STATS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("group", "region", "parameter"):
        df[c] = df[c].astype(str).str.strip()
    df["source_file"] = Path(path).name
    return df


class Delivery:
    def __init__(self, summary: pd.DataFrame | None = None, conditions: ConditionTable | None = None,
                 uses: dict | None = None):
        self.summary = summary if summary is not None else pd.DataFrame(columns=list(REQUIRED) + ["source_file"])
        self.conditions = conditions if conditions is not None else ConditionTable(self.groups())
        self.uses: dict[str, str] = dict(uses or {})     # delivery group -> particle formation group

    def groups(self) -> list[str]:
        return list(pd.unique(self.summary["group"])) if len(self.summary) else []

    def empty(self) -> bool:
        return len(self.summary) == 0

    def add(self, df: pd.DataFrame) -> list[str]:
        """Append a summary table. Groups already present are replaced by the new rows. Returns notes."""
        notes = []
        if not self.empty():
            old = set(self.groups())
            again = [g for g in pd.unique(df["group"]) if g in old]
            if again:
                notes.append(f"세포 전달 그룹 {len(again)}개는 새 파일의 값으로 바꿨습니다: " + ", ".join(again[:5])
                             + ("…" if len(again) > 5 else ""))
                self.summary = self.summary[~self.summary["group"].isin(again)]
        self.summary = pd.concat([self.summary, df], ignore_index=True, sort=False)
        self.conditions.set_groups(self.groups())
        self.uses = {g: u for g, u in self.uses.items() if g in set(self.groups())}
        return notes

    def remove_groups(self, groups):
        groups = set(groups)
        self.summary = self.summary[~self.summary["group"].isin(groups)].reset_index(drop=True)
        self.conditions.set_groups(self.groups())
        self.uses = {g: u for g, u in self.uses.items() if g not in groups}

    def regions(self) -> list[str]:
        return list(pd.unique(self.summary["region"])) if len(self.summary) else []

    def parameters(self) -> list[str]:
        if self.empty():
            return []
        return [p for p in pd.unique(self.summary["parameter"]) if p not in META_PARAMETERS]

    def channels(self) -> list[str]:
        """Intensity parameters (CH1_Mean, ...)."""
        return [p for p in self.parameters() if re.search(r"(mean|intensity|int)", p, re.I)]

    def value(self, stat="mean") -> pd.DataFrame:
        """group x (region, parameter) table of one statistic."""
        if self.empty() or stat not in self.summary.columns:
            return pd.DataFrame()
        d = self.summary[~self.summary["parameter"].isin(META_PARAMETERS)]
        t = d.pivot_table(index="group", columns=["region", "parameter"], values=stat, aggfunc="first")
        return t.reindex(self.groups())

    def ratio(self, num_region=None, den_region="particle") -> pd.DataFrame:
        """Per intensity channel: mean in the cytoplasm region / mean inside particles."""
        m = self.value("mean")
        if m.empty:
            return pd.DataFrame()
        regions = [r for r in self.regions() if r != den_region]
        num_region = num_region or next((r for r in regions if "cyto" in r), regions[0] if regions else None)
        out = {}
        for ch in self.channels():
            if (num_region, ch) in m.columns and (den_region, ch) in m.columns:
                den = m[(den_region, ch)].replace(0, np.nan)
                out[ch] = m[(num_region, ch)] / den
        return pd.DataFrame(out, index=m.index)

    def measures(self) -> pd.DataFrame:
        """Flat group x measure table for comparison: '<region> · <parameter>' means + ratios."""
        m = self.value("mean")
        if m.empty:
            return m
        flat = pd.DataFrame({f"{region_label(r)} · {p}": m[(r, p)] for r, p in m.columns}, index=m.index)
        for ch, v in self.ratio().items():
            flat[f"{RATIO_LABEL} · {ch}"] = v
        return flat

    # ---- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        return {"conditions": self.conditions.to_dict(), "uses": dict(self.uses)}

    def summary_csv(self) -> str:
        buf = io.StringIO()
        self.summary.to_csv(buf, index=False)
        return buf.getvalue()

    @classmethod
    def from_parts(cls, csv_text: str | None, d: dict | None) -> "Delivery":
        summary = None
        if csv_text:
            summary = pd.read_csv(io.StringIO(csv_text), dtype={"group": str, "region": str, "parameter": str})
        out = cls(summary)
        if d:
            out.conditions = ConditionTable.from_dict(d.get("conditions", {}), out.groups())
            out.uses = {g: u for g, u in d.get("uses", {}).items() if g in set(out.groups())}
        return out


# "1uM AHP17999-r568, 5uM FITC-PRD, 10% PEG, 0min": amount + unit + name, or a time.
_AMOUNT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(nM|uM|µM|mM|M|%|mg/ml|mg/mL|ug/ml|µg/ml)\s+(.+?)\s*$", re.I)
_TIME = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(s|sec|min|h|hr|hour|hours)\s*$", re.I)
_UNIT = {"um": "µM", "µm": "µM", "nm": "nM", "mm": "mM", "m": "M", "%": "%"}


def parse_delivery_name(group: str) -> dict:
    """{condition name: (value, unit)} from comma-separated tokens like '1uM X' or '10min'."""
    out = {}
    for tok in str(group).split(","):
        m = _AMOUNT.match(tok)
        if m:
            unit = _UNIT.get(m.group(2).lower(), m.group(2))
            out[m.group(3).strip()] = (m.group(1), unit)
            continue
        m = _TIME.match(tok)
        if m:
            unit = {"s": "s", "sec": "s", "min": "min"}.get(m.group(2).lower(), "h")
            out["시간"] = (m.group(1), unit)
    return out


def fill_delivery_from_names(table: ConditionTable) -> list[str]:
    """Add the conditions found in delivery group names and fill blank cells only."""
    parsed = {g: parse_delivery_name(g) for g in table.values.index}
    filled = []
    names = []
    for p in parsed.values():
        for k in p:
            if k not in names:
                names.append(k)
    for name in names:
        unit = next(p[name][1] for p in parsed.values() if name in p)
        if name not in table.names():
            table.add_condition(ConditionDef(name, "numeric", unit))
        new = [g for g, p in parsed.items() if name in p and table.value(g, name) is None]
        for g in new:
            table.set_value(g, name, parsed[g][name][0])
        if new:
            filled.append(name)
    return filled
