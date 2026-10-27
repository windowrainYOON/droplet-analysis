"""User-defined experimental conditions, one row per group. Blank means "not recorded"."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import io
import json

import numpy as np
import pandas as pd

KINDS = ("numeric", "categorical")
KIND_LABELS = {"numeric": "수치", "categorical": "범주"}


@dataclass
class ConditionDef:
    name: str
    kind: str = "categorical"
    unit: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} ({self.unit})" if self.unit else self.name


class ConditionTable:
    """Schema + values. Values are stored as text exactly as typed; blanks are NaN."""

    def __init__(self, groups=(), schema=None, values=None, references=()):
        self.schema: list[ConditionDef] = list(schema or [])
        self.values = pd.DataFrame(index=pd.Index(list(groups), name="group"), dtype=object)
        for c in self.schema:
            self.values[c.name] = pd.Series(np.nan, index=self.values.index, dtype=object)
        if values is not None:
            for g, row in values.items():
                for k, v in row.items():
                    if g in self.values.index and k in self.values.columns:
                        self.values.at[g, k] = v
        self.references: set[str] = set(references)

    # ---- schema -------------------------------------------------------
    def names(self) -> list[str]:
        return [c.name for c in self.schema]

    def get(self, name: str) -> ConditionDef:
        for c in self.schema:
            if c.name == name:
                return c
        raise KeyError(name)

    def add_condition(self, cond: ConditionDef):
        if cond.name in self.names():
            raise ValueError(f"'{cond.name}' 조건이 이미 있습니다.")
        self.schema.append(cond)
        self.values[cond.name] = pd.Series(np.nan, index=self.values.index, dtype=object)

    def remove_condition(self, name: str):
        self.schema = [c for c in self.schema if c.name != name]
        self.values = self.values.drop(columns=[name])

    def update_condition(self, old: str, cond: ConditionDef):
        i = self.names().index(old)
        self.schema[i] = cond
        if old != cond.name:
            self.values = self.values.rename(columns={old: cond.name})

    def set_groups(self, groups):
        groups = list(groups)
        self.values = self.values.reindex(groups)
        self.values.index.name = "group"
        self.references &= set(groups)

    # ---- values -------------------------------------------------------
    def set_value(self, group: str, name: str, value):
        if value is None or (isinstance(value, str) and value.strip() == ""):
            value = np.nan
        elif isinstance(value, str):
            value = value.strip()
        if self.values[name].dtype != object:      # pandas may narrow an all-blank column to float
            self.values[name] = self.values[name].astype(object)
        self.values.at[group, name] = value

    def value(self, group: str, name: str):
        v = self.values.at[group, name]
        return None if _blank(v) else v

    def numeric(self, name: str) -> pd.Series:
        """Numeric view of a condition; text that is not a number becomes blank."""
        return pd.to_numeric(self.values[name].map(_to_number), errors="coerce")

    def categorical(self, name: str) -> pd.Series:
        return self.values[name].map(lambda v: np.nan if _blank(v) else _fmt(v))

    def recorded_count(self, name: str) -> int:
        return int(self.values[name].map(lambda v: not _blank(v)).sum())

    # ---- import / export ---------------------------------------------
    def to_dict(self) -> dict:
        vals = {g: {k: (None if _blank(v) else v) for k, v in row.items()} for g, row in self.values.iterrows()}
        return {"schema": [asdict(c) for c in self.schema], "values": vals, "references": sorted(self.references)}

    @classmethod
    def from_dict(cls, d: dict, groups=None) -> "ConditionTable":
        schema = [ConditionDef(**c) for c in d.get("schema", [])]
        vals = d.get("values", {})
        groups = list(groups) if groups is not None else list(vals)
        return cls(groups, schema, vals, d.get("references", []))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_csv(self) -> str:
        out = self.values.copy()
        out.insert(0, "기준 그룹", ["★" if g in self.references else "" for g in out.index])
        return out.to_csv(encoding=None)

    def merge_csv(self, text: str, add_missing_conditions=True) -> list[str]:
        """Read a table whose first column is the group name. Returns groups that did not match."""
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        return self.merge_frame(df, add_missing_conditions)

    def merge_frame(self, df: pd.DataFrame, add_missing_conditions=True) -> list[str]:
        gcol = df.columns[0]
        unmatched = []
        for col in df.columns[1:]:
            if col == "기준 그룹":
                continue
            if col not in self.names() and add_missing_conditions:
                self.add_condition(ConditionDef(col, guess_kind(df[col])))
        for _, row in df.iterrows():
            g = str(row[gcol]).strip()
            if g not in self.values.index:
                unmatched.append(g)
                continue
            for col in df.columns[1:]:
                if col == "기준 그룹":
                    if str(row[col]).strip():
                        self.references.add(g)
                    continue
                if col in self.names():
                    self.set_value(g, col, row[col])
        return unmatched


def guess_kind(series: pd.Series) -> str:
    vals = [v for v in series if not _blank(v)]
    if vals and all(_to_number(v) is not None for v in vals):
        return "numeric"
    return "categorical"


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "")


def _to_number(v):
    if _blank(v):
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


def _fmt(v) -> str:
    n = _to_number(v)
    if n is not None and float(n).is_integer():
        return str(int(n))
    return str(v)
