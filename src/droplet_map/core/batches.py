"""Purification batches: the level above groups.

Every group (particle formation or cell delivery) belongs to exactly one batch and inherits the
batch's conditions (purification date, protein concentration, ...). Group-level conditions only
describe what was done with that batch's protein.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .conditions import ConditionDef, ConditionTable

UNASSIGNED = "배치 미지정"
BATCH_COLUMN = "배치"                       # name of the batch column in merged condition views
BATCH_PREFIX = "배치: "                     # batch-level conditions in merged views
OLD_BATCH_NAMES = ("정제 배치", "배치", "batch", "Batch", "purification batch", "Purification batch")


class Batches:
    """Batch table (rows = batches, columns = batch conditions) + group -> batch membership."""

    def __init__(self, table: ConditionTable | None = None, of: dict | None = None):
        self.table = table if table is not None else ConditionTable([])
        self.table.values.index.name = "batch"
        self.of: dict[str, str] = dict(of or {})
        self.references: set[str] = set()       # reference batches (score against matched groups)

    # ---- batches --------------------------------------------------------
    def names(self) -> list[str]:
        return list(self.table.values.index)

    def add(self, name: str):
        name = name.strip()
        if not name:
            raise ValueError("배치 이름을 입력하세요.")
        if name not in self.names():
            self.table.set_groups(self.names() + [name])
            self.table.values.index.name = "batch"

    def rename(self, old: str, new: str):
        new = new.strip()
        if not new or new == old:
            return
        if new in self.names():
            raise ValueError(f"'{new}' 배치가 이미 있습니다.")
        self.table.values = self.table.values.rename(index={old: new})
        self.table.values.index.name = "batch"
        self.of = {g: (new if b == old else b) for g, b in self.of.items()}
        if old in self.references:
            self.references.discard(old); self.references.add(new)

    def remove(self, name: str, move_to: str = UNASSIGNED):
        """Remove a batch; its groups move to another batch (default: unassigned)."""
        if move_to != UNASSIGNED and move_to not in self.names():
            self.add(move_to)
        for g, b in list(self.of.items()):
            if b == name:
                self.of[g] = move_to
        self.table.set_groups([b for b in self.names() if b != name])
        self.table.values.index.name = "batch"
        self.references.discard(name)
        self._ensure_unassigned()

    # ---- membership -----------------------------------------------------
    def batch_of(self, group: str) -> str:
        return self.of.get(group, UNASSIGNED)

    def assign(self, groups, batch: str):
        batch = batch.strip() or UNASSIGNED
        if batch != UNASSIGNED:
            self.add(batch)
        for g in groups:
            self.of[g] = batch
        self._ensure_unassigned()

    def groups_in(self, batch: str, among=None) -> list[str]:
        among = list(self.of) if among is None else list(among)
        return [g for g in among if self.batch_of(g) == batch]

    def sync_groups(self, groups):
        """Known groups only; new groups start unassigned (or in the only batch, when there is one)."""
        groups = list(groups)
        real = [b for b in self.names() if b != UNASSIGNED]
        default = real[0] if len(real) == 1 else UNASSIGNED
        self.of = {g: self.of.get(g, default) for g in groups}
        self._ensure_unassigned()

    def unassigned(self, among=None) -> list[str]:
        return self.groups_in(UNASSIGNED, among)

    def _ensure_unassigned(self):
        """The 'unassigned' pseudo batch exists only while some group is in it."""
        used = UNASSIGNED in set(self.of.values())
        have = UNASSIGNED in self.names()
        if used and not have:
            self.table.set_groups(self.names() + [UNASSIGNED])
        elif have and not used:
            self.table.set_groups([b for b in self.names() if b != UNASSIGNED])
        self.table.values.index.name = "batch"

    def series(self, groups) -> pd.Series:
        return pd.Series({g: self.batch_of(g) for g in groups}, dtype=object)

    # ---- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        return {"table": self.table.to_dict(), "of": dict(self.of), "references": sorted(self.references)}

    @classmethod
    def from_dict(cls, d: dict) -> "Batches":
        t = d.get("table", {})
        b = cls(ConditionTable.from_dict(t, list(t.get("values", {}))), d.get("of", {}))
        b.references = set(d.get("references", []))
        return b


def batches_from_conditions(cond: ConditionTable) -> tuple[Batches, str | None]:
    """Old projects kept the batch as a group condition (e.g. '정제 배치'). Turn that column into
    batch membership and drop it from the group conditions. Returns (batches, moved column name)."""
    col = next((c.name for c in cond.schema if c.name.strip() in OLD_BATCH_NAMES), None)
    b = Batches()
    groups = list(cond.values.index)
    if col is None:
        b.of = {g: UNASSIGNED for g in groups}
        b._ensure_unassigned()
        return b, None
    vals = cond.categorical(col)
    for g in groups:
        v = vals.get(g)
        b.assign([g], "" if pd.isna(v) else str(v))
    cond.remove_condition(col)
    return b, col


def merged_conditions(groups, cond: ConditionTable, batches: Batches) -> ConditionTable:
    """Read-only view for analysis and display: batch, the batch's conditions, then the group's own.
    Shares the reference set with the group table."""
    groups = list(groups)
    schema = [ConditionDef(BATCH_COLUMN, "categorical")]
    schema += [ConditionDef(BATCH_PREFIX + c.name, c.kind, c.unit) for c in batches.table.schema]
    own = [c for c in cond.schema if c.name not in {s.name for s in schema}]
    schema += own
    out = ConditionTable(groups, schema)
    for g in groups:
        b = batches.batch_of(g)
        out.values.at[g, BATCH_COLUMN] = np.nan if b == UNASSIGNED else b
        for c in batches.table.schema:
            if b in batches.table.values.index:
                v = batches.table.value(b, c.name)
                if v is not None:
                    out.values.at[g, BATCH_PREFIX + c.name] = v
        for c in own:
            if g in cond.values.index:
                v = cond.value(g, c.name)
                if v is not None:
                    out.values.at[g, c.name] = v
    out.references = cond.references
    return out

