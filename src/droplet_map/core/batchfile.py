"""Batch files: one Excel workbook per purification batch.

A batch file holds everything recorded for one batch, one sheet per kind of record. Each sheet
has one of four shapes:
- 배치 값 (batch values): one row per item (purification date, zeta potential, ...), one value each.
- 조건표 (condition table): one row per particle formation condition, keyed by 조건 ID.
- 입자별 측정 (per-particle): one row per particle from the image analysis, keyed by 조건 ID.
- 조건별 측정 (per-condition): one row per measurement (FRAP, cell delivery, ...), keyed by 조건 ID.

Column headers are "name (unit)". Sheet shapes, categories and column roles live in a hidden
sheet (_droplet); a file made or changed in Excel without it is read by guessing from names.
Blank cells mean "not recorded" and are never read as zero.
"""
from __future__ import annotations

import copy
import datetime as _dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KIND_BATCH = "배치 값"
KIND_CONDITIONS = "조건표"
KIND_PARTICLES = "입자별 측정"
KIND_MEASURE = "조건별 측정"
SHEET_KINDS = (KIND_BATCH, KIND_CONDITIONS, KIND_PARTICLES, KIND_MEASURE)

ROLE_ID = "조건 ID"
ROLE_CONDITION = "조건"         # condition table: a formation condition (pH, NaCl, ...)
ROLE_REPLICATE = "반복"         # measurement sheets: image / replicate label
ROLE_SUB = "하위 조건"          # measurement sheets: a condition of the measurement itself (time, dose)
ROLE_VALUE = "측정값"
ROLE_NOTE = "메모"
ROLES = (ROLE_ID, ROLE_CONDITION, ROLE_REPLICATE, ROLE_SUB, ROLE_VALUE, ROLE_NOTE)
ROLES_FOR = {
    KIND_CONDITIONS: (ROLE_ID, ROLE_CONDITION, ROLE_NOTE),
    KIND_PARTICLES: (ROLE_ID, ROLE_REPLICATE, ROLE_VALUE, ROLE_NOTE),
    KIND_MEASURE: (ROLE_ID, ROLE_REPLICATE, ROLE_SUB, ROLE_VALUE, ROLE_NOTE),
}

COND_ID = "조건 ID"
# batch-value sheet: fixed columns
B_ITEM, B_VALUE, B_UNIT, B_GROUP, B_NOTE = "항목", "값", "단위", "구분", "메모"
BATCH_COLUMNS = (B_ITEM, B_VALUE, B_UNIT, B_GROUP, B_NOTE)
ITEM_NAME = "배치 이름"
CAT_PURIFICATION = "정제 조건"
CAT_PROPERTY = "배치 성질"

SHEET_BATCH = "배치 정보"
SHEET_CONDITIONS = "형성 조건"
SHEET_PARTICLES = "입자 형태"
SHEET_INVITRO = "in vitro 실험"
SHEET_DELIVERY = "세포 전달"
META_SHEET = "_droplet"
TEMPLATE_NAME = "_템플릿.xlsx"
SUFFIX = ".xlsx"

RATIO_PREFIX = "세포질/입자 "     # a blank ratio column is filled from "세포질 X" / "입자 안 X"
CYTO_PREFIX = "세포질 "
INSIDE_PREFIX = "입자 안 "

_HEADER = re.compile(r"^\s*(.*?)\s*\(([^()]*)\)\s*$")


def split_header(text) -> tuple[str, str]:
    """'NaCl (mM)' -> ('NaCl', 'mM'); 'circularity' -> ('circularity', '')."""
    text = "" if text is None else str(text).strip()
    m = _HEADER.match(text)
    if m and m.group(1):
        return m.group(1), m.group(2).strip()
    return text, ""


def blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() == "") \
        or (v is pd.NA) or (isinstance(v, (np.floating,)) and np.isnan(v))


def to_number(v):
    if blank(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


@dataclass
class Column:
    name: str
    unit: str = ""
    role: str = ROLE_VALUE

    @property
    def label(self) -> str:
        return f"{self.name} ({self.unit})" if self.unit else self.name


@dataclass
class Sheet:
    name: str
    kind: str
    category: str = ""
    columns: list[Column] = field(default_factory=list)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)    # columns = column names

    def __post_init__(self):
        if not self.category:
            self.category = self.name
        self.data = self.data.reindex(columns=[c.name for c in self.columns])

    # ---- columns ----------------------------------------------------------
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def has(self, name: str) -> bool:
        return name in self.names()

    def by_role(self, *roles) -> list[str]:
        return [c.name for c in self.columns if c.role in roles]

    def add_column(self, col: Column, pos: int | None = None):
        if self.kind == KIND_BATCH:
            raise ValueError("배치 값 시트의 열은 고정입니다. 항목은 행으로 추가하세요.")
        col.name = col.name.strip()
        if not col.name:
            raise ValueError("열 이름을 입력하세요.")
        if self.has(col.name):
            raise ValueError(f"'{col.name}' 열이 이미 있습니다.")
        pos = len(self.columns) if pos is None else pos
        self.columns.insert(pos, col)
        self.data.insert(min(pos, self.data.shape[1]), col.name, pd.Series(np.nan, index=self.data.index, dtype=object))

    def update_column(self, old: str, col: Column):
        if self.kind == KIND_BATCH:
            raise ValueError("배치 값 시트의 열은 고정입니다.")
        col.name = col.name.strip()
        if not col.name:
            raise ValueError("열 이름을 입력하세요.")
        if col.name != old and self.has(col.name):
            raise ValueError(f"'{col.name}' 열이 이미 있습니다.")
        if old == COND_ID and col.role != ROLE_ID:
            raise ValueError("조건 ID 열은 다른 시트와 이어 주는 열이라 바꿀 수 없습니다.")
        i = self.names().index(old)
        self.columns[i] = col
        if old != col.name:
            self.data = self.data.rename(columns={old: col.name})

    def remove_column(self, name: str):
        if self.kind == KIND_BATCH:
            raise ValueError("배치 값 시트의 열은 고정입니다.")
        if name == COND_ID:
            raise ValueError("조건 ID 열은 지울 수 없습니다.")
        self.columns = [c for c in self.columns if c.name != name]
        self.data = self.data.drop(columns=[name])

    # ---- values -----------------------------------------------------------
    def numeric(self, name: str) -> pd.Series:
        return pd.to_numeric(self.data[name].map(to_number), errors="coerce")

    def text(self, name: str) -> pd.Series:
        return self.data[name].map(lambda v: np.nan if blank(v) else fmt(v))

    def set_value(self, row: int, name: str, value):
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                value = np.nan
        if self.data[name].dtype != object:
            self.data[name] = self.data[name].astype(object)
        self.data.iat[row, self.names().index(name)] = value

    def add_rows(self, n=1):
        empty = pd.DataFrame(np.nan, index=range(n), columns=self.data.columns, dtype=object)
        self.data = pd.concat([self.data, empty], ignore_index=True)

    def remove_rows(self, rows):
        self.data = self.data.drop(index=self.data.index[list(rows)]).reset_index(drop=True)

    def append_frame(self, df: pd.DataFrame):
        """Rows from another table; columns the sheet lacks are added as measurements."""
        for c in df.columns:
            if not self.has(c):
                num = pd.to_numeric(df[c], errors="coerce").notna().sum() >= max(1, df[c].notna().sum() // 2)
                self.add_column(Column(c, "", ROLE_VALUE if num else ROLE_NOTE))
        self.data = pd.concat([self.data, df.reindex(columns=self.names())], ignore_index=True)

    def is_numeric_column(self, name: str) -> bool:
        vals = [v for v in self.data[name] if not blank(v)]
        return bool(vals) and all(to_number(v) is not None for v in vals)


@dataclass
class Issue:
    sheet: str
    row: int | None      # 0-based data row, None = whole sheet / column
    column: str | None
    message: str
    serious: bool = True


@dataclass
class BatchFile:
    sheets: list[Sheet] = field(default_factory=list)
    path: Path | None = None

    # ---- sheets -----------------------------------------------------------
    def sheet(self, name: str) -> Sheet:
        for s in self.sheets:
            if s.name == name:
                return s
        raise KeyError(name)

    def of_kind(self, kind: str) -> list[Sheet]:
        return [s for s in self.sheets if s.kind == kind]

    def first(self, kind: str) -> Sheet | None:
        k = self.of_kind(kind)
        return k[0] if k else None

    @property
    def batch_sheet(self) -> Sheet | None:
        return self.first(KIND_BATCH)

    @property
    def conditions(self) -> Sheet | None:
        return self.first(KIND_CONDITIONS)

    @property
    def particles(self) -> Sheet | None:
        return self.first(KIND_PARTICLES)

    def measure_sheets(self) -> list[Sheet]:
        return self.of_kind(KIND_MEASURE)

    def add_sheet(self, name: str, kind: str = KIND_MEASURE, category: str = "") -> Sheet:
        name = name.strip()
        if not name:
            raise ValueError("시트 이름을 입력하세요.")
        if name.startswith("_"):
            raise ValueError("'_'로 시작하는 이름은 앱이 쓰는 시트라 쓸 수 없습니다.")
        if name in [s.name for s in self.sheets]:
            raise ValueError(f"'{name}' 시트가 이미 있습니다.")
        if kind in (KIND_BATCH, KIND_CONDITIONS, KIND_PARTICLES) and self.first(kind) is not None:
            raise ValueError(f"'{kind}' 모양의 시트는 하나만 둘 수 있습니다.")
        if kind == KIND_BATCH:
            s = _batch_sheet(name)
        else:
            cols = [Column(COND_ID, "", ROLE_ID)]
            if kind == KIND_MEASURE:
                cols.append(Column("반복", "", ROLE_REPLICATE))
            if kind == KIND_PARTICLES:
                cols.append(Column("이미지", "", ROLE_REPLICATE))
            s = Sheet(name, kind, category or name, cols)
        self.sheets.append(s)
        return s

    def rename_sheet(self, old: str, new: str):
        new = new.strip()
        if not new or new == old:
            return
        if new.startswith("_") or new in [s.name for s in self.sheets]:
            raise ValueError(f"'{new}' 이름은 쓸 수 없습니다.")
        s = self.sheet(old)
        if s.category == old:
            s.category = new
        s.name = new

    def remove_sheet(self, name: str):
        s = self.sheet(name)
        if s.kind in (KIND_BATCH, KIND_CONDITIONS):
            raise ValueError(f"'{s.kind}' 시트는 지울 수 없습니다. 다른 시트가 이 시트에 기대고 있습니다.")
        self.sheets = [x for x in self.sheets if x.name != name]

    # ---- batch values -----------------------------------------------------
    @property
    def name(self) -> str:
        v = self.item(ITEM_NAME)
        if v is not None and str(v).strip():
            return str(v).strip()
        return self.path.stem if self.path else "새 배치"

    def item(self, name: str):
        s = self.batch_sheet
        if s is None:
            return None
        hit = s.data[s.data[B_ITEM].astype(str).str.strip() == name]
        if not len(hit):
            return None
        v = hit[B_VALUE].iloc[0]
        return None if blank(v) else v

    def set_item(self, name: str, value, unit="", group=CAT_PURIFICATION):
        s = self.batch_sheet
        rows = s.data.index[s.data[B_ITEM].astype(str).str.strip() == name]
        if len(rows):
            s.set_value(int(rows[0]), B_VALUE, value)
            return
        s.add_rows(1)
        r = len(s.data) - 1
        for k, v in ((B_ITEM, name), (B_VALUE, value), (B_UNIT, unit), (B_GROUP, group)):
            s.set_value(r, k, v)

    def items(self) -> list[dict]:
        """Batch values other than the batch name: {name, value, unit, category, note}."""
        s = self.batch_sheet
        if s is None:
            return []
        out = []
        for _, r in s.data.iterrows():
            n = r[B_ITEM]
            if blank(n) or str(n).strip() == ITEM_NAME:
                continue
            out.append(dict(name=str(n).strip(), value=None if blank(r[B_VALUE]) else r[B_VALUE],
                            unit="" if blank(r[B_UNIT]) else str(r[B_UNIT]).strip(),
                            category=CAT_PURIFICATION if blank(r[B_GROUP]) else str(r[B_GROUP]).strip(),
                            note="" if blank(r[B_NOTE]) else str(r[B_NOTE])))
        return out

    # ---- conditions -------------------------------------------------------
    def condition_ids(self) -> list[str]:
        c = self.conditions
        if c is None or not c.has(COND_ID):
            return []
        return [i for i in c.text(COND_ID) if isinstance(i, str)]

    def fill_condition_ids(self) -> int:
        """Blank 조건 ID cells in the condition table get C01, C02, ... Returns how many."""
        c = self.conditions
        if c is None:
            return 0
        taken = set(self.condition_ids())
        n, k = 0, 1
        for r in range(len(c.data)):
            if blank(c.data.at[r, COND_ID]) and any(not blank(v) for v in c.data.iloc[r]):
                while f"C{k:02d}" in taken:
                    k += 1
                c.set_value(r, COND_ID, f"C{k:02d}")
                taken.add(f"C{k:02d}")
                n += 1
        return n

    def condition_label(self, cid: str) -> str:
        """'pH 7 · NaCl 150 mM · PEG 10 %' for a condition id (for menus and tables)."""
        c = self.conditions
        if c is None:
            return cid
        rows = c.data.index[c.text(COND_ID) == cid]
        if not len(rows):
            return cid
        r = rows[0]
        parts = []
        for col in c.columns:
            if col.role != ROLE_CONDITION or blank(c.data.at[r, col.name]):
                continue
            v = fmt(c.data.at[r, col.name])
            parts.append(f"{col.name} {v}{(' ' + col.unit) if col.unit else ''}")
        return " · ".join(parts) or cid

    # ---- derived values ---------------------------------------------------
    def fill_ratios(self):
        """Blank '세포질/입자 X' cells get '세포질 X' / '입자 안 X' when both are there."""
        for s in self.measure_sheets():
            for c in s.names():
                if not c.startswith(RATIO_PREFIX):
                    continue
                rest = c[len(RATIO_PREFIX):]
                num, den = CYTO_PREFIX + rest, INSIDE_PREFIX + rest
                if s.has(num) and s.has(den):
                    r = s.numeric(num) / s.numeric(den).replace(0, np.nan)
                    miss = s.data[c].map(blank)
                    s.data.loc[miss & r.notna(), c] = r[miss & r.notna()]

    # ---- checks -----------------------------------------------------------
    def check(self) -> list[Issue]:
        out: list[Issue] = []
        if self.batch_sheet is None:
            out.append(Issue("", None, None, "배치 값 시트가 없습니다."))
        cs = self.conditions
        if cs is None:
            out.append(Issue("", None, None, "형성 조건 시트(조건표)가 없습니다."))
            ids = set()
        else:
            ids_series = cs.text(COND_ID) if cs.has(COND_ID) else pd.Series(dtype=object)
            ids = set(ids_series.dropna())
            dup = ids_series[ids_series.duplicated() & ids_series.notna()]
            for r, v in dup.items():
                out.append(Issue(cs.name, int(r), COND_ID, f"조건 ID '{v}'가 두 번 있습니다."))
            for r in range(len(cs.data)):
                if ids_series.isna().iloc[r] and any(not blank(v) for v in cs.data.iloc[r]):
                    out.append(Issue(cs.name, r, COND_ID, "조건 ID가 비어 있습니다."))
        for s in self.sheets:
            if s.kind == KIND_BATCH:
                for r, v in s.data[B_ITEM].items():
                    if blank(v) and not blank(s.data.at[r, B_VALUE]):
                        out.append(Issue(s.name, int(r), B_ITEM, "항목 이름이 비어 있습니다."))
                continue
            if not s.has(COND_ID):
                out.append(Issue(s.name, None, None, "조건 ID 열이 없습니다."))
                continue
            for col in s.columns:      # text conditions are fine (e.g. a buffer name); measurements are numbers
                if s.kind != KIND_CONDITIONS and col.role in (ROLE_VALUE, ROLE_SUB):
                    bad = [(r, v) for r, v in s.data[col.name].items() if not blank(v) and to_number(v) is None]
                    for r, v in bad[:20]:
                        out.append(Issue(s.name, int(r), col.name, f"숫자가 아닌 값 '{v}'"))
                    if len(bad) > 20:
                        out.append(Issue(s.name, None, col.name, f"숫자가 아닌 값이 {len(bad)}칸 더 있습니다."))
            if s.kind == KIND_CONDITIONS:
                continue
            sid = s.text(COND_ID)
            unknown = sorted(set(sid.dropna()) - ids)
            for v in unknown[:20]:
                r = int(sid.index[sid == v][0])
                out.append(Issue(s.name, r, COND_ID, f"조건표에 없는 조건 ID '{v}'"))
            nid = int(sid.isna().sum() - s.data.apply(lambda row: all(blank(x) for x in row), axis=1).sum())
            if nid > 0:
                r = int(sid.index[sid.isna()][0])
                out.append(Issue(s.name, r, COND_ID, f"조건 ID가 빈 행이 {nid}개 있습니다 (분석에서 빠짐)."))
        return out

    # ---- copies -----------------------------------------------------------
    def copy_for_new_batch(self, name: str) -> "BatchFile":
        """Same sheets and columns, batch values and condition table kept, measurements emptied."""
        out = copy.deepcopy(self)
        out.path = None
        for s in out.sheets:
            if s.kind in (KIND_PARTICLES, KIND_MEASURE):
                s.data = s.data.iloc[0:0].copy()
        out.set_item(ITEM_NAME, name)
        return out

    def layout_only(self) -> "BatchFile":
        """For a template: sheets, columns and batch items without values or rows."""
        out = copy.deepcopy(self)
        out.path = None
        for s in out.sheets:
            if s.kind == KIND_BATCH:
                s.data[B_VALUE] = np.nan
                s.data[B_NOTE] = np.nan
            else:
                s.data = s.data.iloc[0:0].copy()
        return out


def fmt(v) -> str:
    if isinstance(v, (_dt.datetime, _dt.date, pd.Timestamp)):
        return v.strftime("%Y-%m-%d")
    n = to_number(v)
    if n is not None and float(n).is_integer() and not isinstance(v, str):
        return str(int(n))
    if n is not None and isinstance(v, str):
        return v.strip()
    return str(v).strip()


# ---------------------------------------------------------------- template
def _batch_sheet(name=SHEET_BATCH, rows=None) -> Sheet:
    rows = rows if rows is not None else [
        (ITEM_NAME, "", "", CAT_PURIFICATION, ""),
        ("정제일", "", "", CAT_PURIFICATION, "예: 2026-04-30"),
        ("컬럼", "", "", CAT_PURIFICATION, ""),
        ("단백질 농도", "", "mg/mL", CAT_PURIFICATION, ""),
        ("제타 포텐셜", "", "mV", CAT_PROPERTY, ""),
    ]
    df = pd.DataFrame(rows, columns=list(BATCH_COLUMNS), dtype=object).replace("", np.nan)
    return Sheet(name, KIND_BATCH, CAT_PURIFICATION, [Column(c, "", ROLE_NOTE) for c in BATCH_COLUMNS], df)


def template() -> BatchFile:
    """A new batch file with the default sheets and columns."""
    sheets = [
        _batch_sheet(),
        Sheet(SHEET_CONDITIONS, KIND_CONDITIONS, SHEET_CONDITIONS, [
            Column(COND_ID, "", ROLE_ID), Column("pH", "", ROLE_CONDITION), Column("NaCl", "mM", ROLE_CONDITION),
            Column("PEG", "%", ROLE_CONDITION), Column("단백질", "µM", ROLE_CONDITION), Column("메모", "", ROLE_NOTE)]),
        Sheet(SHEET_PARTICLES, KIND_PARTICLES, SHEET_PARTICLES, [
            Column(COND_ID, "", ROLE_ID), Column("이미지", "", ROLE_REPLICATE)]),
        Sheet(SHEET_INVITRO, KIND_MEASURE, SHEET_INVITRO, [
            Column(COND_ID, "", ROLE_ID), Column("반복", "", ROLE_REPLICATE),
            Column("FRAP 회복률", "%", ROLE_VALUE), Column("FRAP t½", "s", ROLE_VALUE), Column("탁도", "OD600", ROLE_VALUE)]),
        Sheet(SHEET_DELIVERY, KIND_MEASURE, SHEET_DELIVERY, [
            Column(COND_ID, "", ROLE_ID), Column("반복", "", ROLE_REPLICATE),
            Column("시간", "min", ROLE_SUB), Column("항체", "µM", ROLE_SUB),
            Column(CYTO_PREFIX + "항체 신호", "", ROLE_VALUE), Column(INSIDE_PREFIX + "항체 신호", "", ROLE_VALUE),
            Column(RATIO_PREFIX + "항체 신호", "", ROLE_VALUE), Column("이미지 수", "", ROLE_NOTE)]),
    ]
    return BatchFile(sheets)


# ---------------------------------------------------------------- reading
def _guess_kind(name: str, headers: list[str]) -> str | None:
    names = [split_header(h)[0] for h in headers]
    if name == SHEET_BATCH or (B_ITEM in names and B_VALUE in names):
        return KIND_BATCH
    if COND_ID not in names:
        return None
    if name == SHEET_CONDITIONS or "조건" in name and "형성" in name:
        return KIND_CONDITIONS
    if name == SHEET_PARTICLES or "입자" in name and ("형태" in name or "모양" in name):
        return KIND_PARTICLES
    return KIND_MEASURE


def _guess_role(kind: str, name: str, values) -> str:
    low = name.lower()
    if name == COND_ID:
        return ROLE_ID
    if low in ("메모", "note", "notes", "memo", "비고"):
        return ROLE_NOTE
    if kind == KIND_CONDITIONS:
        return ROLE_CONDITION
    if low in ("반복", "이미지", "image", "replicate", "rep", "particle", "입자", "source_file"):
        return ROLE_REPLICATE if low in ("반복", "이미지", "image", "replicate", "rep") else ROLE_NOTE
    if low in ("시간", "time", "항체", "dose", "농도"):
        return ROLE_SUB
    vals = [v for v in values if not blank(v)]
    if vals and sum(to_number(v) is not None for v in vals) >= 0.8 * len(vals):
        return ROLE_VALUE
    return ROLE_NOTE


def _cell(v):
    if isinstance(v, _dt.datetime):
        return v.strftime("%Y-%m-%d") if (v.hour, v.minute, v.second) == (0, 0, 0) else v.isoformat(sep=" ")
    if isinstance(v, _dt.date):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, str):
        v = v.strip()
        return np.nan if v == "" else v
    return np.nan if v is None else v


def read_batch_file(path) -> tuple[BatchFile, list[str]]:
    """Read a batch file. Returns (file, notes about sheets that were skipped or guessed)."""
    path = Path(path)
    names, rows_of, close = _open_workbook(path)
    notes = []
    meta_sheets, meta_cols = {}, {}
    if META_SHEET in names:
        for row in list(rows_of(META_SHEET))[1:]:
            row = list(row) + [None] * 5
            sname, col, kind, cat, _unit = [_cell(v) for v in row[:5]]
            if blank(sname):
                continue
            if blank(col):
                meta_sheets[str(sname)] = (str(kind) if not blank(kind) else None, None if blank(cat) else str(cat))
            else:
                meta_cols[(str(sname), str(col))] = str(kind) if not blank(kind) else None
    sheets = []
    for ws_name in names:
        if ws_name.startswith("_"):
            continue
        rows = iter(rows_of(ws_name))
        header = next(rows, None)
        if header is None:
            continue
        keep = [i for i, h in enumerate(header) if not blank(h)]
        headers = [str(header[i]).strip() for i in keep]
        if not headers:
            continue
        body = []
        for r in rows:
            r = list(r) + [None] * (len(header) - len(r))
            vals = [_cell(r[i]) for i in keep]
            if all(blank(v) for v in vals):
                continue
            body.append(vals)
        kind, cat = meta_sheets.get(ws_name, (None, None))
        if kind not in SHEET_KINDS:
            kind = _guess_kind(ws_name, headers)
            if kind is None:
                notes.append(f"'{ws_name}' 시트: 조건 ID 열이 없어 읽지 않았습니다.")
                continue
        names = [split_header(h) for h in headers]
        df = pd.DataFrame(body, columns=[n for n, _u in names], dtype=object) if body else \
            pd.DataFrame(columns=[n for n, _u in names], dtype=object)
        df = df.loc[:, ~df.columns.duplicated()]
        if kind == KIND_BATCH:
            cols = [Column(c, "", ROLE_NOTE) for c in BATCH_COLUMNS]
            df = df.reindex(columns=list(BATCH_COLUMNS))
        else:
            cols, seen = [], set()
            for n, u in names:
                if n in seen:
                    continue
                seen.add(n)
                role = meta_cols.get((ws_name, n))
                if role not in ROLES_FOR[kind]:
                    role = _guess_role(kind, n, df[n] if n in df else [])
                cols.append(Column(n, u, role))
            if COND_ID not in df.columns:
                notes.append(f"'{ws_name}' 시트: 조건 ID 열이 없어 읽지 않았습니다.")
                continue
            df[COND_ID] = df[COND_ID].map(lambda v: np.nan if blank(v) else fmt(v))
        sheets.append(Sheet(ws_name, kind, cat or ws_name if kind != KIND_BATCH else CAT_PURIFICATION, cols, df))
    close()
    bf = BatchFile(sheets, path)
    for kind in (KIND_BATCH, KIND_CONDITIONS, KIND_PARTICLES):
        extra = bf.of_kind(kind)[1:]
        for s in extra:                 # one of each; later ones are read as per-condition sheets
            if kind == KIND_PARTICLES:
                s.kind = KIND_MEASURE
            else:
                notes.append(f"'{s.name}' 시트: '{kind}' 시트가 둘이라 첫 번째만 씁니다.")
                bf.sheets.remove(s)
    bf.fill_ratios()
    return bf, notes


def _open_workbook(path: Path):
    """(sheet names, rows(name) -> iterable of row tuples, close). python-calamine is about ten
    times faster than openpyxl on large particle sheets; openpyxl is the fallback."""
    try:
        from python_calamine import CalamineWorkbook
        wb = CalamineWorkbook.from_path(str(path))
        return list(wb.sheet_names), lambda n: wb.get_sheet_by_name(n).to_python(skip_empty_area=False), lambda: None
    except ImportError:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        return list(wb.sheetnames), lambda n: wb[n].iter_rows(values_only=True), wb.close


# ---------------------------------------------------------------- writing
def _out(v, role):
    if blank(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, str) and role not in (ROLE_ID, ROLE_NOTE, ROLE_REPLICATE):
        n = to_number(v)
        if n is not None and re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", v.strip().replace(",", "")):
            return int(n) if n.is_integer() and "." not in v else n
    return v


def write_batch_file(bf: BatchFile, path=None):
    """Write the workbook next to the target first, then replace it (a failed save never leaves a
    broken file)."""
    import os
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font, PatternFill
    path = Path(path or bf.path)
    if path.suffix.lower() != SUFFIX:
        path = path.with_suffix(SUFFIX)
    wb = Workbook(write_only=True)
    bold = Font(bold=True)
    fill = PatternFill("solid", fgColor="EEF2F7")
    for s in bf.sheets:
        ws = wb.create_sheet(s.name)
        ws.freeze_panes = "B2" if s.kind != KIND_BATCH else "A2"
        for i, c in enumerate(s.columns):
            letter = _letter(i)
            ws.column_dimensions[letter].width = max(8, min(40, len(c.label) * 1.6 + 2)) if s.kind != KIND_BATCH \
                else (24 if c.name in (B_ITEM, B_VALUE) else 14)
        head = []
        for c in s.columns:
            cell = WriteOnlyCell(ws, value=c.label if s.kind != KIND_BATCH else c.name)
            cell.font = bold
            cell.fill = fill
            head.append(cell)
        ws.append(head)
        roles = [c.role for c in s.columns]
        arr = s.data.reindex(columns=s.names()).to_numpy(dtype=object)
        for row in arr:
            ws.append([_out(v, roles[i] if s.kind != KIND_BATCH else (ROLE_VALUE if i == 1 else ROLE_NOTE))
                       for i, v in enumerate(row)])
    meta = wb.create_sheet(META_SHEET)
    meta.sheet_state = "hidden"
    meta.append(["시트", "열", "모양 / 역할", "범주", "단위"])
    for s in bf.sheets:
        meta.append([s.name, None, s.kind, s.category, None])
        if s.kind != KIND_BATCH:
            for c in s.columns:
                meta.append([s.name, c.name, c.role, None, c.unit or None])
    tmp = path.with_name(path.name + ".saving")
    try:
        wb.save(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    bf.path = path
    return path


def _letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


# ---------------------------------------------------------------- folders
def batch_paths(folder) -> list[Path]:
    """Batch files in a folder (not the template, not Excel's lock files)."""
    folder = Path(folder)
    return sorted(p for p in folder.glob("*" + SUFFIX)
                  if not p.name.startswith(("_", "~$", ".")) and p.is_file())


def folder_template(folder) -> BatchFile:
    p = Path(folder) / TEMPLATE_NAME
    if p.exists():
        try:
            bf, _ = read_batch_file(p)
            return bf.layout_only()
        except Exception:
            pass
    return template()


def safe_file_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip().strip(".")
    return name or "배치"
