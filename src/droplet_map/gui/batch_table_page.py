"""Batch table (read-only): one row per purification batch. Top: the values from each batch file's
batch sheet (purification conditions, zeta potential, ...), each in a category. Bottom: the items computed from each batch's
experiments (trends per condition unit, matched conditions, batch medians) that, together with the
typed values, place the batch on the batch map."""
from __future__ import annotations

import pandas as pd
from PySide6 import QtCore, QtWidgets

from ..core.batch_map import KIND_DIRECT, KIND_LEVEL, KIND_MATCHED, KIND_TREND, kind_on
from ..core.batches import UNASSIGNED
from .common import fill_table, flow_buttons, info_label, scroll_panel, section, title_label

ALL = "모두"
KIND_HELP = {
    KIND_TREND: "배치 안에서 조건 한 단위당 측정값이 얼마나 변하는지 (예: pH 1 오를 때 circularity 변화). "
                "배치마다 실험한 조건이 달라도 비교할 수 있습니다.",
    KIND_MATCHED: "두 배치 이상이 같은 조건 세트로 실험한 경우, 그 조건에서의 값.",
    KIND_LEVEL: "배치의 모든 그룹 값의 중앙값. 배치마다 실험한 조건이 다르면 그 차이도 섞입니다.",
}


class BatchTablePage(QtCore.QObject):
    kinds_changed = QtCore.Signal()    # which computed items are made
    edit_requested = QtCore.Signal()   # open a batch file in Excel
    title = "배치 표"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.desc = None
        self.view = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.view)
        lay.addWidget(title_label("배치 표"))
        lay.addWidget(info_label("배치 하나 = 단백질 정제 한 번. 위쪽은 배치마다 직접 넣는 값(정제 조건, 제타 포텐셜, 항체 전달 효율 …), "
                                 "아래쪽은 그 배치의 실험 데이터에서 계산한 값입니다. 둘을 합쳐 배치 지도의 위치를 정합니다."))
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        top = QtWidgets.QWidget(); tl = QtWidgets.QVBoxLayout(top); tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(QtWidgets.QLabel("<b>배치 파일의 배치 값</b> (열 머리 = [범주] 항목, 고치는 곳은 엑셀)"))
        tl.addWidget(self.table, 1)
        split.addWidget(top)

        bottom = QtWidgets.QWidget(); bl = QtWidgets.QVBoxLayout(bottom); bl.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("<b>실험 데이터에서 계산한 값</b>"))
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("범주"))
        self.f_cat = QtWidgets.QComboBox(); row.addWidget(self.f_cat)
        row.addWidget(QtWidgets.QLabel("종류"))
        self.f_kind = QtWidgets.QComboBox(); self.f_kind.addItems([ALL, KIND_TREND, KIND_MATCHED, KIND_LEVEL])
        row.addWidget(self.f_kind)
        self.f_text = QtWidgets.QLineEdit(); self.f_text.setPlaceholderText("항목 이름으로 찾기"); self.f_text.setClearButtonEnabled(True)
        row.addWidget(self.f_text)
        for w in (self.f_cat, self.f_kind):
            w.currentIndexChanged.connect(self.fill_derived)
        self.f_text.textChanged.connect(self.fill_derived)
        bl.addLayout(row)
        self.derived = QtWidgets.QTableWidget()
        self.derived.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.derived.setAlternatingRowColors(True)
        bl.addWidget(self.derived, 1)
        self.derived_note = QtWidgets.QLabel(""); self.derived_note.setStyleSheet("color:#666;"); self.derived_note.setWordWrap(True)
        bl.addWidget(self.derived_note)
        split.addWidget(bottom)
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)

        edit = section("배치 값 고치기",
                       info_label("배치 값과 범주(구분)는 배치 파일(엑셀)에서 고칩니다. 엑셀에서 저장하면 여기에 바로 다시 읽힙니다. "
                                  "★ 기준 배치는 왼쪽 목록에서 고릅니다."),
                       flow_buttons((("엑셀에서 열기…", self.edit_requested.emit),)))
        self.kind_checks = {}
        self.kinds_w = QtWidgets.QWidget(); self.kinds_lay = QtWidgets.QVBoxLayout(self.kinds_w)
        self.kinds_lay.setContentsMargins(0, 0, 0, 0)
        calc = section("실험 데이터에서 계산 (시트마다)", self.kinds_w,
                       info_label(f"경향: {KIND_HELP[KIND_TREND]}"),
                       info_label(f"같은 조건: {KIND_HELP[KIND_MATCHED]}"),
                       info_label(f"대표값: {KIND_HELP[KIND_LEVEL]}"),
                       info_label("입자 값은 전체 입자 기준 robust z(형태값은 크기 신뢰도 가중)의 그룹 중앙값입니다. "
                                  "형성 조건은 배치 파일의 '형성 조건' 시트에서 옵니다."))
        self.panel = scroll_panel(edit, calc)

    # ---- project ----------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.desc = None
        self.refresh()

    def batch_frame(self) -> pd.DataFrame:
        b = self.project.batches
        names = [n for n in b.names() if n != UNASSIGNED]
        out = pd.DataFrame(index=names)
        for c in b.table.schema:
            label = f"[{b.category(c.name)}] {c.name}" + (f" ({c.unit})" if c.unit else "")
            out[label] = [b.table.value(n, c.name) if n in b.table.values.index else None for n in names]
        out.index = [("★ " if n in b.references else "") + n for n in names]
        return out

    def _sources(self):
        """(source name, label, particle?) for every sheet that gives computed items."""
        p = self.project
        out = []
        if len(p.particles):
            name, cat = p.particle_sheet()
            out.append((name, f"{name} (입자별, 범주 {cat})", True))
        out += [(m.name, f"{m.name} (범주 {m.category})", False) for m in p.measures]
        return out

    def _fill_kind_checks(self):
        while self.kinds_lay.count():
            w = self.kinds_lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.kind_checks = {}
        kinds = self.project.settings.bm_kinds
        for src, label, particle in self._sources():
            self.kinds_lay.addWidget(QtWidgets.QLabel(f"<b>{label}</b>"))
            for kind in (KIND_TREND, KIND_MATCHED, KIND_LEVEL):
                cb = QtWidgets.QCheckBox(kind)
                cb.setToolTip(KIND_HELP[kind])
                cb.setChecked(kind_on(kinds, src, kind, particle))
                cb.toggled.connect(self._kinds_toggled)
                self.kind_checks[f"{src}:{kind}"] = cb
                self.kinds_lay.addWidget(cb)
        if not self.kind_checks:
            self.kinds_lay.addWidget(info_label("조건별 측정 시트나 입자 데이터가 아직 없습니다."))

    def refresh(self):
        p = self.project
        if p is None:
            return
        fill_table(self.table, self.batch_frame(), index_label="배치")
        self.table.resizeColumnsToContents()
        self._fill_kind_checks()
        cur = self.f_cat.currentText()
        self.f_cat.blockSignals(True); self.f_cat.clear(); self.f_cat.addItems([ALL] + list(p.batches.categories))
        i = self.f_cat.findText(cur); self.f_cat.setCurrentIndex(max(i, 0)); self.f_cat.blockSignals(False)
        self.fill_derived()

    def conditions_changed(self):
        self.refresh()

    def set_descriptors(self, desc):
        self.desc = desc
        self.fill_derived()

    def fill_derived(self):
        t = self.derived
        d = self.desc
        if d is None:
            t.clear(); t.setRowCount(0); t.setColumnCount(0)
            self.derived_note.setText("배치 지도를 계산하면 여기에 나옵니다 (배치 지도 탭 또는 ⌘B).")
            return
        meta = d.meta[d.meta["kind"] != KIND_DIRECT]
        if self.f_cat.currentText() not in ("", ALL):
            meta = meta[meta["category"] == self.f_cat.currentText()]
        if self.f_kind.currentText() != ALL:
            meta = meta[meta["kind"] == self.f_kind.currentText()]
        q = self.f_text.text().strip().lower()
        if q:
            meta = meta[[q in str(i).lower() for i in meta.index]]
        batches = list(d.values.index)
        cols = ["범주", "종류", "단위"] + batches
        t.setRowCount(len(meta)); t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setVerticalHeaderLabels([str(i) for i in meta.index])
        for r, (it, m) in enumerate(meta.iterrows()):
            for c, v in enumerate((m["category"], m["kind"], m["unit"])):
                t.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
            for c, b in enumerate(batches, start=3):
                v = d.values.at[b, it]
                txt = "—" if pd.isna(v) else (f"{v:.3g}" if isinstance(v, float) else str(v))
                cell = QtWidgets.QTableWidgetItem(txt)
                cell.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
                t.setItem(r, c, cell)
        t.resizeColumnsToContents()
        allm = d.meta[d.meta["kind"] != KIND_DIRECT]
        by = allm.groupby("kind").size()
        self.derived_note.setText(f"계산한 항목 {len(allm)}개 (" + ", ".join(f"{k} {n}" for k, n in by.items()) + f") · 표시 {len(meta)}개")

    def ui_state(self):
        return {"cat": self.f_cat.currentText(), "kind": self.f_kind.currentText()}

    def restore_ui(self, st):
        if not st:
            return
        for cb, k in ((self.f_cat, "cat"), (self.f_kind, "kind")):
            i = cb.findText(st.get(k, ""))
            if i >= 0:
                cb.setCurrentIndex(i)

    def _kinds_toggled(self):
        self.project.settings.bm_kinds = {k: cb.isChecked() for k, cb in self.kind_checks.items()}
        self.kinds_changed.emit()

    def descriptors_frame(self):
        """Batch x every item (typed and computed), for saving."""
        if self.desc is None:
            return None
        return self.desc.values
