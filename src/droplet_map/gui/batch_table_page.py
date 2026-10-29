"""Batch table: one row per purification batch. Top: values typed per batch (purification
conditions, zeta potential, ...), each in a category. Bottom: the items computed from each batch's
experiments (trends per condition unit, matched conditions, batch medians) that, together with the
typed values, place the batch on the batch map."""
from __future__ import annotations

import pandas as pd
from PySide6 import QtCore, QtWidgets

from ..core.batch_map import DEFAULT_SOURCES, KIND_DIRECT, KIND_LEVEL, KIND_MATCHED, KIND_TREND, SRC_DELIVERY, SRC_PARTICLE
from ..core.batches import UNASSIGNED
from .common import flow_buttons, info_label, scroll_panel, section, title_label, warn
from .conditions_page import ConditionDialog, ConditionTableView, GroupTableModel, TableEditing

ALL = "모두"
KIND_HELP = {
    KIND_TREND: "배치 안에서 조건 한 단위당 측정값이 얼마나 변하는지 (예: pH 1 오를 때 circularity 변화). "
                "배치마다 실험한 조건이 달라도 비교할 수 있습니다.",
    KIND_MATCHED: "두 배치 이상이 같은 조건 세트로 실험한 경우, 그 조건에서의 값.",
    KIND_LEVEL: "배치의 모든 그룹 값의 중앙값. 배치마다 실험한 조건이 다르면 그 차이도 섞입니다.",
}


class ItemDialog(ConditionDialog):
    """Name, kind, unit + the category the item describes."""

    def __init__(self, parent, categories, cond=None, category=None):
        super().__init__(parent, cond, "배치 표")
        self.setWindowTitle("항목 편집" if cond else "새 항목 추가")
        self.name.setPlaceholderText("예: 제타 포텐셜")
        self.cat = QtWidgets.QComboBox()
        self.cat.addItems(categories)
        if category in categories:
            self.cat.setCurrentText(category)
        self.layout().insertRow(0, "범주", self.cat)

    def category(self) -> str:
        return self.cat.currentText()


class BatchTablePage(QtCore.QObject, TableEditing):
    changed = QtCore.Signal()          # batch values, items or categories changed
    kinds_changed = QtCore.Signal()    # which computed items are made
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
        self.model = GroupTableModel("batch", self)
        self.model.edited.connect(self._edited)
        self.table = ConditionTableView()
        self.table.parent_page = self
        self.table.setModel(self.model)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self.edit_item)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.active = self.table
        top = QtWidgets.QWidget(); tl = QtWidgets.QVBoxLayout(top); tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(QtWidgets.QLabel("<b>직접 입력한 값</b> (열 머리 = [범주] 항목, 두 번 누르면 편집)"))
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

        batches = section("배치",
                          info_label("배치 이름은 표에서 두 번 눌러 바꿉니다. ★ = 기준 배치 (배치 지도의 score 기준)."),
                          flow_buttons((("+ 배치 추가", self.add_batch), ("배치 삭제", self.remove_batch))))
        self.cat_list = QtWidgets.QListWidget(); self.cat_list.setMaximumHeight(120)
        cats = section("범주",
                       info_label("배치를 설명하는 값의 큰 묶음입니다. 배치 지도에서 범주마다 무게를 줄 수 있습니다."),
                       self.cat_list,
                       flow_buttons((("+ 범주 추가", self.add_category), ("이름 바꾸기", self.rename_category),
                                     ("범주 삭제", self.remove_category))))
        items = section("직접 입력 항목",
                        info_label("배치마다 값이 하나인 항목입니다 (예: 정제일, 단백질 농도, 제타 포텐셜, 항체 전달 효율). "
                                   "날짜(2026-03-02)는 날짜 차이로 계산합니다."),
                        flow_buttons((("+ 항목 추가", self.add_item), ("항목 편집", self.edit_item),
                                      ("항목 삭제", self.remove_item))))
        vals = section("값 입력",
                       info_label("빈칸(—)은 '기록 안 함'이고 0으로 채우지 않습니다. 엑셀 표를 붙여넣을 수 있습니다(⌘V)."),
                       flow_buttons((("선택 칸 한 번에 입력…", self.fill_selected), ("선택 칸 비우기", lambda: self.clear_selected(None)))))
        self.kind_checks = {}
        kinds_w = QtWidgets.QWidget(); kl = QtWidgets.QVBoxLayout(kinds_w); kl.setContentsMargins(0, 0, 0, 0)
        for src, label in ((SRC_PARTICLE, "입자 데이터 (in vitro 관측)"), (SRC_DELIVERY, "세포 전달 데이터")):
            kl.addWidget(QtWidgets.QLabel(f"<b>{label}</b>"))
            for kind in (KIND_TREND, KIND_MATCHED, KIND_LEVEL):
                cb = QtWidgets.QCheckBox(kind)
                cb.setToolTip(KIND_HELP[kind])
                cb.toggled.connect(self._kinds_toggled)
                self.kind_checks[f"{src}:{kind}"] = cb
                kl.addWidget(cb)
        calc = section("실험 데이터에서 계산", kinds_w,
                       info_label(f"경향: {KIND_HELP[KIND_TREND]}"),
                       info_label(f"같은 조건: {KIND_HELP[KIND_MATCHED]}"),
                       info_label(f"대표값: {KIND_HELP[KIND_LEVEL]}"),
                       info_label("입자 값은 전체 입자 기준 robust z(형태값은 크기 신뢰도 가중)의 그룹 중앙값입니다. "
                                  "그룹 조건은 '그룹 조건' 탭에서 넣습니다."))
        self.panel = scroll_panel(batches, cats, items, vals, calc)

    # ---- project ----------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.desc = None
        self.model.set_project(project)
        self.refresh()

    def refresh(self):
        p = self.project
        if p is None:
            return
        self.model.refresh()
        self.table.resizeColumnsToContents()
        self.cat_list.clear()
        for c in p.batches.categories:
            n = sum(1 for col in p.batches.table.names() if p.batches.category(col) == c)
            self.cat_list.addItem(f"{c}  (직접 입력 {n}개)")
        kinds = {**DEFAULT_SOURCES, **p.settings.bm_kinds}
        for k, cb in self.kind_checks.items():
            cb.blockSignals(True); cb.setChecked(bool(kinds.get(k, True))); cb.blockSignals(False)
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

    # ---- editing ------------------------------------------------------------
    def _edited(self):
        self.refresh()
        self.changed.emit()

    _after_edit = _edited

    def _kinds_toggled(self):
        self.project.settings.bm_kinds = {k: cb.isChecked() for k, cb in self.kind_checks.items()}
        self.kinds_changed.emit()

    def add_batch(self):
        name, ok = QtWidgets.QInputDialog.getText(self.view, "배치 추가", "새 배치 이름 (예: 정제 로트 번호):")
        if ok and name.strip():
            try:
                self.project.batches.add(name)
            except ValueError as e:
                warn(self.view, str(e)); return
            self._edited()

    def remove_batch(self):
        sel = [b for b in self._selected_rows(self.table) if b != UNASSIGNED]
        if not sel:
            warn(self.view, "표에서 지울 배치를 먼저 고르세요."); return
        b = sel[0]
        p = self.project
        others = [x for x in p.batches.names() if x not in (b, UNASSIGNED)]
        n = len(p.batches.groups_in(b, p.all_groups()))
        target = UNASSIGNED
        if n:
            target, ok = QtWidgets.QInputDialog.getItem(self.view, "배치 삭제", f"'{b}'의 그룹 {n}개를 어느 배치로 옮길까요?",
                                                        [UNASSIGNED] + others, 0, False)
            if not ok:
                return
        elif QtWidgets.QMessageBox.question(self.view, "배치 삭제", f"'{b}' 배치를 지울까요?") != QtWidgets.QMessageBox.Yes:
            return
        p.batches.remove(b, target)
        self._edited()

    def _selected_category(self):
        r = self.cat_list.currentRow()
        cats = self.project.batches.categories
        return cats[r] if 0 <= r < len(cats) else None

    def add_category(self):
        name, ok = QtWidgets.QInputDialog.getText(self.view, "범주 추가", "새 범주 이름 (예: 안정성, 구조 분석):")
        if ok and name.strip():
            try:
                self.project.batches.add_category(name)
            except ValueError as e:
                warn(self.view, str(e)); return
            self._edited()

    def rename_category(self):
        c = self._selected_category()
        if c is None:
            warn(self.view, "범주 목록에서 먼저 고르세요."); return
        name, ok = QtWidgets.QInputDialog.getText(self.view, "범주 이름 바꾸기", "새 이름:", text=c)
        if ok and name.strip():
            try:
                self.project.batches.rename_category(c, name)
            except ValueError as e:
                warn(self.view, str(e)); return
            w = self.project.settings.bm_weights
            if c in w:
                w[name.strip()] = w.pop(c)
            self._edited()

    def remove_category(self):
        c = self._selected_category()
        if c is None:
            warn(self.view, "범주 목록에서 먼저 고르세요."); return
        b = self.project.batches
        if c in ("in vitro 관측", "세포 전달"):
            warn(self.view, f"'{c}'에는 실험 데이터에서 계산한 값이 들어가므로 지울 수 없습니다. 이름은 바꿀 수 있습니다."); return
        if QtWidgets.QMessageBox.question(self.view, "범주 삭제", f"'{c}' 범주를 지울까요? 그 범주의 직접 입력 항목은 "
                                          f"'{[x for x in b.categories if x != c][0]}'(으)로 옮깁니다.") != QtWidgets.QMessageBox.Yes:
            return
        try:
            b.remove_category(c)
        except ValueError as e:
            warn(self.view, str(e)); return
        self._edited()

    def _current_item(self, section=None):
        col = section if isinstance(section, int) else self.table.currentIndex().column()
        if col is None or col < 0 or col >= self.model.columnCount():
            return None
        return self.model.condition(col)

    def add_item(self):
        b = self.project.batches
        dlg = ItemDialog(self.view, b.categories, category=self._selected_category())
        if dlg.exec():
            c = dlg.result_def()
            try:
                b.table.add_condition(c)
            except ValueError as e:
                warn(self.view, str(e)); return
            b.set_category(c.name, dlg.category())
            self._edited()

    def edit_item(self, section=None):
        c = self._current_item(section)
        if c is None:
            if section is None or section is False:
                warn(self.view, "편집할 항목 열의 칸을 먼저 고르세요.")
            return
        b = self.project.batches
        dlg = ItemDialog(self.view, b.categories, c, b.category(c.name))
        if dlg.exec():
            new = dlg.result_def()
            if new.name != c.name and new.name in b.table.names():
                warn(self.view, "같은 이름의 항목이 이미 있습니다."); return
            b.table.update_condition(c.name, new)
            b.rename_column(c.name, new.name)
            b.set_category(new.name, dlg.category())
            self._edited()

    def remove_item(self):
        c = self._current_item()
        if c is None:
            warn(self.view, "삭제할 항목 열의 칸을 먼저 고르세요."); return
        if QtWidgets.QMessageBox.question(self.view, "항목 삭제", f"'{c.name}' 항목과 입력한 값을 모두 지울까요?") == QtWidgets.QMessageBox.Yes:
            self.project.batches.table.remove_condition(c.name)
            self.project.batches.category_of.pop(c.name, None)
            self._edited()

    def descriptors_frame(self):
        """Batch x every item (typed and computed), for saving."""
        if self.desc is None:
            return None
        return self.desc.values
