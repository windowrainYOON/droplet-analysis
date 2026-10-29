"""Group condition editor: the groups made from each batch's protein, in two tabs (particle
formation, cell delivery), each with its own conditions. A group always belongs to one batch; the
batch's values (edited on the batch table page) show here as grey, read-only columns. Blank cells
mean "not recorded". The table model and editing helpers are shared with the batch table page."""
from __future__ import annotations

import json

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.batches import UNASSIGNED
from ..core.conditions import ConditionDef, KINDS, KIND_LABELS, fill_from_group_names
from ..core.delivery import fill_delivery_from_names
from .common import flow_buttons, info_label, scroll_panel, section, title_label, warn

ALL_BATCHES = "모든 배치"
INHERITED = QtGui.QColor("#8a8f99")
BLANK = QtGui.QColor("#b8bcc4")
UNASSIGNED_BG = QtGui.QColor("#fff3d6")


class GroupTableModel(QtCore.QAbstractTableModel):
    """One of three tables: 'batch' (rows = batches), 'formation' or 'delivery' (rows = groups).
    Columns are (kind, key): ref, name, batch, count, inh <batch condition>, own <condition>, uses."""
    edited = QtCore.Signal()

    def __init__(self, which, parent=None):
        super().__init__(parent)
        self.which = which
        self.project = None
        self.filter = ALL_BATCHES
        self.rows, self.cols = [], []

    # ---- structure ----------------------------------------------------
    def table(self):
        p = self.project
        return {"batch": p.batches.table, "formation": p.conditions, "delivery": p.delivery.conditions}[self.which]

    def set_project(self, project):
        self.project = project
        self.refresh()

    def refresh(self):
        self.beginResetModel()
        p = self.project
        self.rows, self.cols = [], []
        if p is not None:
            if self.which == "batch":
                self.rows = p.batches.names()
                self.cols = [("ref", None), ("name", None), ("n_form", None), ("n_deliv", None)]
                self.cols += [("own", c.name) for c in p.batches.table.schema]
            else:
                groups = p.groups if self.which == "formation" else p.delivery.groups()
                if self.filter != ALL_BATCHES:
                    groups = [g for g in groups if p.batch_of(g) == self.filter]
                self.rows = groups
                self.cols = ([("ref", None)] if self.which == "formation" else []) + [("name", None), ("batch", None), ("count", None)]
                self.cols += [("inh", c.name) for c in p.batches.table.schema]
                if self.which == "delivery":
                    self.cols.append(("uses", None))
                self.cols += [("own", c.name) for c in self.table().schema]
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.cols)

    def editable(self, col) -> bool:
        kind = self.cols[col][0]
        return kind in ("own", "batch", "uses") or (kind == "name" and self.which == "batch")

    def condition(self, col):
        kind, key = self.cols[col]
        return self.table().get(key) if kind == "own" else None

    # ---- display ------------------------------------------------------
    def headerData(self, section, orient, role=QtCore.Qt.DisplayRole):
        if orient == QtCore.Qt.Vertical:
            return str(section + 1) if role == QtCore.Qt.DisplayRole else None
        kind, key = self.cols[section]
        if role == QtCore.Qt.ToolTipRole:
            return {"inh": "배치 조건 (배치 표에서 고칩니다)", "batch": "이 그룹을 만든 단백질의 정제 배치",
                    "uses": "이 세포 전달 실험에 쓴 입자를 만든 입자 형성 그룹",
                    "ref": "기준: 원하는 상태"}.get(kind)
        if role == QtCore.Qt.ForegroundRole and kind == "inh":
            return INHERITED
        if role != QtCore.Qt.DisplayRole:
            return None
        if kind == "ref":
            return "기준 ★"
        if kind == "name":
            return "배치" if self.which == "batch" else "그룹"
        if kind == "n_form":
            return "입자 형성\n그룹"
        if kind == "n_deliv":
            return "세포 전달\n그룹"
        if kind == "batch":
            return "배치"
        if kind == "count":
            return "입자 수" if self.which == "formation" else "측정 항목"
        if kind == "uses":
            return "사용한 입자\n형성 그룹"
        if kind == "inh":
            c = self.project.batches.table.get(key)
            return f"[배치]\n{c.label}"
        c = self.table().get(key)
        if self.which == "batch":
            return f"[{self.project.batches.category(key)}]\n{c.label}"
        return f"[{KIND_LABELS[c.kind]}]\n{c.label}"

    def data(self, idx, role=QtCore.Qt.DisplayRole):
        if not idx.isValid():
            return None
        p, r = self.project, self.rows[idx.row()]
        kind, key = self.cols[idx.column()]
        if role == QtCore.Qt.BackgroundRole and self.which != "batch" and p.batch_of(r) == UNASSIGNED:
            return UNASSIGNED_BG
        if kind == "ref":
            if role == QtCore.Qt.CheckStateRole:
                refs = p.batches.references if self.which == "batch" else p.conditions.references
                return QtCore.Qt.Checked if r in refs else QtCore.Qt.Unchecked
            return None
        if kind == "name":
            return r if role in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole, QtCore.Qt.EditRole) else None
        if kind in ("count", "n_form", "n_deliv"):
            if role == QtCore.Qt.TextAlignmentRole:
                return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            if role != QtCore.Qt.DisplayRole:
                return None
            if kind == "n_form":
                return str(len(p.batches.groups_in(r, p.groups)))
            if kind == "n_deliv":
                return str(len(p.batches.groups_in(r, p.delivery.groups())))
            if self.which == "formation":
                return f"{self.parent_counts().get(r, 0):,}"
            return str(int((p.delivery.summary["group"] == r).sum()))
        if kind == "batch":
            b = p.batch_of(r)
            if role == QtCore.Qt.DisplayRole:
                return b
            if role == QtCore.Qt.EditRole:
                return "" if b == UNASSIGNED else b
            if role == QtCore.Qt.ForegroundRole and b == UNASSIGNED:
                return QtGui.QColor("#b05a00")
            return None
        if kind == "uses":
            v = p.delivery.uses.get(r)
            if role == QtCore.Qt.DisplayRole:
                return v or "—"
            if role == QtCore.Qt.EditRole:
                return v or ""
            if role == QtCore.Qt.ForegroundRole and not v:
                return BLANK
            return None
        if kind == "inh":
            b = p.batch_of(r)
            t = p.batches.table
            v = t.value(b, key) if b in t.values.index else None
            if role == QtCore.Qt.DisplayRole:
                return "—" if v is None else str(v)
            if role == QtCore.Qt.ForegroundRole:
                return BLANK if v is None else INHERITED
            return None
        t = self.table()
        v = t.value(r, key)
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            if v is None:
                return "—" if role == QtCore.Qt.DisplayRole else ""
            return str(v)
        if role == QtCore.Qt.ForegroundRole and v is None:
            return BLANK
        if role == QtCore.Qt.ForegroundRole and t.get(key).kind == "numeric":
            try:
                float(str(v).replace(",", ""))
            except ValueError:
                return QtGui.QColor("#d6456b")
        return None

    def parent_counts(self):
        if not hasattr(self, "_counts") or self._counts_for is not self.project.particles:
            self._counts = self.project.particles.groupby("group").size().to_dict()
            self._counts_for = self.project.particles
        return self._counts

    def flags(self, idx):
        f = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if self.cols[idx.column()][0] == "ref":
            return f | QtCore.Qt.ItemIsUserCheckable
        if self.editable(idx.column()):
            return f | QtCore.Qt.ItemIsEditable
        return f

    def setData(self, idx, value, role=QtCore.Qt.EditRole):
        p, r = self.project, self.rows[idx.row()]
        kind, key = self.cols[idx.column()]
        if kind == "ref" and role == QtCore.Qt.CheckStateRole:
            checked = value == QtCore.Qt.Checked or value == QtCore.Qt.Checked.value or value == 2
            refs = p.batches.references if self.which == "batch" else p.conditions.references
            (refs.add if checked else refs.discard)(r)
        elif role != QtCore.Qt.EditRole:
            return False
        elif kind == "own":
            self.table().set_value(r, key, value)
        elif kind == "batch":
            p.batches.assign([r], str(value or ""))
        elif kind == "uses":
            v = str(value or "").strip()
            if v and v not in p.groups:
                return False
            if v:
                p.delivery.uses[r] = v
            else:
                p.delivery.uses.pop(r, None)
        elif kind == "name" and self.which == "batch":
            try:
                p.batches.rename(r, str(value))
            except ValueError as e:
                warn(None, str(e)); return False
            self.refresh(); self.edited.emit()
            return True
        else:
            return False
        self.dataChanged.emit(idx, idx)
        self.edited.emit()
        return True


class ChoiceDelegate(QtWidgets.QStyledItemDelegate):
    """Drop-down for the batch column (editable: typing a new name makes a new batch) and for the
    'used particle group' column."""

    def __init__(self, choices, editable, parent=None):
        super().__init__(parent)
        self.choices, self.editable_combo = choices, editable

    def createEditor(self, parent, option, index):
        cb = QtWidgets.QComboBox(parent)
        cb.setEditable(self.editable_combo)
        cb.addItems(self.choices())
        return cb

    def setEditorData(self, editor, index):
        v = index.data(QtCore.Qt.EditRole) or ""
        i = editor.findText(v)
        if i >= 0:
            editor.setCurrentIndex(i)
        else:
            editor.setEditText(v)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText())


class ConditionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, cond: ConditionDef | None = None, where=""):
        super().__init__(parent)
        self.setWindowTitle(("조건 편집" if cond else "새 조건 추가") + (f" — {where}" if where else ""))
        form = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit(cond.name if cond else "")
        self.name.setPlaceholderText("예: Glycerol")
        self.kind = QtWidgets.QComboBox()
        for k in KINDS:
            self.kind.addItem(KIND_LABELS[k], k)
        if cond:
            self.kind.setCurrentIndex(KINDS.index(cond.kind))
        self.unit = QtWidgets.QLineEdit(cond.unit if cond else "")
        self.unit.setPlaceholderText("예: %, mM, µM, °C (비워도 됨)")
        form.addRow("이름", self.name)
        form.addRow("종류", self.kind)
        form.addRow("단위", self.unit)
        form.addRow(info_label("수치: 농도·pH·시간처럼 크기 순서가 있는 값. 범주: 변이체·세포 종류처럼 이름으로 구분하는 값. 칸은 비워 둘 수 있습니다."))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok); bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _ok(self):
        if not self.name.text().strip():
            warn(self, "조건 이름을 입력하세요.")
            return
        self.accept()

    def result_def(self) -> ConditionDef:
        return ConditionDef(self.name.text().strip(), self.kind.currentData(), self.unit.text().strip())


class ConditionTableView(QtWidgets.QTableView):
    """Paste blocks from Excel, clear with Delete."""
    focused = QtCore.Signal(object)

    def keyPressEvent(self, e):
        if e.matches(QtGui.QKeySequence.Paste):
            self.parent_page.paste(self); return
        if e.matches(QtGui.QKeySequence.Copy):
            self.parent_page.copy(self); return
        if e.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self.parent_page.clear_selected(self); return
        super().keyPressEvent(e)

    def focusInEvent(self, e):
        self.focused.emit(self)
        super().focusInEvent(e)


TABLE_LABELS = {"batch": "배치 표", "formation": "입자 형성 그룹 표", "delivery": "세포 전달 그룹 표"}


class TableEditing:
    """Copy / paste / fill / clear on ConditionTableView tables. Needs self.view, self.active,
    self._after_edit."""

    def _selected_rows(self, table):
        m = table.model()
        return [m.rows[r] for r in sorted({i.row() for i in table.selectedIndexes()})]

    def _editable_selection(self, table=None):
        table = table or self.active
        m = table.model()
        return [i for i in table.selectedIndexes() if m.cols[i.column()][0] in ("own", "uses")]

    def fill_selected(self):
        sel = self._editable_selection()
        if not sel:
            warn(self.view, "값을 넣을 칸을 먼저 고르세요 (⌘ 또는 Shift로 여러 칸)."); return
        text, ok = QtWidgets.QInputDialog.getText(self.view, "선택 칸 한 번에 입력", f"{len(sel)}칸에 넣을 값 (비우면 '기록 안 함'):")
        if ok:
            m = self.active.model()
            for i in sel:
                m.setData(i, text)
            self._after_edit()

    def clear_selected(self, table=None):
        table = table or self.active
        for i in self._editable_selection(table):
            table.model().setData(i, "")
        self._after_edit()

    def copy(self, table=None):
        table = table or self.active
        sel = table.selectedIndexes()
        if not sel:
            return
        m = table.model()
        rows = sorted({i.row() for i in sel}); cols = sorted({i.column() for i in sel})
        lines = []
        for r in rows:
            vals = []
            for c in cols:
                v = m.data(m.index(r, c), QtCore.Qt.EditRole)
                if v is None:
                    v = m.data(m.index(r, c), QtCore.Qt.DisplayRole)
                vals.append("" if v is None else str(v))
            lines.append("\t".join(vals))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def paste(self, table=None):
        table = table or self.active
        text = QtWidgets.QApplication.clipboard().text()
        if not text:
            return
        m = table.model()
        cur = table.currentIndex()
        if not cur.isValid() or not m.editable(cur.column()):
            warn(self.view, "붙여넣을 위치(조건 칸)를 먼저 고르세요."); return
        rows = [r.split("\t") for r in text.rstrip("\n").split("\n")]
        for dr, vals in enumerate(rows):
            for dc, v in enumerate(vals):
                r, c = cur.row() + dr, cur.column() + dc
                if r < m.rowCount() and c < m.columnCount() and m.editable(c) and m.cols[c][0] != "name":
                    m.setData(m.index(r, c), v.strip())
        self._after_edit()


class ConditionsPage(QtCore.QObject, TableEditing):
    """Center: group tables (formation, delivery). Right: editing tools. Files live in the 데이터 menu."""
    changed = QtCore.Signal()
    title = "그룹 조건"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.models, self.tables = {}, {}
        self.view = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.view)
        lay.addWidget(title_label("그룹 조건"))
        bottom = QtWidgets.QWidget(); bl = QtWidgets.QVBoxLayout(bottom); bl.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("<b>그룹</b> (배치에서 나온 단백질로 한 실험)"))
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("보기:"))
        self.batch_filter = QtWidgets.QComboBox()
        self.batch_filter.currentIndexChanged.connect(self._filter_changed)
        row.addWidget(self.batch_filter)
        bl.addLayout(row)
        self.group_tabs = QtWidgets.QTabWidget()
        self.group_tabs.addTab(self._make_table("formation"), "입자 형성")
        self.group_tabs.addTab(self._make_table("delivery"), "세포 전달")
        self.group_tabs.currentChanged.connect(lambda i: self._set_active(self.tables["formation" if i == 0 else "delivery"]))
        bl.addWidget(self.group_tabs, 1)
        lay.addWidget(bottom, 1)
        self.coverage = QtWidgets.QLabel(); self.coverage.setStyleSheet("color:#666;"); self.coverage.setWordWrap(True)
        lay.addWidget(self.coverage)
        self.active = self.tables["formation"]

        self.active_label = QtWidgets.QLabel(""); self.active_label.setWordWrap(True)
        batches = section("배치",
                          info_label("그룹은 모두 정제 배치 하나에 속합니다. 배치 추가·삭제와 배치 값(정제 조건, 제타 포텐셜 …)은 "
                                     "'배치 표' 탭에서 고칩니다. 회색 [배치] 열은 그 값입니다. 그룹 조건은 배치마다 "
                                     "'조건에 따른 변화(경향)'를 계산하는 데 쓰입니다."),
                          flow_buttons((("선택 그룹 → 배치 지정…", self.assign_batch),)))
        cols = section("조건 열",
                       info_label("아래 버튼은 마지막으로 누른 표에 적용됩니다."), self.active_label,
                       flow_buttons((("+ 조건 추가", self.add_condition), ("조건 편집", self.edit_condition),
                                     ("조건 삭제", self.remove_condition))))
        vals = section("값 입력",
                       info_label("빈칸(—)은 '기록 안 함'으로 저장되며 0으로 채우지 않습니다. 엑셀 표를 붙여넣을 수 있고(⌘V), "
                                  "여러 칸을 골라 한 번에 입력하거나 Delete로 비울 수 있습니다. 회색 [배치] 열은 배치 표에서 고칩니다."),
                       flow_buttons((("선택 칸 한 번에 입력…", self.fill_selected), ("선택 칸 비우기", lambda: self.clear_selected(None)),
                                     ("그룹 이름에서 조건 채우기…", self.fill_from_names))))
        deliv = section("세포 전달",
                        info_label("'사용한 입자 형성 그룹'에 이 실험에 쓴 입자를 만든 그룹을 고르면, 세포 전달 결과와 "
                                   "그 입자의 모양·score를 나란히 볼 수 있습니다. 세포 전달 요약 CSV는 '파일 › 불러오기'나 "
                                   "끌어 놓기로 넣습니다."))
        ref = section("기준", info_label("입자 형성 표의 ★ = 기준 그룹(원하는 상태, 입자 수준 score에 씁니다). "
                                        "왼쪽 목록에서 오른쪽 클릭으로도 지정합니다."))
        files = section("파일", info_label("조건 CSV·템플릿 불러오기와 저장은 메뉴 막대의 '데이터' 메뉴에 있습니다."))
        self.panel = scroll_panel(batches, cols, vals, deliv, ref, files)
        self._set_active(self.active)

    @property
    def model(self):
        """Particle formation table (kept for callers that refresh after reference changes)."""
        return _Refresher(self)

    def _make_table(self, which):
        m = GroupTableModel(which, self)
        m.edited.connect(self._edited)
        t = ConditionTableView()
        t.parent_page = self
        t.setModel(m)
        t.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        t.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)
        t.horizontalHeader().sectionDoubleClicked.connect(lambda s, tt=t: self._header_double(tt, s))
        t.setAlternatingRowColors(True)
        t.setWordWrap(True)
        t.focused.connect(self._set_active)
        t.clicked.connect(lambda _i, tt=t: self._set_active(tt))
        self.models[which], self.tables[which] = m, t
        return t

    def _set_active(self, table):
        self.active = table
        which = table.model().which
        self.active_label.setText(f"지금 표: <b>{TABLE_LABELS[which]}</b>")

    def _install_delegates(self):
        p = self.project
        for which in ("formation", "delivery"):
            m, t = self.models[which], self.tables[which]
            for c in range(m.columnCount()):
                t.setItemDelegateForColumn(c, None)
            for c, (kind, _k) in enumerate(m.cols):
                if kind == "batch":
                    t.setItemDelegateForColumn(c, ChoiceDelegate(lambda: [b for b in p.batches.names() if b != UNASSIGNED], True, t))
                elif kind == "uses":
                    t.setItemDelegateForColumn(c, ChoiceDelegate(lambda: [""] + p.groups, False, t))

    def set_project(self, project):
        self.project = project
        for m in self.models.values():
            m.project = project
        self._fill_filter()
        self.refresh_all()

    def _fill_filter(self):
        cur = self.batch_filter.currentText()
        self.batch_filter.blockSignals(True); self.batch_filter.clear()
        self.batch_filter.addItem(ALL_BATCHES)
        for b in self.project.batches.names():
            self.batch_filter.addItem(b)
        i = self.batch_filter.findText(cur); self.batch_filter.setCurrentIndex(max(i, 0))
        self.batch_filter.blockSignals(False)

    def _filter_changed(self):
        f = self.batch_filter.currentText() or ALL_BATCHES
        for which in ("formation", "delivery"):
            self.models[which].filter = f
        self.refresh_all()

    def refresh_all(self):
        if self.project is None:
            return
        for m in self.models.values():
            m.refresh()
        self._install_delegates()
        for which, t in self.tables.items():
            t.resizeColumnsToContents()
            name_col = next((i for i, c in enumerate(self.models[which].cols) if c[0] == "name"), None)
            if name_col is not None:
                t.setColumnWidth(name_col, min(t.columnWidth(name_col), 340))
        n_d = len(self.project.delivery.groups())
        self.group_tabs.setTabText(1, f"세포 전달 ({n_d})" if n_d else "세포 전달")
        self._update_coverage()

    def _update_coverage(self):
        p = self.project
        t = p.conditions
        un = p.batches.unassigned(p.all_groups())
        parts = [f"배치 {len([b for b in p.batches.names() if b != UNASSIGNED])}개"]
        if un:
            parts.append(f"<span style='color:#b05a00'>배치 미지정 그룹 {len(un)}개</span>")
        n = len(t.values)
        parts += [f"{c.name}: {t.recorded_count(c.name)}/{n}" for c in t.schema]
        parts.append(f"기준 그룹 {len(t.references)}개")
        if p.batches.references:
            parts.append("기준 배치: " + ", ".join(sorted(p.batches.references)))
        self.coverage.setText(" · ".join(parts))

    def _edited(self):
        self._fill_filter()
        self.refresh_all()
        self.changed.emit()

    def _after_edit(self):
        self._edited()

    # ---- batches --------------------------------------------------------
    def assign_batch(self):
        table = self.tables["formation" if self.group_tabs.currentIndex() == 0 else "delivery"]
        groups = self._selected_rows(table)
        if not groups:
            warn(self.view, "아래 그룹 표에서 배치를 정할 그룹을 먼저 고르세요 (⌘ 또는 Shift로 여러 개)."); return
        names = [b for b in self.project.batches.names() if b != UNASSIGNED]
        b, ok = QtWidgets.QInputDialog.getItem(self.view, "배치 지정", f"그룹 {len(groups)}개를 넣을 배치 (새 이름도 됩니다):",
                                               names, 0, True)
        if ok and b.strip():
            self.project.batches.assign(groups, b)
            self._after_edit()

    # ---- condition columns ----------------------------------------------
    def _active_table(self):
        return self.active.model().table(), TABLE_LABELS[self.active.model().which]

    def add_condition(self):
        t, where = self._active_table()
        dlg = ConditionDialog(self.view, where=where)
        if dlg.exec():
            try:
                t.add_condition(dlg.result_def())
            except ValueError as e:
                warn(self.view, str(e)); return
            self._after_edit()

    def _current_condition(self, table=None, section=None):
        table = table or self.active
        m = table.model()
        col = section if isinstance(section, int) else table.currentIndex().column()
        if col is None or col < 0 or col >= m.columnCount():
            return None
        return m.condition(col)

    def _header_double(self, table, section):
        self._set_active(table)
        if table.model().cols[section][0] == "inh":
            warn(self.view, "회색 [배치] 열은 '배치 표' 탭에서 고칩니다.")
            return
        self.edit_condition(section)

    def edit_condition(self, section=None):
        c = self._current_condition(section=section)
        if c is None:
            warn(self.view, "편집할 조건 열의 칸을 먼저 고르세요."); return
        t, where = self._active_table()
        dlg = ConditionDialog(self.view, c, where)
        if dlg.exec():
            new = dlg.result_def()
            if new.name != c.name and new.name in t.names():
                warn(self.view, "같은 이름의 조건이 이미 있습니다."); return
            t.update_condition(c.name, new)
            self._after_edit()

    def remove_condition(self):
        c = self._current_condition()
        if c is None:
            warn(self.view, "삭제할 조건 열의 칸을 먼저 고르세요."); return
        t, where = self._active_table()
        if QtWidgets.QMessageBox.question(self.view, "조건 삭제", f"{where}의 '{c.name}' 조건과 입력한 값을 모두 지울까요?") == QtWidgets.QMessageBox.Yes:
            t.remove_condition(c.name)
            self._after_edit()

    def fill_from_names(self):
        which = "formation" if self.group_tabs.currentIndex() == 0 else "delivery"
        if which == "formation":
            text = ("입자 형성 그룹 이름의 pH · NaCl(mM) · PEG(%) 값을 빈칸에만 넣습니다.\n"
                    "이미 입력한 값은 그대로 두고, 이름에서 찾지 못한 칸은 빈칸(—)으로 남깁니다.")
        else:
            text = ("세포 전달 그룹 이름에서 '1uM 물질', '10% PEG', '10min' 같은 값을 찾아 빈칸에만 넣습니다.\n"
                    "이미 입력한 값은 그대로 둡니다.")
        if QtWidgets.QMessageBox.question(self.view, "그룹 이름에서 조건 채우기", text + "\n계속할까요?") != QtWidgets.QMessageBox.Yes:
            return
        if which == "formation":
            filled = fill_from_group_names(self.project.conditions)
        else:
            filled = fill_delivery_from_names(self.project.delivery.conditions)
        self._after_edit()
        if not filled:
            warn(self.view, "새로 채울 빈칸이 없습니다 (이름에서 값을 찾지 못했거나 이미 입력됨).")

    # ---- files ----------------------------------------------------------
    def import_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.view, "조건 CSV 불러오기", "", "CSV (*.csv *.txt)")
        if path:
            self.import_csv_path(path)

    def import_csv_path(self, path) -> list:
        """A table whose first column names groups (formation or delivery) or batches. A 'batch'/'배치'
        column assigns groups to batches."""
        import io as _io
        import pandas as pd
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
        df = pd.read_csv(_io.StringIO(text), dtype=str, keep_default_na=False)
        p = self.project
        first = df.columns[0]
        keys = set(df[first].astype(str).str.strip())
        bcol = next((c for c in df.columns[1:] if c.strip().lower() in ("batch", "배치", "정제 배치")), None)
        if bcol:
            for _, row in df.iterrows():
                g = str(row[first]).strip()
                if g in set(p.all_groups()) and str(row[bcol]).strip():
                    p.batches.assign([g], str(row[bcol]).strip())
            df = df.drop(columns=[bcol])
        if keys & set(p.batches.names()) and not keys & set(p.all_groups()):
            unmatched = p.batches.table.merge_frame(df)
        elif keys & set(p.delivery.groups()) and not keys & set(p.groups):
            unmatched = p.delivery.conditions.merge_frame(df)
        else:
            unmatched = p.conditions.merge_frame(df)
        self._after_edit()
        if unmatched:
            warn(self.view, "이 이름은 데이터에 없어 건너뛰었습니다:\n" + "\n".join(unmatched[:20]))
        return unmatched

    def export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.view, "조건 CSV 저장", "conditions.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(self.conditions_csv())

    def conditions_csv(self) -> str:
        from .export_page import _conditions_csv
        return _conditions_csv(self.project)

    def save_template(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.view, "조건 목록 템플릿 저장", "conditions_template.json", "JSON (*.json)")
        if path:
            p = self.project
            data = {"format": "droplet-map-condition-template",
                    "schema": [c.__dict__ for c in p.conditions.schema],
                    "batch_schema": [c.__dict__ for c in p.batches.table.schema],
                    "delivery_schema": [c.__dict__ for c in p.delivery.conditions.schema]}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def load_template(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.view, "템플릿 불러오기", "", "JSON (*.json)")
        if path:
            self.load_template_path(path)

    def load_template_path(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        p = self.project
        for key, table in (("schema", p.conditions), ("batch_schema", p.batches.table),
                           ("delivery_schema", p.delivery.conditions)):
            for c in data.get(key, []):
                if c["name"] not in table.names():
                    table.add_condition(ConditionDef(**c))
        self._after_edit()


class _Refresher:
    def __init__(self, page):
        self.page = page

    def refresh(self):
        self.page.refresh_all()
