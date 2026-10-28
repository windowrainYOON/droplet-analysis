"""Condition editor: groups x user-defined conditions. Blank cells mean "not recorded"."""
from __future__ import annotations

import json

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.conditions import ConditionDef, ConditionTable, KINDS, KIND_LABELS, fill_from_group_names
from .common import flow_buttons, info_label, scroll_panel, section, title_label, warn

FIXED = 3  # star, group, particle count


class ConditionModel(QtCore.QAbstractTableModel):
    edited = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table: ConditionTable | None = None
        self.counts = {}

    def set_table(self, table, counts):
        self.beginResetModel()
        self.table, self.counts = table, counts
        self.endResetModel()

    def groups(self):
        return list(self.table.values.index) if self.table is not None else []

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if self.table is None else len(self.table.values)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if self.table is None else FIXED + len(self.table.schema)

    def headerData(self, section, orient, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orient == QtCore.Qt.Vertical:
            return str(section + 1)
        if section == 0:
            return "기준 ★"
        if section == 1:
            return "그룹"
        if section == 2:
            return "입자 수"
        c = self.table.schema[section - FIXED]
        return f"[{KIND_LABELS[c.kind]}]\n{c.label}"

    def data(self, idx, role=QtCore.Qt.DisplayRole):
        if not idx.isValid():
            return None
        g = self.groups()[idx.row()]
        col = idx.column()
        if col == 0:
            if role == QtCore.Qt.CheckStateRole:
                return QtCore.Qt.Checked if g in self.table.references else QtCore.Qt.Unchecked
            return None
        if col == 1:
            return g if role in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole) else None
        if col == 2:
            if role == QtCore.Qt.DisplayRole:
                return f"{self.counts.get(g, 0):,}"
            if role == QtCore.Qt.TextAlignmentRole:
                return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            return None
        name = self.table.schema[col - FIXED].name
        v = self.table.value(g, name)
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            if v is None:
                return "—" if role == QtCore.Qt.DisplayRole else ""
            return str(v)
        if role == QtCore.Qt.ForegroundRole and v is None:
            return QtGui.QColor("#b8bcc4")
        if role == QtCore.Qt.ForegroundRole and self.table.schema[col - FIXED].kind == "numeric":
            try:
                float(str(v).replace(",", ""))
            except ValueError:
                return QtGui.QColor("#d6456b")
        return None

    def flags(self, idx):
        f = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if idx.column() == 0:
            return f | QtCore.Qt.ItemIsUserCheckable
        if idx.column() >= FIXED:
            return f | QtCore.Qt.ItemIsEditable
        return f

    def setData(self, idx, value, role=QtCore.Qt.EditRole):
        g = self.groups()[idx.row()]
        if idx.column() == 0 and role == QtCore.Qt.CheckStateRole:
            checked = value == QtCore.Qt.Checked or value == QtCore.Qt.Checked.value or value == 2
            (self.table.references.add if checked else self.table.references.discard)(g)
        elif idx.column() >= FIXED and role == QtCore.Qt.EditRole:
            self.table.set_value(g, self.table.schema[idx.column() - FIXED].name, value)
        else:
            return False
        self.dataChanged.emit(idx, idx)
        self.edited.emit()
        return True

    def refresh(self):
        self.beginResetModel(); self.endResetModel()


class ConditionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, cond: ConditionDef | None = None):
        super().__init__(parent)
        self.setWindowTitle("조건 편집" if cond else "새 조건 추가")
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
        form.addRow(info_label("수치: 농도·pH·시간처럼 크기 순서가 있는 값. 범주: 배치·변이체처럼 이름으로 구분하는 값. 칸은 비워 둘 수 있습니다."))
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

    def keyPressEvent(self, e):
        if e.matches(QtGui.QKeySequence.Paste):
            self.parent_page.paste(); return
        if e.matches(QtGui.QKeySequence.Copy):
            self.parent_page.copy(); return
        if e.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self.parent_page.clear_selected(); return
        super().keyPressEvent(e)


class ConditionsPage(QtCore.QObject):
    """Center: the condition table. Right: editing tools. File import/export lives in the 조건 menu."""
    changed = QtCore.Signal()
    title = "조건"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.view = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.view)
        lay.addWidget(title_label("조건 편집기"))
        self.model = ConditionModel(self)
        self.model.edited.connect(self.changed)
        self.table = ConditionTableView()
        self.table.parent_page = self
        self.table.setModel(self.model)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)
        self.table.horizontalHeader().sectionDoubleClicked.connect(lambda s: self.edit_condition(s))
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        lay.addWidget(self.table, 1)
        self.coverage = QtWidgets.QLabel()
        self.coverage.setStyleSheet("color:#666;")
        self.coverage.setWordWrap(True)
        lay.addWidget(self.coverage)

        edit = section("조건 열", flow_buttons((("+ 조건 추가", self.add_condition), ("조건 편집", self.edit_condition),
                                                ("조건 삭제", self.remove_condition))))
        vals = section("값 입력",
                       info_label("빈칸(—)은 '기록 안 함'으로 저장되며 0으로 채우지 않습니다. 엑셀 표를 붙여넣을 수 있고(⌘V), "
                                  "여러 칸을 골라 한 번에 입력하거나 Delete로 비울 수 있습니다."),
                       flow_buttons((("선택 칸 한 번에 입력…", self.fill_selected), ("선택 칸 비우기", self.clear_selected),
                                     ("그룹 이름에서 조건 채우기…", self.fill_from_names))))
        ref = section("기준 그룹", info_label("표의 ★ 칸을 체크하거나 왼쪽 그룹 목록에서 오른쪽 클릭으로 지정합니다. "
                                            "기준 그룹 = 원하는 배치. 기준 score는 이 그룹들과의 거리입니다."))
        files = section("파일", info_label("조건 CSV·템플릿 불러오기와 저장은 메뉴 막대의 '조건' 메뉴에 있습니다."))
        self.panel = scroll_panel(edit, vals, ref, files)

    def set_project(self, project):
        self.project = project
        counts = project.particles.groupby("group").size().to_dict()
        self.model.set_table(project.conditions, counts)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(1, min(self.table.columnWidth(1), 340))
        self._update_coverage()

    def _update_coverage(self):
        t = self.project.conditions
        if not t.schema:
            self.coverage.setText("조건이 아직 없습니다. '+ 조건 추가'로 시작하세요.")
            return
        n = len(t.values)
        parts = [f"{c.name}: {t.recorded_count(c.name)}/{n}" for c in t.schema]
        self.coverage.setText("기록된 그룹 수 — " + " · ".join(parts) + f"   |   기준 그룹 {len(t.references)}개")

    def _after_edit(self):
        self.model.refresh()
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(1, min(self.table.columnWidth(1), 340))
        self._update_coverage()
        self.changed.emit()

    # ---- schema ---------------------------------------------------------
    def add_condition(self):
        dlg = ConditionDialog(self.view)
        if dlg.exec():
            try:
                self.project.conditions.add_condition(dlg.result_def())
            except ValueError as e:
                warn(self.view, str(e)); return
            self._after_edit()

    def _current_condition(self, section=None):
        t = self.project.conditions
        col = section if isinstance(section, int) else self.table.currentIndex().column()
        if col is None or col < FIXED or col - FIXED >= len(t.schema):
            return None
        return t.schema[col - FIXED]

    def edit_condition(self, section=None):
        c = self._current_condition(section)
        if c is None:
            warn(self.view, "편집할 조건 열의 칸을 먼저 고르세요."); return
        dlg = ConditionDialog(self.view, c)
        if dlg.exec():
            new = dlg.result_def()
            if new.name != c.name and new.name in self.project.conditions.names():
                warn(self.view, "같은 이름의 조건이 이미 있습니다."); return
            self.project.conditions.update_condition(c.name, new)
            self._after_edit()

    def remove_condition(self):
        c = self._current_condition()
        if c is None:
            warn(self.view, "삭제할 조건 열의 칸을 먼저 고르세요."); return
        if QtWidgets.QMessageBox.question(self.view, "조건 삭제", f"'{c.name}' 조건과 입력한 값을 모두 지울까요?") == QtWidgets.QMessageBox.Yes:
            self.project.conditions.remove_condition(c.name)
            self._after_edit()

    def fill_from_names(self):
        batch, ok = QtWidgets.QInputDialog.getText(
            self.view, "그룹 이름에서 조건 채우기",
            "그룹 이름의 pH · NaCl(mM) · PEG(%) 값을 빈칸에만 넣습니다. 이미 입력한 값은 그대로 두고,\n"
            "이름에서 찾지 못한 칸은 빈칸(—)으로 남깁니다. 표에서 결과를 확인하세요.\n"
            "빈 Batch 칸에 넣을 이름 (비우면 건너뜀):",
            text=self._last_batch())
        if not ok:
            return
        filled = fill_from_group_names(self.project.conditions, batch)
        self._after_edit()
        if not filled:
            warn(self.view, "새로 채울 빈칸이 없습니다 (이름에서 값을 찾지 못했거나 이미 입력됨).")

    def _last_batch(self):
        t = self.project.conditions
        if "Batch" in t.names():
            vals = t.categorical("Batch").dropna()
            if len(vals):
                return str(vals.iloc[0])
        return ""

    # ---- values ---------------------------------------------------------
    def _editable_selection(self):
        return [i for i in self.table.selectedIndexes() if i.column() >= FIXED]

    def fill_selected(self):
        sel = self._editable_selection()
        if not sel:
            warn(self.view, "값을 넣을 칸을 먼저 고르세요 (⌘ 또는 Shift로 여러 칸)."); return
        text, ok = QtWidgets.QInputDialog.getText(self.view, "선택 칸 한 번에 입력", f"{len(sel)}칸에 넣을 값 (비우면 '기록 안 함'):")
        if ok:
            for i in sel:
                self.model.setData(i, text)
            self._after_edit()

    def clear_selected(self):
        for i in self._editable_selection():
            self.model.setData(i, "")
        self._after_edit()

    def copy(self):
        sel = self.table.selectedIndexes()
        if not sel:
            return
        rows = sorted({i.row() for i in sel}); cols = sorted({i.column() for i in sel})
        lines = []
        for r in rows:
            vals = []
            for c in cols:
                v = self.model.data(self.model.index(r, c), QtCore.Qt.EditRole)
                vals.append("" if v is None else str(v))
            lines.append("\t".join(vals))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def paste(self):
        text = QtWidgets.QApplication.clipboard().text()
        if not text:
            return
        cur = self.table.currentIndex()
        if not cur.isValid() or cur.column() < FIXED:
            warn(self.view, "붙여넣을 위치(조건 칸)를 먼저 고르세요."); return
        rows = [r.split("\t") for r in text.rstrip("\n").split("\n")]
        for dr, vals in enumerate(rows):
            for dc, v in enumerate(vals):
                r, c = cur.row() + dr, cur.column() + dc
                if r < self.model.rowCount() and c < self.model.columnCount():
                    self.model.setData(self.model.index(r, c), v.strip())
        self._after_edit()

    # ---- files ----------------------------------------------------------
    def import_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.view, "조건 CSV 불러오기", "", "CSV (*.csv *.txt)")
        if path:
            self.import_csv_path(path)

    def import_csv_path(self, path) -> list:
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
        unmatched = self.project.conditions.merge_csv(text)
        self._after_edit()
        if unmatched:
            warn(self.view, "이 그룹 이름은 데이터에 없어 건너뛰었습니다:\n" + "\n".join(unmatched[:20]))
        return unmatched

    def export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.view, "조건 CSV 저장", "conditions.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(self.project.conditions.to_csv())

    def save_template(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.view, "조건 목록 템플릿 저장", "conditions_template.json", "JSON (*.json)")
        if path:
            data = {"format": "droplet-map-condition-template", "schema": [c.__dict__ for c in self.project.conditions.schema]}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def load_template(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.view, "템플릿 불러오기", "", "JSON (*.json)")
        if path:
            self.load_template_path(path)

    def load_template_path(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for c in data.get("schema", []):
            if c["name"] not in self.project.conditions.names():
                self.project.conditions.add_condition(ConditionDef(**c))
        self._after_edit()
