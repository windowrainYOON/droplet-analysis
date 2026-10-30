"""Batch note: a separate window that edits the batch files of a folder (one Excel workbook per
purification batch). Left: the batch files; center: one tab per sheet; right: sheet, column and row
tools, CSV import and the file check. Droplet Map reads the folder again after each save."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.batchfile import (B_GROUP, B_ITEM, B_NOTE, B_UNIT, KIND_BATCH,
                              KIND_CONDITIONS, KIND_MEASURE, KIND_PARTICLES, ROLE_ID, ROLE_NOTE, ROLE_VALUE, ROLES_FOR,
                              SHEET_DELIVERY, SHEET_PARTICLES, TEMPLATE_NAME, BatchFile, Column, batch_paths, blank,
                              fmt, folder_template, read_batch_file, safe_file_name, template, to_number,
                              write_batch_file)
from ..core.importing import (add_delivery, add_particles, conditions_from_names, read_delivery_files,
                              read_particle_files, suggest_conditions)
from ..core.io import classify_file
from .common import app_icon, flow_buttons, info_label, scroll_panel, section, warn

MAX_UNDO = 30
CHECK_DELAY_MS = 600
TEXT_BATCH_COLUMNS = (B_ITEM, B_UNIT, B_GROUP, B_NOTE)
ISSUE_BG = QtGui.QColor("#ffe1dc")
ROLE_BG = {ROLE_ID: QtGui.QColor("#eef3fb")}


def cell_value(sheet, name, text):
    """Typed text -> stored value: numbers where the column holds numbers, text otherwise."""
    if isinstance(text, str):
        text = text.strip()
    if blank(text):
        return np.nan
    if sheet.kind == KIND_BATCH:
        if name in TEXT_BATCH_COLUMNS:
            return str(text)
    elif sheet.column(name).role in (ROLE_ID, ROLE_NOTE):
        return fmt(text) if not isinstance(text, str) else text
    n = to_number(text)
    return text if n is None else n


class SheetModel(QtCore.QAbstractTableModel):
    def __init__(self, note, sheet):
        super().__init__()
        self.note, self.sheet = note, sheet
        self.issues = set()         # (row, column name)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.sheet.data)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.sheet.columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        col = self.sheet.columns[index.column()]
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            v = self.sheet.data.iat[index.row(), index.column()]
            return "" if blank(v) else fmt(v)
        if role == QtCore.Qt.BackgroundRole:
            if (index.row(), col.name) in self.issues:
                return ISSUE_BG
            return ROLE_BG.get(col.role) if self.sheet.kind != KIND_BATCH else None
        if role == QtCore.Qt.TextAlignmentRole:
            v = self.sheet.data.iat[index.row(), index.column()]
            if isinstance(v, (int, float, np.integer, np.floating)) and not blank(v):
                return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Vertical:
            return str(section + 1)
        if section >= len(self.sheet.columns):
            return None
        c = self.sheet.columns[section]
        return c.label if self.sheet.kind == KIND_BATCH else f"{c.label}\n[{c.role}]"

    def flags(self, index):
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if role != QtCore.Qt.EditRole or not index.isValid():
            return False
        name = self.sheet.columns[index.column()].name
        new = cell_value(self.sheet, name, value)
        old = self.sheet.data.iat[index.row(), index.column()]
        if (blank(old) and blank(new)) or (not blank(old) and not blank(new) and fmt(old) == fmt(new)):
            return False
        self.note.before_edit()
        self.sheet.set_value(index.row(), name, new)
        self.dataChanged.emit(index, index)
        self.note.after_edit(structure=False)
        return True

    def reset(self):
        self.beginResetModel(); self.endResetModel()


class SheetView(QtWidgets.QTableView):
    """Excel-like: ⌘C / ⌘V with tab-separated text, Delete clears the selection."""

    def __init__(self, note):
        super().__init__()
        self.note = note
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed
                             | QtWidgets.QAbstractItemView.AnyKeyPressed)
        self.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)
        self.horizontalHeader().setResizeContentsPrecision(200)     # widths from the first rows: fast on big sheets
        self.horizontalHeader().sectionDoubleClicked.connect(lambda c: note.edit_column(c))
        self.verticalHeader().setDefaultSectionSize(22)

    def keyPressEvent(self, e):
        if e.matches(QtGui.QKeySequence.Copy):
            self.note.copy_cells(); return
        if e.matches(QtGui.QKeySequence.Paste):
            self.note.paste_cells(); return
        if e.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace) and self.state() != QtWidgets.QAbstractItemView.EditingState:
            self.note.clear_cells(); return
        super().keyPressEvent(e)


class ColumnDialog(QtWidgets.QDialog):
    def __init__(self, parent, kind, col: Column | None = None):
        super().__init__(parent)
        self.setWindowTitle("열 편집" if col else "새 열")
        f = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit(col.name if col else "")
        self.name.setPlaceholderText("예: FRAP 회복률, pH, 시간")
        self.unit = QtWidgets.QLineEdit(col.unit if col else "")
        self.unit.setPlaceholderText("예: %, mM, min (없으면 비움)")
        self.role = QtWidgets.QComboBox()
        roles = [r for r in ROLES_FOR.get(kind, ()) if r != ROLE_ID or (col and col.role == ROLE_ID)]
        self.role.addItems(roles)
        if col and col.role in roles:
            self.role.setCurrentText(col.role)
        elif kind == KIND_CONDITIONS:
            self.role.setCurrentText("조건")
        else:
            self.role.setCurrentText(ROLE_VALUE)
        if col and col.role == ROLE_ID:
            self.role.setEnabled(False)
        f.addRow("이름", self.name)
        f.addRow("단위", self.unit)
        f.addRow("역할", self.role)
        f.addRow(info_label("조건 = 입자 형성 조건(형성 조건 시트) · 측정값 = 분석에 쓰는 숫자 · 하위 조건 = 측정 자체의 "
                            "조건(시간, 처리 양 …) · 반복 = 이미지나 반복 번호 · 메모 = 분석에 안 씀"))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        f.addRow(bb)

    def column(self) -> Column:
        return Column(self.name.text().strip(), self.unit.text().strip(), self.role.currentText())


class SheetDialog(QtWidgets.QDialog):
    def __init__(self, parent, categories, sheet=None, kinds=(KIND_MEASURE,)):
        super().__init__(parent)
        self.setWindowTitle("시트 편집" if sheet else "새 시트")
        f = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit(sheet.name if sheet else "")
        self.name.setPlaceholderText("예: DLS, 안정성, 동물 실험")
        self.kind = QtWidgets.QComboBox(); self.kind.addItems(list(kinds))
        if sheet:
            self.kind.addItem(sheet.kind) if sheet.kind not in kinds else None
            self.kind.setCurrentText(sheet.kind); self.kind.setEnabled(False)
        self.cat = QtWidgets.QComboBox(); self.cat.setEditable(True); self.cat.addItems(categories)
        self.cat.setEditText(sheet.category if sheet else "")
        self.cat.lineEdit().setPlaceholderText("비우면 시트 이름")
        f.addRow("이름", self.name)
        f.addRow("모양", self.kind)
        f.addRow("범주", self.cat)
        f.addRow(info_label("조건별 측정 = 조건 ID마다 한 행 이상 (FRAP, 세포 전달 …) · 입자별 측정 = 입자 하나에 한 행. "
                            "범주는 배치 지도에서 무게를 따로 주는 묶음입니다."))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        f.addRow(bb)


class NewBatchDialog(QtWidgets.QDialog):
    def __init__(self, parent, batches):
        super().__init__(parent)
        self.setWindowTitle("새 배치")
        f = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit(); self.name.setPlaceholderText("예: 정제 로트 번호 (B06)")
        self.src = QtWidgets.QComboBox()
        self.src.addItem("템플릿 (_템플릿.xlsx)", None)
        for b, path in batches:
            self.src.addItem(f"'{b}' 복사 (시트·열·형성 조건·배치 값, 측정값은 비움)", path)
        if batches:
            self.src.setCurrentIndex(len(batches))
        f.addRow("배치 이름", self.name)
        f.addRow("시작", self.src)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        f.addRow(bb)


class ImportDialog(QtWidgets.QDialog):
    """CSV groups -> formation conditions (조건 ID) of this batch. Values are read from group names
    only when the user presses the buttons, and only blanks are filled."""

    def __init__(self, parent, bf: BatchFile, groups: dict, what: str, sheets=None):
        super().__init__(parent)
        self.bf = bf
        self.groups = list(groups)
        self.setWindowTitle(f"{what} 넣기 — 어느 형성 조건인가요?")
        self.resize(760, 520)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(info_label(f"CSV의 그룹마다 이 배치({bf.name})의 형성 조건(조건 ID)을 고르세요. 비워 둔 그룹은 "
                                 + ("넣지 않습니다." if what == "입자 CSV" else "조건 ID 없이 넣습니다 (분석에서 빠짐).")))
        self.sheet = None
        if sheets:
            row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel("넣을 시트"))
            self.sheet = QtWidgets.QComboBox(); self.sheet.addItems(sheets)
            if SHEET_DELIVERY in sheets:
                self.sheet.setCurrentText(SHEET_DELIVERY)
            row.addWidget(self.sheet, 1); lay.addLayout(row)
            self.subs = QtWidgets.QCheckBox("그룹 이름의 시간 · 처리 양을 하위 조건 칸에 채우기")
            lay.addWidget(self.subs)
        self.table = QtWidgets.QTableWidget(len(self.groups), 3)
        self.table.setHorizontalHeaderLabels(["CSV 그룹", "행 수", "조건 ID"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.combos = []
        for r, g in enumerate(self.groups):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(g)))
            n = QtWidgets.QTableWidgetItem(f"{groups[g]:,}"); n.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            self.table.setItem(r, 1, n)
            cb = QtWidgets.QComboBox(); cb.setEditable(True)
            self.combos.append(cb)
            self.table.setCellWidget(r, 2, cb)
        self._fill_choices()
        lay.addWidget(self.table, 1)
        lay.addWidget(flow_buttons((("그룹 이름에서 찾기 (빈칸만)", self.suggest),
                                    ("못 찾은 그룹은 이름으로 새 조건 만들기", self.make_from_names),
                                    ("모두 비우기", lambda: [cb.setEditText("") for cb in self.combos]))))
        self.note = QtWidgets.QLabel(""); self.note.setStyleSheet("color:#666;"); self.note.setWordWrap(True)
        lay.addWidget(self.note)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("넣기")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _label(self, cid):
        lab = self.bf.condition_label(cid)
        return cid if lab == cid else f"{cid} — {lab}"

    def _fill_choices(self):
        items = [""] + [self._label(c) for c in self.bf.condition_ids()]
        for cb in self.combos:
            cur = cb.currentText()
            cb.blockSignals(True); cb.clear(); cb.addItems(items); cb.setEditText(cur); cb.blockSignals(False)

    def suggest(self):
        blank_rows = [r for r, cb in enumerate(self.combos) if not cb.currentText().strip()]
        hits = suggest_conditions(self.bf, [self.groups[r] for r in blank_rows])
        n = 0
        for r in blank_rows:
            cid = hits.get(self.groups[r])
            if cid:
                self.combos[r].setEditText(self._label(cid)); n += 1
        self.note.setText(f"이름의 pH · NaCl · PEG 값이 조건표의 한 행과 맞는 그룹 {n}개를 채웠습니다 "
                          f"(빈칸 {len(blank_rows)}개 중).")

    def make_from_names(self):
        blank_rows = [r for r, cb in enumerate(self.combos) if not cb.currentText().strip()]
        made = conditions_from_names(self.bf, [self.groups[r] for r in blank_rows])
        self._fill_choices()
        for r in blank_rows:
            cid = made.get(self.groups[r])
            if cid:
                self.combos[r].setEditText(self._label(cid))
        self.note.setText(f"형성 조건 시트에 새 조건 {len(made)}개를 만들었습니다 (값은 그룹 이름에서). "
                          "넣은 뒤 형성 조건 시트에서 확인하세요.")

    def mapping(self) -> dict:
        out = {}
        for g, cb in zip(self.groups, self.combos):
            t = cb.currentText().strip()
            out[g] = t.split(" — ")[0].strip() if t else None
        return out


class Note:
    """One open batch file with its undo history."""

    def __init__(self, path: Path, bf: BatchFile, is_template=False):
        self.path, self.bf, self.is_template = path, bf, is_template
        self.dirty = False
        self.undo, self.redo = [], []
        self.mtime = path.stat().st_mtime if path.exists() else None


class BatchNoteWindow(QtWidgets.QMainWindow):
    saved = QtCore.Signal(str)        # path of a batch file just written

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowIcon(app_icon())
        self.setWindowTitle("배치 노트")
        self.resize(1400, 860)
        self.setAcceptDrops(True)
        self.folder: Path | None = None
        self.notes: dict = {}         # path str -> Note (open files keep their edits when switching)
        self.cur: Note | None = None
        self.models = []
        self.settings = QtCore.QSettings("droplet-map", "Droplet Map")

        self.files = QtWidgets.QListWidget()
        self.files.currentRowChanged.connect(self._file_chosen)
        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(6, 6, 6, 6)
        self.folder_label = QtWidgets.QLabel(""); self.folder_label.setWordWrap(True); self.folder_label.setStyleSheet("color:#555;")
        ll.addWidget(QtWidgets.QLabel("<b>배치 파일</b>"))
        ll.addWidget(self.folder_label)
        ll.addWidget(self.files, 1)
        ll.addWidget(flow_buttons((("+ 새 배치", self.new_batch), ("휴지통으로", self.trash_batch))))
        left.setMinimumWidth(200)

        self.tabs = QtWidgets.QTabWidget(); self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._tab_changed)
        center = QtWidgets.QWidget(); cl = QtWidgets.QVBoxLayout(center); cl.setContentsMargins(0, 0, 0, 0)
        self.title = QtWidgets.QLabel(""); self.title.setStyleSheet("font-size:15px;font-weight:600;padding:6px;")
        cl.addWidget(self.title)
        cl.addWidget(self.tabs, 1)

        self.sheet_info = info_label("")
        sheet_box = section("시트", self.sheet_info,
                            flow_buttons((("+ 시트 추가", self.add_sheet), ("이름 · 범주", self.edit_sheet),
                                          ("시트 삭제", self.remove_sheet))))
        col_box = section("열", info_label("열 머리를 두 번 누르면 이름 · 단위 · 역할을 바꿉니다."),
                          flow_buttons((("+ 열 추가", self.add_column), ("열 편집", lambda: self.edit_column()),
                                        ("열 삭제", self.remove_column), ("◀", lambda: self.move_column(-1)),
                                        ("▶", lambda: self.move_column(1)))))
        row_box = section("행", info_label("엑셀에서 복사한 표를 붙여넣을 수 있습니다 (⌘V). 빈칸 = 기록 안 함, 0이 아닙니다."),
                          flow_buttons((("+ 행 추가", self.add_rows), ("선택 행 삭제", self.remove_rows),
                                        ("빈 조건 ID 채우기 (C01 …)", self.fill_ids))))
        csv_box = section("이미지 분석 CSV 넣기",
                          info_label("입자 CSV는 입자 형태 시트에, 세포 전달 요약 CSV는 세포 전달 시트에 넣습니다. "
                                     "CSV 그룹마다 형성 조건을 고릅니다. 창에 끌어다 놓아도 됩니다."),
                          flow_buttons((("입자 CSV 넣기…", lambda: self.import_csvs(kind="particles")),
                                        ("세포 전달 CSV 넣기…", lambda: self.import_csvs(kind="delivery")))))
        self.issues = QtWidgets.QListWidget(); self.issues.setMinimumHeight(140)
        self.issues.itemDoubleClicked.connect(self._jump_to_issue)
        check_box = section("확인", info_label("두 번 누르면 그 칸으로 갑니다. 빨간 칸은 고쳐야 할 칸입니다."), self.issues)
        tpl_box = section("템플릿",
                          info_label("새 배치는 템플릿(폴더의 _템플릿.xlsx)이나 이전 배치에서 시작합니다. "
                                     "목록 맨 아래 '템플릿'을 골라 직접 고칠 수도 있습니다."),
                          flow_buttons((("이 배치 구성을 템플릿으로", self.save_as_template),)))
        right = scroll_panel(sheet_box, col_box, row_box, csv_box, check_box, tpl_box)
        right.setMinimumWidth(300)

        self.split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        for w in (left, center, right):
            self.split.addWidget(w)
        self.split.setStretchFactor(1, 1)
        self.split.setCollapsible(1, False)
        sizes = self.settings.value("note_split_sizes")
        self.split.setSizes([int(x) for x in sizes] if sizes else [230, 860, 330])
        self.split.splitterMoved.connect(lambda *_: self.settings.setValue("note_split_sizes", self.split.sizes()))
        self.setCentralWidget(self.split)

        self.check_timer = QtCore.QTimer(self); self.check_timer.setSingleShot(True)
        self.check_timer.setInterval(CHECK_DELAY_MS); self.check_timer.timeout.connect(self.run_check)
        self._build_menus()
        self.statusBar().showMessage("고친 뒤 ⌘S로 저장하면 Droplet Map이 다시 읽습니다.")

    # ---- menus ---------------------------------------------------------------
    def _act(self, menu, text, fn, shortcut=None):
        a = QtGui.QAction(text, self)
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
        a.triggered.connect(fn)
        menu.addAction(a)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        f = mb.addMenu("배치 파일")
        self._act(f, "새 배치…", self.new_batch, QtGui.QKeySequence.New)
        self._act(f, "저장", self.save, QtGui.QKeySequence.Save)
        self._act(f, "모두 저장", self.save_all, "Ctrl+Alt+S")
        self._act(f, "디스크에서 다시 읽기 (고친 내용 버림)", self.revert)
        self._act(f, "엑셀에서 열기", self.open_in_excel)
        f.addSeparator()
        self._act(f, "입자 CSV 넣기…", lambda: self.import_csvs(kind="particles"))
        self._act(f, "세포 전달 CSV 넣기…", lambda: self.import_csvs(kind="delivery"))
        f.addSeparator()
        self._act(f, "이 배치 구성을 템플릿으로 저장", self.save_as_template)
        f.addSeparator()
        self._act(f, "창 닫기", self.close, QtGui.QKeySequence.Close)
        e = mb.addMenu("편집")
        self.a_undo = self._act(e, "실행 취소", self.do_undo, QtGui.QKeySequence.Undo)
        self.a_redo = self._act(e, "다시 실행", self.do_redo, QtGui.QKeySequence.Redo)
        e.addSeparator()
        self._act(e, "복사", self.copy_cells)
        self._act(e, "붙여넣기", self.paste_cells)
        self._act(e, "선택 칸 비우기", self.clear_cells)
        e.addSeparator()
        self._act(e, "행 추가", self.add_rows)
        self._act(e, "선택 행 삭제", self.remove_rows)
        s = mb.addMenu("시트 · 열")
        self._act(s, "시트 추가…", self.add_sheet)
        self._act(s, "시트 이름 · 범주…", self.edit_sheet)
        self._act(s, "시트 삭제", self.remove_sheet)
        s.addSeparator()
        self._act(s, "열 추가…", self.add_column)
        self._act(s, "열 편집…", lambda: self.edit_column())
        self._act(s, "열 삭제", self.remove_column)

    # ---- folder and files ------------------------------------------------------
    def set_folder(self, folder):
        folder = Path(folder) if folder else None
        if folder != self.folder:
            if self.folder is not None and not self._confirm_all():
                return
            self.notes, self.cur = {}, None
        self.folder = folder
        self.refresh_files()

    def _paths(self) -> list[Path]:
        if self.folder is None or not self.folder.is_dir():
            return []
        out = batch_paths(self.folder)
        out += [Path(k) for k, n in self.notes.items() if not n.is_template and Path(k) not in out]   # new, unsaved
        return out + [self.folder / TEMPLATE_NAME]

    def refresh_files(self):
        self.folder_label.setText(str(self.folder) if self.folder else "폴더 없음")
        keep = str(self.cur.path) if self.cur else None
        for k, n in list(self.notes.items()):       # changed outside (Excel) and not edited here: read again
            p = Path(k)
            if not n.dirty and p.exists() and n.mtime != p.stat().st_mtime:
                self.notes.pop(k)
        self.files.blockSignals(True)
        self.files.clear()
        row = -1
        for p in self._paths():
            n = self.notes.get(str(p))
            if p.name == TEMPLATE_NAME:
                text = "📐 템플릿"
            else:
                name = n.bf.name if n else p.stem
                text = name if name == p.stem else f"{name}  ({p.name})"
            if n and n.dirty:
                text = "● " + text
            it = QtWidgets.QListWidgetItem(text)
            it.setData(QtCore.Qt.UserRole, str(p))
            it.setToolTip(str(p))
            self.files.addItem(it)
            if str(p) == keep:
                row = self.files.count() - 1
        self.files.blockSignals(False)
        if row < 0 and self.files.count() > 1:
            row = 0
        if row >= 0:
            self.files.setCurrentRow(row)
            if self.cur is None or str(self.cur.path) != self.files.item(row).data(QtCore.Qt.UserRole) \
                    or str(self.cur.path) not in self.notes:
                self._file_chosen(row)
        else:
            self.cur = None
            self._show()

    def _file_chosen(self, row):
        if row < 0:
            return
        path = Path(self.files.item(row).data(QtCore.Qt.UserRole))
        n = self.notes.get(str(path))
        if n is None:
            try:
                if path.exists():
                    bf, msgs = read_batch_file(path)
                    for m in msgs:
                        self.statusBar().showMessage(f"{path.name}: {m}")
                else:
                    bf = template()
            except Exception as e:
                warn(self, f"{path.name}을(를) 읽지 못했습니다 (엑셀에서 열려 있으면 닫고 다시 고르세요).\n{e}")
                return
            n = Note(path, bf, is_template=path.name == TEMPLATE_NAME)
            self.notes[str(path)] = n
        self.cur = n
        self._show()

    def _show(self):
        cur_tab = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else None
        self.tabs.blockSignals(True)
        while self.tabs.count():
            w = self.tabs.widget(0); self.tabs.removeTab(0); w.deleteLater()
        self.models = []
        n = self.cur
        if n is None:
            self.title.setText("배치 파일이 없습니다. '+ 새 배치'로 시작하세요.")
            self.tabs.blockSignals(False)
            self.issues.clear()
            return
        for s in n.bf.sheets:
            m = SheetModel(self, s)
            v = SheetView(self); v.setModel(m)
            v.resizeColumnsToContents()
            for c in range(m.columnCount()):
                v.setColumnWidth(c, min(max(v.columnWidth(c), 70), 260))
            self.models.append(m)
            self.tabs.addTab(v, s.name)
        names = [s.name for s in n.bf.sheets]
        self.tabs.setCurrentIndex(names.index(cur_tab) if cur_tab in names else 0)
        self.tabs.blockSignals(False)
        self._tab_changed(self.tabs.currentIndex())
        self._update_title()
        self.run_check()

    def _update_title(self):
        n = self.cur
        if n is None:
            return
        what = "템플릿 (새 배치의 시작 모양)" if n.is_template else f"배치 {n.bf.name}"
        self.title.setText(f"{what}  —  {n.path.name}" + ("  ● 저장 안 함" if n.dirty else ""))
        self.setWindowTitle(f"배치 노트 — {n.path.name}[*]")
        self.setWindowModified(n.dirty)
        self.a_undo.setEnabled(bool(n.undo)); self.a_redo.setEnabled(bool(n.redo))
        for i in range(self.files.count()):
            it = self.files.item(i)
            if it.data(QtCore.Qt.UserRole) == str(n.path):
                base = it.text().removeprefix("● ")
                if not n.is_template:
                    base = n.bf.name if n.bf.name == n.path.stem else f"{n.bf.name}  ({n.path.name})"
                it.setText(("● " if n.dirty else "") + base)

    def _tab_changed(self, i):
        s = self.sheet()
        if s is None:
            self.sheet_info.setText("")
            return
        ncol = len(s.columns)
        hint = {KIND_BATCH: "배치마다 값이 하나인 항목 (정제 조건, 제타 포텐셜 …). '구분'이 배치 지도의 범주입니다.",
                KIND_CONDITIONS: "입자 형성 조건표. 조건 ID로 다른 시트와 이어집니다.",
                KIND_PARTICLES: "이미지 분석의 입자 하나에 한 행.",
                KIND_MEASURE: "조건 ID마다 한 행 이상 (반복 측정)."}[s.kind]
        self.sheet_info.setText(f"<b>{s.name}</b> · 모양 {s.kind} · 범주 {s.category} · 열 {ncol}개 · 행 {len(s.data):,}개<br>{hint}")

    def sheet(self):
        i = self.tabs.currentIndex()
        if self.cur is None or i < 0 or i >= len(self.cur.bf.sheets):
            return None
        return self.cur.bf.sheets[i]

    def view(self) -> SheetView | None:
        return self.tabs.currentWidget()

    def model(self) -> SheetModel | None:
        i = self.tabs.currentIndex()
        return self.models[i] if 0 <= i < len(self.models) else None

    # ---- undo ---------------------------------------------------------------------
    def before_edit(self):
        n = self.cur
        n.undo.append(copy.deepcopy(n.bf.sheets))
        del n.undo[:-MAX_UNDO]
        n.redo.clear()

    def after_edit(self, structure=True):
        n = self.cur
        n.dirty = True
        if structure:
            self._show()
        else:
            self._update_title()
            self._tab_changed(self.tabs.currentIndex())
            self.check_timer.start()

    def _restore(self, src, dst):
        n = self.cur
        if n is None or not src:
            return
        dst.append(copy.deepcopy(n.bf.sheets))
        n.bf.sheets = src.pop()
        n.dirty = True
        self._show()

    def do_undo(self):
        if self.cur:
            self._restore(self.cur.undo, self.cur.redo)

    def do_redo(self):
        if self.cur:
            self._restore(self.cur.redo, self.cur.undo)

    def _drop_last_undo(self):
        """Cancelled dialog: put the sheets back as they were and forget the snapshot."""
        n = self.cur
        n.bf.sheets = n.undo.pop()
        self._show()

    # ---- saving --------------------------------------------------------------------
    def save(self, note=None) -> bool:
        n = note or self.cur
        if n is None:
            return False
        p = n.path
        if p.exists() and n.mtime is not None and p.stat().st_mtime != n.mtime:
            r = QtWidgets.QMessageBox.question(self, "배치 노트", f"{p.name}이(가) 이 창 밖(엑셀 등)에서 바뀌었습니다. "
                                               "여기서 고친 내용으로 덮어쓸까요?")
            if r != QtWidgets.QMessageBox.Yes:
                return False
        try:
            write_batch_file(n.bf, p)
        except PermissionError:
            warn(self, f"{p.name}을(를) 쓰지 못했습니다. 엑셀에서 열려 있으면 닫고 다시 저장하세요."); return False
        except Exception as e:
            warn(self, f"{p.name}을(를) 저장하지 못했습니다. 이전 파일은 그대로 있습니다.\n{e}"); return False
        n.dirty = False
        n.mtime = p.stat().st_mtime
        self._update_title()
        self.statusBar().showMessage(f"저장: {p.name}")
        if not n.is_template:
            self.saved.emit(str(p))
        return True

    def save_all(self):
        for n in list(self.notes.values()):
            if n.dirty and not self.save(n):
                return False
        return True

    def revert(self):
        n = self.cur
        if n is None or not n.path.exists():
            return
        if n.dirty and QtWidgets.QMessageBox.question(self, "배치 노트", "고친 내용을 버리고 파일에서 다시 읽을까요?") \
                != QtWidgets.QMessageBox.Yes:
            return
        self.notes.pop(str(n.path))
        self.cur = None
        self.refresh_files()

    def _confirm_all(self) -> bool:
        dirty = [n for n in self.notes.values() if n.dirty]
        if not dirty:
            return True
        r = QtWidgets.QMessageBox.question(
            self, "배치 노트", "저장하지 않은 배치 파일이 있습니다:\n" + "\n".join(n.path.name for n in dirty) + "\n\n저장할까요?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        if r == QtWidgets.QMessageBox.Save:
            return self.save_all()
        if r == QtWidgets.QMessageBox.Discard:
            for n in dirty:
                self.notes.pop(str(n.path), None)
            return True
        return False

    def closeEvent(self, e):
        if self._confirm_all():
            e.accept()
        else:
            e.ignore()

    def open_in_excel(self):
        n = self.cur
        if n is None:
            return
        if n.dirty and not self.save(n):
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(n.path)))

    # ---- batches -------------------------------------------------------------------
    def new_batch(self):
        if self.folder is None:
            warn(self, "먼저 Droplet Map에서 배치 폴더를 여세요."); return
        batches = [(self.notes[str(p)].bf.name if str(p) in self.notes else p.stem, p)
                   for p in batch_paths(self.folder)]
        dlg = NewBatchDialog(self, batches)
        if not dlg.exec():
            return
        self.create_batch(dlg.name.text().strip(), dlg.src.currentData())

    def create_batch(self, name, src=None):
        """A new batch file from the folder template (src None) or as a copy of another batch file."""
        if not name:
            warn(self, "배치 이름을 입력하세요."); return None
        path = self.folder / (safe_file_name(name) + ".xlsx")
        if path.exists() or str(path) in self.notes:
            warn(self, f"'{path.name}' 파일이 이미 있습니다. 다른 이름을 쓰세요."); return None
        if src is None:
            bf = folder_template(self.folder).copy_for_new_batch(name)
        else:
            n = self.notes.get(str(src))
            base = n.bf if n else read_batch_file(src)[0]
            bf = base.copy_for_new_batch(name)
        bf.path = path
        n = Note(path, bf)
        n.dirty = True
        self.notes[str(path)] = n
        self.cur = n
        self.refresh_files()
        self.save(n)
        self.statusBar().showMessage(f"새 배치 {name}: {path.name}")
        return n

    def trash_batch(self):
        n = self.cur
        if n is None or n.is_template:
            return
        if QtWidgets.QMessageBox.question(self, "배치 노트", f"'{n.bf.name}' 배치 파일({n.path.name})을 휴지통으로 옮길까요?") \
                != QtWidgets.QMessageBox.Yes:
            return
        if n.path.exists() and not QtCore.QFile.moveToTrash(str(n.path)):
            warn(self, "휴지통으로 옮기지 못했습니다."); return
        self.notes.pop(str(n.path), None)
        self.cur = None
        self.refresh_files()
        self.saved.emit(str(n.path))

    def save_as_template(self):
        n = self.cur
        if n is None or self.folder is None:
            return
        tpl = self.folder / TEMPLATE_NAME
        if QtWidgets.QMessageBox.question(self, "템플릿", f"'{n.bf.name}'의 시트 · 열 · 배치 항목(값은 비움)을 새 배치의 "
                                          "템플릿으로 저장할까요?") != QtWidgets.QMessageBox.Yes:
            return
        write_batch_file(n.bf.layout_only(), tpl)
        self.notes.pop(str(tpl), None)
        self.statusBar().showMessage(f"템플릿 저장: {tpl.name}")

    # ---- sheets ----------------------------------------------------------------------
    def _categories(self):
        out = []
        for n in self.notes.values():
            for s in n.bf.sheets:
                if s.kind != KIND_BATCH and s.category not in out:
                    out.append(s.category)
            for it in n.bf.items():
                if it["category"] not in out:
                    out.append(it["category"])
        return out

    def add_sheet(self):
        n = self.cur
        if n is None:
            return
        kinds = [KIND_MEASURE] + ([KIND_PARTICLES] if n.bf.particles is None else [])
        dlg = SheetDialog(self, self._categories(), kinds=kinds)
        if not dlg.exec():
            return
        self.before_edit()
        try:
            n.bf.add_sheet(dlg.name.text(), dlg.kind.currentText(), dlg.cat.currentText().strip())
        except ValueError as e:
            n.undo.pop(); warn(self, str(e)); return
        self.after_edit()
        self.tabs.setCurrentIndex(self.tabs.count() - 1)

    def edit_sheet(self):
        s = self.sheet()
        if s is None:
            return
        dlg = SheetDialog(self, self._categories(), s)
        if not dlg.exec():
            return
        self.before_edit()
        try:
            self.cur.bf.rename_sheet(s.name, dlg.name.text())
        except ValueError as e:
            self.cur.undo.pop(); warn(self, str(e)); return
        cat = dlg.cat.currentText().strip()
        if cat and s.kind != KIND_BATCH:
            s.category = cat
        self.after_edit()

    def remove_sheet(self):
        s = self.sheet()
        if s is None:
            return
        if QtWidgets.QMessageBox.question(self, "시트 삭제", f"'{s.name}' 시트와 그 안의 값 {len(s.data):,}행을 지울까요? "
                                          "(⌘Z로 되돌릴 수 있습니다)") != QtWidgets.QMessageBox.Yes:
            return
        self.before_edit()
        try:
            self.cur.bf.remove_sheet(s.name)
        except ValueError as e:
            self.cur.undo.pop(); warn(self, str(e)); return
        self.after_edit()

    # ---- columns ---------------------------------------------------------------------
    def _current_column(self, section=None):
        s, v = self.sheet(), self.view()
        if s is None or v is None:
            return None, None
        c = section if isinstance(section, int) and section >= 0 else v.currentIndex().column()
        if c is None or c < 0 or c >= len(s.columns):
            return s, None
        return s, c

    def add_column(self):
        s, c = self._current_column()
        if s is None:
            return
        if s.kind == KIND_BATCH:
            warn(self, "배치 정보 시트는 열이 고정입니다. 항목(제타 포텐셜 등)은 행으로 추가하세요."); return
        dlg = ColumnDialog(self, s.kind)
        if not dlg.exec():
            return
        self.before_edit()
        try:
            s.add_column(dlg.column(), None if c is None else c + 1)
        except ValueError as e:
            self.cur.undo.pop(); warn(self, str(e)); return
        self.after_edit()

    def edit_column(self, section=None):
        s, c = self._current_column(section)
        if s is None or c is None:
            return
        if s.kind == KIND_BATCH:
            warn(self, "배치 정보 시트는 열이 고정입니다 (항목 · 값 · 단위 · 구분 · 메모)."); return
        col = s.columns[c]
        dlg = ColumnDialog(self, s.kind, col)
        if not dlg.exec():
            return
        self.before_edit()
        try:
            s.update_column(col.name, dlg.column())
        except ValueError as e:
            self.cur.undo.pop(); warn(self, str(e)); return
        self.after_edit()

    def remove_column(self):
        s, c = self._current_column()
        if s is None or c is None:
            return
        name = s.columns[c].name
        if QtWidgets.QMessageBox.question(self, "열 삭제", f"'{name}' 열과 그 값을 지울까요? (⌘Z로 되돌릴 수 있습니다)") \
                != QtWidgets.QMessageBox.Yes:
            return
        self.before_edit()
        try:
            s.remove_column(name)
        except ValueError as e:
            self.cur.undo.pop(); warn(self, str(e)); return
        self.after_edit()

    def move_column(self, step):
        s, c = self._current_column()
        if s is None or c is None or s.kind == KIND_BATCH:
            return
        j = c + step
        if not 0 <= j < len(s.columns):
            return
        self.before_edit()
        s.columns[c], s.columns[j] = s.columns[j], s.columns[c]
        s.data = s.data[s.names()]
        self.after_edit()
        v = self.view()
        v.setCurrentIndex(v.model().index(max(v.currentIndex().row(), 0), j))

    # ---- rows and cells ------------------------------------------------------------------
    def add_rows(self):
        s = self.sheet()
        if s is None:
            return
        n, ok = QtWidgets.QInputDialog.getInt(self, "행 추가", "추가할 행 수:", 1, 1, 10000)
        if not ok:
            return
        self.before_edit()
        s.add_rows(n)
        self.after_edit()
        self.view().scrollToBottom()

    def _selected_rows(self):
        v = self.view()
        return sorted({i.row() for i in v.selectionModel().selectedIndexes()}) if v else []

    def remove_rows(self):
        s = self.sheet()
        rows = self._selected_rows()
        if s is None or not rows:
            return
        if len(rows) > 1 and QtWidgets.QMessageBox.question(self, "행 삭제", f"선택한 {len(rows)}행을 지울까요?") \
                != QtWidgets.QMessageBox.Yes:
            return
        self.before_edit()
        s.remove_rows(rows)
        self.after_edit()

    def fill_ids(self):
        if self.cur is None:
            return
        self.before_edit()
        k = self.cur.bf.fill_condition_ids()
        if not k:
            self.cur.undo.pop()
            self.statusBar().showMessage("빈 조건 ID가 없습니다."); return
        self.after_edit()
        self.statusBar().showMessage(f"조건 ID {k}개를 채웠습니다.")

    def copy_cells(self):
        v, s = self.view(), self.sheet()
        if v is None:
            return
        idx = v.selectionModel().selectedIndexes()
        if not idx:
            return
        rows = sorted({i.row() for i in idx}); cols = sorted({i.column() for i in idx})
        sel = {(i.row(), i.column()) for i in idx}
        lines = []
        for r in rows:
            cells = []
            for c in cols:
                val = s.data.iat[r, c] if (r, c) in sel else None
                cells.append("" if blank(val) else fmt(val))
            lines.append("\t".join(cells))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def paste_cells(self, text=None):
        v, s = self.view(), self.sheet()
        if v is None or s is None:
            return
        text = QtWidgets.QApplication.clipboard().text() if text is None else text
        if not text:
            return
        grid = [line.split("\t") for line in text.rstrip("\r\n").replace("\r\n", "\n").split("\n")]
        if grid and len(grid) > 1 and all(c.strip() in s.names() + [col.label for col in s.columns] for c in grid[0] if c.strip()):
            grid = grid[1:]                  # pasted with its header row
        cur = v.currentIndex()
        r0, c0 = max(cur.row(), 0), max(cur.column(), 0)
        ncol = min(max(len(r) for r in grid), len(s.columns) - c0)
        self.before_edit()
        need = r0 + len(grid) - len(s.data)
        if need > 0:
            s.add_rows(need)
        names = s.names()[c0:c0 + ncol]
        block = [[cell_value(s, names[j], row[j]) if j < len(row) else np.nan for j in range(ncol)] for row in grid]
        for j, name in enumerate(names):
            if s.data[name].dtype != object:
                s.data[name] = s.data[name].astype(object)
            s.data.iloc[r0:r0 + len(grid), c0 + j] = [b[j] for b in block]
        self.model().reset()
        self.after_edit(structure=False)
        cut = max(len(r) for r in grid) - ncol
        self.statusBar().showMessage(f"{len(grid)}행 × {ncol}열을 붙여넣었습니다" + (f" (오른쪽 {cut}열은 열이 모자라 뺐습니다)" if cut > 0 else ""))

    def clear_cells(self):
        v, s = self.view(), self.sheet()
        if v is None:
            return
        idx = v.selectionModel().selectedIndexes()
        if not idx:
            return
        self.before_edit()
        for i in idx:
            s.set_value(i.row(), s.columns[i.column()].name, np.nan)
        self.model().reset()
        self.after_edit(structure=False)

    # ---- check ----------------------------------------------------------------------------
    def run_check(self):
        self.issues.clear()
        n = self.cur
        if n is None:
            return
        issues = n.bf.check() if not n.is_template else []
        names = [s.name for s in n.bf.sheets]
        for m in self.models:
            m.issues = {(i.row, i.column) for i in issues if i.sheet == m.sheet.name and i.row is not None}
            m.layoutChanged.emit()
        if not issues:
            it = QtWidgets.QListWidgetItem("✓ 고칠 곳이 없습니다.")
            it.setForeground(QtGui.QColor("#2a7a2a"))
            self.issues.addItem(it)
            return
        for i in issues:
            where = i.sheet + (f" {i.row + 1}행" if i.row is not None else "") + (f" · {i.column}" if i.column else "")
            it = QtWidgets.QListWidgetItem(f"{where}: {i.message}" if where else i.message)
            it.setData(QtCore.Qt.UserRole, (names.index(i.sheet) if i.sheet in names else -1, i.row, i.column))
            if i.serious:
                it.setForeground(QtGui.QColor("#a02020"))
            self.issues.addItem(it)

    def _jump_to_issue(self, it):
        d = it.data(QtCore.Qt.UserRole)
        if not d or d[0] < 0:
            return
        si, row, col = d
        self.tabs.setCurrentIndex(si)
        v, s = self.view(), self.sheet()
        c = s.names().index(col) if col in s.names() else 0
        if row is not None:
            ix = v.model().index(row, c)
            v.setCurrentIndex(ix); v.scrollTo(ix)

    # ---- CSV import -------------------------------------------------------------------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.import_csvs(paths)

    def import_csvs(self, paths=None, kind=None):
        """Particle CSVs or cell delivery summary CSVs into the open batch file."""
        n = self.cur
        if n is None or n.is_template:
            warn(self, "왼쪽에서 CSV를 넣을 배치를 먼저 고르세요 (없으면 '+ 새 배치')."); return
        if not paths:
            title = "입자 CSV 넣기" if kind == "particles" else "세포 전달 요약 CSV 넣기"
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, title, str(Path.home()), "CSV (*.csv *.txt *.tsv)")
            if not paths:
                return
        kinds = {p: classify_file(p) for p in paths}
        parts = [p for p, k in kinds.items() if k == "particles" or (kind == "particles" and k not in ("delivery",))]
        deliv = [p for p, k in kinds.items() if k == "delivery"]
        other = [p for p in paths if p not in parts + deliv]
        if parts:
            self._import_particles(parts)
        if deliv:
            self._import_delivery(deliv)
        if other:
            warn(self, "입자 CSV나 세포 전달 요약 CSV로 보이지 않아 건너뛰었습니다:\n" + "\n".join(Path(p).name for p in other))

    def _import_particles(self, paths):
        n = self.cur
        try:
            frame, notes = read_particle_files(paths)
        except Exception as e:
            warn(self, f"입자 CSV를 읽지 못했습니다.\n{e}"); return
        if not len(frame):
            warn(self, "\n".join(notes) or "입자 행이 없습니다."); return
        counts = frame.groupby("group", sort=False).size().to_dict()
        self.before_edit()
        if n.bf.particles is None:
            n.bf.add_sheet(SHEET_PARTICLES, KIND_PARTICLES)
        dlg = ImportDialog(self, n.bf, counts, "입자 CSV")
        if not dlg.exec():
            self._drop_last_undo(); return
        try:
            notes += add_particles(n.bf, frame, dlg.mapping())
        except ValueError as e:
            self._drop_last_undo(); warn(self, str(e)); return
        self.after_edit()
        self.tabs.setCurrentIndex([s.name for s in n.bf.sheets].index(n.bf.particles.name))
        self._report(notes, "입자 CSV")

    def _import_delivery(self, paths):
        n = self.cur
        try:
            d, notes = read_delivery_files(paths)
        except Exception as e:
            warn(self, f"세포 전달 CSV를 읽지 못했습니다.\n{e}"); return
        groups = d.groups()
        if not groups:
            warn(self, "세포 전달 그룹이 없습니다."); return
        self.before_edit()
        if not n.bf.measure_sheets():
            n.bf.add_sheet(SHEET_DELIVERY, KIND_MEASURE)
        counts = d.summary.groupby("group").size().to_dict()
        dlg = ImportDialog(self, n.bf, counts, "세포 전달 CSV", sheets=[s.name for s in n.bf.measure_sheets()])
        if not dlg.exec():
            self._drop_last_undo(); return
        sheet = dlg.sheet.currentText()
        try:
            notes += add_delivery(n.bf, sheet, d, dlg.mapping(), sub_from_names=dlg.subs.isChecked())
        except ValueError as e:
            self._drop_last_undo(); warn(self, str(e)); return
        self.after_edit()
        self.tabs.setCurrentIndex([s.name for s in n.bf.sheets].index(sheet))
        self._report(notes, "세포 전달 CSV")

    def _report(self, notes, what):
        if notes:
            QtWidgets.QMessageBox.information(self, f"{what} 넣기", "\n\n".join(notes) + "\n\n저장(⌘S)하면 Droplet Map에 반영됩니다.")
