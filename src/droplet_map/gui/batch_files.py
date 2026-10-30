"""Batch files from the app's side: the data are edited in Excel, the app only makes new batch
files (from the folder template or as a copy of another batch), opens them in Excel, puts
image-analysis CSVs into them and lists what needs fixing. Droplet Map reads the folder again
whenever a file changes."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.batchfile import (KIND_MEASURE, KIND_PARTICLES, SHEET_DELIVERY, SHEET_PARTICLES, BatchFile, batch_paths,
                              folder_template, read_batch_file, safe_file_name, write_batch_file)
from ..core.importing import (add_delivery, add_particles, conditions_from_names, read_delivery_files,
                              read_particle_files, suggest_conditions)
from ..core.io import classify_file
from .common import flow_buttons, info_label, warn


def open_in_excel(path):
    """Open a batch file in the app the system uses for .xlsx (Excel). Replaced in the self-test."""
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))


def create_batch(folder, name, src=None) -> Path:
    """A new batch file from the folder template (src None) or as a copy of another batch file:
    sheets, columns, formation conditions and batch values kept, measurements emptied."""
    folder = Path(folder)
    name = name.strip()
    if not name:
        raise ValueError("배치 이름을 입력하세요.")
    path = folder / (safe_file_name(name) + ".xlsx")
    if path.exists():
        raise ValueError(f"'{path.name}' 파일이 이미 있습니다. 다른 이름을 쓰세요.")
    base = folder_template(folder) if src is None else read_batch_file(src)[0]
    return write_batch_file(base.copy_for_new_batch(name), path)


def batch_choices(folder) -> list[tuple[str, Path]]:
    out = []
    for p in batch_paths(folder):
        try:
            out.append((read_batch_file(p)[0].name, p))
        except Exception:
            out.append((p.stem, p))
    return out


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


def import_csvs(parent, folder, paths, batch_path=None) -> Path | None:
    """Particle CSVs or cell delivery summary CSVs into one batch file (asked which). Returns the
    file written, or None."""
    files = batch_choices(folder)
    if not files:
        warn(parent, "CSV를 넣을 배치 파일이 없습니다. 먼저 '새 배치'를 만드세요."); return None
    if batch_path is None:
        labels = [f"{n}  ({p.name})" for n, p in files]
        pick, ok = QtWidgets.QInputDialog.getItem(parent, "CSV 넣기", "어느 배치의 데이터인가요?", labels, 0, False)
        if not ok:
            return None
        batch_path = files[labels.index(pick)][1]
    try:
        bf, _ = read_batch_file(batch_path)
    except Exception as e:
        warn(parent, f"{Path(batch_path).name}을(를) 읽지 못했습니다 (엑셀에서 열려 있으면 저장 후 다시 하세요).\n{e}"); return None
    kinds = {p: classify_file(p) for p in paths}
    parts = [p for p, k in kinds.items() if k == "particles"]
    deliv = [p for p, k in kinds.items() if k == "delivery"]
    other = [p for p in paths if p not in parts + deliv]
    notes = []
    try:
        if parts:
            notes += _particles(parent, bf, parts)
        if deliv:
            notes += _delivery(parent, bf, deliv)
    except _Cancel:
        return None
    except ValueError as e:
        warn(parent, str(e)); return None
    if other:
        warn(parent, "입자 CSV나 세포 전달 요약 CSV로 보이지 않아 건너뛰었습니다:\n" + "\n".join(Path(p).name for p in other))
    if not (parts or deliv):
        return None
    try:
        write_batch_file(bf, batch_path)
    except PermissionError:
        warn(parent, f"{Path(batch_path).name}을(를) 쓰지 못했습니다. 엑셀에서 열려 있으면 닫고 다시 하세요."); return None
    if notes:
        QtWidgets.QMessageBox.information(parent, "CSV 넣기", "\n\n".join(notes))
    return Path(batch_path)


class _Cancel(Exception):
    pass


def _particles(parent, bf, paths):
    frame, notes = read_particle_files(paths)
    if not len(frame):
        raise ValueError("\n".join(notes) or "입자 행이 없습니다.")
    if bf.particles is None:
        bf.add_sheet(SHEET_PARTICLES, KIND_PARTICLES)
    dlg = ImportDialog(parent, bf, frame.groupby("group", sort=False).size().to_dict(), "입자 CSV")
    if not dlg.exec():
        raise _Cancel
    return notes + add_particles(bf, frame, dlg.mapping())


def _delivery(parent, bf, paths):
    d, notes = read_delivery_files(paths)
    if not d.groups():
        raise ValueError("세포 전달 그룹이 없습니다.")
    if not bf.measure_sheets():
        bf.add_sheet(SHEET_DELIVERY, KIND_MEASURE)
    counts = d.summary.groupby("group").size().to_dict()
    dlg = ImportDialog(parent, bf, counts, "세포 전달 CSV", sheets=[s.name for s in bf.measure_sheets()])
    if not dlg.exec():
        raise _Cancel
    return notes + add_delivery(bf, dlg.sheet.currentText(), d, dlg.mapping(), sub_from_names=dlg.subs.isChecked())


def issues_text(folder) -> str:
    """What needs fixing in each batch file (duplicate / unknown condition IDs, text in number columns)."""
    out = []
    for name, p in batch_choices(folder):
        try:
            bf, notes = read_batch_file(p)
        except Exception as e:
            out.append(f"■ {p.name}: 읽지 못했습니다 ({e})"); continue
        issues = bf.check()
        lines = [f"  • {n}" for n in notes]
        for i in issues[:40]:
            where = i.sheet + (f" {i.row + 2}행" if i.row is not None else "") + (f" · {i.column}" if i.column else "")
            lines.append(f"  • {where}: {i.message}" if where else f"  • {i.message}")
        if len(issues) > 40:
            lines.append(f"  … 그 밖에 {len(issues) - 40}개")
        out.append(f"■ {p.name} ({name}): " + ("고칠 곳 없음" if not lines else "") + ("\n" + "\n".join(lines) if lines else ""))
    return "\n\n".join(out) or "배치 파일이 없습니다."
