"""Left pane (group list with condition filter) and the shared measurement-weight panel."""
from __future__ import annotations

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from .common import FlowLayout, info_label, section, short_group


class GroupPanel(QtWidgets.QWidget):
    """Groups with a check box each (checked = shown on the particle map), ★ = reference group.
    A condition filter checks only the groups with a chosen condition value."""
    visible_changed = QtCore.Signal()
    references_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 4, 6)
        head = QtWidgets.QLabel("그룹")
        f = head.font(); f.setBold(True); f.setPointSize(f.pointSize() + 1); head.setFont(f)
        lay.addWidget(head)
        lay.addWidget(info_label("체크한 그룹만 지도에 표시합니다. ★ = 기준 그룹 (오른쪽 클릭으로 지정)."))
        self.list = QtWidgets.QListWidget()
        self.list.setWordWrap(True)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        self.list.itemChanged.connect(lambda _it: self.visible_changed.emit())
        self.list.setTextElideMode(QtCore.Qt.ElideNone)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.setResizeMode(QtWidgets.QListView.Adjust)
        self.list.setStyleSheet("QListWidget::item{padding:3px 2px;}")
        lay.addWidget(self.list, 1)

        self.cond = QtWidgets.QComboBox()
        self.value = QtWidgets.QComboBox()
        self.cond.currentIndexChanged.connect(self._fill_values)
        flt = QtWidgets.QWidget(); fl = QtWidgets.QVBoxLayout(flt); fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self.cond); fl.addWidget(self.value)
        btns = QtWidgets.QWidget(); bl = FlowLayout(btns)
        for text, fn in (("이 값만", self._only_value), ("모두", lambda: self._check_all(True)),
                         ("없음", lambda: self._check_all(False))):
            b = QtWidgets.QPushButton(text); b.clicked.connect(fn); bl.addWidget(b)
        fl.addWidget(btns)
        lay.addWidget(section("조건으로 거르기", flt))
        self.count = QtWidgets.QLabel(""); self.count.setStyleSheet("color:#666;")
        lay.addWidget(self.count)

    def set_project(self, project):
        self.project = project
        self.refresh()

    def refresh(self):
        p = self.project
        if p is None:
            return
        prev = self.visible_groups() if self.list.count() else None
        counts = p.particles.groupby("group").size()
        t = p.conditions
        self.list.blockSignals(True)
        self.list.clear()
        for g in p.groups:
            star = "★ " if g in t.references else ""
            it = QtWidgets.QListWidgetItem(f"{star}{g}")
            it.setData(QtCore.Qt.UserRole, g)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if prev is None or g in prev else QtCore.Qt.Unchecked)
            tip = [f"<b>{g}</b>", f"입자 {int(counts.get(g, 0)):,}개"]
            for c in t.schema:
                v = t.value(g, c.name)
                tip.append(f"{c.label}: {'—' if v is None else v}")
            it.setToolTip("<br>".join(tip))
            if star:
                it.setForeground(QtGui.QColor("#8a6d00"))
            self.list.addItem(it)
        self.list.blockSignals(False)
        cur = self.cond.currentText()
        self.cond.blockSignals(True); self.cond.clear()
        for c in t.schema:
            self.cond.addItem(c.name)
        i = self.cond.findText(cur); self.cond.setCurrentIndex(max(i, 0))
        self.cond.blockSignals(False)
        self._fill_values()
        self._update_count()

    def _fill_values(self):
        self.value.clear()
        if self.project is None or not self.cond.currentText():
            return
        vals = self.project.conditions.categorical(self.cond.currentText())
        levels = sorted(vals.dropna().unique(), key=_sort_key)
        for v in levels:
            self.value.addItem(str(v))
        if vals.isna().any():
            self.value.addItem("기록 안 함")

    def _only_value(self):
        name, val = self.cond.currentText(), self.value.currentText()
        if not name:
            return
        vals = self.project.conditions.categorical(name)
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            v = vals.get(it.data(QtCore.Qt.UserRole))
            ok = (pd.isna(v) and val == "기록 안 함") or (not pd.isna(v) and str(v) == val)
            it.setCheckState(QtCore.Qt.Checked if ok else QtCore.Qt.Unchecked)
        self.list.blockSignals(False)
        self._update_count()
        self.visible_changed.emit()

    def _check_all(self, on):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
        self.list.blockSignals(False)
        self._update_count()
        self.visible_changed.emit()

    def _update_count(self):
        n = len(self.visible_groups())
        self.count.setText(f"지도에 표시: {n}/{self.list.count()}개 그룹")

    def visible_groups(self) -> set:
        return {self.list.item(i).data(QtCore.Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == QtCore.Qt.Checked}

    def set_visible_groups(self, groups):
        groups = set(groups)
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(QtCore.Qt.Checked if it.data(QtCore.Qt.UserRole) in groups else QtCore.Qt.Unchecked)
        self.list.blockSignals(False)
        self._update_count()

    def _menu(self, pos):
        sel = [it.data(QtCore.Qt.UserRole) for it in self.list.selectedItems()]
        if not sel or self.project is None:
            return
        m = QtWidgets.QMenu(self)
        refs = self.project.conditions.references
        a_on = m.addAction("기준 그룹(★)으로 지정")
        a_off = m.addAction("기준 그룹 해제")
        m.addSeparator()
        a_show = m.addAction("이 그룹만 지도에 표시")
        act = m.exec(self.list.mapToGlobal(pos))
        if act is a_on:
            refs.update(sel)
        elif act is a_off:
            refs.difference_update(sel)
        elif act is a_show:
            self.set_visible_groups(sel)
            self.visible_changed.emit()
            return
        else:
            return
        self.refresh()
        self.references_changed.emit()


class WeightsPanel(QtWidgets.QWidget):
    """Per-measurement weights shared by the particle map and the reference score."""
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.rows = {}
        self._timer = QtCore.QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(500)
        self._timer.timeout.connect(self.changed)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(info_label("각 측정값이 지도와 기준 score에 얼마나 들어갈지 정합니다. "
                                 "0 = 빼기, 1 = 기본, 2 = 두 배. 특정 값의 가중치를 올리면 지도가 그 값을 따라 더 뚜렷이 "
                                 "갈라지고(응집도 상승), 내리면 그 값의 차이가 덜 반영됩니다. "
                                 "score는 바로 다시 계산되고, 지도는 '지도 다시 계산'(또는 자동 재계산)으로 반영됩니다."))
        self.grid_box = QtWidgets.QWidget()
        self.grid = QtWidgets.QVBoxLayout(self.grid_box); self.grid.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.grid_box)
        b = QtWidgets.QPushButton("모두 1로"); b.clicked.connect(self.reset)
        row = QtWidgets.QWidget(); fl = FlowLayout(row); fl.addWidget(b)
        lay.addWidget(row)
        lay.addStretch(1)

    def set_project(self, project):
        self.project = project
        self.rebuild()

    def rebuild(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.rows = {}
        if self.project is None:
            return
        fw = self.project.settings.feature_weights
        for f in self.project.roles.features():
            spec = self.project.roles.spec(f)
            box = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(box); v.setContentsMargins(0, 2, 0, 2); v.setSpacing(1)
            lab = QtWidgets.QLabel(f + ("  · 형태값 (크기 신뢰도 적용)" if spec.shape else ""))
            lab.setWordWrap(True)
            v.addWidget(lab)
            h = QtWidgets.QHBoxLayout()
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal); sl.setRange(0, 300); sl.setSingleStep(10); sl.setPageStep(50)
            sp = QtWidgets.QDoubleSpinBox(); sp.setRange(0, 3); sp.setSingleStep(0.1); sp.setDecimals(2); sp.setFixedWidth(70)
            val = float(fw.get(f, 1.0))
            sl.setValue(int(round(val * 100))); sp.setValue(val)
            sl.valueChanged.connect(lambda x, s=sp: (s.blockSignals(True), s.setValue(x / 100), s.blockSignals(False)))
            sp.valueChanged.connect(lambda x, s=sl: (s.blockSignals(True), s.setValue(int(round(x * 100))), s.blockSignals(False)))
            sl.valueChanged.connect(lambda _x, name=f: self._edited(name))
            sp.valueChanged.connect(lambda _x, name=f: self._edited(name))
            h.addWidget(sl, 1); h.addWidget(sp)
            v.addLayout(h)
            self.grid.addWidget(box)
            self.rows[f] = sp

    def _edited(self, name):
        self.project.settings.feature_weights[name] = float(self.rows[name].value())
        self._timer.start()

    def reset(self):
        for f, sp in self.rows.items():
            sp.setValue(1.0)
        self.project.settings.feature_weights = {f: 1.0 for f in self.rows}
        self._timer.start()


def _sort_key(v):
    try:
        return (0, float(str(v)), "")
    except ValueError:
        return (1, 0.0, str(v))
