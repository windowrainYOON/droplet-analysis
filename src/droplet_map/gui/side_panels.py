"""Left pane (group list with condition filter) and the shared measurement-weight panel."""
from __future__ import annotations

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.batches import UNASSIGNED
from .common import FlowLayout, info_label, section, short_group

KIND = QtCore.Qt.UserRole + 1       # tree item kind: batch / group / delivery-head / delivery


class GroupPanel(QtWidgets.QWidget):
    """Batches with their groups underneath (checked = shown on the particle map), ★ = reference.
    A condition filter checks only the groups with a chosen condition value."""
    visible_changed = QtCore.Signal()
    references_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 4, 6)
        head = QtWidgets.QLabel("배치 › 그룹")
        f = head.font(); f.setBold(True); f.setPointSize(f.pointSize() + 1); head.setFont(f)
        lay.addWidget(head)
        lay.addWidget(info_label("체크한 입자 형성 그룹만 지도에 표시합니다. 배치를 체크하면 그 아래 그룹이 함께 바뀝니다. "
                                 "★ = 기준 (오른쪽 클릭으로 지정)."))
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setWordWrap(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.setTextElideMode(QtCore.Qt.ElideNone)
        self.tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tree.setStyleSheet("QTreeWidget::item{padding:2px 1px;}")
        lay.addWidget(self.tree, 1)

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

    # ---- items --------------------------------------------------------
    def _group_items(self):
        out = []
        for i in range(self.tree.topLevelItemCount()):
            b = self.tree.topLevelItem(i)
            for j in range(b.childCount()):
                it = b.child(j)
                if it.data(0, KIND) == "group":
                    out.append(it)
        return out

    def set_project(self, project):
        self.project = project
        self.refresh()

    def refresh(self):
        p = self.project
        if p is None:
            return
        prev = self.visible_groups() if self.tree.topLevelItemCount() else None
        known = getattr(self, "_known", set()) if prev is not None else set()
        self._known = set(p.groups)
        counts = p.particles.groupby("group").size()
        view = p.view
        refs, brefs = p.conditions.references, p.batches.references
        self.tree.blockSignals(True)
        self.tree.clear()
        for b in p.batches.names():
            form = p.batches.groups_in(b, p.groups)
            deliv = p.batches.groups_in(b, p.delivery.groups())
            if not form and not deliv and b == UNASSIGNED:
                continue
            bi = QtWidgets.QTreeWidgetItem([("★ " if b in brefs else "") + f"{b}  ({len(form)})"])
            bi.setData(0, QtCore.Qt.UserRole, b); bi.setData(0, KIND, "batch")
            bf = bi.font(0); bf.setBold(True); bi.setFont(0, bf)
            bi.setFlags(bi.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsAutoTristate)
            if b == UNASSIGNED:
                bi.setForeground(0, QtGui.QColor("#b05a00"))
            tip = [f"<b>{b}</b>", f"입자 형성 그룹 {len(form)}개 · 세포 전달 그룹 {len(deliv)}개"]
            if b in p.batches.table.values.index:
                for c in p.batches.table.schema:
                    v = p.batches.table.value(b, c.name)
                    tip.append(f"{c.label}: {'—' if v is None else v}")
            bi.setToolTip(0, "<br>".join(tip))
            self.tree.addTopLevelItem(bi)
            for g in form:
                star = "★ " if g in refs else ""
                it = QtWidgets.QTreeWidgetItem([f"{star}{short_group(g, 60)}"])     # full name in the tooltip
                it.setData(0, QtCore.Qt.UserRole, g); it.setData(0, KIND, "group")
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                on = prev is None or g in prev or g not in known      # new groups start shown
                it.setCheckState(0, QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
                tip = [f"<b>{g}</b>", f"입자 {int(counts.get(g, 0)):,}개"]
                for c in view.schema:
                    v = view.value(g, c.name)
                    tip.append(f"{c.label}: {'—' if v is None else v}")
                it.setToolTip(0, "<br>".join(tip))
                if star:
                    it.setForeground(0, QtGui.QColor("#8a6d00"))
                bi.addChild(it)
            if deliv:
                di = QtWidgets.QTreeWidgetItem([f"세포 전달 ({len(deliv)})"])
                di.setData(0, KIND, "delivery-head")
                di.setFlags(QtCore.Qt.ItemIsEnabled)
                di.setForeground(0, QtGui.QColor("#5b6b8a"))
                for g in deliv:
                    used = p.delivery.uses.get(g)
                    ci = QtWidgets.QTreeWidgetItem([g])
                    ci.setData(0, QtCore.Qt.UserRole, g); ci.setData(0, KIND, "delivery")
                    ci.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                    ci.setForeground(0, QtGui.QColor("#5b6b8a"))
                    ci.setToolTip(0, f"<b>{g}</b><br>사용한 입자: {used or '—'}")
                    di.addChild(ci)
                bi.addChild(di)
            bi.setExpanded(True)
        self.tree.blockSignals(False)
        cur = self.cond.currentText()
        self.cond.blockSignals(True); self.cond.clear()
        for c in view.schema:
            self.cond.addItem(c.name)
        i = self.cond.findText(cur); self.cond.setCurrentIndex(max(i, 0))
        self.cond.blockSignals(False)
        self._fill_values()
        self._update_count()

    def _item_changed(self, it, _col):
        if it.data(0, KIND) in ("group", "batch"):
            self._update_count()
            self.visible_changed.emit()

    def _fill_values(self):
        self.value.clear()
        if self.project is None or not self.cond.currentText():
            return
        vals = self.project.view.categorical(self.cond.currentText())
        levels = sorted(vals.dropna().unique(), key=_sort_key)
        for v in levels:
            self.value.addItem(str(v))
        if vals.isna().any():
            self.value.addItem("기록 안 함")

    def _only_value(self):
        name, val = self.cond.currentText(), self.value.currentText()
        if not name:
            return
        vals = self.project.view.categorical(name)
        self.tree.blockSignals(True)
        for it in self._group_items():
            v = vals.get(it.data(0, QtCore.Qt.UserRole))
            ok = (pd.isna(v) and val == "기록 안 함") or (not pd.isna(v) and str(v) == val)
            it.setCheckState(0, QtCore.Qt.Checked if ok else QtCore.Qt.Unchecked)
        self.tree.blockSignals(False)
        self._update_count()
        self.visible_changed.emit()

    def _check_all(self, on):
        self.tree.blockSignals(True)
        for it in self._group_items():
            it.setCheckState(0, QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
        self.tree.blockSignals(False)
        self._update_count()
        self.visible_changed.emit()

    def _update_count(self):
        n = len(self.visible_groups())
        self.count.setText(f"지도에 표시: {n}/{len(self._group_items())}개 그룹")

    def visible_groups(self) -> set:
        return {it.data(0, QtCore.Qt.UserRole) for it in self._group_items()
                if it.checkState(0) == QtCore.Qt.Checked}

    def set_visible_groups(self, groups):
        groups = set(groups)
        self.tree.blockSignals(True)
        for it in self._group_items():
            it.setCheckState(0, QtCore.Qt.Checked if it.data(0, QtCore.Qt.UserRole) in groups else QtCore.Qt.Unchecked)
        self.tree.blockSignals(False)
        self._update_count()

    def _menu(self, pos):
        items = self.tree.selectedItems()
        groups = [it.data(0, QtCore.Qt.UserRole) for it in items if it.data(0, KIND) == "group"]
        batches = [it.data(0, QtCore.Qt.UserRole) for it in items if it.data(0, KIND) == "batch"
                   and it.data(0, QtCore.Qt.UserRole) != UNASSIGNED]
        if (not groups and not batches) or self.project is None:
            return
        m = QtWidgets.QMenu(self)
        acts = {}
        if groups:
            acts[m.addAction("기준 그룹(★)으로 지정")] = ("ref", groups)
            acts[m.addAction("기준 그룹 해제")] = ("unref", groups)
            m.addSeparator()
            acts[m.addAction("이 그룹만 지도에 표시")] = ("show", groups)
        if batches:
            if groups:
                m.addSeparator()
            acts[m.addAction("기준 배치(★)로 지정")] = ("bref", batches)
            acts[m.addAction("기준 배치 해제")] = ("bunref", batches)
            acts[m.addAction("이 배치만 지도에 표시")] = ("bshow", batches)
        act = m.exec(self.tree.mapToGlobal(pos))
        if act not in acts:
            return
        what, names = acts[act]
        p = self.project
        if what == "ref":
            p.conditions.references.update(names)
        elif what == "unref":
            p.conditions.references.difference_update(names)
        elif what == "bref":
            p.batches.references.update(names)
        elif what == "bunref":
            p.batches.references.difference_update(names)
        else:
            show = names if what == "show" else [g for b in names for g in p.batches.groups_in(b, p.groups)]
            self.set_visible_groups(show)
            self.visible_changed.emit()
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
