"""Data page: which CSVs are loaded, how many particles per group, and what each column is for."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from ..core.io import ROLES, ROLE_LABELS
from .common import info_label, title_label


class DataPage(QtWidgets.QWidget):
    changed = QtCore.Signal()
    open_requested = QtCore.Signal()
    add_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(title_label("데이터"))
        lay.addWidget(info_label("이미지 분석에서 나온 입자별 CSV를 불러옵니다. group 열이 그룹(=조건 1세트), image 열이 이미지입니다. "
                                 "아래 표에서 어떤 측정값을 분석에 쓸지, 로그 변환과 형태값(크기 기반 신뢰도 적용) 여부를 고릅니다."))
        bar = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("CSV 열기 (새 프로젝트)…"); b1.clicked.connect(self.open_requested)
        b2 = QtWidgets.QPushButton("CSV 추가…"); b2.clicked.connect(self.add_requested)
        bar.addWidget(b1); bar.addWidget(b2)
        bar.addSpacing(20)
        bar.addWidget(QtWidgets.QLabel("최소 픽셀 수(QC):"))
        self.min_px = QtWidgets.QSpinBox(); self.min_px.setRange(0, 1000); self.min_px.setValue(3)
        self.min_px.setToolTip("area_px가 이보다 작은 입자는 잡음으로 보고 뺍니다.")
        self.min_px.valueChanged.connect(self._min_px_changed)
        bar.addWidget(self.min_px)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.summary = QtWidgets.QLabel("CSV를 불러오세요.")
        lay.addWidget(self.summary)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.groups = QtWidgets.QTableWidget()
        self.groups.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.groups.setSortingEnabled(True)
        self.roles = QtWidgets.QTableWidget()
        self.roles.setColumnCount(5)
        self.roles.setHorizontalHeaderLabels(["열", "역할", "로그 변환", "형태값", "중앙값"])
        split.addWidget(self._boxed("그룹", self.groups))
        split.addWidget(self._boxed("열 역할", self.roles))
        split.setSizes([500, 500])
        lay.addWidget(split, 1)

    def _boxed(self, title, w):
        box = QtWidgets.QGroupBox(title)
        l = QtWidgets.QVBoxLayout(box); l.addWidget(w)
        return box

    def _min_px_changed(self, v):
        if self.project is not None:
            self.project.settings.min_px = int(v)
            self.changed.emit()

    def set_project(self, project):
        self.project = project
        self.min_px.blockSignals(True); self.min_px.setValue(project.settings.min_px); self.min_px.blockSignals(False)
        self.refresh()

    def refresh(self):
        p = self.project
        if p is None:
            return
        df = p.particles
        files = df["source_file"].nunique() if "source_file" in df.columns else 1
        self.summary.setText(f"파일 {files}개 · 그룹 {df['group'].nunique()}개 · 이미지 {df.groupby(['group', 'image']).ngroups}장 · 입자 {len(df):,}개")
        g = df.groupby("group", sort=False)
        size_col = p.roles.size_column if p.roles.size_column in df.columns else None
        tab = pd.DataFrame({"이미지": g["image"].nunique(), "입자": g.size()})
        if size_col:
            tab["지름 중앙값 (µm)"] = g[size_col].median().round(3)
            tab["1.2 µm 이상"] = g[size_col].apply(lambda s: int((s >= 1.2).sum()))
        if "area_px" in df.columns:
            tab["QC 제외 비율"] = g["area_px"].apply(lambda s: f"{(s < p.settings.min_px).mean():.0%}")
        self.groups.setSortingEnabled(False)
        self.groups.clear()
        self.groups.setRowCount(len(tab)); self.groups.setColumnCount(len(tab.columns) + 1)
        self.groups.setHorizontalHeaderLabels(["그룹"] + list(tab.columns))
        for i, (name, row) in enumerate(tab.iterrows()):
            self.groups.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            for j, v in enumerate(row):
                it = QtWidgets.QTableWidgetItem()
                if isinstance(v, (int, float, np.integer, np.floating)):
                    it.setData(QtCore.Qt.DisplayRole, float(v) if isinstance(v, (float, np.floating)) else int(v))
                else:
                    it.setText(str(v))
                self.groups.setItem(i, j + 1, it)
        self.groups.setSortingEnabled(True)
        self.groups.resizeColumnsToContents()
        self._fill_roles()

    def _fill_roles(self):
        p = self.project
        cols = p.roles.columns
        self.roles.blockSignals(True)
        self.roles.setRowCount(len(cols))
        for i, spec in enumerate(cols):
            it = QtWidgets.QTableWidgetItem(spec.name)
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.roles.setItem(i, 0, it)
            combo = QtWidgets.QComboBox()
            for r in ROLES:
                combo.addItem(ROLE_LABELS[r], r)
            combo.setCurrentIndex(ROLES.index(spec.role))
            numeric = pd.api.types.is_numeric_dtype(p.particles[spec.name]) if spec.name in p.particles else False
            if not numeric:
                for r in ("feature", "measure"):
                    combo.model().item(ROLES.index(r)).setEnabled(False)
            combo.currentIndexChanged.connect(lambda _i, s=spec, c=combo: self._set_role(s, c.currentData()))
            self.roles.setCellWidget(i, 1, combo)
            for j, attr in ((2, "log"), (3, "shape")):
                chk = QtWidgets.QTableWidgetItem()
                chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                chk.setCheckState(QtCore.Qt.Checked if getattr(spec, attr) else QtCore.Qt.Unchecked)
                chk.setData(QtCore.Qt.UserRole, (spec.name, attr))
                self.roles.setItem(i, j, chk)
            med = ""
            if numeric:
                med = f"{p.particles[spec.name].median():.4g}"
            mi = QtWidgets.QTableWidgetItem(med); mi.setFlags(mi.flags() & ~QtCore.Qt.ItemIsEditable)
            self.roles.setItem(i, 4, mi)
        self.roles.blockSignals(False)
        try:
            self.roles.itemChanged.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.roles.itemChanged.connect(self._check_changed)
        self.roles.resizeColumnsToContents()

    def _set_role(self, spec, role):
        spec.role = role
        self.changed.emit()

    def _check_changed(self, item):
        data = item.data(QtCore.Qt.UserRole)
        if not data:
            return
        name, attr = data
        setattr(self.project.roles.spec(name), attr, item.checkState() == QtCore.Qt.Checked)
        self.changed.emit()
