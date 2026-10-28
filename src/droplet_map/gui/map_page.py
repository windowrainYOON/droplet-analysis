"""Particle map: UMAP scatter with selectable dot unit (particle / image / group / condition),
colouring, region selection, hierarchical sub-maps and a cohesion readout for weight tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.analysis import cohesion
from .common import (FlowLayout, MplCanvas, color, flow_buttons, form, info_label, scroll_panel, section,
                     short_group, title_label)

MAX_LEGEND = 16
SYMBOLS = ["o", "s", "t", "d", "+", "x", "star", "p", "h", "t1"]
UNITS = (("입자", "particle"), ("이미지 (입자 중앙 위치)", "image"), ("그룹 (입자 중앙 위치)", "group"),
         ("조건 값별 묶음", "cond"))


class MapPage(QtCore.QObject):
    run_requested = QtCore.Signal()                 # recompute the current map level
    submap_requested = QtCore.Signal(object, str)   # particle row indices, name
    inputs_changed = QtCore.Signal()                # measurements / groups going into the map
    batch_maps_requested = QtCore.Signal()          # one map per batch, each computed on its own
    weights_note = "가중치는 오른쪽 '측정값 가중치' 탭에서 바꿉니다."
    title = "입자 지도"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.result = None
        self.levels = []            # ParticleMap stack: [root, sub, subsub, ...]
        self.visible = None         # set of groups to show (None = all)
        self.sel_mask = None
        self.roi = None
        self.stale = False
        self._skip_values = {}      # condition -> set of values left out of the map ("" = not recorded)
        self._build_view()
        self._build_panel()

    # ---- layout ----------------------------------------------------------
    def _build_view(self):
        self.view = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.view)
        lay.addWidget(title_label("입자 지도"))
        self.crumbs = QtWidgets.QWidget()
        self.crumb_lay = FlowLayout(self.crumbs)
        lay.addWidget(self.crumbs)
        self.stale_label = QtWidgets.QLabel("")
        self.stale_label.setStyleSheet("color:#b05a00;")
        self.stale_label.setWordWrap(True)
        lay.addWidget(self.stale_label)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget(); tl = QtWidgets.QVBoxLayout(top); tl.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(background="w")
        self.plot.hideAxis("left"); self.plot.hideAxis("bottom")
        self.legend = self.plot.addLegend(offset=(5, 5), labelTextColor="#333", labelTextSize="8pt",
                                          brush=pg.mkBrush(255, 255, 255, 200))
        tl.addWidget(self.plot, 1)
        self.colorbar_label = QtWidgets.QLabel(""); self.colorbar_label.setStyleSheet("color:#555;")
        self.colorbar_label.setWordWrap(True)
        tl.addWidget(self.colorbar_label)
        split.addWidget(top)
        self.bottom = QtWidgets.QTabWidget()
        self.cluster_canvas = MplCanvas(None, 5, 3)
        self.comp_canvas = MplCanvas(None, 5, 3)
        self.sel_canvas = MplCanvas(None, 5, 3)
        self.batch_canvas = MplCanvas(None, 6, 4)
        self.bottom.addTab(self.cluster_canvas, "군집 해석")
        self.bottom.addTab(self.comp_canvas, "그룹별 군집 구성")
        self.bottom.addTab(self.sel_canvas, "선택 영역 요약")
        self.bottom.addTab(self.batch_canvas, "배치별 지도")
        self.batch_maps = None      # {batch: ParticleMap} computed per batch, or None
        self.batch_canvas.message("배치가 둘 이상이면 지금 지도를 배치마다 나눠 보여 줍니다.")
        split.addWidget(self.bottom)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)
        for c in (self.cluster_canvas, self.comp_canvas):
            c.message("'분석 실행'을 누르면 여기에 결과가 나옵니다.")
        self.sel_canvas.message("'영역 선택'을 켜고 상자를 움직이세요.")
        self._draw_crumbs()

    def _build_panel(self):
        self.d50 = self._dspin(0.1, 20, 1.2, 0.1, " µm", "형태값 신뢰도가 0.5가 되는 지름")
        self.kk = self._dspin(1, 12, 4, 0.5, "", "신뢰도 곡선의 가파름")
        self.nn = QtWidgets.QSpinBox(); self.nn.setRange(5, 100); self.nn.setValue(15)
        self.nn.setToolTip("UMAP 이웃 수. 크면 큰 구조, 작으면 세부 구조를 봅니다.")
        self.res = self._dspin(0.05, 5, 0.5, 0.05, "", "높을수록 군집이 잘게 나뉩니다")
        self.auto_rerun = QtWidgets.QCheckBox("가중치를 바꾸면 지도 자동 재계산")
        self.auto_rerun.setToolTip("지도 계산은 입자 수에 따라 수십 초 걸립니다.")
        calc = section("지도 계산",
                       form([("크기 신뢰도 d50", self.d50), ("곡선 k", self.kk), ("이웃 수", self.nn), ("군집 해상도", self.res)]),
                       info_label(self.weights_note),
                       self.auto_rerun,
                       flow_buttons((("지도 다시 계산 (현재 단계)", self.run_requested.emit),)))

        self.unit = QtWidgets.QComboBox()
        for label, key in UNITS:
            self.unit.addItem(label, key)
        self.unit_cond = QtWidgets.QComboBox()
        self.color_by = QtWidgets.QComboBox()
        self.shape_by = QtWidgets.QComboBox()
        self.shape_by.setToolTip("두 번째 조건을 점 모양으로 나타냅니다 (값이 많으면 앞의 10가지만 구분).")
        self.dot = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.dot.setRange(1, 20); self.dot.setValue(4)
        self.labels = QtWidgets.QCheckBox("묶음 점에 이름 표시"); self.labels.setChecked(True)
        for w in (self.unit, self.unit_cond, self.color_by, self.shape_by):
            w.currentIndexChanged.connect(self.redraw)
        self.dot.valueChanged.connect(self.redraw)
        self.labels.toggled.connect(self.redraw)
        self.unit.currentIndexChanged.connect(lambda _i: self.unit_cond.setEnabled(self.unit.currentData() == "cond"))
        self.unit_cond.setEnabled(False)
        self.cohesion = QtWidgets.QLabel("")
        self.cohesion.setWordWrap(True)
        self.cohesion.setToolTip("지도 위에서 각 입자의 가까운 이웃 15개 중 같은 색(범주)인 비율을, "
                                 "무작위로 섞였을 때를 0, 완전히 갈라졌을 때를 1로 맞춘 값입니다.")
        dots = section("점 표시",
                       form([("점 하나 =", self.unit), ("묶을 조건", self.unit_cond), ("색칠", self.color_by), ("모양", self.shape_by), ("점 크기", self.dot)]),
                       info_label("조건 값 하나만 보려면 왼쪽 '조건으로 거르기'를 씁니다 (나머지는 옅은 회색)."),
                       self.labels,
                       self.cohesion)

        self.map_feats = QtWidgets.QListWidget()
        self.map_feats.setMinimumHeight(110); self.map_feats.setMaximumHeight(170)
        self.map_feats.itemChanged.connect(self._inputs_edited)
        self.skip_cond = QtWidgets.QComboBox()
        self.skip_cond.currentIndexChanged.connect(self._fill_skip_values)
        self.skip_vals = QtWidgets.QListWidget()
        self.skip_vals.setMinimumHeight(90); self.skip_vals.setMaximumHeight(150)
        self.skip_vals.itemChanged.connect(self._skip_value_edited)
        self.only_visible = QtWidgets.QCheckBox("왼쪽에서 체크한 그룹만으로 지도 계산")
        self.only_visible.setToolTip("끄면 왼쪽 체크는 표시만 바꾸고, 지도는 모든 그룹으로 계산합니다.")
        self.only_visible.toggled.connect(self._inputs_edited)
        self.inputs_summary = QtWidgets.QLabel(""); self.inputs_summary.setWordWrap(True)
        self.inputs_summary.setStyleSheet("color:#555;")
        inputs = section("지도에 넣을 값",
                         info_label("체크를 끈 측정값은 지도 좌표와 군집 계산에서 빠집니다. 체크를 끈 조건 값의 그룹은 "
                                    "지도 계산에서 빠집니다. 기준 score와 그룹 비교에는 영향이 없습니다. "
                                    "QC까지 포함해 분석 전체에서 빼려면 데이터 탭의 '열 역할'을 바꾸세요."),
                         QtWidgets.QLabel("측정값"), self.map_feats,
                         form([("조건", self.skip_cond)]), self.skip_vals,
                         self.only_visible,
                         flow_buttons((("모두 넣기", self._include_all),
                                       ("지도 다시 계산", self.run_requested.emit))),
                         self.inputs_summary)

        self.cluster_list = QtWidgets.QListWidget()
        self.cluster_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.cluster_list.setMinimumHeight(140)
        self.select_btn = QtWidgets.QPushButton("영역 선택"); self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(self._toggle_roi)
        sel_row = QtWidgets.QWidget(); fl = FlowLayout(sel_row); fl.addWidget(self.select_btn)
        self.batch_pick = QtWidgets.QComboBox()
        self.batch_mode = QtWidgets.QComboBox()
        self.batch_mode.addItem("지금 지도를 배치마다 나눠 보기 (같은 좌표·같은 군집)", "split")
        self.batch_mode.addItem("배치마다 따로 계산한 지도 (군집 번호는 배치마다 다름)", "own")
        self.batch_mode.currentIndexChanged.connect(self._batch_mode_changed)
        batch_sec = section("배치",
                            info_label("전체 지도 = 모든 배치의 입자를 한 지도에 (색칠을 '조건: 배치'로 두면 배치 차이가 보입니다). "
                                       "배치 안 지도 = 고른 배치의 입자만으로 다시 계산해 그 배치 안의 군집을 봅니다. "
                                       "배치별 지도 = 아래 '배치별 지도' 탭."),
                            form([("배치", self.batch_pick)]),
                            flow_buttons((("이 배치만으로 지도", self._submap_batch),)),
                            form([("배치별 지도", self.batch_mode)]),
                            flow_buttons((("배치별 지도 그리기", self._show_batch_maps),)))
        hier = section("계층 지도 (하위 지도)",
                       info_label("고른 군집이나 선택 영역의 입자만으로 UMAP과 군집을 다시 계산합니다. "
                                  "하위 지도에서는 측정값을 그 입자들 안에서 다시 표준화하므로 작은 차이가 더 벌어져 보입니다. "
                                  "위쪽 경로를 눌러 상위 지도로 돌아갑니다."),
                       self.cluster_list,
                       flow_buttons((("선택 군집 → 하위 지도", self._submap_clusters),
                                     ("선택 영역 → 하위 지도", self._submap_roi),
                                     ("상위 지도로", self.go_up))),
                       sel_row)
        self.panel = scroll_panel(calc, inputs, dots, batch_sec, hier)

    def _dspin(self, lo, hi, val, step, suffix, tip):
        s = QtWidgets.QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
        s.setSuffix(suffix); s.setToolTip(tip)
        return s

    # ---- wiring ----------------------------------------------------------
    def set_project(self, project):
        self.project = project
        s = project.settings
        self.d50.setValue(s.d50); self.kk.setValue(s.k); self.nn.setValue(s.n_neighbors); self.res.setValue(s.resolution)
        self.result = None
        self.levels = []
        self.plot.clear(); self.legend.clear()
        self._skip_values = {k: set(map(str, v)) for k, v in s.map_skip_values.items()}
        self.only_visible.blockSignals(True); self.only_visible.setChecked(bool(s.map_skip_groups)); self.only_visible.blockSignals(False)
        self._fill_options()
        self._fill_batches()
        self.batch_maps = None
        self.fill_inputs()
        self._draw_crumbs()

    def apply_settings(self):
        s = self.project.settings
        s.d50, s.k, s.n_neighbors, s.resolution = self.d50.value(), self.kk.value(), self.nn.value(), self.res.value()
        s.map_skip_features = [self.map_feats.item(i).data(QtCore.Qt.UserRole) for i in range(self.map_feats.count())
                               if self.map_feats.item(i).checkState() != QtCore.Qt.Checked]
        s.map_skip_values = {k: sorted(v) for k, v in self._skip_values.items() if v}
        if self.only_visible.isChecked() and self.visible is not None:
            s.map_skip_groups = [g for g in self.project.groups if g not in self.visible]
        else:
            s.map_skip_groups = []

    # ---- what goes into the map -------------------------------------------
    def fill_inputs(self):
        """Measurement and condition-value check lists, from the project settings."""
        if self.project is None:
            return
        skip = set(self.project.settings.map_skip_features)
        self.map_feats.blockSignals(True); self.map_feats.clear()
        for f in self.project.roles.features():
            it = QtWidgets.QListWidgetItem(f)
            it.setData(QtCore.Qt.UserRole, f)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked if f in skip else QtCore.Qt.Checked)
            self.map_feats.addItem(it)
        self.map_feats.blockSignals(False)
        cur = self.skip_cond.currentText()
        self.skip_cond.blockSignals(True); self.skip_cond.clear()
        for c in self.project.view.schema:
            self.skip_cond.addItem(c.name)
        i = self.skip_cond.findText(cur); self.skip_cond.setCurrentIndex(max(i, 0))
        self.skip_cond.blockSignals(False)
        self._fill_skip_values()

    def _fill_skip_values(self):
        self.skip_vals.blockSignals(True); self.skip_vals.clear()
        name = self.skip_cond.currentText()
        if self.project is not None and name:
            vals = self.project.view.categorical(name)
            skip = self._skip_values.get(name, set())
            levels = [str(v) for v in sorted(vals.dropna().unique(), key=_level_key)]
            if vals.isna().any():
                levels.append("")
            for v in levels:
                n = int((vals.isna() if v == "" else vals.astype(str).eq(v)).sum())
                it = QtWidgets.QListWidgetItem(f"{v or '기록 안 함'}  (그룹 {n}개)")
                it.setData(QtCore.Qt.UserRole, v)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Unchecked if v in skip else QtCore.Qt.Checked)
                self.skip_vals.addItem(it)
        self.skip_vals.blockSignals(False)
        self._update_inputs_summary()

    def _skip_value_edited(self, _it=None):
        name = self.skip_cond.currentText()
        if not name:
            return
        self._skip_values[name] = {self.skip_vals.item(i).data(QtCore.Qt.UserRole) for i in range(self.skip_vals.count())
                                   if self.skip_vals.item(i).checkState() != QtCore.Qt.Checked}
        self._inputs_edited()

    def _inputs_edited(self, *_a):
        if self.project is None:
            return
        self.apply_settings()
        self._update_inputs_summary()
        if self.result is not None:
            self.set_stale("지도에 넣을 값이 바뀌었습니다. '지도 다시 계산'을 누르면 반영됩니다.")
        self.inputs_changed.emit()

    def _include_all(self):
        self._skip_values = {}
        self.only_visible.blockSignals(True); self.only_visible.setChecked(False); self.only_visible.blockSignals(False)
        self.map_feats.blockSignals(True)
        for i in range(self.map_feats.count()):
            self.map_feats.item(i).setCheckState(QtCore.Qt.Checked)
        self.map_feats.blockSignals(False)
        self._fill_skip_values()
        self._inputs_edited()

    def visible_changed_for_map(self):
        """Left-pane checks changed: matters for the map calculation only when 'only checked groups' is on."""
        if self.only_visible.isChecked():
            self._inputs_edited()

    def _update_inputs_summary(self):
        if self.project is None:
            return
        from ..core.analysis import map_skipped_groups
        s = self.project.settings
        feats = self.project.roles.features()
        n_f = len([f for f in feats if f not in s.map_skip_features])
        skipped = map_skipped_groups(self.project.view, s)
        groups = self.project.groups
        n_g = len([g for g in groups if g not in skipped])
        self.inputs_summary.setText(f"지도 계산에 쓰는 값: 측정값 {n_f}/{len(feats)}개 · 그룹 {n_g}/{len(groups)}개")

    def ui_state(self) -> dict:
        return {"unit": self.unit.currentData(), "unit_cond": self.unit_cond.currentText(),
                "color_by": self.color_by.currentText(), "shape_by": self.shape_by.currentText(), "dot": self.dot.value(),
                "labels": self.labels.isChecked(), "auto_rerun": self.auto_rerun.isChecked(),
                "skip_cond": self.skip_cond.currentText(),
                "bottom_tab": self.bottom.currentIndex()}

    def restore_ui(self, st: dict):
        if not st:
            return
        for w in (self.unit, self.unit_cond, self.color_by, self.shape_by):
            w.blockSignals(True)
        i = self.unit.findData(st.get("unit")); self.unit.setCurrentIndex(max(i, 0))
        self.unit_cond.setEnabled(self.unit.currentData() == "cond")
        i = self.unit_cond.findText(st.get("unit_cond", "")); self.unit_cond.setCurrentIndex(max(i, 0))
        i = self.color_by.findText(st.get("color_by", "")); self.color_by.setCurrentIndex(max(i, 0))
        i = self.shape_by.findText(st.get("shape_by", "")); self.shape_by.setCurrentIndex(max(i, 0))
        for w in (self.unit, self.unit_cond, self.color_by, self.shape_by):
            w.blockSignals(False)
        self.dot.setValue(int(st.get("dot", 4)))
        self.labels.setChecked(bool(st.get("labels", True)))
        self.auto_rerun.setChecked(bool(st.get("auto_rerun", False)))
        self.bottom.setCurrentIndex(int(st.get("bottom_tab", 0)))
        i = self.skip_cond.findText(st.get("skip_cond", ""))
        if i >= 0:
            self.skip_cond.setCurrentIndex(i)

    def _fill_options(self):
        cur = self.color_by.currentText()
        self.color_by.blockSignals(True)
        self.color_by.clear()
        self.color_by.addItem("그룹", ("group", None))
        self.color_by.addItem("군집", ("cluster", None))
        self.color_by.addItem("형태값 신뢰도 w", ("weight", None))
        names = []
        if self.project:
            for c in self.project.view.schema:
                if c.kind == "numeric":
                    self.color_by.addItem(f"조건: {c.label} (단계별)", ("cond_lv", c.name))
                self.color_by.addItem(f"조건: {c.label}", ("cond", c.name))
                names.append(c.name)
            for f in self.project.roles.features():
                self.color_by.addItem(f"측정값: {f}", ("feat", f))
        i = self.color_by.findText(cur)
        self.color_by.setCurrentIndex(max(i, 0))
        self.color_by.blockSignals(False)
        cur = self.shape_by.currentText()
        self.shape_by.blockSignals(True); self.shape_by.clear()
        self.shape_by.addItem("없음", None)
        for n in names:
            self.shape_by.addItem(f"조건: {n}", n)
        i = self.shape_by.findText(cur); self.shape_by.setCurrentIndex(max(i, 0))
        self.shape_by.blockSignals(False)
        cur = self.unit_cond.currentText()
        self.unit_cond.blockSignals(True); self.unit_cond.clear()
        for n in names:
            self.unit_cond.addItem(n)
        i = self.unit_cond.findText(cur); self.unit_cond.setCurrentIndex(max(i, 0))
        self.unit_cond.blockSignals(False)

    def _fill_batches(self):
        cur = self.batch_pick.currentText()
        self.batch_pick.clear()
        if self.project is not None:
            from ..core.batches import UNASSIGNED
            self.batch_pick.addItems([b for b in self.project.batches.names() if b != UNASSIGNED])
        i = self.batch_pick.findText(cur); self.batch_pick.setCurrentIndex(max(i, 0))

    def conditions_changed(self):
        self._fill_options()
        self._fill_batches()
        self.fill_inputs()
        self.redraw()

    def set_visible_groups(self, groups):
        self.visible = set(groups) if groups is not None else None
        self.redraw()

    def set_stale(self, text):
        self.stale = bool(text)
        self.stale_label.setText(text)

    def set_result(self, result):
        """New full analysis: the map hierarchy restarts at the root."""
        self.result = result
        self.batch_maps = None
        self.levels = [result.pmap] if result.pmap is not None else []
        self.set_stale("")
        self._fill_options()
        self._level_changed()

    def push_level(self, pmap):
        self.levels.append(pmap)
        self.set_stale("")
        self._level_changed()

    def replace_level(self, pmap):
        """Recomputed current level (e.g. after a weight change)."""
        if not self.levels:
            return
        self.levels[-1] = pmap
        if len(self.levels) == 1 and self.result is not None:
            self.result.pmap = pmap
        self.set_stale("")
        self._level_changed()

    def go_up(self):
        if len(self.levels) > 1:
            self.levels.pop()
            self._level_changed()

    def go_to(self, depth):
        del self.levels[depth + 1:]
        self._level_changed()

    def current(self):
        return self.levels[-1] if self.levels else None

    def current_rows(self) -> np.ndarray:
        pm = self.current()
        n = len(self.result.prepared.data)
        return np.arange(n) if pm is None or pm.idx is None else pm.idx

    def _level_changed(self):
        if self.roi is not None:
            self.select_btn.setChecked(False)
        self._draw_crumbs()
        self._fill_clusters()
        self.redraw()
        self._draw_clusters()
        self._draw_composition()
        if self.batch_mode.currentData() == "split":
            self.draw_batch_maps()
        self.plot.autoRange()

    def _draw_crumbs(self):
        while self.crumb_lay.count():
            it = self.crumb_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if not self.levels:
            lab = QtWidgets.QLabel("지도 단계: 아직 없음")
            lab.setStyleSheet("color:#777;")
            self.crumb_lay.addWidget(lab)
            return
        lab = QtWidgets.QLabel("지도 단계:")
        self.crumb_lay.addWidget(lab)
        for i, pm in enumerate(self.levels):
            if i:
                self.crumb_lay.addWidget(QtWidgets.QLabel("›"))
            n = len(pm.clusters)
            b = QtWidgets.QToolButton(); b.setText(f"{pm.name} ({n:,})")
            b.setAutoRaise(True)
            if i == len(self.levels) - 1:
                f = b.font(); f.setBold(True); b.setFont(f)
            b.clicked.connect(lambda _c=False, d=i: self.go_to(d))
            self.crumb_lay.addWidget(b)

    def _fill_clusters(self):
        self.cluster_list.clear()
        pm = self.current()
        if pm is None:
            return
        counts = pd.Series(pm.clusters).value_counts().sort_index()
        for c, n in counts.items():
            it = QtWidgets.QListWidgetItem(f"군집 {self._cname(c)}  ({n:,}개)")
            it.setData(QtCore.Qt.UserRole, int(c))
            it.setForeground(QtGui.QColor(color(int(c))))
            self.cluster_list.addItem(it)

    def _cname(self, c):
        pm = self.current()
        prefix = pm.name.replace("군집 ", "") + "." if pm is not None and pm.name.startswith("군집 ") else ""
        return f"{prefix}{c}"

    # ---- hierarchy requests ----------------------------------------------
    def _submap_clusters(self):
        pm = self.current()
        sel = [it.data(QtCore.Qt.UserRole) for it in self.cluster_list.selectedItems()]
        if pm is None or not sel:
            QtWidgets.QMessageBox.information(self.view, "계층 지도", "오른쪽 목록에서 군집을 하나 이상 고르세요.")
            return
        m = np.isin(pm.clusters, sel)
        m &= self._visible_mask()
        name = "군집 " + "+".join(self._cname(c) for c in sel)
        self.submap_requested.emit(self.current_rows()[m], name)

    def _submap_batch(self):
        b = self.batch_pick.currentText()
        if self.result is None or not b:
            QtWidgets.QMessageBox.information(self.view, "배치 안 지도", "'분석 실행' 후에 배치를 고르세요."); return
        d = self.result.prepared.data
        rows = np.flatnonzero(d["group"].map(self.project.batch_of).eq(b).to_numpy())
        if not len(rows):
            QtWidgets.QMessageBox.information(self.view, "배치 안 지도", f"'{b}' 배치에 분석된 입자가 없습니다."); return
        self.submap_requested.emit(rows, f"배치 {b}")

    def _batch_mode_changed(self):
        if self.batch_mode.currentData() == "split":
            self.draw_batch_maps()

    def _show_batch_maps(self):
        self.bottom.setCurrentWidget(self.batch_canvas)
        if self.batch_mode.currentData() == "own":
            self.batch_maps_requested.emit()
        else:
            self.draw_batch_maps()

    def set_batch_maps(self, maps):
        self.batch_maps = maps
        self.bottom.setCurrentWidget(self.batch_canvas)
        self.draw_batch_maps()

    def _batches_present(self, rows):
        bs = self.result.prepared.data["group"].iloc[rows].map(self.project.batch_of)
        return [b for b in self.project.batches.names() if (bs == b).any()], bs.to_numpy()

    def draw_batch_maps(self):
        c = self.batch_canvas
        if self.result is None or self.current() is None:
            c.message("'분석 실행' 후에 그립니다."); return
        own = self.batch_mode.currentData() == "own"
        if own and not self.batch_maps:
            c.message("'배치별 지도 그리기'를 누르면 배치마다 지도를 따로 계산합니다."); return
        c.clear()
        if own:
            panels = [(b, pm.umap, pm.clusters, None) for b, pm in self.batch_maps.items()]
            title = "배치마다 따로 계산한 지도 — 색 = 그 배치 안의 군집"
        else:
            pm = self.current()
            names, bs = self._batches_present(self.current_rows())
            if len(names) < 2:
                c.message("지금 지도에 배치가 하나뿐입니다."); return
            panels = [(b, pm.umap, pm.clusters, bs == b) for b in names]
            title = f"{pm.name} 지도를 배치마다 나눔 — 같은 좌표·같은 군집 번호"
        n = len(panels)
        cols = min(n, 3)
        rows_n = int(np.ceil(n / cols))
        for i, (b, xy, cl, mask) in enumerate(panels):
            ax = c.fig.add_subplot(rows_n, cols, i + 1)
            if mask is not None:
                ax.scatter(xy[~mask, 0], xy[~mask, 1], s=1, color="#dddddd", rasterized=True)
                xy, cl = xy[mask], cl[mask]
            ax.scatter(xy[:, 0], xy[:, 1], s=2, c=[color(int(k)) for k in cl], rasterized=True)
            comp = pd.Series(cl).value_counts(normalize=True).head(3)
            top = ", ".join(f"{self._cname(k) if not own else k}:{v:.0%}" for k, v in comp.items())
            ax.set_title(f"{b}  (입자 {len(cl):,})\n많은 군집 {top}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        c.fig.suptitle(title, fontsize=9)
        c.draw_idle()

    def batch_rows(self) -> dict:
        """Rows of the prepared data per batch (for per-batch maps)."""
        d = self.result.prepared.data
        b = d["group"].map(self.project.batch_of).to_numpy()
        return {name: np.flatnonzero(b == name) for name in self.project.batches.names() if (b == name).any()}

    def _submap_roi(self):
        if self.sel_mask is None or self.roi is None or not self.sel_mask.any():
            QtWidgets.QMessageBox.information(self.view, "계층 지도", "'영역 선택'을 켜고 상자 안에 입자를 담으세요.")
            return
        m = self.sel_mask & self._visible_mask()
        self.submap_requested.emit(self.current_rows()[m], f"선택 영역 {len(self.levels)}")

    def _visible_mask(self):
        d = self.result.prepared.data.iloc[self.current_rows()]
        if self.visible is None:
            return np.ones(len(d), bool)
        return d["group"].isin(self.visible).to_numpy()

    # ---- scatter ---------------------------------------------------------
    def _values(self, kind, key, d, pm, rows):
        """Per-particle colour values for the current level."""
        if kind == "group":
            return d["group"].to_numpy(), "cat"
        if kind == "cluster":
            return np.array([f"군집 {self._cname(c)}" for c in pm.clusters]), "cat"
        if kind == "cond_lv":
            vals = self.project.view.categorical(key)
            return d["group"].map(vals).fillna("기록 안 함").to_numpy(), "cat"
        if kind == "cond":
            if self.project.view.get(key).kind == "numeric":
                return d["group"].map(self.project.view.numeric(key)).to_numpy(float), "num"
            vals = self.project.view.categorical(key)
            return d["group"].map(vals).fillna("기록 안 함").to_numpy(), "cat"
        if kind == "weight":
            return self.result.prepared.w[rows], "num"
        v = d[key].to_numpy(float)
        if self.project.roles.spec(key).log:
            v = np.log10(np.clip(v, 1e-12, None))
        return v, "num"

    def redraw(self):
        self.plot.clear(); self.legend.clear()
        self.colorbar_label.setText("")
        self.cohesion.setText("")
        pm = self.current()
        if pm is None or self.result is None:
            return
        rows = self.current_rows()
        d = self.result.prepared.data.iloc[rows].reset_index(drop=True)
        xy = pm.umap
        kind, key = self.color_by.currentData() or ("group", None)
        if kind in ("cond", "cond_lv") and key not in self.project.view.names():
            kind, key = "group", None
        vals, vtype = self._values(kind, key, d, pm, rows)
        vis = self._visible_mask()
        shape_name = self.shape_by.currentData()
        shp = None
        if shape_name and shape_name in self.project.view.names():
            shp = d["group"].map(self.project.view.categorical(shape_name)).fillna("기록 안 함").to_numpy()
            shp_levels = sorted(pd.unique(shp[vis]), key=_level_key)
            sym_of = {v: SYMBOLS[i % len(SYMBOLS)] for i, v in enumerate(shp_levels)}
        unit = self.unit.currentData()
        size = self.dot.value()

        # faint background so hidden groups still show the map's shape
        if (~vis).any() or unit != "particle":
            bg = pg.ScatterPlotItem(xy[:, 0], xy[:, 1], size=max(2, size - 1), pen=None, brush=pg.mkBrush(200, 200, 200, 60))
            self.plot.addItem(bg)

        if unit == "particle":
            idx = np.flatnonzero(vis)
            idx = idx[np.random.default_rng(0).permutation(len(idx))]
            px, py, pv, psize, plabel = xy[idx, 0], xy[idx, 1], vals[idx], np.full(len(idx), size), None
            psym = [sym_of[v] for v in shp[idx]] if shp is not None else "o"
        else:
            keys = self._unit_keys(unit, d)
            if keys is None:
                self.colorbar_label.setText("'묶을 조건'을 고르세요.")
                return
            df = pd.DataFrame({"k": keys, "x": xy[:, 0], "y": xy[:, 1], "v": vals})[vis]
            df = df[df["k"].notna()]
            if df.empty:
                return
            agg = df.groupby("k", sort=False).agg(x=("x", "median"), y=("y", "median"), n=("x", "size"))
            if vtype == "cat":
                agg["v"] = df.groupby("k", sort=False)["v"].agg(lambda s: s.value_counts().index[0])
            else:
                agg["v"] = df.groupby("k", sort=False)["v"].median()
            nmax = agg["n"].max()
            px, py, pv = agg["x"].to_numpy(), agg["y"].to_numpy(), agg["v"].to_numpy()
            psize = size * 2 + 22 * np.sqrt(agg["n"].to_numpy() / nmax)
            plabel = [self._key_label(unit, k) for k in agg.index]
            psym = "o"
            if shp is not None:
                sdf = pd.DataFrame({"k": keys, "s": shp})[vis]
                maj = sdf[sdf["k"].notna()].groupby("k", sort=False)["s"].agg(lambda x: x.value_counts().index[0])
                psym = [sym_of[maj[k]] for k in agg.index]

        if vtype == "cat":
            cats = list(pd.unique(vals[vis])) if vis.any() else []
            if kind == "cluster":
                cats = sorted(cats, key=lambda s: [float(x) for x in s.split()[-1].split(".")])
            elif kind in ("cond", "cond_lv"):
                cats = sorted(cats, key=_level_key)
            brushes = {}
            for i, c in enumerate(cats):
                col = QtGui.QColor("#cfcfcf") if c == "기록 안 함" else QtGui.QColor(color(i))
                col.setAlpha(170 if unit == "particle" else 230)
                brushes[c] = pg.mkBrush(col)
            b = [brushes.get(v, pg.mkBrush("#cfcfcf")) for v in pv]
            pen = None if unit == "particle" else pg.mkPen("#333", width=0.8)
            self.plot.addItem(pg.ScatterPlotItem(px, py, size=psize, pen=pen, brush=b, symbol=psym))
            for c in cats[:MAX_LEGEND]:
                self.legend.addItem(pg.ScatterPlotItem(size=8, pen=None, brush=brushes[c]), short_group(str(c), 34))
            if len(cats) > MAX_LEGEND:
                self.legend.addItem(pg.ScatterPlotItem(size=0), f"… 외 {len(cats) - MAX_LEGEND}개")
            self._show_cohesion(xy[vis], vals[vis], self.color_by.currentText())
        else:
            pv = np.asarray(pv, float)
            finite = np.isfinite(pv)
            lo, hi = (np.nanpercentile(pv[finite], [2, 98]) if finite.any() else (0, 1))
            if hi <= lo:
                hi = lo + 1
            cmap = pg.colormap.get("viridis")
            t = np.clip((pv - lo) / (hi - lo), 0, 1)
            cols = cmap.map(np.nan_to_num(t), mode="qcolor")
            brushes = [pg.mkBrush(QtGui.QColor("#d9d9d9")) if not finite[i] else pg.mkBrush(c) for i, c in enumerate(cols)]
            pen = None if unit == "particle" else pg.mkPen("#333", width=0.8)
            self.plot.addItem(pg.ScatterPlotItem(px, py, size=psize, pen=pen, brush=brushes, symbol=psym))
            label = self.color_by.currentText().split(": ", 1)[-1]
            if kind == "cond":     # numeric condition: its recorded levels are the categories
                cat = d["group"].map(self.project.view.categorical(key)).fillna("기록 안 함").to_numpy()
                self._show_cohesion(xy[vis], cat[vis], self.color_by.currentText())
            self.colorbar_label.setText(f"{label}: 어두운 보라 = {lo:.3g}  →  노랑 = {hi:.3g}   (회색 = 값 없음)")

        if shp is not None:
            for v in shp_levels[:len(SYMBOLS)]:
                self.legend.addItem(pg.ScatterPlotItem(size=9, pen=pg.mkPen("#444"), brush=pg.mkBrush("#999"), symbol=sym_of[v]),
                                    f"{shape_name} {v}")
        if plabel is not None and self.labels.isChecked():
            for x, y, t in zip(px, py, plabel):
                ti = pg.TextItem(t, color="#222", anchor=(0, 1))
                ti.setPos(x, y)
                self.plot.addItem(ti)
        if kind == "cluster" and unit == "particle":
            for c in np.unique(pm.clusters):
                m = (pm.clusters == c) & vis
                if m.sum() == 0:
                    continue
                cx, cy = np.median(xy[m], axis=0)
                t = pg.TextItem(self._cname(c), color="#111", anchor=(.5, .5), fill=pg.mkBrush(255, 255, 255, 180))
                t.setPos(cx, cy); self.plot.addItem(t)
        if self.roi is not None:
            self.plot.addItem(self.roi)

    def _unit_keys(self, unit, d):
        if unit == "image":
            return (d["group"].astype(str) + " | " + d["image"].astype(str)).to_numpy()
        if unit == "group":
            return d["group"].to_numpy()
        name = self.unit_cond.currentText()
        if not name:
            return None
        vals = self.project.view.categorical(name)
        return d["group"].map(vals).to_numpy(object)

    def _key_label(self, unit, k):
        if unit == "image":
            g, im = str(k).split(" | ", 1)
            return f"{short_group(g, 24)} · {im}"
        if unit == "group":
            return short_group(str(k), 28)
        return f"{self.unit_cond.currentText()} {k}"

    def _show_cohesion(self, xy, labels, what):
        lab = pd.Series(labels, dtype=object).replace("기록 안 함", None).to_numpy()
        score = cohesion(xy, lab)
        if score is None:
            self.cohesion.setText("")
            return
        self.cohesion.setText(f"응집도 ({what.split(': ', 1)[-1]} 기준): <b>{score:.2f}</b>  "
                              f"(0 = 무작위로 섞임, 1 = 완전히 갈라짐)")
        self.last_cohesion = float(score)

    # ---- selection -------------------------------------------------------
    def _toggle_roi(self, on):
        pm = self.current()
        if on and pm is not None:
            xy = pm.umap
            cx, cy = np.median(xy, axis=0)
            w = (xy[:, 0].max() - xy[:, 0].min()) / 4
            h = (xy[:, 1].max() - xy[:, 1].min()) / 4
            self.roi = pg.RectROI([cx - w / 2, cy - h / 2], [w, h], pen=pg.mkPen("#d6456b", width=2))
            self.roi.addScaleHandle([0, 0], [1, 1])
            self.roi.sigRegionChangeFinished.connect(self._roi_changed)
            self.plot.addItem(self.roi)
            self.bottom.setCurrentWidget(self.sel_canvas)
            self._roi_changed()
        else:
            if self.roi is not None:
                self.plot.removeItem(self.roi)
            self.roi = None
            self.sel_mask = None
            if on:
                self.select_btn.setChecked(False)

    def _roi_changed(self):
        pm = self.current()
        if self.roi is None or pm is None:
            return
        pos, size = self.roi.pos(), self.roi.size()
        xy = pm.umap
        m = (xy[:, 0] >= pos.x()) & (xy[:, 0] <= pos.x() + size.x()) & (xy[:, 1] >= pos.y()) & (xy[:, 1] <= pos.y() + size.y())
        m &= self._visible_mask()
        self.sel_mask = m
        self._draw_selection(m)

    def _draw_selection(self, m):
        c = self.sel_canvas
        c.clear()
        rows = self.current_rows()
        d = self.result.prepared.data.iloc[rows]
        if m.sum() == 0:
            c.message("상자 안에 입자가 없습니다."); return
        ax1 = c.fig.add_subplot(1, 2, 1)
        share = d.loc[d.index[m], "group"].value_counts() / d["group"].value_counts()
        share = share.dropna().sort_values(ascending=True).tail(12)
        ax1.barh([short_group(g, 26) for g in share.index], share.values * 100, color="#2a6fdb")
        ax1.set_xlabel("그룹 입자 중 선택 영역 비율 (%)")
        ax1.set_title(f"선택 {int(m.sum()):,}개 ({m.mean():.1%}) — 어느 그룹에서 왔나", fontsize=9)
        ax1.tick_params(axis="y", labelsize=7)
        ax2 = c.fig.add_subplot(1, 2, 2)
        Z = self.result.prepared.Z.iloc[rows].reset_index(drop=True)
        diff = Z[m].median() - Z[~m].median()
        cols = ["#d6456b" if v > 0 else "#2a6fdb" for v in diff.values]
        ax2.bar(range(len(diff)), diff.values, color=cols)
        ax2.axhline(0, color="#999", lw=.8)
        ax2.set_xticks(range(len(diff)), diff.index, rotation=30, ha="right", fontsize=7)
        ax2.set_ylabel("나머지 대비 중앙값 차이 (z)")
        c.draw_idle()

    # ---- cluster plots ---------------------------------------------------
    def _draw_clusters(self):
        pm = self.current()
        c = self.cluster_canvas
        c.clear()
        if pm is None:
            c.draw_idle(); return
        p = self.result.prepared
        rows = self.current_rows()
        cl = pm.clusters
        Z = p.Z.iloc[rows].reset_index(drop=True).copy()
        for f in p.shape_features:
            Z[f] = Z[f] * p.w[rows]
        prof = Z.groupby(cl).median()
        ax = c.fig.add_subplot(111)
        im = ax.imshow(prof.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        labs = [f + ("*" if f in p.shape_features else "") for f in prof.columns]
        ax.set_xticks(range(prof.shape[1]), labs, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(prof.shape[0]), [f"군집 {self._cname(k)} (n={int((cl == k).sum()):,})" for k in prof.index], fontsize=7)
        c.fig.colorbar(im, ax=ax, label="군집 중앙값 (전체 기준 z)", shrink=.8)
        ax.set_title("군집마다 무엇이 다른가 (*형태값은 신뢰도 가중)", fontsize=9)
        c.draw_idle()

    def _draw_composition(self):
        pm = self.current()
        c = self.comp_canvas
        c.clear()
        if pm is None:
            c.draw_idle(); return
        d = self.result.prepared.data.iloc[self.current_rows()]
        comp = pd.crosstab(d["group"].to_numpy(), pm.clusters, normalize="index")
        order = [g for g in self.project.groups if g in comp.index]
        comp = comp.reindex(order)
        ax = c.fig.add_subplot(111)
        left = np.zeros(len(comp))
        for k in comp.columns:
            ax.barh(range(len(comp)), comp[k].values, left=left, color=color(int(k)), label=f"군집 {self._cname(k)}", height=.8)
            left += comp[k].values
        ax.set_yticks(range(len(comp)), [short_group(g, 30) for g in comp.index], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("입자 비율")
        ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
        ax.set_title(f"그룹별 군집 구성비 — {pm.name}", fontsize=9)
        c.draw_idle()

    # ---- exports -----------------------------------------------------------
    def particles_frame(self) -> pd.DataFrame:
        pm = self.current()
        rows = self.current_rows()
        d = self.result.prepared.data.iloc[rows].copy()
        d["shape_weight_w"] = self.result.prepared.w[rows]
        if pm is not None:
            d["umap_1"], d["umap_2"] = pm.umap[:, 0], pm.umap[:, 1]
            d["cluster"] = [self._cname(c) for c in pm.clusters]
            d["map_level"] = pm.name
        return d

    def save_map_image(self, path):
        from pyqtgraph.exporters import ImageExporter
        ex = ImageExporter(self.plot.plotItem)
        ex.parameters()["width"] = 2000
        ex.export(path)


def _level_key(v):
    try:
        return (0, float(str(v)), "")
    except ValueError:
        return (1, 0.0, str(v))
