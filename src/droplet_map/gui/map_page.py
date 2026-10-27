"""Particle map: UMAP scatter coloured by group / cluster / condition / measurement, with a
rectangle selection that summarises the chosen particles, plus cluster interpretation plots."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .common import MplCanvas, color, info_label, title_label, short_group

MAX_LEGEND = 16


class MapPage(QtWidgets.QWidget):
    run_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.result = None
        self.sel_mask = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(title_label("입자 지도"))
        lay.addWidget(info_label("비슷한 입자끼리 가까이 놓은 지도(UMAP)와 Leiden 군집입니다. 멀리 떨어진 덩어리 사이의 거리는 해석하지 않습니다. "
                                 "'영역 선택'을 켜고 상자를 움직이면 그 안의 입자를 요약합니다."))
        bar = QtWidgets.QHBoxLayout()
        self.d50 = self._dspin(0.1, 20, 1.2, 0.1, " µm", "형태값 신뢰도가 0.5가 되는 지름")
        self.kk = self._dspin(1, 12, 4, 0.5, "", "신뢰도 곡선의 가파름")
        self.nn = QtWidgets.QSpinBox(); self.nn.setRange(5, 100); self.nn.setValue(15)
        self.res = self._dspin(0.05, 5, 0.5, 0.05, "", "높을수록 군집이 잘게 나뉩니다")
        for lab, w in (("d50", self.d50), ("k", self.kk), ("이웃 수", self.nn), ("군집 해상도", self.res)):
            bar.addWidget(QtWidgets.QLabel(lab)); bar.addWidget(w)
        run = QtWidgets.QPushButton("분석 실행"); run.setDefault(True)
        run.clicked.connect(self.run_requested)
        bar.addWidget(run)
        bar.addSpacing(16)
        bar.addWidget(QtWidgets.QLabel("색칠:"))
        self.color_by = QtWidgets.QComboBox(); self.color_by.setMinimumWidth(200)
        self.color_by.currentIndexChanged.connect(self.redraw)
        bar.addWidget(self.color_by)
        self.select_btn = QtWidgets.QPushButton("영역 선택"); self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(self._toggle_roi)
        bar.addWidget(self.select_btn)
        bar.addStretch(1)
        lay.addLayout(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setAspectLocked(False)
        self.plot.hideAxis("left"); self.plot.hideAxis("bottom")
        self.legend = self.plot.addLegend(offset=(5, 5), labelTextColor="#333", labelTextSize="8pt")
        ll.addWidget(self.plot, 1)
        self.colorbar_label = QtWidgets.QLabel(""); self.colorbar_label.setStyleSheet("color:#555;")
        ll.addWidget(self.colorbar_label)
        split.addWidget(left)
        right = QtWidgets.QTabWidget()
        self.cluster_canvas = MplCanvas(self, 5, 4)
        self.comp_canvas = MplCanvas(self, 5, 4)
        self.sel_canvas = MplCanvas(self, 5, 4)
        right.addTab(self.cluster_canvas, "군집 해석")
        right.addTab(self.comp_canvas, "그룹별 군집 구성")
        right.addTab(self.sel_canvas, "선택 영역 요약")
        self.right_tabs = right
        split.addWidget(right)
        split.setSizes([620, 520])
        lay.addWidget(split, 1)
        self.roi = None
        self.scatter = None
        for c in (self.cluster_canvas, self.comp_canvas):
            c.message("'분석 실행'을 누르면 여기에 결과가 나옵니다.")
        self.sel_canvas.message("'영역 선택'을 켜고 상자를 움직이세요.")

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
        self.plot.clear(); self.legend.clear()
        self._fill_color_options()

    def apply_settings(self):
        s = self.project.settings
        s.d50, s.k, s.n_neighbors, s.resolution = self.d50.value(), self.kk.value(), self.nn.value(), self.res.value()

    def _fill_color_options(self):
        cur = self.color_by.currentText()
        self.color_by.blockSignals(True)
        self.color_by.clear()
        self.color_by.addItem("그룹", ("group", None))
        self.color_by.addItem("군집", ("cluster", None))
        self.color_by.addItem("형태값 신뢰도 w", ("weight", None))
        if self.project:
            for c in self.project.conditions.schema:
                self.color_by.addItem(f"조건: {c.label}", ("cond", c.name))
            for f in self.project.roles.features():
                self.color_by.addItem(f"측정값: {f}", ("feat", f))
        i = self.color_by.findText(cur)
        self.color_by.setCurrentIndex(max(i, 0))
        self.color_by.blockSignals(False)

    def conditions_changed(self):
        self._fill_color_options()
        self.redraw()

    def set_result(self, result):
        self.result = result
        self._fill_color_options()
        self.redraw()
        self._draw_clusters()
        self._draw_composition()
        if self.roi is not None:
            self._roi_changed()

    # ---- scatter ---------------------------------------------------------
    def redraw(self):
        if self.result is None or self.result.pmap is None:
            return
        r = self.result
        xy = r.pmap.umap
        d = r.prepared.data
        kind, key = self.color_by.currentData() or ("group", None)
        self.plot.clear(); self.legend.clear()
        self.colorbar_label.setText("")
        order = np.random.default_rng(0).permutation(len(xy))  # avoid one group painting over another
        if kind in ("group", "cluster", "cond") and not (kind == "cond" and self.project.conditions.get(key).kind == "numeric"):
            if kind == "group":
                labels = d["group"].to_numpy()
            elif kind == "cluster":
                labels = np.array([f"군집 {c}" for c in r.pmap.clusters])
            else:
                vals = self.project.conditions.categorical(key)
                labels = d["group"].map(vals).fillna("기록 안 함").to_numpy()
            cats = list(pd.unique(labels))
            if kind == "cluster":
                cats = sorted(cats, key=lambda s: int(s.split()[-1]))
            brushes = {}
            for i, c in enumerate(cats):
                col = QtGui.QColor("#cfcfcf") if c == "기록 안 함" else QtGui.QColor(color(i))
                col.setAlpha(170)
                brushes[c] = pg.mkBrush(col)
            b = [brushes[l] for l in labels[order]]
            self.scatter = pg.ScatterPlotItem(xy[order, 0], xy[order, 1], size=3.5, pen=None, brush=b)
            self.plot.addItem(self.scatter)
            for i, c in enumerate(cats[:MAX_LEGEND]):
                item = pg.ScatterPlotItem(size=8, pen=None, brush=brushes[c])
                self.legend.addItem(item, short_group(str(c), 34))
            if len(cats) > MAX_LEGEND:
                self.legend.addItem(pg.ScatterPlotItem(size=0), f"… 외 {len(cats) - MAX_LEGEND}개")
            if kind == "cluster":
                for c in np.unique(r.pmap.clusters):
                    m = r.pmap.clusters == c
                    cx, cy = np.median(xy[m], axis=0)
                    t = pg.TextItem(str(c), color="#111", anchor=(.5, .5), fill=pg.mkBrush(255, 255, 255, 180))
                    t.setPos(cx, cy); self.plot.addItem(t)
        else:
            if kind == "weight":
                v = r.prepared.w; label = "형태값 신뢰도 w"
            elif kind == "feat":
                v = d[key].to_numpy(float); label = key
                if self.project.roles.spec(key).log:
                    v = np.log10(np.clip(v, 1e-12, None)); label = f"log10 {key}"
            else:
                v = d["group"].map(self.project.conditions.numeric(key)).to_numpy(float); label = key
            finite = np.isfinite(v)
            lo, hi = (np.nanpercentile(v[finite], [2, 98]) if finite.any() else (0, 1))
            if hi <= lo:
                hi = lo + 1
            cmap = pg.colormap.get("viridis")
            t = np.clip((v - lo) / (hi - lo), 0, 1)
            cols = cmap.map(np.nan_to_num(t[order]), mode="qcolor")
            brushes = [pg.mkBrush(QtGui.QColor("#d9d9d9")) if not finite[order][i] else pg.mkBrush(c) for i, c in enumerate(cols)]
            self.scatter = pg.ScatterPlotItem(xy[order, 0], xy[order, 1], size=3.5, pen=None, brush=brushes)
            self.plot.addItem(self.scatter)
            self.colorbar_label.setText(f"{label}: 어두운 보라 = {lo:.3g}  →  노랑 = {hi:.3g}   (회색 = 값 없음)")
        if self.roi is not None:
            self.plot.addItem(self.roi)

    # ---- selection -------------------------------------------------------
    def _toggle_roi(self, on):
        if on and self.result is not None and self.result.pmap is not None:
            xy = self.result.pmap.umap
            cx, cy = np.median(xy, axis=0)
            w = (xy[:, 0].max() - xy[:, 0].min()) / 4
            h = (xy[:, 1].max() - xy[:, 1].min()) / 4
            self.roi = pg.RectROI([cx - w / 2, cy - h / 2], [w, h], pen=pg.mkPen("#d6456b", width=2))
            self.roi.addScaleHandle([0, 0], [1, 1])
            self.roi.sigRegionChangeFinished.connect(self._roi_changed)
            self.plot.addItem(self.roi)
            self.right_tabs.setCurrentWidget(self.sel_canvas)
            self._roi_changed()
        else:
            if self.roi is not None:
                self.plot.removeItem(self.roi)
            self.roi = None
            if on:
                self.select_btn.setChecked(False)

    def _roi_changed(self):
        if self.roi is None or self.result is None:
            return
        pos, size = self.roi.pos(), self.roi.size()
        xy = self.result.pmap.umap
        m = (xy[:, 0] >= pos.x()) & (xy[:, 0] <= pos.x() + size.x()) & (xy[:, 1] >= pos.y()) & (xy[:, 1] <= pos.y() + size.y())
        self.sel_mask = m
        self._draw_selection(m)

    def _draw_selection(self, m):
        c = self.sel_canvas
        c.clear()
        r = self.result
        d = r.prepared.data
        if m.sum() == 0:
            c.message("상자 안에 입자가 없습니다."); return
        ax1 = c.fig.add_subplot(2, 1, 1)
        share = d.loc[m, "group"].value_counts() / d["group"].value_counts()
        share = share.dropna().sort_values(ascending=True).tail(12)
        ax1.barh([short_group(g, 34) for g in share.index], share.values * 100, color="#2a6fdb")
        ax1.set_xlabel("그룹 입자 중 선택 영역에 든 비율 (%)")
        ax1.set_title(f"선택한 입자 {int(m.sum()):,}개 ({m.mean():.1%}) — 어느 그룹에서 왔나", fontsize=9)
        ax1.tick_params(axis="y", labelsize=7)
        ax2 = c.fig.add_subplot(2, 1, 2)
        Z = r.prepared.Z
        diff = Z[m].median() - Z[~m].median()
        cols = ["#d6456b" if v > 0 else "#2a6fdb" for v in diff.values]
        ax2.bar(range(len(diff)), diff.values, color=cols)
        ax2.axhline(0, color="#999", lw=.8)
        ax2.set_xticks(range(len(diff)), diff.index, rotation=30, ha="right", fontsize=7)
        ax2.set_ylabel("나머지 대비 중앙값 차이 (z)")
        c.draw_idle()

    # ---- cluster plots ---------------------------------------------------
    def _draw_clusters(self):
        r = self.result
        c = self.cluster_canvas
        c.clear()
        if r is None or r.pmap is None:
            return
        cl = r.pmap.clusters
        Z = r.prepared.Z.copy()
        for f in r.prepared.shape_features:
            Z[f] = Z[f] * r.prepared.w
        prof = Z.groupby(cl).median()
        ax = c.fig.add_subplot(111)
        im = ax.imshow(prof.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        labs = [f + ("*" if f in r.prepared.shape_features else "") for f in prof.columns]
        ax.set_xticks(range(prof.shape[1]), labs, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(prof.shape[0]), [f"군집 {k} (n={int((cl == k).sum()):,})" for k in prof.index], fontsize=8)
        c.fig.colorbar(im, ax=ax, label="군집 중앙값 (robust z)", shrink=.8)
        ax.set_title("군집마다 무엇이 다른가 (*형태값은 신뢰도 가중)", fontsize=9)
        c.draw_idle()

    def _draw_composition(self):
        r = self.result
        c = self.comp_canvas
        c.clear()
        if r is None or r.profiles.composition is None:
            return
        comp = r.profiles.composition
        ax = c.fig.add_subplot(111)
        left = np.zeros(len(comp))
        for k in comp.columns:
            ax.barh(range(len(comp)), comp[k].values, left=left, color=color(int(k)), label=f"군집 {k}", height=.8)
            left += comp[k].values
        ax.set_yticks(range(len(comp)), [short_group(g, 34) for g in comp.index], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("입자 비율")
        ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
        ax.set_title("그룹별 군집 구성비", fontsize=9)
        c.draw_idle()
