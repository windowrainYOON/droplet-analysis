"""Particle map: UMAP scatter coloured / shaped / filtered by group, cluster, condition or
measurement; measurement weights that reshape the map; drill-down sub-maps (hierarchical UMAP)
of one cluster or a selected region; and cluster interpretation plots for the current level."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.analysis import ParticleMap, cohesion, particle_map
from .common import MplCanvas, WeightsTable, color, info_label, run_in_thread, short_group, title_label, warn

MAX_LEGEND = 16
SYMBOLS = ["o", "t", "s", "d", "star", "p", "h", "t1", "+", "x"]
NOT_RECORDED = "기록 안 함"


class MapPage(QtWidgets.QWidget):
    run_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.result = None
        self.sel_mask = None
        self.levels = []          # [{"idx": rows of prepared.data, "pmap": ParticleMap, "name": str}]
        self._replay = []         # sub-levels to recompute after a re-run with new weights
        self._bg = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(title_label("입자 지도"))
        lay.addWidget(info_label("비슷한 입자끼리 가까이 놓은 지도(UMAP)와 Leiden 군집입니다. 멀리 떨어진 덩어리 사이의 거리는 해석하지 않습니다. "
                                 "점의 색·모양을 조건별로 고르고, '표시'로 일부 조건만 강조할 수 있습니다. "
                                 "군집이나 선택 영역 '안으로' 들어가면 그 입자들만으로 UMAP을 다시 그립니다(계층적 지도). "
                                 "'가중치' 탭에서 측정값의 비중을 바꾸면 지도가 그 측정값 위주로 다시 뭉칩니다."))
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
        bar.addStretch(1)
        lay.addLayout(bar)

        bar2 = QtWidgets.QHBoxLayout()
        bar2.addWidget(QtWidgets.QLabel("색:"))
        self.color_by = QtWidgets.QComboBox(); self.color_by.setMinimumWidth(190)
        bar2.addWidget(self.color_by)
        bar2.addWidget(QtWidgets.QLabel("모양:"))
        self.shape_by = QtWidgets.QComboBox(); self.shape_by.setMinimumWidth(140)
        self.shape_by.setToolTip("두 번째 조건을 점 모양으로 나타냅니다 (값이 많으면 앞의 10가지만 구분).")
        bar2.addWidget(self.shape_by)
        bar2.addWidget(QtWidgets.QLabel("표시:"))
        self.show_only = QtWidgets.QComboBox(); self.show_only.setMinimumWidth(160)
        self.show_only.setToolTip("고른 조건 값의 입자만 색칠하고 나머지는 연한 회색으로 둡니다.")
        bar2.addWidget(self.show_only)
        for cb in (self.color_by, self.shape_by, self.show_only):
            cb.currentIndexChanged.connect(self.redraw)
        self.select_btn = QtWidgets.QPushButton("영역 선택"); self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(self._toggle_roi)
        bar2.addWidget(self.select_btn)
        bar2.addStretch(1)
        lay.addLayout(bar2)

        bar3 = QtWidgets.QHBoxLayout()
        self.crumb = QtWidgets.QLabel("지도: 전체")
        self.crumb.setStyleSheet("font-weight:bold;")
        bar3.addWidget(self.crumb)
        self.up_btn = QtWidgets.QPushButton("◂ 상위로"); self.up_btn.clicked.connect(self.go_up)
        bar3.addWidget(self.up_btn)
        bar3.addSpacing(12)
        self.cluster_pick = QtWidgets.QComboBox(); self.cluster_pick.setMinimumWidth(150)
        bar3.addWidget(self.cluster_pick)
        self.into_cluster_btn = QtWidgets.QPushButton("군집 안으로 ▸"); self.into_cluster_btn.clicked.connect(self.into_cluster)
        bar3.addWidget(self.into_cluster_btn)
        self.into_sel_btn = QtWidgets.QPushButton("선택 영역 안으로 ▸"); self.into_sel_btn.clicked.connect(self.into_selection)
        bar3.addWidget(self.into_sel_btn)
        self.busy = QtWidgets.QLabel(""); self.busy.setStyleSheet("color:#b05a00;")
        bar3.addWidget(self.busy)
        bar3.addStretch(1)
        lay.addLayout(bar3)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setAspectLocked(False)
        self.plot.hideAxis("left"); self.plot.hideAxis("bottom")
        self.legend = self.plot.addLegend(offset=(5, 5), labelTextColor="#333", labelTextSize="8pt")
        ll.addWidget(self.plot, 1)
        self.colorbar_label = QtWidgets.QLabel(""); self.colorbar_label.setStyleSheet("color:#555;")
        ll.addWidget(self.colorbar_label)
        self.cohesion_label = QtWidgets.QLabel(""); self.cohesion_label.setStyleSheet("color:#555;")
        self.cohesion_label.setToolTip("실루엣 점수(-1–1): 같은 색의 점끼리 얼마나 한곳에 모여 있고 다른 색과 떨어져 있는지. "
                                       "가중치를 바꾼 뒤 이 값이 오르면 그 색(조건)으로 더 잘 뭉친 것입니다.")
        ll.addWidget(self.cohesion_label)
        split.addWidget(left)

        right = QtWidgets.QTabWidget()
        wtab = QtWidgets.QWidget(); wl = QtWidgets.QVBoxLayout(wtab)
        wl.addWidget(info_label("지도를 만들 때 각 측정값(표준화 값)에 곱하는 가중치입니다. 높이면 그 측정값이 비슷한 입자끼리 더 강하게 뭉치고, "
                                "0이면 지도에서 빠집니다. 기준 score 페이지의 가중치와 같은 값입니다."))
        self.weights = WeightsTable()
        self.weights.changed.connect(self._weights_edited)
        wl.addWidget(self.weights, 1)
        wb = QtWidgets.QHBoxLayout()
        self.apply_w = QtWidgets.QPushButton("가중치 적용 → 지도 다시 계산"); self.apply_w.clicked.connect(self.apply_weights)
        reset = QtWidgets.QPushButton("모두 1로"); reset.clicked.connect(self.weights.reset)
        wb.addWidget(self.apply_w); wb.addWidget(reset); wb.addStretch(1)
        wl.addLayout(wb)
        self.w_note = QtWidgets.QLabel(""); self.w_note.setStyleSheet("color:#b05a00;"); self.w_note.setWordWrap(True)
        wl.addWidget(self.w_note)
        right.addTab(wtab, "가중치")
        self.cluster_canvas = MplCanvas(self, 5, 4)
        self.comp_canvas = MplCanvas(self, 5, 4)
        self.sel_canvas = MplCanvas(self, 5, 4)
        right.addTab(self.cluster_canvas, "군집 해석")
        right.addTab(self.comp_canvas, "그룹별 군집 구성")
        right.addTab(self.sel_canvas, "선택 영역 요약")
        self.right_tabs = right
        split.addWidget(right)
        split.setSizes([640, 500])
        lay.addWidget(split, 1)
        self.roi = None
        self.scatter = None
        for c in (self.cluster_canvas, self.comp_canvas):
            c.message("'분석 실행'을 누르면 여기에 결과가 나옵니다.")
        self.sel_canvas.message("'영역 선택'을 켜고 상자를 움직이세요.")
        self._update_level_controls()

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
        self.levels = []; self._replay = []
        self.plot.clear(); self.legend.clear()
        self.features_changed()
        self._fill_options()
        self._update_level_controls()

    def features_changed(self):
        if self.project is None:
            return
        roles = self.project.roles
        feats = roles.features()
        self.weights.set_features(feats, [f for f in feats if roles.spec(f).shape], self.project.settings)

    def apply_settings(self):
        s = self.project.settings
        s.d50, s.k, s.n_neighbors, s.resolution = self.d50.value(), self.kk.value(), self.nn.value(), self.res.value()

    def _fill_options(self):
        cond = self.project.conditions if self.project else None
        cur = [cb.currentText() for cb in (self.color_by, self.shape_by, self.show_only)]
        for cb in (self.color_by, self.shape_by, self.show_only):
            cb.blockSignals(True); cb.clear()
        self.color_by.addItem("그룹", ("group", None))
        self.color_by.addItem("군집", ("cluster", None))
        self.color_by.addItem("형태값 신뢰도 w", ("weight", None))
        self.shape_by.addItem("없음", (None, None))
        self.show_only.addItem("전체", (None, None))
        if self.project:
            for c in cond.schema:
                if c.kind == "numeric":
                    self.color_by.addItem(f"조건: {c.label} (단계별)", ("cond_lv", c.name))
                    self.color_by.addItem(f"조건: {c.label} (연속)", ("cond", c.name))
                else:
                    self.color_by.addItem(f"조건: {c.label}", ("cond_lv", c.name))
                self.shape_by.addItem(f"조건: {c.label}", ("cond_lv", c.name))
                for v in _sorted_levels(cond.categorical(c.name).dropna().unique()):
                    self.show_only.addItem(f"{c.name} = {_level_text(c, v)}", (c.name, v))
            for f in self.project.roles.features():
                self.color_by.addItem(f"측정값: {f}", ("feat", f))
        for cb, t in zip((self.color_by, self.shape_by, self.show_only), cur):
            i = cb.findText(t)
            cb.setCurrentIndex(max(i, 0))
            cb.blockSignals(False)

    def conditions_changed(self):
        self._fill_options()
        self.redraw()

    def set_result(self, result):
        self.result = result
        self.levels = [{"idx": np.arange(len(result.prepared.data)), "pmap": result.pmap, "name": "전체"}] if result.pmap is not None else []
        self.w_note.setText("")
        self._fill_options()
        self._level_changed()
        replay, self._replay = self._replay, []
        if replay and self.levels and all(lv["idx"].max() < len(result.prepared.data) for lv in replay):
            self._replay_levels(replay)

    # ---- current level ---------------------------------------------------
    def _level(self):
        return self.levels[-1] if self.levels else None

    def _data(self):
        """Particles on the current level, re-indexed 0..n-1 so they line up with the map rows."""
        return self.result.prepared.data.iloc[self._level()["idx"]].reset_index(drop=True)

    def _level_changed(self):
        if self.roi is not None:
            self.select_btn.setChecked(False)
        self._update_level_controls()
        self.redraw()
        self._draw_clusters()
        self._draw_composition()

    def _update_level_controls(self):
        lv = self._level()
        busy = self._bg is not None
        self.crumb.setText("지도: " + (" ▸ ".join(l["name"] for l in self.levels) if self.levels else "전체")
                           + (f"  ({len(lv['idx']):,}개 입자)" if lv else ""))
        self.up_btn.setEnabled(len(self.levels) > 1 and not busy)
        self.cluster_pick.clear()
        if lv is not None:
            cl = lv["pmap"].clusters
            for c in np.unique(cl):
                self.cluster_pick.addItem(f"군집 {c} ({int((cl == c).sum()):,}개)", int(c))
        self.into_cluster_btn.setEnabled(lv is not None and not busy)
        self.into_sel_btn.setEnabled(lv is not None and not busy)
        self.apply_w.setEnabled(not busy)

    # ---- hierarchy -------------------------------------------------------
    def go_up(self):
        if len(self.levels) > 1:
            self.levels.pop()
            self._level_changed()

    def into_cluster(self):
        lv = self._level()
        if lv is None:
            return
        c = self.cluster_pick.currentData()
        m = lv["pmap"].clusters == c
        self._drill(lv["idx"][m], f"군집 {c}")

    def into_selection(self):
        lv = self._level()
        if lv is None:
            return
        if self.roi is None or self.sel_mask is None or not self.sel_mask.any():
            warn(self, "'영역 선택'을 켜고 상자로 입자를 고른 뒤 누르세요."); return
        self._drill(lv["idx"][self.sel_mask], "선택 영역")

    def _drill(self, idx, name, then=None):
        if len(idx) < 30:
            warn(self, f"입자가 {len(idx)}개뿐이라 하위 지도를 그리지 않습니다 (30개 이상 필요)."); return
        self.apply_settings()
        p, s = self.result.prepared, self.project.settings
        self.busy.setText(f"'{name}' 하위 지도 계산 중… ({len(idx):,}개 입자)")

        def done(pm: ParticleMap):
            self.busy.setText("")
            self.levels.append({"idx": idx, "pmap": pm, "name": name})
            self._level_changed()
            if then:
                then()

        def failed(msg):
            self.busy.setText("")
            self._update_level_controls()
            warn(self, "하위 지도를 계산하지 못했습니다.\n" + msg.split("\n\n")[0])

        run_in_thread(self, lambda: particle_map(p, s, idx=idx), done, failed)
        self._update_level_controls()

    def _replay_levels(self, levels):
        """Recompute the same drill-down path (same particles) on the new full map."""
        if not levels:
            return
        head, rest = levels[0], levels[1:]
        self._drill(head["idx"], head["name"], then=lambda: self._replay_levels(rest))

    # ---- weights ---------------------------------------------------------
    def _weights_edited(self):
        if self.result is not None:
            self.w_note.setText("가중치가 바뀌었습니다. '가중치 적용'을 누르면 지도(와 유사도·score)를 다시 계산합니다.")

    def apply_weights(self):
        if self.project is None:
            return
        # keep the drill-down path: the same particles are re-mapped with the new weights
        self._replay = [dict(lv) for lv in self.levels[1:]]
        self.w_note.setText("")
        self.run_requested.emit()

    # ---- scatter ---------------------------------------------------------
    def _cond_labels(self, d, name):
        c = self.project.conditions.get(name)
        vals = self.project.conditions.categorical(name)
        lab = d["group"].map(vals)
        return lab.map(lambda v: NOT_RECORDED if pd.isna(v) else _level_text(c, v)).to_numpy(object), \
            [_level_text(c, v) for v in _sorted_levels(vals.dropna().unique())]

    def redraw(self):
        lv = self._level()
        if self.result is None or lv is None:
            return
        pm = lv["pmap"]
        xy = pm.umap
        d = self._data()
        kind, key = self.color_by.currentData() or ("group", None)
        self.plot.clear(); self.legend.clear()
        self.colorbar_label.setText(""); self.cohesion_label.setText("")

        fname, fval = self.show_only.currentData() or (None, None)
        shown = np.ones(len(xy), bool)
        if fname is not None and fname in self.project.conditions.names():
            shown = (d["group"].map(self.project.conditions.categorical(fname)) == fval).to_numpy()
        rng = np.random.default_rng(0)
        order = rng.permutation(np.flatnonzero(shown))     # avoid one group painting over another
        hidden = np.flatnonzero(~shown)
        if len(hidden):
            self.plot.addItem(pg.ScatterPlotItem(xy[hidden, 0], xy[hidden, 1], size=3, pen=None,
                                                 brush=pg.mkBrush(200, 200, 200, 70)))

        skind, skey = self.shape_by.currentData() or (None, None)
        symbols = "o"
        sym_cats = []
        if skind == "cond_lv" and skey in self.project.conditions.names():
            slab, sym_cats = self._cond_labels(d, skey)
            sym_of = {c: SYMBOLS[i] if i < len(SYMBOLS) else "o" for i, c in enumerate(sym_cats)}
            symbols = [sym_of.get(l, "o") for l in slab[order]]
        size = 5 if skind else 3.5

        labels = None
        if kind == "group":
            labels = d["group"].to_numpy(object)
            cats = list(pd.unique(labels))
        elif kind == "cluster":
            labels = np.array([f"군집 {c}" for c in pm.clusters], dtype=object)
            cats = [f"군집 {c}" for c in sorted(np.unique(pm.clusters))]
        elif kind == "cond_lv":
            labels, cats = self._cond_labels(d, key)
            if (labels == NOT_RECORDED).any():
                cats = cats + [NOT_RECORDED]
        if labels is not None:
            brushes = {}
            for i, c in enumerate(cats):
                col = QtGui.QColor("#cfcfcf") if c == NOT_RECORDED else QtGui.QColor(color(i))
                col.setAlpha(170)
                brushes[c] = pg.mkBrush(col)
            b = [brushes[l] for l in labels[order]]
            self.scatter = pg.ScatterPlotItem(xy[order, 0], xy[order, 1], size=size, pen=None, brush=b, symbol=symbols)
            self.plot.addItem(self.scatter)
            for c in cats[:MAX_LEGEND]:
                self.legend.addItem(pg.ScatterPlotItem(size=8, pen=None, brush=brushes[c]), short_group(str(c), 34))
            if len(cats) > MAX_LEGEND:
                self.legend.addItem(pg.ScatterPlotItem(size=0), f"… 외 {len(cats) - MAX_LEGEND}개")
            if kind == "cluster":
                for c in np.unique(pm.clusters):
                    m = pm.clusters == c
                    cx, cy = np.median(xy[m], axis=0)
                    t = pg.TextItem(str(c), color="#111", anchor=(.5, .5), fill=pg.mkBrush(255, 255, 255, 180))
                    t.setPos(cx, cy); self.plot.addItem(t)
            lab = pd.Series(labels[order]).where(lambda s: s != NOT_RECORDED)
            coh = cohesion(xy[order], lab.to_numpy(object))
            if coh is not None:
                what = self.color_by.currentText()
                self.cohesion_label.setText(f"응집도 (실루엣, '{what}' 기준): {coh:.2f}   — 1에 가까울수록 같은 색끼리 뭉쳐 있고, 0 근처면 섞여 있습니다.")
        else:
            if kind == "weight":
                v = self.result.prepared.w[lv["idx"]]; label = "형태값 신뢰도 w"
            elif kind == "feat":
                v = d[key].to_numpy(float); label = key
                if self.project.roles.spec(key).log:
                    v = np.log10(np.clip(v, 1e-12, None)); label = f"log10 {key}"
            else:
                v = d["group"].map(self.project.conditions.numeric(key)).to_numpy(float); label = key
            v = v[order]
            finite = np.isfinite(v)
            lo, hi = (np.nanpercentile(v[finite], [2, 98]) if finite.any() else (0, 1))
            if hi <= lo:
                hi = lo + 1
            cmap = pg.colormap.get("viridis")
            t = np.clip((v - lo) / (hi - lo), 0, 1)
            cols = cmap.map(np.nan_to_num(t), mode="qcolor")
            brushes = [pg.mkBrush(c) if finite[i] else pg.mkBrush(QtGui.QColor("#d9d9d9")) for i, c in enumerate(cols)]
            self.scatter = pg.ScatterPlotItem(xy[order, 0], xy[order, 1], size=size, pen=None, brush=brushes, symbol=symbols)
            self.plot.addItem(self.scatter)
            self.colorbar_label.setText(f"{label}: 어두운 보라 = {lo:.3g}  →  노랑 = {hi:.3g}   (회색 = 값 없음)")
        if sym_cats:
            for i, c in enumerate(sym_cats[:len(SYMBOLS)]):
                self.legend.addItem(pg.ScatterPlotItem(size=9, pen=pg.mkPen("#444"), brush=pg.mkBrush("#888"), symbol=SYMBOLS[i]),
                                    f"모양 · {c}")
        if self.roi is not None:
            self.plot.addItem(self.roi)

    # ---- selection -------------------------------------------------------
    def _toggle_roi(self, on):
        lv = self._level()
        if on and self.result is not None and lv is not None:
            xy = lv["pmap"].umap
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
            self.sel_mask = None
            if on:
                self.select_btn.setChecked(False)

    def _roi_changed(self):
        lv = self._level()
        if self.roi is None or self.result is None or lv is None:
            return
        pos, size = self.roi.pos(), self.roi.size()
        xy = lv["pmap"].umap
        m = (xy[:, 0] >= pos.x()) & (xy[:, 0] <= pos.x() + size.x()) & (xy[:, 1] >= pos.y()) & (xy[:, 1] <= pos.y() + size.y())
        self.sel_mask = m
        self._draw_selection(m)

    def _draw_selection(self, m):
        c = self.sel_canvas
        c.clear()
        d = self._data()
        if m.sum() == 0:
            c.message("상자 안에 입자가 없습니다."); return
        ax1 = c.fig.add_subplot(2, 1, 1)
        share = d.loc[m, "group"].value_counts() / d["group"].value_counts()
        share = share.dropna().sort_values(ascending=True).tail(12)
        ax1.barh([short_group(g, 34) for g in share.index], share.values * 100, color="#2a6fdb")
        ax1.set_xlabel("이 지도의 그룹 입자 중 선택 영역에 든 비율 (%)")
        ax1.set_title(f"선택한 입자 {int(m.sum()):,}개 ({m.mean():.1%}) — 어느 그룹에서 왔나", fontsize=9)
        ax1.tick_params(axis="y", labelsize=7)
        ax2 = c.fig.add_subplot(2, 1, 2)
        Z = self.result.prepared.Z.iloc[self._level()["idx"]].reset_index(drop=True)
        diff = Z[m].median() - Z[~m].median() if (~m).any() else Z[m].median() * 0
        cols = ["#d6456b" if v > 0 else "#2a6fdb" for v in diff.values]
        ax2.bar(range(len(diff)), diff.values, color=cols)
        ax2.axhline(0, color="#999", lw=.8)
        ax2.set_xticks(range(len(diff)), diff.index, rotation=30, ha="right", fontsize=7)
        ax2.set_ylabel("나머지 대비 중앙값 차이 (z)")
        c.draw_idle()

    # ---- cluster plots ---------------------------------------------------
    def _draw_clusters(self):
        c = self.cluster_canvas
        c.clear()
        lv = self._level()
        if self.result is None or lv is None:
            return
        p = self.result.prepared
        cl = lv["pmap"].clusters
        Z = p.Z.iloc[lv["idx"]].reset_index(drop=True)
        w = p.w[lv["idx"]]
        for f in p.shape_features:
            Z[f] = Z[f] * w
        prof = Z.groupby(cl).median()
        ax = c.fig.add_subplot(111)
        im = ax.imshow(prof.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        fw = self.project.settings.feature_weights
        labs = [f + ("*" if f in p.shape_features else "") + (f" ×{fw[f]:g}" if fw.get(f, 1.0) != 1.0 else "") for f in prof.columns]
        ax.set_xticks(range(prof.shape[1]), labs, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(prof.shape[0]), [f"군집 {k} (n={int((cl == k).sum()):,})" for k in prof.index], fontsize=8)
        c.fig.colorbar(im, ax=ax, label="군집 중앙값 (robust z)", shrink=.8)
        ax.set_title(f"[{lv['name']}] 군집마다 무엇이 다른가 (*형태값은 신뢰도 가중, ×가중치)", fontsize=9)
        c.draw_idle()

    def _draw_composition(self):
        c = self.comp_canvas
        c.clear()
        lv = self._level()
        if self.result is None or lv is None:
            return
        groups = [g for g in self.result.profiles.groups]
        d = self._data()
        comp = pd.crosstab(d["group"], lv["pmap"].clusters, normalize="index")
        comp = comp.reindex([g for g in groups if g in comp.index])
        ax = c.fig.add_subplot(111)
        left = np.zeros(len(comp))
        for k in comp.columns:
            ax.barh(range(len(comp)), comp[k].values, left=left, color=color(int(k)), label=f"군집 {k}", height=.8)
            left += comp[k].values
        ax.set_yticks(range(len(comp)), [short_group(g, 34) for g in comp.index], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("입자 비율")
        ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
        ax.set_title(f"[{lv['name']}] 그룹별 군집 구성비", fontsize=9)
        c.draw_idle()


def _sorted_levels(values):
    vals = [str(v) for v in values]
    try:
        return sorted(vals, key=float)
    except ValueError:
        return sorted(vals)


def _level_text(cond, v) -> str:
    return f"{v} {cond.unit}".strip() if cond.kind == "numeric" else str(v)
