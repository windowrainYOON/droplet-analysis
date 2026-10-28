"""Reference score: how close each group is to the chosen reference (desired) batch."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from ..core.analysis import reference_score
from .common import MplCanvas, WeightsTable, color, fill_table, info_label, short_group, title_label, warn


class ScorePage(QtWidgets.QWidget):
    score_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.result = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(title_label("기준 score"))
        lay.addWidget(info_label("조건 편집기에서 ★로 고른 기준 그룹(원하는 배치)과 분포가 얼마나 가까운지 0–100으로 매깁니다. "
                                 "기준 그룹끼리의 차이만큼은 허용 범위로 보고 감점하지 않습니다. 작은 입자가 많은 그룹은 형태값의 비중이 줄어듭니다."))
        top = QtWidgets.QHBoxLayout()
        self.refs_label = QtWidgets.QLabel("기준 그룹: 없음")
        self.refs_label.setWordWrap(True)
        top.addWidget(self.refs_label, 1)
        top.addWidget(QtWidgets.QLabel("허용 폭(z)"))
        self.scale = QtWidgets.QDoubleSpinBox(); self.scale.setRange(0.05, 5); self.scale.setSingleStep(0.05); self.scale.setValue(0.5)
        self.scale.setToolTip("기준과의 차이가 이만큼(z 단위) 더 벌어질 때마다 score가 e배 줄어듭니다.")
        top.addWidget(self.scale)
        b = QtWidgets.QPushButton("score 다시 계산"); b.clicked.connect(self.recompute)
        top.addWidget(b)
        lay.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        wbox = QtWidgets.QGroupBox("측정값 가중치 (0 = 무시)")
        wl = QtWidgets.QVBoxLayout(wbox)
        self.weights = WeightsTable()
        wl.addWidget(self.weights)
        wl.addWidget(info_label("입자 지도 페이지의 가중치와 같은 값입니다. 바꾼 뒤 'score 다시 계산'을 누르세요."))
        split.addWidget(wbox)
        tabs = QtWidgets.QTabWidget()
        self.rank = MplCanvas(self, 6, 6)
        tabs.addTab(self.rank, "순위")
        dw = QtWidgets.QWidget(); dl = QtWidgets.QVBoxLayout(dw)
        bar = QtWidgets.QHBoxLayout(); bar.addWidget(QtWidgets.QLabel("비교할 그룹"))
        self.cmp_group = QtWidgets.QComboBox(); self.cmp_group.setMinimumWidth(320)
        self.cmp_group.currentIndexChanged.connect(self.draw_dist)
        bar.addWidget(self.cmp_group); bar.addStretch(1); dl.addLayout(bar)
        self.dist = MplCanvas(self, 6, 5)
        dl.addWidget(self.dist, 1)
        tabs.addTab(dw, "기준과 분포 비교")
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tabs.addTab(self.table, "score 표 (Δ = 기준 대비 z 차이)")
        split.addWidget(tabs)
        split.setSizes([260, 900])
        lay.addWidget(split, 1)
        self.rank.message("조건 편집기에서 기준 그룹(★)을 고른 뒤 '분석 실행'을 누르세요.")

    def set_project(self, project):
        self.project = project
        self.result = None
        self.scale.setValue(project.settings.score_scale)
        self.features_changed()
        self._update_refs()

    def _update_refs(self):
        refs = sorted(self.project.conditions.references) if self.project else []
        self.refs_label.setText("기준 그룹: " + (", ".join(short_group(r, 40) for r in refs) if refs else "없음 (조건 편집기에서 ★ 체크)"))

    def conditions_changed(self):
        self._update_refs()

    def features_changed(self):
        if self.project is None:
            return
        roles = self.project.roles
        feats = roles.features()
        self.weights.set_features(feats, [f for f in feats if roles.spec(f).shape], self.project.settings)

    def apply_settings(self):
        self.project.settings.score_scale = float(self.scale.value())

    def set_result(self, result):
        self.result = result
        self.weights.refresh()
        groups = result.profiles.groups
        cur = self.cmp_group.currentText()
        self.cmp_group.blockSignals(True); self.cmp_group.clear()
        for g in groups:
            self.cmp_group.addItem(g)
        if cur in groups:
            self.cmp_group.setCurrentText(cur)
        elif result.score is not None:
            non_ref = [g for g in result.score.index if result.score.at[g, "기준"] != "★"]
            if non_ref:
                self.cmp_group.setCurrentText(non_ref[0])
        self.cmp_group.blockSignals(False)
        self._update_refs()
        self.draw()

    def recompute(self):
        if self.result is None:
            warn(self, "먼저 '분석 실행'을 누르세요."); return
        self.apply_settings()
        refs = sorted(self.project.conditions.references)
        if not refs:
            warn(self, "조건 편집기에서 기준 그룹(★)을 하나 이상 고르세요."); return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.result.score = reference_score(self.result.prepared, self.project.settings, refs)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._update_refs()
        self.draw()
        self.score_changed.emit()

    def draw(self):
        sc = self.result.score if self.result is not None else None
        if sc is None:
            self.rank.message("조건 편집기에서 기준 그룹(★)을 고른 뒤 'score 다시 계산'을 누르세요.")
            self.dist.message("")
            self.table.clear(); self.table.setRowCount(0)
            return
        c = self.rank
        c.clear()
        ax = c.fig.add_subplot(111)
        s = sc["score"].iloc[::-1]
        cols = ["#c9a227" if sc.at[g, "기준"] == "★" else "#2a6fdb" for g in s.index]
        ax.barh(range(len(s)), s.to_numpy(), color=cols)
        for i, v in enumerate(s.to_numpy()):
            ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=7.5)
        ax.set_yticks(range(len(s)), [short_group(g, 44) for g in s.index], fontsize=7.5)
        ax.set_xlim(0, 108); ax.set_xlabel("score (100 = 기준과 같은 분포)")
        ax.set_title("노란 막대 = 기준 그룹", fontsize=8.5, loc="left", color="#777")
        c.draw_idle()
        show = sc.copy()
        show.insert(1, "평균 신뢰도 w", self.result.profiles.mean_weight.reindex(show.index).round(3))
        show.insert(2, "입자", self.result.profiles.n_particles.reindex(show.index))
        fill_table(self.table, show)
        self.draw_dist()

    def draw_dist(self):
        r = self.result
        if r is None or r.score is None:
            return
        g = self.cmp_group.currentText()
        refs = [x for x in r.score.index if r.score.at[x, "기준"] == "★"]
        p = r.prepared
        d = p.data
        c = self.dist
        c.clear()
        feats = p.features
        ncol = min(3, len(feats)); nrow = int(np.ceil(len(feats) / ncol))
        gm = (d["group"] == g).to_numpy()
        rm = d["group"].isin(refs).to_numpy()
        for k, f in enumerate(feats):
            ax = c.fig.add_subplot(nrow, ncol, k + 1)
            z = p.Z[f].to_numpy()
            lo, hi = np.nanpercentile(z, [0.5, 99.5])
            bins = np.linspace(lo, hi, 40)
            wts = p.w if f in p.shape_features else np.ones(len(z))
            for m, col, lab in ((rm, "#c9a227", "기준"), (gm, "#2a6fdb", short_group(g, 28))):
                if m.sum() and wts[m].sum() > 0:
                    ax.hist(z[m], bins=bins, weights=wts[m], density=True, histtype="stepfilled", alpha=.35, color=col, label=lab)
                    ax.hist(z[m], bins=bins, weights=wts[m], density=True, histtype="step", color=col, lw=1.2)
            dv = r.score.at[g, f"Δ {f}"] if f"Δ {f}" in r.score.columns else np.nan
            ax.set_title(f"{f}  (Δ {dv:+.2f})" + (" · w 가중" if f in p.shape_features else ""), fontsize=8)
            ax.set_yticks([]); ax.tick_params(labelsize=7)
            if k == 0:
                ax.legend(fontsize=7, frameon=False)
        c.fig.suptitle(f"score {r.score.at[g, 'score']:.0f}  —  가로축: 표준화 값(z)", fontsize=9)
        c.draw_idle()
