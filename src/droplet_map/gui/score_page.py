"""Reference score: how close each group is to the chosen reference (desired) batch."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from .common import MplCanvas, fill_table, flow_buttons, form, info_label, scroll_panel, section, short_group, title_label


class ScorePage(QtCore.QObject):
    """Center: ranking / distribution / table. Right: reference groups and tolerance.
    Measurement weights come from the shared weight panel."""
    recompute_requested = QtCore.Signal()
    title = "기준 score"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.result = None
        self.view = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.view)
        lay.addWidget(title_label("기준 score"))
        self.tabs = QtWidgets.QTabWidget()
        self.rank = MplCanvas(None, 6, 6)
        self.tabs.addTab(self.rank, "순위")
        self.dist = MplCanvas(None, 6, 5)
        self.tabs.addTab(self.dist, "기준과 분포 비교")
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.table, "score 표 (Δ = 기준 대비 z 차이)")
        lay.addWidget(self.tabs, 1)
        self.rank.message("기준 그룹(★)을 고른 뒤 '분석 실행'을 누르세요.")

        self.refs_label = QtWidgets.QLabel("없음")
        self.refs_label.setWordWrap(True)
        refs = section("기준 그룹 (원하는 배치)", self.refs_label,
                       info_label("왼쪽 그룹 목록에서 오른쪽 클릭, 또는 조건 표의 ★ 칸으로 지정합니다."))
        self.scale = QtWidgets.QDoubleSpinBox(); self.scale.setRange(0.05, 5); self.scale.setSingleStep(0.05); self.scale.setValue(0.5)
        self.scale.setToolTip("기준과의 차이가 이만큼(z 단위) 더 벌어질 때마다 score가 e배 줄어듭니다.")
        self.scale.valueChanged.connect(lambda _v: self._scale_timer.start())
        self._scale_timer = QtCore.QTimer(self); self._scale_timer.setSingleShot(True); self._scale_timer.setInterval(500)
        self._scale_timer.timeout.connect(self.recompute)
        calc = section("score 계산",
                       form([("허용 폭 (z)", self.scale)]),
                       info_label("기준 그룹끼리의 차이만큼은 허용 범위로 보고 감점하지 않습니다. 작은 입자가 많은 그룹은 "
                                  "형태값의 비중이 줄어듭니다. 측정값별 비중은 '측정값 가중치' 탭에서 정합니다."),
                       flow_buttons((("score 다시 계산", self.recompute),)))
        self.cmp_group = QtWidgets.QComboBox()
        self.cmp_group.currentIndexChanged.connect(self.draw_dist)
        cmp = section("분포 비교", form([("비교할 그룹", self.cmp_group)]))
        self.panel = scroll_panel(refs, calc, cmp)

    def set_project(self, project):
        self.project = project
        self.result = None
        self.scale.blockSignals(True); self.scale.setValue(project.settings.score_scale); self.scale.blockSignals(False)
        self._update_refs()

    def _update_refs(self):
        refs = sorted(self.project.conditions.references) if self.project else []
        self.refs_label.setText("\n".join(f"★ {r}" for r in refs) if refs else "없음")

    def conditions_changed(self):
        self._update_refs()

    def apply_settings(self):
        self.project.settings.score_scale = float(self.scale.value())

    def ui_state(self):
        return {"tab": self.tabs.currentIndex(), "cmp": self.cmp_group.currentText()}

    def restore_ui(self, st):
        if st:
            self.tabs.setCurrentIndex(int(st.get("tab", 0)))
            i = self.cmp_group.findText(st.get("cmp", ""))
            if i >= 0:
                self.cmp_group.setCurrentIndex(i)

    def set_result(self, result):
        self.result = result
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
            return
        self.apply_settings()
        self.recompute_requested.emit()

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
