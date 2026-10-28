"""Batch comparison: groups paired across batches by identical formation conditions, so what
differs is the protein batch, not the formation condition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from ..core.batch_compare import (batch_distance, batch_feature_delta, batch_reference_score, batch_summary,
                                  matched_groups)
from ..core.batches import UNASSIGNED
from .common import MplCanvas, color, fill_table, flow_buttons, form, info_label, scroll_panel, section, title_label


class BatchPage(QtCore.QObject):
    title = "배치 비교"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.result = None
        self.matched = pd.DataFrame()
        self.score = None
        self.view = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.view)
        lay.addWidget(title_label("배치 비교"))
        self.note = info_label("")
        lay.addWidget(self.note)
        self.tabs = QtWidgets.QTabWidget()
        self.pairs = QtWidgets.QTableWidget(); self.pairs.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.pairs, "같은 조건의 그룹 짝")
        self.dist = MplCanvas(None, 5, 4)
        self.tabs.addTab(self.dist, "배치 간 거리")
        self.delta = MplCanvas(None, 6, 4)
        self.tabs.addTab(self.delta, "두 배치의 측정값 차이")
        self.score_canvas = MplCanvas(None, 6, 4)
        self.tabs.addTab(self.score_canvas, "기준 배치 score")
        self.score_table = QtWidgets.QTableWidget(); self.score_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.score_table, "score 표")
        lay.addWidget(self.tabs, 1)

        self.a = QtWidgets.QComboBox(); self.b = QtWidgets.QComboBox()
        self.a.currentIndexChanged.connect(self.draw_delta); self.b.currentIndexChanged.connect(self.draw_delta)
        pair = section("두 배치 비교", form([("배치 A", self.a), ("배치 B", self.b)]),
                       info_label("같은 형성 조건(그룹 조건 값이 모두 같은 그룹)끼리만 짝지어, B − A의 중앙값 차이(z)를 "
                                  "조건마다 점으로, 평균을 막대로 보여 줍니다."))
        self.ref = QtWidgets.QComboBox()
        score = section("기준 배치 score",
                        form([("기준 배치", self.ref)]),
                        info_label("각 그룹을 기준 배치의 같은 조건 그룹과 비교한 score(0–100)입니다. 기준 배치에 없는 조건의 "
                                   "그룹은 빠집니다. 배치 표나 왼쪽 목록에서 ★로 기준 배치를 정할 수 있습니다."),
                        flow_buttons((("score 계산", self.compute_score),)))
        about = section("읽는 법", info_label(
            "형성 조건을 맞춘 뒤에도 남는 차이가 배치(단백질) 차이입니다. 배치가 많아지면 조건 비교 탭에서 "
            "[배치] 조건(정제일, 농도 …)과 결과의 관계도 볼 수 있습니다."))
        self.panel = scroll_panel(pair, score, about)
        self._empty("'분석 실행' 후에 배치를 비교합니다.")

    def _empty(self, text):
        for c in (self.dist, self.delta, self.score_canvas):
            c.message(text)

    def set_project(self, project):
        self.project = project
        self.result = None
        self.score = None
        self.conditions_changed()

    def conditions_changed(self):
        if self.project is None:
            return
        names = [b for b in self.project.batches.names() if b != UNASSIGNED]
        for cb, pick in ((self.a, 0), (self.b, 1), (self.ref, None)):
            cur = cb.currentText()
            cb.blockSignals(True); cb.clear(); cb.addItems(names)
            i = cb.findText(cur)
            if i < 0 and cb is self.ref and self.project.batches.references:
                i = cb.findText(sorted(self.project.batches.references)[0])
            if i < 0 and pick is not None:
                i = min(pick, len(names) - 1)
            cb.setCurrentIndex(max(i, 0))
            cb.blockSignals(False)
        if self.result is not None:
            self.refresh()

    def set_result(self, result):
        self.result = result
        self.score = None
        self.refresh()

    def ui_state(self):
        return {"tab": self.tabs.currentIndex(), "a": self.a.currentText(), "b": self.b.currentText(),
                "ref": self.ref.currentText()}

    def restore_ui(self, st):
        if not st:
            return
        self.tabs.setCurrentIndex(int(st.get("tab", 0)))
        for cb, k in ((self.a, "a"), (self.b, "b"), (self.ref, "ref")):
            i = cb.findText(st.get(k, ""))
            if i >= 0:
                cb.setCurrentIndex(i)

    # ---- drawing ----------------------------------------------------------
    def refresh(self):
        p = self.project
        n_b = len([b for b in p.batches.names() if b != UNASSIGNED])
        self.matched = matched_groups(p.groups, p.conditions, p.batches)
        if n_b < 2:
            self.note.setText("배치가 하나뿐입니다. 다른 배치의 입자 CSV를 추가하고 같은 형성 조건을 입력하면 여기서 비교합니다.")
        elif self.matched.empty:
            self.note.setText("배치 사이에 형성 조건이 모두 같은 그룹이 없습니다. 조건 탭에서 두 배치의 그룹 조건을 같은 값으로 "
                              "입력했는지 확인하세요.")
        else:
            self.note.setText(f"배치 {n_b}개 · 두 배치 이상에 있는 형성 조건 {len(self.matched)}가지")
        fill_table(self.pairs, self.matched.fillna("—"), index_label="형성 조건")
        self.draw_distance()
        self.draw_delta()
        self.draw_score()

    def draw_distance(self):
        c = self.dist
        if self.result is None or self.matched.empty:
            c.message("비교할 짝이 없습니다."); return
        D, N = batch_distance(self.result.distance, self.matched)
        c.clear()
        ax = c.fig.add_subplot(111)
        im = ax.imshow(D.to_numpy(float), cmap="viridis_r")
        ax.set_xticks(range(len(D))); ax.set_xticklabels(D.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(D))); ax.set_yticklabels(D.index)
        for i in range(len(D)):
            for j in range(len(D)):
                if i != j and N.iat[i, j]:
                    ax.text(j, i, f"{D.iat[i, j]:.2f}\nn={N.iat[i, j]}", ha="center", va="center", fontsize=8, color="w")
        c.fig.colorbar(im, ax=ax, label="같은 조건 그룹 간 평균 거리 (z)")
        ax.set_title("배치 간 거리 — 같은 형성 조건끼리만", fontsize=9)
        c.draw_idle()

    def draw_delta(self):
        c = self.delta
        a, b = self.a.currentText(), self.b.currentText()
        if self.result is None or self.matched.empty or not a or not b or a == b \
                or a not in self.matched.columns or b not in self.matched.columns:
            c.message("서로 다른 두 배치를 고르세요 (같은 조건의 그룹이 있어야 합니다)."); return
        t = batch_feature_delta(self.result.prepared, self.matched, a, b)
        if t.empty:
            c.message(f"{a}와 {b}에 같은 조건의 그룹이 없습니다."); return
        c.clear()
        ax = c.fig.add_subplot(111)
        feats = list(t.columns)
        y = np.arange(len(feats))
        mean = t.loc["평균"]
        ax.barh(y, mean.to_numpy(float), color=[color(0) if v >= 0 else color(1) for v in mean], alpha=.5)
        keys = [k for k in t.index if k != "평균"]
        for i, k in enumerate(keys):
            ax.plot(t.loc[k].to_numpy(float), y + (i - (len(keys) - 1) / 2) * 0.08, "o", ms=3, color="#333", alpha=.6)
        ax.axvline(0, color="#999", lw=.8)
        ax.set_yticks(y); ax.set_yticklabels(feats); ax.invert_yaxis()
        ax.set_xlabel(f"{b} − {a}  (중앙값 z 차이)")
        ax.set_title(f"같은 조건 {len(keys)}가지에서 본 배치 차이", fontsize=9)
        c.draw_idle()

    def compute_score(self):
        if self.result is None:
            self.score_canvas.message("먼저 '분석 실행'을 누르세요."); return
        ref = self.ref.currentText()
        if not ref:
            self.score_canvas.message("기준 배치를 고르세요."); return
        p = self.project
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.score = batch_reference_score(self.result.prepared, p.settings, p.groups, p.conditions, p.batches, ref)
        except ValueError as e:
            self.score = None
            self.score_canvas.message(str(e)); return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        p.batches.references = {ref}
        self.draw_score()

    def draw_score(self):
        c = self.score_canvas
        if self.score is None:
            c.message("기준 배치를 고르고 'score 계산'을 누르세요.")
            self.score_table.clear(); self.score_table.setRowCount(0)
            return
        s = self.score
        fill_table(self.score_table, s)
        summ = batch_summary(s)
        c.clear()
        ax = c.fig.add_subplot(111)
        bs = list(summ.index)
        for i, b in enumerate(bs):
            v = s[(s["배치"] == b)]["score"].dropna()
            ax.scatter(np.full(len(v), i) + np.linspace(-.15, .15, len(v)) if len(v) > 1 else [i] * len(v), v,
                       s=18, color=color(i), alpha=.8)
            ax.hlines(summ.at[b, "평균 score"], i - .3, i + .3, color=color(i), lw=2)
        ax.set_xticks(range(len(bs))); ax.set_xticklabels([f"{b}\n(그룹 {summ.at[b, '비교한 그룹 수']}개)" for b in bs])
        ax.set_ylim(0, 105); ax.set_ylabel("score (같은 조건의 기준 배치 그룹 대비)")
        ax.set_title(f"기준 배치: {self.ref.currentText()} — 선 = 배치 평균", fontsize=9)
        c.draw_idle()

    def score_frame(self):
        return self.score
