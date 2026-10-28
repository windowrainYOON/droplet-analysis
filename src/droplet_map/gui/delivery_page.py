"""Cell delivery: intensity inside particles and in the cytoplasm per group, and the link to the
particle formation group whose particles were used."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from ..core.delivery import RATIO_LABEL, region_label
from .common import MplCanvas, color, fill_table, form, info_label, scroll_panel, section, title_label

SHOW = (("강도 (영역별 평균 ± SD)", "mean"), (f"{RATIO_LABEL} 비율", "ratio"))


class DeliveryPage(QtCore.QObject):
    title = "세포 전달"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.result = None
        self.view = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.view)
        lay.addWidget(title_label("세포 전달"))
        self.note = info_label("")
        lay.addWidget(self.note)
        self.tabs = QtWidgets.QTabWidget()
        self.plot = MplCanvas(None, 7, 4)
        self.tabs.addTab(self.plot, "그룹별 강도")
        self.link = QtWidgets.QTableWidget(); self.link.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.link, "사용한 입자와 나란히")
        self.table = QtWidgets.QTableWidget(); self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.table, "요약 표")
        lay.addWidget(self.tabs, 1)

        self.channel = QtWidgets.QComboBox()
        self.show = QtWidgets.QComboBox()
        for label, key in SHOW:
            self.show.addItem(label, key)
        self.order = QtWidgets.QComboBox()
        for w in (self.channel, self.show, self.order):
            w.currentIndexChanged.connect(self.redraw)
        view = section("보기", form([("채널", self.channel), ("값", self.show), ("가로축 순서", self.order)]),
                       info_label("입자 안 = 세포 속 입자 영역의 평균 강도, 세포질 (입자 제외) = 입자를 뺀 세포질의 평균 강도. "
                                  "비율이 크면 입자 밖(세포질)으로 더 퍼졌다는 뜻입니다. 막대 = 그룹 평균, 선 = SD."))
        link = section("입자 형성과 잇기",
                       info_label("조건 탭의 세포 전달 표에서 '사용한 입자 형성 그룹'을 고르면, '사용한 입자와 나란히' 탭에 "
                                  "그 입자의 측정값 중앙값과 기준 score가 함께 나옵니다 ('분석 실행' 후)."))
        self.panel = scroll_panel(view, link)
        self.plot.message("세포 전달 요약 CSV를 불러오면 여기에 나옵니다 (파일 › 불러오기 또는 끌어 놓기).")

    def set_project(self, project):
        self.project = project
        self.result = None
        self.refresh()

    def set_result(self, result):
        self.result = result
        self.refresh_link()

    def conditions_changed(self):
        self.refresh()

    def ui_state(self):
        return {"tab": self.tabs.currentIndex(), "channel": self.channel.currentText(),
                "show": self.show.currentData(), "order": self.order.currentText()}

    def restore_ui(self, st):
        if not st:
            return
        self.tabs.setCurrentIndex(int(st.get("tab", 0)))
        for cb, k in ((self.channel, "channel"), (self.order, "order")):
            i = cb.findText(st.get(k, ""))
            if i >= 0:
                cb.setCurrentIndex(i)
        i = self.show.findData(st.get("show"))
        if i >= 0:
            self.show.setCurrentIndex(i)

    # ---- content ---------------------------------------------------------
    def refresh(self):
        p = self.project
        if p is None:
            return
        d = p.delivery
        for cb, items in ((self.channel, d.channels()),
                          (self.order, ["불러온 순서", "배치"] + d.conditions.names())):
            cur = cb.currentText()
            cb.blockSignals(True); cb.clear(); cb.addItems(items)
            i = cb.findText(cur); cb.setCurrentIndex(max(i, 0))
            cb.blockSignals(False)
        if d.empty():
            self.note.setText("세포 전달 데이터가 없습니다.")
        else:
            linked = sum(1 for g in d.groups() if d.uses.get(g))
            self.note.setText(f"세포 전달 그룹 {len(d.groups())}개 · 영역 {', '.join(region_label(r) for r in d.regions())} · "
                              f"사용한 입자 그룹을 정한 그룹 {linked}개")
            fill_table(self.table, d.measures())
        self.redraw()
        self.refresh_link()

    def _ordered_groups(self):
        p = self.project
        gs = p.delivery.groups()
        key = self.order.currentText()
        if key == "배치":
            return sorted(gs, key=lambda g: (p.batch_of(g), gs.index(g)))
        if key in p.delivery.conditions.names():
            t = p.delivery.conditions
            num = t.numeric(key) if t.get(key).kind == "numeric" else None

            def k(g):
                v = num.get(g) if num is not None else t.categorical(key).get(g)
                return (pd.isna(v), v if not pd.isna(v) else 0, gs.index(g))
            return sorted(gs, key=k)
        return gs

    def _label(self, g):
        """The delivery conditions that differ between groups, else the name."""
        t = self.project.delivery.conditions
        parts = []
        for c in t.schema:
            vals = t.categorical(c.name)
            if vals.nunique() > 1:
                v = vals.get(g)
                parts.append(f"{c.name} {v}{c.unit}" if not pd.isna(v) else f"{c.name} —")
        if parts:
            return "\n".join(parts)
        return g if len(g) <= 28 else g[:27] + "…"

    def redraw(self):
        c = self.plot
        p = self.project
        if p is None or p.delivery.empty():
            c.message("세포 전달 요약 CSV를 불러오면 여기에 나옵니다 (파일 › 불러오기 또는 끌어 놓기)."); return
        ch = self.channel.currentText()
        if not ch:
            c.message("강도 채널(…_Mean) 열이 없습니다."); return
        gs = self._ordered_groups()
        x = np.arange(len(gs))
        c.clear()
        ax = c.fig.add_subplot(111)
        if self.show.currentData() == "ratio":
            r = p.delivery.ratio().reindex(gs)
            if ch not in r.columns:
                c.message("비율을 계산할 두 영역이 없습니다."); return
            ax.bar(x, r[ch].to_numpy(float), color=[color(i) for i in range(len(gs))], alpha=.75)
            ax.set_ylabel(f"{RATIO_LABEL} ({ch})")
        else:
            m, sd = p.delivery.value("mean"), p.delivery.value("sd")
            regions = [r for r in p.delivery.regions() if (r, ch) in m.columns]
            w = 0.8 / max(len(regions), 1)
            for i, reg in enumerate(regions):
                vals = m[(reg, ch)].reindex(gs).to_numpy(float)
                err = sd[(reg, ch)].reindex(gs).to_numpy(float) if (reg, ch) in sd.columns else None
                ax.bar(x + (i - (len(regions) - 1) / 2) * w, vals, w, yerr=err, capsize=3, color=color(i), alpha=.8,
                       label=region_label(reg))
            ax.legend(fontsize=8, frameon=False)
            ax.set_ylabel(f"{ch} 평균 강도")
        ax.set_xticks(x); ax.set_xticklabels([self._label(g) for g in gs], fontsize=8)
        batches = [p.batch_of(g) for g in gs]
        if len(set(batches)) > 1:
            for i, b in enumerate(batches):
                ax.annotate(b, (i, 0), xytext=(0, -38), textcoords="offset points", ha="center", fontsize=7, color="#777")
        ax.set_title(f"{ch} — 그룹 {len(gs)}개", fontsize=9)
        c.draw_idle()

    def refresh_link(self):
        p = self.project
        if p is None or p.delivery.empty():
            self.link.clear(); self.link.setRowCount(0); return
        rows = {}
        meas = p.delivery.measures()
        ratio_cols = [c for c in meas.columns if c.startswith(RATIO_LABEL)]
        for g in p.delivery.groups():
            used = p.delivery.uses.get(g)
            row = {"배치": p.batch_of(g), "사용한 입자 형성 그룹": used or "—"}
            for c in ratio_cols:
                row[c] = meas.at[g, c]
            if used and self.result is not None:
                prof = self.result.profiles.median_raw
                if used in prof.index:
                    for f in prof.columns:
                        row[f"입자 중앙값 · {f}"] = prof.at[used, f]
                if self.result.score is not None and used in self.result.score.index:
                    row["입자 기준 score"] = self.result.score.at[used, "score"]
            rows[g] = row
        fill_table(self.link, pd.DataFrame(rows).T)
