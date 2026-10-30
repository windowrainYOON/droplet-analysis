"""Per-condition measurement sheets (in vitro experiments such as FRAP, cell delivery, and any sheet
the user adds): a measurement against a formation condition or a sub-condition, one line per
batch, and the particles of the formation condition next to it."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from ..core.batchfile import COND_ID
from .common import MplCanvas, color, fill_table, form, info_label, scroll_panel, section, title_label


class MeasurePage(QtCore.QObject):
    title = "측정 시트"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.result = None
        self.view = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.view)
        lay.addWidget(title_label("측정 시트"))
        self.note = info_label("")
        lay.addWidget(self.note)
        self.tabs = QtWidgets.QTabWidget()
        self.plot = MplCanvas(None, 7, 4)
        self.tabs.addTab(self.plot, "조건에 따른 변화")
        self.table = QtWidgets.QTableWidget(); self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.table, "조건별 요약")
        self.link = QtWidgets.QTableWidget(); self.link.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.link, "그 조건의 입자와 나란히")
        lay.addWidget(self.tabs, 1)

        self.sheet = QtWidgets.QComboBox()
        self.measure = QtWidgets.QComboBox()
        self.xaxis = QtWidgets.QComboBox()
        self.sheet.currentIndexChanged.connect(self._sheet_changed)
        for w in (self.measure, self.xaxis):
            w.currentIndexChanged.connect(self.redraw)
        view = section("보기", form([("시트", self.sheet), ("측정값", self.measure), ("가로축", self.xaxis)]),
                       info_label("점 = 반복 측정 하나, 선 = 배치별 중앙값. 가로축은 형성 조건(pH, NaCl …)이나 "
                                  "그 시트의 하위 조건(시간 …)입니다. 값은 배치 파일(엑셀)에서 고칩니다."))
        self.panel = scroll_panel(view)
        self.plot.message("배치 파일에 조건별 측정 시트(FRAP, 세포 전달 …)가 있으면 여기에 나옵니다.")

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
        return {"tab": self.tabs.currentIndex(), "sheet": self.sheet.currentText(),
                "measure": self.measure.currentText(), "x": self.xaxis.currentText()}

    def restore_ui(self, st):
        if not st:
            return
        self.tabs.setCurrentIndex(int(st.get("tab", 0)))
        for cb, k in ((self.sheet, "sheet"), (self.measure, "measure"), (self.xaxis, "x")):
            i = cb.findText(st.get(k, ""))
            if i >= 0:
                cb.setCurrentIndex(i)

    # ---- content ---------------------------------------------------------
    def current(self):
        if self.project is None:
            return None
        return self.project.measure_set(self.sheet.currentText())

    @staticmethod
    def _set_items(cb, items):
        cur = cb.currentText()
        cb.blockSignals(True); cb.clear(); cb.addItems(items)
        i = cb.findText(cur); cb.setCurrentIndex(max(i, 0))
        cb.blockSignals(False)

    def refresh(self):
        p = self.project
        if p is None:
            return
        self._set_items(self.sheet, [m.name for m in p.measures])
        self._sheet_changed()

    def _sheet_changed(self):
        m = self.current()
        if m is None:
            self.note.setText("조건별 측정 시트에 값이 없습니다.")
            self._set_items(self.measure, []); self._set_items(self.xaxis, [])
            fill_table(self.table, pd.DataFrame())
            self.redraw(); self.refresh_link()
            return
        self._set_items(self.measure, m.values)
        xs = [c.name for c in m.cond.schema if m.cond.numeric(c.name).notna().any()]
        xs = m.subs + [x for x in xs if x not in m.subs]
        self._set_items(self.xaxis, xs + ["배치"])
        self.note.setText(f"'{m.name}' 시트 · 범주 '{m.category}' · 배치 {len(set(m.batch_of.values()))}개 · "
                          f"측정 {len(m.rows)}행 → 조건별 {len(m.measures)}개")
        t = m.measures.copy()
        t.insert(0, "배치", [m.batch_of[g] for g in t.index])
        t.insert(1, "형성 조건", [self._cond_label(m.formation_of[g]) for g in t.index])
        fill_table(self.table, t)
        self.redraw()
        self.refresh_link()

    def _cond_label(self, fg):
        p = self.project
        b, _, cid = fg.partition(" · ")
        bf = p.source.files.get(b)
        return f"{cid}: {bf.condition_label(cid)}" if bf is not None else fg

    def redraw(self):
        c = self.plot
        m = self.current()
        meas, x = self.measure.currentText(), self.xaxis.currentText()
        if m is None or not meas or not x:
            c.message("배치 파일에 조건별 측정 시트(FRAP, 세포 전달 …)가 있으면 여기에 나옵니다."); return
        rows = m.rows.copy()
        rows["_y"] = pd.to_numeric(rows[meas], errors="coerce")
        if x == "배치":
            rows["_x"] = rows["배치"]
        elif x in rows.columns:
            rows["_x"] = pd.to_numeric(rows[x], errors="coerce")
        else:                                    # a formation condition: from the summary group's conditions
            v = m.cond.numeric(x)
            rows["_x"] = rows["그룹"].map(v)
        rows = rows.dropna(subset=["_x", "_y"])
        if rows.empty:
            c.message(f"'{meas}'와(과) '{x}' 값이 함께 있는 행이 없습니다."); return
        c.clear()
        ax = c.fig.add_subplot(111)
        batches = list(pd.unique(rows["배치"]))
        if x == "배치":
            pos = {b: i for i, b in enumerate(batches)}
            for i, b in enumerate(batches):
                y = rows.loc[rows["배치"] == b, "_y"].to_numpy(float)
                ax.scatter(np.full(len(y), pos[b]) + np.random.default_rng(i).uniform(-.12, .12, len(y)), y,
                           s=16, color=color(i), alpha=.7)
                ax.hlines(np.median(y), pos[b] - .3, pos[b] + .3, color=color(i), lw=2)
            ax.set_xticks(range(len(batches))); ax.set_xticklabels(batches, fontsize=8)
        else:
            for i, b in enumerate(batches):
                d = rows[rows["배치"] == b]
                ax.scatter(d["_x"], d["_y"], s=14, color=color(i), alpha=.45)
                med = d.groupby("_x")["_y"].median().sort_index()
                ax.plot(med.index, med.to_numpy(float), "-o", ms=4, color=color(i), label=b)
            unit = m.units.get(x) or next((cc.unit for cc in m.cond.schema if cc.name == x), "")
            ax.set_xlabel(f"{x} ({unit})" if unit else x)
            ax.legend(fontsize=8, frameon=False, title="배치", title_fontsize=8)
        u = m.units.get(meas, "")
        ax.set_ylabel(f"{meas} ({u})" if u else meas)
        ax.set_title(f"{m.name}: {meas}", fontsize=9)
        c.draw_idle()

    def refresh_link(self):
        p, m = self.project, self.current()
        if p is None or m is None:
            self.link.clear(); self.link.setRowCount(0); return
        out = {}
        for g in m.measures.index:
            fg = m.formation_of[g]
            row = {"배치": m.batch_of[g], "형성 조건": self._cond_label(fg)}
            for v in m.values:
                row[v] = m.measures.at[g, v]
            if self.result is not None:
                prof = self.result.profiles.median_raw
                if fg in prof.index:
                    for f in prof.columns:
                        row[f"입자 중앙값 · {f}"] = prof.at[fg, f]
                if self.result.score is not None and fg in self.result.score.index:
                    row["입자 기준 score"] = self.result.score.at[fg, "score"]
            out[g] = row
        fill_table(self.link, pd.DataFrame(out).T)
        if self.result is None:
            self.link.setToolTip("입자 값은 '입자 분석 실행' 뒤에 나옵니다.")

    def summary_frame(self):
        m = self.current()
        if m is None:
            return None
        t = m.measures.copy()
        t.insert(0, "배치", [m.batch_of[g] for g in t.index])
        t.insert(1, COND_ID, [m.formation_of[g].partition(" · ")[2] for g in t.index])
        return t
