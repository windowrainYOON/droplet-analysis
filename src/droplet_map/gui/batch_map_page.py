"""Batch map (main view): each purification batch is a point; close points = batches alike in the
values that describe them (batch table + trends and matched values from their experiments).
Double-click a point to open that batch's particle map (particle level)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets
from scipy.cluster.hierarchy import dendrogram

from ..core.batch_map import Descriptors, batch_distance_map, batch_scores, pair_differences
from ..core.batches import UNASSIGNED
from .common import MplCanvas, color, flow_buttons, form, info_label, scroll_panel, section, title_label

NO_COLOR = "(색 없음)"


def subset(desc: Descriptors, batches) -> Descriptors:
    return Descriptors(desc.values.loc[[b for b in desc.values.index if b in set(batches)]], desc.meta)


def short_item(it: str, n=48) -> str:
    return it if len(it) <= n else it[: n - 1] + "…"


class BatchMapPage(QtCore.QObject):
    title = "배치 지도"
    descriptors_ready = QtCore.Signal(object)       # Descriptors, for the batch table page
    particle_map_requested = QtCore.Signal(str)     # batch
    compare_requested = QtCore.Signal(str, str)
    settings_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.desc: Descriptors | None = None
        self.bm = None
        self.visible = None            # batches shown (None = all)
        self.view = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.view)
        lay.addWidget(title_label("배치 지도"))
        self.note = info_label("")
        lay.addWidget(self.note)
        self.tabs = QtWidgets.QTabWidget()
        self.canvas = MplCanvas(None, 7, 5.5)
        self.canvas.mpl_connect("button_press_event", self._clicked)
        self.tabs.addTab(self.canvas, "배치 지도")
        self.cat_canvas = MplCanvas(None, 7, 5)
        self.tabs.addTab(self.cat_canvas, "범주별 지도")
        self.dist_canvas = MplCanvas(None, 6, 5)
        self.tabs.addTab(self.dist_canvas, "거리 표")
        self.tree_canvas = MplCanvas(None, 6, 4)
        self.tabs.addTab(self.tree_canvas, "배치 군집")
        self.score_canvas = MplCanvas(None, 6, 4)
        self.tabs.addTab(self.score_canvas, "기준 배치 score")
        lay.addWidget(self.tabs, 1)
        self.axis_note = info_label("")
        lay.addWidget(self.axis_note)

        self.status = info_label("")
        calc = section("계산", info_label("배치 표의 값과 각 배치의 실험 데이터로 배치 사이 거리를 계산합니다. "
                                          "무게나 항목을 바꾸면 바로 다시 그립니다. 데이터·조건을 바꾼 뒤에는 '다시 계산'을 누르세요."),
                       flow_buttons((("배치 지도 다시 계산", self.compute),)), self.status)
        self.weight_box = QtWidgets.QWidget(); self.weight_lay = QtWidgets.QFormLayout(self.weight_box)
        self.weight_lay.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        weights = section("범주 무게", info_label("범주마다 거리를 따로 구해 이 무게로 평균합니다. 0 = 빼기. "
                                                 "항목 수가 많은 범주가 적은 범주를 덮지 않습니다."), self.weight_box)
        self.part_box = QtWidgets.QWidget(); self.part_lay = QtWidgets.QVBoxLayout(self.part_box)
        self.part_lay.setContentsMargins(0, 0, 0, 0)
        parts = section("항목 묶음", info_label("범주 안에서 묶음(직접 입력, 경향, 같은 조건, 대표값)마다 평균한 뒤 범주 거리를 냅니다."),
                        self.part_box)
        self.item_find = QtWidgets.QLineEdit(); self.item_find.setPlaceholderText("항목 찾기"); self.item_find.setClearButtonEnabled(True)
        self.item_find.textChanged.connect(self._filter_items)
        self.items = QtWidgets.QListWidget(); self.items.setMinimumHeight(160)
        self.items.itemChanged.connect(self._item_toggled)
        all_btn = QtWidgets.QPushButton("모두 넣기"); all_btn.clicked.connect(self._all_items)
        items = section("항목", info_label("체크를 끄면 그 항목을 거리에서 뺍니다."), self.item_find, self.items, all_btn)
        self.color_by = QtWidgets.QComboBox()
        self.color_by.currentIndexChanged.connect(self._color_changed)
        self.show_names = QtWidgets.QCheckBox("배치 이름 표시"); self.show_names.setChecked(True)
        self.show_names.toggled.connect(self.draw)
        view = section("보기", form([("점 색", self.color_by)]), self.show_names)
        self.ref = QtWidgets.QComboBox()
        self.ref.currentIndexChanged.connect(self._ref_changed)
        ref = section("기준 배치", form([("기준", self.ref)]),
                      info_label("score = 100 × (1 − 기준 배치와의 거리). 100 = 기준 배치와 같음."))
        self.cmp_a = QtWidgets.QComboBox(); self.cmp_b = QtWidgets.QComboBox()
        cmp = section("두 배치 비교", form([("배치 A", self.cmp_a), ("배치 B", self.cmp_b)]),
                      flow_buttons((("배치 비교 탭에서 보기", lambda: self.compare_requested.emit(self.cmp_a.currentText(), self.cmp_b.currentText())),)))
        self.drill = QtWidgets.QComboBox()
        sub = section("입자 수준으로", info_label("지도의 점을 두 번 누르거나 아래에서 배치를 골라, 그 배치 입자들의 군집 지도를 엽니다."),
                      form([("배치", self.drill)]),
                      flow_buttons((("이 배치의 입자 지도", lambda: self.drill.currentText() and self.particle_map_requested.emit(self.drill.currentText())),)))
        self.panel = scroll_panel(calc, weights, parts, items, view, ref, cmp, sub)
        self.canvas.message("입자 CSV나 프로젝트를 열면 배치 지도를 계산합니다.")

    # ---- project / computing ------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.desc = None
        self.bm = None
        self._fill_batch_combos()        # computed by the main window once every page has the project

    def conditions_changed(self):
        self._fill_batch_combos()
        self.set_stale("배치 값이나 조건이 바뀌었습니다. '배치 지도 다시 계산'을 누르세요.")

    def set_stale(self, text):
        self.status.setText(f"<span style='color:#b05a00'>{text}</span>")

    def set_visible_batches(self, batches):
        self.visible = None if batches is None else set(batches)
        self.redraw_map()

    def _real_batches(self):
        return [b for b in self.project.batches.names() if b != UNASSIGNED]

    def _fill_batch_combos(self):
        names = self._real_batches()
        refs = sorted(self.project.batches.references)
        for cb, pick in ((self.ref, None), (self.cmp_a, 0), (self.cmp_b, 1), (self.drill, 0)):
            cur = cb.currentText()
            cb.blockSignals(True); cb.clear(); cb.addItems(names)
            i = cb.findText(cur)
            if i < 0 and cb is self.ref and refs:
                i = cb.findText(refs[0])
            if i < 0 and pick is not None:
                i = min(pick, len(names) - 1)
            cb.setCurrentIndex(max(i, 0))
            cb.blockSignals(False)

    def compute(self):
        p = self.project
        if p is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.desc = p.batch_descriptors()
        except Exception as e:     # shown, the app keeps working
            self.desc = None
            self.canvas.message(f"배치 지도를 계산하지 못했습니다.\n{e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.status.setText(f"항목 {len(self.desc.meta)}개 · 배치 {len(self.desc.values)}개로 계산했습니다.")
        self._fill_controls()
        self.descriptors_ready.emit(self.desc)
        self.redraw_map()

    def _fill_controls(self):
        p, d = self.project, self.desc
        s = p.settings
        # category weights
        while self.weight_lay.rowCount():
            self.weight_lay.removeRow(0)
        self.weight_spins = {}
        cats = [c for c in p.batches.categories if c in set(d.meta["category"])] + \
               [c for c in d.categories() if c not in p.batches.categories]
        for c in cats:
            sp = QtWidgets.QDoubleSpinBox(); sp.setRange(0, 10); sp.setSingleStep(0.5); sp.setDecimals(1)
            sp.setValue(float(s.bm_weights.get(c, 1.0)))
            sp.valueChanged.connect(self._weights_changed)
            n = int((d.meta["category"] == c).sum())
            self.weight_lay.addRow(f"{c} ({n})", sp)
            self.weight_spins[c] = sp
        # parts
        while self.part_lay.count():
            w = self.part_lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.part_checks = {}
        for part, n in d.meta.groupby("part", sort=False).size().items():
            cb = QtWidgets.QCheckBox(f"{part} ({n})")
            cb.setChecked(part not in s.bm_skip)
            cb.toggled.connect(self._parts_changed)
            self.part_lay.addWidget(cb)
            self.part_checks[part] = cb
        # items
        self.items.blockSignals(True); self.items.clear()
        for it, m in d.meta.iterrows():
            li = QtWidgets.QListWidgetItem(f"[{m['kind']}] {it}")
            li.setData(QtCore.Qt.UserRole, it)
            li.setToolTip(f"{it}\n범주: {m['category']} · 단위: {m['unit']}")
            li.setFlags(li.flags() | QtCore.Qt.ItemIsUserCheckable)
            li.setCheckState(QtCore.Qt.Unchecked if it in s.bm_skip else QtCore.Qt.Checked)
            self.items.addItem(li)
        self.items.blockSignals(False)
        self._filter_items()
        # colour
        cur = s.bm_color or self.color_by.currentText()
        self.color_by.blockSignals(True); self.color_by.clear()
        self.color_by.addItem(NO_COLOR)
        for it in d.meta.index:
            self.color_by.addItem(short_item(str(it), 60), it)
        i = self.color_by.findData(cur)
        self.color_by.setCurrentIndex(max(i, 0))
        self.color_by.blockSignals(False)
        for w in (self.weight_box, self.part_box, self.panel.widget()):
            w.updateGeometry()

    # ---- settings -----------------------------------------------------------
    def _weights_changed(self):
        self.project.settings.bm_weights = {c: sp.value() for c, sp in self.weight_spins.items()}
        self.settings_changed.emit()
        self.redraw_map()

    def _parts_changed(self):
        s = self.project.settings
        items = [x for x in s.bm_skip if x not in self.part_checks]
        s.bm_skip = items + [p for p, cb in self.part_checks.items() if not cb.isChecked()]
        self.settings_changed.emit()
        self.redraw_map()

    def _item_toggled(self, li):
        s = self.project.settings
        it = li.data(QtCore.Qt.UserRole)
        skip = [x for x in s.bm_skip if x != it]
        if li.checkState() != QtCore.Qt.Checked:
            skip.append(it)
        s.bm_skip = skip
        self.settings_changed.emit()
        self.redraw_map()

    def _all_items(self):
        s = self.project.settings
        s.bm_skip = [x for x in s.bm_skip if x in getattr(self, "part_checks", {})]
        self._fill_controls()
        self.settings_changed.emit()
        self.redraw_map()

    def _filter_items(self):
        q = self.item_find.text().strip().lower()
        for i in range(self.items.count()):
            li = self.items.item(i)
            li.setHidden(bool(q) and q not in li.text().lower())

    def _color_changed(self):
        self.project.settings.bm_color = self.color_by.currentData() or ""
        self.settings_changed.emit()
        self.draw()

    def _ref_changed(self):
        r = self.ref.currentText()
        if r:
            self.project.batches.references = {r}
            self.settings_changed.emit()
        self.draw_score()
        self.draw()

    def ui_state(self):
        return {"tab": self.tabs.currentIndex(), "names": self.show_names.isChecked(),
                "a": self.cmp_a.currentText(), "b": self.cmp_b.currentText()}

    def restore_ui(self, st):
        if not st:
            return
        self.tabs.setCurrentIndex(int(st.get("tab", 0)))
        self.show_names.setChecked(bool(st.get("names", True)))
        for cb, k in ((self.cmp_a, "a"), (self.cmp_b, "b")):
            i = cb.findText(st.get(k, ""))
            if i >= 0:
                cb.setCurrentIndex(i)

    # ---- map ----------------------------------------------------------------
    def shown_batches(self):
        names = list(self.desc.values.index)
        return names if self.visible is None else [b for b in names if b in self.visible]

    def redraw_map(self):
        if self.desc is None:
            return
        s = self.project.settings
        shown = self.shown_batches()
        self.bm = batch_distance_map(subset(self.desc, shown), s.bm_weights, s.bm_skip)
        n = len(shown)
        used = len(self.bm.used)
        hidden = len(self.desc.values) - n
        parts = [f"배치 {n}개" + (f" (왼쪽 목록에서 뺀 배치 {hidden}개)" if hidden else ""),
                 f"거리에 쓴 항목 {used}개 / 전체 {len(self.desc.meta)}개"]
        if n >= 2:
            off = self.bm.coverage.where(~np.eye(n, dtype=bool)).stack()
            if len(off):
                parts.append(f"두 배치가 함께 가진 항목 {int(off.min())}–{int(off.max())}개")
        self.note.setText(" · ".join(parts))
        self.draw()
        self.draw_categories()
        self.draw_distance()
        self.draw_tree()
        self.draw_score()

    def _colors(self, names):
        it = self.color_by.currentData()
        if not it or it not in self.desc.values.columns:
            return [color(0)] * len(names), None
        v = self.desc.values.loc[names, it]
        if self.desc.meta.at[it, "type"] == "numeric":
            x = pd.to_numeric(v, errors="coerce").to_numpy(float)
            return x, "num"
        cats = list(pd.unique(v.dropna().astype(str)))
        return [color(cats.index(str(a))) if pd.notna(a) else "#cccccc" for a in v], cats

    def draw(self):
        c = self.canvas
        bm = self.bm
        if bm is None:
            return
        names = bm.batches
        if len(names) < 3:
            if len(names) == 2:
                d = bm.distance.iat[0, 1]
                c.message(f"배치가 2개입니다: {names[0]} ↔ {names[1]} 거리 {d:.2f} (0 = 같음, 1 = 가장 다름).\n"
                          "배치가 3개 이상이면 지도를 그립니다. '배치 비교' 탭에서 항목별 차이를 보세요.")
            else:
                c.message("배치가 하나뿐입니다. 다른 배치의 데이터를 추가하면 배치 사이의 거리를 지도로 그립니다.\n"
                          "(파일 › 불러오기 · 입자 CSV를 넣을 때 배치 이름을 정합니다. 파일 › 가상 예제 프로젝트로 미리 볼 수 있습니다.)")
            self.axis_note.setText("")
            return
        xy = bm.coords
        c.clear()
        ax = c.fig.add_subplot(111)
        cols, kind = self._colors(names)
        refs = self.project.batches.references
        if kind == "num":
            ok = np.isfinite(cols)
            sc = ax.scatter(xy.iloc[ok, 0], xy.iloc[ok, 1], c=cols[ok], cmap="viridis", s=120, edgecolor="k", lw=.5, zorder=3)
            ax.scatter(xy.iloc[~ok, 0], xy.iloc[~ok, 1], c="#dddddd", s=120, edgecolor="#999", lw=.5, zorder=3)
            it = self.color_by.currentData()
            c.fig.colorbar(sc, ax=ax, label=short_item(f"{it} ({self.desc.meta.at[it, 'unit']})", 50), shrink=.8)
        else:
            ax.scatter(xy.iloc[:, 0], xy.iloc[:, 1], c=cols, s=120, edgecolor="k", lw=.5, zorder=3)
            if kind:
                for i, k in enumerate(kind):
                    ax.scatter([], [], c=color(i), s=60, label=k)
                ax.legend(fontsize=8, frameon=False, title=short_item(self.color_by.currentData(), 30), title_fontsize=8)
        for b in names:
            if b in refs:
                ax.scatter([xy.at[b, "축 1"]], [xy.at[b, "축 2"]], marker="*", s=420, facecolor="none", edgecolor="#c9a227", lw=1.6, zorder=4)
        if self.show_names.isChecked():
            for b in names:
                ax.annotate(b, (xy.at[b, "축 1"], xy.at[b, "축 2"]), xytext=(7, 6), textcoords="offset points", fontsize=9)
        e = bm.explained
        ax.set_xlabel(f"축 1 ({e[0]:.0f}%)")
        ax.set_ylabel(f"축 2 ({e[1]:.0f}%)")
        ax.axhline(0, color="#eee", lw=.8, zorder=0); ax.axvline(0, color="#eee", lw=.8, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")     # distances read the same in both directions
        ax.set_title("가까울수록 비슷한 배치 (PCoA) — 점을 두 번 누르면 그 배치의 입자 지도", fontsize=9)
        c.draw_idle()
        lines = []
        for a, rows in bm.axis_items.items():
            if rows:
                lines.append(f"<b>{a}</b> 방향으로 함께 변하는 항목: " +
                             ", ".join(f"{short_item(it, 60)} ({'+' if r > 0 else '−'}{abs(r):.2f})" for it, r in rows))
        self.axis_note.setText("<br>".join(lines))

    def _clicked(self, ev):
        if not ev.dblclick or self.bm is None or self.bm.coords is None or ev.inaxes is None:
            return
        xy = self.bm.coords
        d = (xy["축 1"] - ev.xdata) ** 2 + (xy["축 2"] - ev.ydata) ** 2
        self.particle_map_requested.emit(str(d.idxmin()))

    def draw_categories(self):
        c = self.cat_canvas
        bm = self.bm
        if bm is None or len(bm.batches) < 3 or not bm.by_category:
            c.message("배치가 3개 이상일 때 범주마다 따로 그린 지도가 나옵니다."); return
        s = self.project.settings
        cats = list(bm.by_category)
        c.clear()
        ncol = min(3, len(cats))
        nrow = int(np.ceil(len(cats) / ncol))
        sub = subset(self.desc, bm.batches)
        for i, cat in enumerate(cats):
            ax = c.fig.add_subplot(nrow, ncol, i + 1)
            w = {k: (1.0 if k == cat else 0.0) for k in cats}
            one = batch_distance_map(sub, w, s.bm_skip)
            if one.coords is None:
                ax.axis("off"); continue
            xy = one.coords
            ax.scatter(xy["축 1"], xy["축 2"], s=50, color=color(i), edgecolor="k", lw=.4)
            for b in xy.index:
                ax.annotate(b, (xy.at[b, "축 1"], xy.at[b, "축 2"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
            ax.set_title(f"{cat} — 항목 {len(one.used)}개, 축 1 {one.explained[0]:.0f}%", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        c.draw_idle()

    def draw_distance(self):
        c = self.dist_canvas
        bm = self.bm
        if bm is None or len(bm.batches) < 2:
            c.message("배치가 2개 이상이면 거리 표가 나옵니다."); return
        D, N = bm.distance, bm.coverage
        c.clear()
        ax = c.fig.add_subplot(111)
        im = ax.imshow(D.to_numpy(float), cmap="viridis_r", vmin=0, vmax=max(0.01, float(np.nanmax(D.to_numpy(float)))))
        ax.set_xticks(range(len(D)), D.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(D)), D.index)
        if len(D) <= 14:
            for i in range(len(D)):
                for j in range(len(D)):
                    if i != j:
                        v = D.iat[i, j]
                        ax.text(j, i, "—" if pd.isna(v) else f"{v:.2f}\nn={N.iat[i, j]}", ha="center", va="center", fontsize=7,
                                color="w" if pd.notna(v) and v > 0.35 else "k")
        c.fig.colorbar(im, ax=ax, label="거리 (0 = 같음, 1 = 가장 다름)", shrink=.8)
        ax.set_title("배치 사이 거리 · n = 함께 가진 항목 수", fontsize=9)
        c.draw_idle()

    def draw_tree(self):
        c = self.tree_canvas
        bm = self.bm
        if bm is None or bm.linkage is None:
            c.message("배치가 3개 이상이면 배치 군집(덴드로그램)이 나옵니다."); return
        c.clear()
        ax = c.fig.add_subplot(111)
        dendrogram(bm.linkage, labels=bm.batches, ax=ax, orientation="right", color_threshold=0, above_threshold_color="#555")
        ax.set_xlabel("거리 (평균 연결)")
        ax.set_title("비슷한 배치끼리 묶기", fontsize=9)
        c.draw_idle()

    def score_frame(self):
        ref = self.ref.currentText()
        if self.bm is None or not ref or ref not in self.bm.distance.index:
            return None
        s = batch_scores(self.bm, ref)
        out = pd.DataFrame({"score": s, "거리": self.bm.distance[ref], "함께 가진 항목 수": self.bm.coverage[ref]})
        return out.sort_values("score", ascending=False)

    def draw_score(self):
        c = self.score_canvas
        f = self.score_frame()
        if f is None or len(f) < 2:
            c.message("기준 배치를 고르면 다른 배치의 score가 나옵니다 (배치 2개 이상)."); return
        c.clear()
        ax = c.fig.add_subplot(111)
        y = np.arange(len(f))
        ax.barh(y, f["score"], color=[("#c9a227" if b == self.ref.currentText() else color(0)) for b in f.index], alpha=.8)
        ax.set_yticks(y, f.index); ax.invert_yaxis()
        for i, v in enumerate(f["score"]):
            ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=8)
        ax.set_xlim(0, 108)
        ax.set_xlabel(f"score (기준 배치 {self.ref.currentText()} = 100)")
        c.draw_idle()

    def pair(self, a, b):
        """Item differences between two batches (scaled by the range over all batches)."""
        return pair_differences(self.desc, a, b, self.bm.used if self.bm is not None else None)
