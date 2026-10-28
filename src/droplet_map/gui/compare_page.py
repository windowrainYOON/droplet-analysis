"""Comparing conditions: group similarity, condition x parameter association, trends, summary table."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from .common import MplCanvas, color, fill_table, form, info_label, scroll_panel, section, short_group, title_label


def image_level_targets(result, project) -> pd.DataFrame:
    """Per-image medians of each feature + image stats; rows = (group, image)."""
    p = result.prepared
    d = p.data
    rows = {}
    for (g, im), idx in d.groupby(["group", "image"], sort=False).groups.items():
        idx = np.asarray(idx)
        row = {}
        for f in p.features:
            v = d.loc[idx, f].to_numpy(float)
            if f in p.shape_features:
                w = p.w[idx]
                if w.sum() > 0:
                    o = np.argsort(v); cw = np.cumsum(w[o]); row[f] = v[o][np.searchsorted(cw, cw[-1] / 2)]
                else:
                    row[f] = np.nan
            else:
                row[f] = float(np.median(v))
        row["이미지당 입자 수"] = len(idx)
        if "area_um2" in d.columns:
            row["이미지당 총 입자 면적 µm²"] = float(d.loc[idx, "area_um2"].sum())
        rows[(g, im)] = row
    return pd.DataFrame(rows).T


class ComparePage(QtCore.QObject):
    """Center: trend / association / similarity / summary tabs. Right: what to plot."""
    title = "조건 비교"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.result = None
        self.img_targets = None
        self.summary_frame = None
        self.view = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.view)
        lay.addWidget(title_label("조건 비교"))
        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.trend = MplCanvas(None, 8, 6)
        self.tabs.addTab(self.trend, "조건별 추세")
        aw = QtWidgets.QWidget(); al = QtWidgets.QVBoxLayout(aw); al.setContentsMargins(0, 0, 0, 0)
        self.assoc = MplCanvas(None, 8, 5)
        al.addWidget(self.assoc, 1)
        self.confound = QtWidgets.QLabel(); self.confound.setStyleSheet("color:#b05a00;"); self.confound.setWordWrap(True)
        al.addWidget(self.confound)
        self.tabs.addTab(aw, "조건 × 파라미터 연관")
        self.sim = MplCanvas(None, 8, 7)
        self.tabs.addTab(self.sim, "그룹 유사도")
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.tabs.addTab(self.table, "그룹 요약 표")
        for c in (self.trend, self.assoc, self.sim):
            c.message("'분석 실행' 후에 표시됩니다.")

        self.x_cond = QtWidgets.QComboBox(); self.hue_cond = QtWidgets.QComboBox(); self.y_target = QtWidgets.QComboBox()
        for w in (self.x_cond, self.hue_cond, self.y_target):
            w.currentIndexChanged.connect(self.draw_trend)
        self.all_targets = QtWidgets.QCheckBox("모든 값 한 번에 보기"); self.all_targets.setChecked(True)
        self.all_targets.toggled.connect(self.draw_trend)
        self.all_targets.toggled.connect(lambda on: self.y_target.setEnabled(not on))
        self.y_target.setEnabled(False)
        trend = section("조건별 추세",
                        form([("가로축 조건", self.x_cond), ("색 구분 조건", self.hue_cond), ("값", self.y_target)]),
                        self.all_targets,
                        info_label("점 = 이미지 1장, 선 = 그룹(이미지 중앙값). 이미지 사이 흩어짐이 크면 그 차이는 조심해서 읽습니다."))
        about = section("읽는 법",
                        info_label("비교는 입자가 아니라 그룹(조건 1세트)과 이미지를 단위로 합니다. 같은 이미지의 입자 수천 개는 "
                                   "독립 실험이 아니기 때문입니다."),
                        info_label("연관: 수치형 조건은 그룹 중앙값과의 Spearman ρ(부호 = 방향), 범주형 조건은 Kruskal-Wallis ε²(0–1). "
                                   "기록된 그룹이 너무 적거나 값이 한 가지뿐인 조건은 숨깁니다. 연관은 원인이 아니라 단서입니다."))
        self.panel = scroll_panel(trend, about)
        self.tabs.currentChanged.connect(lambda i: trend.setVisible(i == 0))

    def ui_state(self):
        return {"x": self.x_cond.currentText(), "hue": self.hue_cond.currentText(), "y": self.y_target.currentText(),
                "all": self.all_targets.isChecked(), "tab": self.tabs.currentIndex()}

    def restore_ui(self, st):
        if not st:
            return
        for w, k in ((self.x_cond, "x"), (self.hue_cond, "hue"), (self.y_target, "y")):
            i = w.findText(st.get(k, ""))
            if i >= 0:
                w.setCurrentIndex(i)
        self.all_targets.setChecked(bool(st.get("all", True)))
        self.tabs.setCurrentIndex(int(st.get("tab", 0)))

    def set_project(self, project):
        self.project = project
        self.result = None
        self._fill_conditions()

    def _fill_conditions(self):
        if self.project is None:
            return
        names = [c.name for c in self.project.view.schema]
        for combo, extra in ((self.x_cond, []), (self.hue_cond, ["(없음)"])):
            cur = combo.currentText()
            combo.blockSignals(True); combo.clear()
            for n in extra + names:
                combo.addItem(n)
            i = combo.findText(cur)
            combo.setCurrentIndex(i if i >= 0 else 0)
            combo.blockSignals(False)
        # sensible defaults: first numeric condition with >1 level on x, second on hue
        t = self.project.view
        varied = [c.name for c in t.schema if t.categorical(c.name).nunique() > 1]
        if self.x_cond.currentText() not in varied and varied:
            self.x_cond.setCurrentText(varied[0])
        if self.hue_cond.currentText() in ("(없음)", self.x_cond.currentText()) and len(varied) > 1:
            self.hue_cond.setCurrentText([v for v in varied if v != self.x_cond.currentText()][0])

    def conditions_changed(self):
        self._fill_conditions()
        if self.result is not None:
            self.set_result(self.result)

    def set_result(self, result):
        self.result = result
        self.img_targets = image_level_targets(result, self.project)
        cur = self.y_target.currentText()
        self.y_target.blockSignals(True); self.y_target.clear()
        for c in self.img_targets.columns:
            self.y_target.addItem(c)
        if result.score is not None:
            self.y_target.addItem("기준 score")
        i = self.y_target.findText(cur); self.y_target.setCurrentIndex(max(i, 0))
        self.y_target.blockSignals(False)
        self.draw_trend()
        self.draw_assoc()
        self.draw_similarity()
        self.fill_summary()

    # ---- trend -----------------------------------------------------------
    def draw_trend(self):
        c = self.trend
        if self.result is None or self.img_targets is None:
            return
        t = self.project.view
        if not t.schema or not self.x_cond.currentText():
            c.message("조건 편집기에서 조건을 먼저 입력하세요."); return
        xname = self.x_cond.currentText()
        hue = self.hue_cond.currentText()
        hue = None if hue in ("(없음)", xname, "") else hue
        xkind = t.get(xname).kind
        xv = t.numeric(xname) if xkind == "numeric" else t.categorical(xname)
        hv = t.categorical(hue) if hue else None
        targets = list(self.img_targets.columns) + (["기준 score"] if self.result.score is not None else [])
        if not self.all_targets.isChecked():
            targets = [self.y_target.currentText()]
        c.clear()
        n = len(targets)
        ncol = min(3, n); nrow = int(np.ceil(n / ncol))
        cats = None
        if xkind != "numeric":
            cats = sorted(xv.dropna().unique(), key=str)
        hue_levels = sorted(hv.dropna().unique(), key=lambda s: (_num(s) is None, _num(s), str(s))) if hv is not None else [None]
        for k, tgt in enumerate(targets):
            ax = c.fig.add_subplot(nrow, ncol, k + 1)
            for hi, h in enumerate(hue_levels):
                gsel = [g for g in xv.dropna().index if (h is None or hv.get(g) == h)]
                pts_x, pts_y, line = [], [], []
                for g in gsel:
                    x = xv[g] if xkind == "numeric" else cats.index(xv[g])
                    if tgt == "기준 score":
                        ys = [self.result.score["score"].get(g, np.nan)]
                    else:
                        ys = self.img_targets.loc[g, tgt].to_numpy(float) if g in self.img_targets.index.get_level_values(0) else []
                    jitter = (np.arange(len(ys)) - (len(ys) - 1) / 2) * 0.0
                    pts_x += list(x + jitter); pts_y += list(ys)
                    if len(ys):
                        line.append((x, np.nanmedian(ys)))
                col = color(hi) if h is not None else "#2a6fdb"
                ax.scatter(pts_x, pts_y, s=10, color=col, alpha=.55, linewidths=0)
                if line:
                    line.sort()
                    lx, ly = zip(*line)
                    ax.plot(lx, ly, "-o", color=col, ms=3.5, lw=1.4, label=(f"{hue} {h}" if h is not None else None))
            ax.set_title(tgt, fontsize=8.5)
            if cats is not None:
                ax.set_xticks(range(len(cats)), [str(x) for x in cats], fontsize=7, rotation=20)
            ax.tick_params(labelsize=7)
            if tgt in ("이미지당 입자 수",) or any(s in tgt for s in ("partition", "diameter", "area", "nnd")):
                vals = np.asarray(pts_y, float)
                if np.nanmin(vals) > 0 and np.nanmax(vals) / max(np.nanmin(vals), 1e-12) > 20:
                    ax.set_yscale("log")
            if k >= n - ncol:
                ax.set_xlabel(t.get(xname).label, fontsize=8)
        if hue:
            handles, labels = c.fig.axes[0].get_legend_handles_labels()
            if handles:
                c.fig.legend(handles, labels, loc="outside upper right", fontsize=7, frameon=False, ncol=min(6, len(labels)))
        c.draw_idle()

    # ---- association -----------------------------------------------------
    def draw_assoc(self):
        c = self.assoc
        a = self.result.assoc
        if a is None:
            c.message("조건을 입력하면 조건 × 파라미터 연관 지도가 나옵니다."); self.confound.setText(""); return
        c.clear()
        ax = c.fig.add_subplot(111)
        M = a.strength.to_numpy(float)
        hidden = a.hidden.reindex(a.strength.index).to_numpy()
        show = np.where(hidden[:, None], np.nan, M)
        cmap = __import__("matplotlib").colormaps["Purples"].copy(); cmap.set_bad("#efefef")
        im = ax.imshow(np.ma.masked_invalid(show), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if hidden[i]:
                    txt = "숨김"; colr = "#999"
                elif np.isfinite(M[i, j]):
                    s = a.sign.iat[i, j]
                    txt = (("+" if s > 0 else "-") if s != 0 else "") + f"{M[i, j]:.2f}"
                    colr = "white" if M[i, j] > .55 else "#222"
                else:
                    continue
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=colr)
        ax.set_xticks(range(M.shape[1]), a.strength.columns, rotation=25, ha="right", fontsize=8)
        ylabels = []
        for name in a.strength.index:
            why = ""
            by_batch = a.level is not None and a.level.get(name) == "batch"
            if hidden[list(a.strength.index).index(name)]:
                if self.project.view.categorical(name).nunique() < 2:
                    why = " · 값 1가지"
                else:
                    why = " · 배치 3개 미만" if by_batch else " · 그룹 부족"
            unit = "배치" if by_batch else "그룹"
            ylabels.append(f"{name} ({unit} n={a.n[name]}){why}")
        ax.set_yticks(range(M.shape[0]), ylabels, fontsize=8)
        c.fig.colorbar(im, ax=ax, label="연관 강도 (|ρ| 또는 ε²)", shrink=.8)
        c.draw_idle()
        if a.confounded:
            self.confound.setText("함께 움직이는 조건 (효과를 구분하기 어려움): " +
                                  ", ".join(f"{x} ↔ {y} ({v:.2f})" for x, y, v in a.confounded))
        else:
            self.confound.setText("")

    # ---- similarity ------------------------------------------------------
    def draw_similarity(self):
        c = self.sim
        D = self.result.distance
        c.clear()
        if len(D) < 2:
            c.message("그룹이 2개 이상이어야 합니다."); return
        Dv = np.nan_to_num(D.to_numpy(), nan=np.nanmax(D.to_numpy()))
        order = leaves_list(linkage(squareform(Dv, checks=False), "average")) if len(D) > 2 else np.arange(len(D))
        g = [D.index[i] for i in order]
        t = self.project.view
        strips = [x.name for x in t.schema if t.categorical(x.name).nunique() > 1][:5]
        hr = [0.35] * len(strips) + [6]
        gs = c.fig.add_gridspec(len(strips) + 1, 2, height_ratios=hr, width_ratios=[30, 1])
        for si, s in enumerate(strips):
            ax = c.fig.add_subplot(gs[si, 0])
            vals = t.categorical(s).reindex(g)
            levels = sorted(vals.dropna().unique(), key=lambda v: (_num(v) is None, _num(v), str(v)))
            for x, v in enumerate(vals):
                ax.add_patch(__import__("matplotlib").patches.Rectangle((x - .5, 0), 1, 1, color="#eeeeee" if pd.isna(v) else color(levels.index(v))))
                if not pd.isna(v):
                    ax.text(x, .5, str(v), ha="center", va="center", fontsize=6, color="white")
            ax.set_xlim(-.5, len(g) - .5); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_ylabel(s, rotation=0, ha="right", va="center", fontsize=8)
        ax = c.fig.add_subplot(gs[len(strips), 0])
        im = ax.imshow(D.loc[g, g].to_numpy(), cmap="viridis_r", aspect="auto")
        lab = [short_group(x, 30) for x in g]
        ax.set_xticks(range(len(g)), lab, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(g)), lab, fontsize=6.5)
        cax = c.fig.add_subplot(gs[len(strips), 1])
        c.fig.colorbar(im, cax=cax, label="그룹 간 거리 (z, 작을수록 비슷)")
        c.draw_idle()

    def fill_summary(self):
        r = self.result
        prof = r.profiles
        df = pd.DataFrame({"이미지": prof.n_images, "입자": prof.n_particles,
                           "형태 유효 입자 n_eff": prof.n_eff_shape.round(1), "평균 신뢰도 w": prof.mean_weight.round(3)})
        df = df.join(prof.median_raw.add_prefix("중앙값 "))
        df = df.join(prof.image_stats)
        if r.score is not None:
            df["기준 score"] = r.score["score"].reindex(df.index)
        t = self.project.view
        for cnd in t.schema:
            df[f"[조건] {cnd.name}"] = t.categorical(cnd.name).reindex(df.index)
        self.table.setSortingEnabled(False)
        fill_table(self.table, df)
        self.table.setSortingEnabled(True)
        self.summary_frame = df


def _num(v):
    try:
        return float(str(v))
    except ValueError:
        return None
