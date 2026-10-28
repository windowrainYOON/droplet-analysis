"""Shared GUI helpers: matplotlib canvas with a Korean font, colours, small widgets."""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

PALETTE = ["#2a6fdb", "#e0703a", "#2e9e6a", "#b04fc0", "#c9a227", "#5b8a9a", "#d6456b", "#7a7a7a",
           "#8fbf3f", "#4b4bb0", "#e39fb7", "#3fb8c9", "#a0522d", "#6b8e23", "#ff8c00", "#708090"]


def _setup_fonts():
    preferred = ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    if not available.intersection(preferred):
        # the font cache may predate an installed Korean font; rebuild once
        for f in font_manager.findSystemFonts():
            if any(k in f for k in ("Nanum", "AppleSDGothic", "AppleGothic", "NotoSansCJK", "malgun")):
                try:
                    font_manager.fontManager.addfont(f)
                except Exception:
                    pass
        available = {f.name for f in font_manager.fontManager.ttflist}
    fam = [f for f in preferred if f in available] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})


_setup_fonts()


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, height=4):
        self.fig = Figure(figsize=(width, height), dpi=100, layout="constrained")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def clear(self):
        self.fig.clear()

    def message(self, text):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(.5, .5, text, ha="center", va="center", color="#777", fontsize=11, wrap=True)
        self.draw_idle()


_COMMON = {"head": 0, "tail": 0}


def set_group_names(groups):
    """Remember comma-separated parts shared by every group name (e.g. 'FITC-PRD', '10% PEG')
    so plots can show only the part that differs."""
    parts = [[p.strip() for p in str(g).split(",")] for g in groups]
    head = tail = 0
    if len(parts) > 1 and min(len(p) for p in parts) > 1:
        m = min(len(p) for p in parts)
        while head < m - 1 and len({p[head] for p in parts}) == 1:
            head += 1
        while tail < m - 1 - head and len({p[-1 - tail] for p in parts}) == 1:
            tail += 1
    _COMMON.update(head=head, tail=tail)


def short_group(name: str, n=38) -> str:
    p = [x.strip() for x in str(name).split(",")]
    h, t = _COMMON["head"], _COMMON["tail"]
    if (h or t) and len(p) > h + t:
        name = ", ".join(p[h: len(p) - t])
    return name if len(name) <= n else name[: n - 1] + "…"


def info_label(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color:#666;")
    return lab


def title_label(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    f = lab.font(); f.setPointSize(f.pointSize() + 5); f.setBold(True)
    lab.setFont(f)
    return lab


def fill_table(table: QtWidgets.QTableWidget, df, float_fmt="{:.3g}", index_label="그룹"):
    table.clear()
    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns) + 1)
    table.setHorizontalHeaderLabels([index_label] + [str(c) for c in df.columns])
    for i, (idx, row) in enumerate(df.iterrows()):
        table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(idx)))
        for j, v in enumerate(row):
            if isinstance(v, float):
                txt = "" if v != v else float_fmt.format(v)
            else:
                txt = "" if v is None else str(v)
            it = QtWidgets.QTableWidgetItem(txt)
            if isinstance(v, (int, float)):
                it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                it.setData(QtCore.Qt.UserRole, v)
            table.setItem(i, j + 1, it)
    table.resizeColumnsToContents()


def is_mac() -> bool:
    return sys.platform == "darwin"


def warn(parent, text):
    QtWidgets.QMessageBox.warning(parent, "Droplet Map", text)


def app_icon() -> QtGui.QIcon:
    from importlib import resources
    try:
        p = resources.files("droplet_map") / "resources" / "droplet_map.png"
        return QtGui.QIcon(str(p))
    except Exception:
        return QtGui.QIcon()


class WeightsTable(QtWidgets.QTableWidget):
    """Per-measurement weights, written straight into settings.feature_weights.
    The map page and the score page each show one; `changed` lets the other one refresh."""
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["측정값", "가중치"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().hide()
        self.settings = None

    def set_features(self, features, shape_features, settings):
        self.settings = settings
        fw = settings.feature_weights
        self.setRowCount(len(features))
        for i, f in enumerate(features):
            it = QtWidgets.QTableWidgetItem(f + (" (형태)" if f in shape_features else ""))
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.setItem(i, 0, it)
            sp = QtWidgets.QDoubleSpinBox(); sp.setRange(0, 10); sp.setSingleStep(0.5); sp.setDecimals(2)
            sp.setValue(float(fw.get(f, 1.0)))
            sp.setProperty("feature", f)
            sp.valueChanged.connect(lambda v, f=f: self._set(f, v))
            self.setCellWidget(i, 1, sp)
        self.resizeColumnToContents(0)

    def _set(self, f, v):
        if self.settings is not None:
            self.settings.feature_weights[f] = float(v)
            self.changed.emit()

    def refresh(self):
        if self.settings is None:
            return
        for i in range(self.rowCount()):
            sp = self.cellWidget(i, 1)
            v = float(self.settings.feature_weights.get(sp.property("feature"), 1.0))
            if sp.value() != v:
                sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)

    def reset(self):
        if self.settings is None:
            return
        for i in range(self.rowCount()):
            self.settings.feature_weights[self.cellWidget(i, 1).property("feature")] = 1.0
        self.refresh()
        self.changed.emit()


class FuncWorker(QtCore.QObject):
    """Runs fn() on a QThread; see run_in_thread."""
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @QtCore.Slot()
    def run(self):
        import traceback
        try:
            self.finished.emit(self.fn())
        except Exception as e:  # shown to the user
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# numba (UMAP) compiles on first use and overflows QThread's 512 KB default stack on macOS,
# which kills the app without a Python traceback.
WORKER_STACK = 64 * 1024 * 1024


def worker_thread(parent) -> QtCore.QThread:
    th = QtCore.QThread(parent)
    th.setStackSize(WORKER_STACK)
    return th


class _Relay(QtCore.QObject):
    """Lives on the GUI thread, so worker signals reach the callbacks there (queued)."""
    done = QtCore.Signal(object)
    fail = QtCore.Signal(str)


def run_in_thread(owner, fn, done, failed):
    """Start fn() in the background; done(result) / failed(message) run on the GUI thread."""
    th = worker_thread(owner)
    wk = FuncWorker(fn)
    wk.moveToThread(th)
    relay = _Relay(owner)
    th.started.connect(wk.run)
    wk.finished.connect(relay.done)
    wk.failed.connect(relay.fail)

    def _finish(cb, arg):
        th.quit(); th.wait()
        owner._bg = None
        relay.deleteLater()
        cb(arg)
    relay.done.connect(lambda r: _finish(done, r))
    relay.fail.connect(lambda m: _finish(failed, m))
    owner._bg = (th, wk, relay)   # keep them alive
    th.start()
