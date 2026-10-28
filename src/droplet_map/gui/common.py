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


class FlowLayout(QtWidgets.QLayout):
    """Lays widgets left to right and wraps to the next line when the pane is narrow."""

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect, test):
        x, y, line_h = rect.x(), rect.y(), 0
        for it in self._items:
            hint = it.sizeHint()
            nx = x + hint.width() + self._spacing
            if nx - self._spacing > rect.right() and line_h > 0:
                x, y = rect.x(), y + line_h + self._spacing
                nx = x + hint.width() + self._spacing
                line_h = 0
            if not test:
                it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = nx
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y()


def flow_buttons(items, parent=None) -> QtWidgets.QWidget:
    """A wrapping row of push buttons from (text, slot) pairs."""
    w = QtWidgets.QWidget(parent)
    fl = FlowLayout(w)
    for text, fn in items:
        b = QtWidgets.QPushButton(text)
        b.clicked.connect(fn)
        fl.addWidget(b)
    return w


def section(title: str, *widgets, layout=None) -> QtWidgets.QGroupBox:
    box = QtWidgets.QGroupBox(title)
    lay = layout or QtWidgets.QVBoxLayout()
    box.setLayout(lay)
    for w in widgets:
        if isinstance(w, QtWidgets.QLayout):
            lay.addLayout(w)
        else:
            lay.addWidget(w)
    return box


def form(rows) -> QtWidgets.QFormLayout:
    """Label/field rows that wrap the field under the label when the pane is narrow."""
    f = QtWidgets.QFormLayout()
    f.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
    f.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    for label, w in rows:
        f.addRow(label, w)
    return f


def scroll_panel(*widgets) -> QtWidgets.QScrollArea:
    """Right-pane panel: widgets stacked top to bottom, scrolls vertically, never forces width."""
    inner = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(inner)
    lay.setContentsMargins(4, 4, 4, 4)
    for w in widgets:
        lay.addWidget(w)
    lay.addStretch(1)
    sa = QtWidgets.QScrollArea()
    sa.setWidget(inner)
    sa.setWidgetResizable(True)
    sa.setFrameShape(QtWidgets.QFrame.NoFrame)
    sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    return sa


def save_figure_dialog(parent, fig, default_name):
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, "그림 저장", default_name,
                                                    "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif)")
    if path:
        fig.savefig(path, dpi=200)
    return path


def save_frame_dialog(parent, df, default_name, index=True):
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, "CSV 저장", default_name, "CSV (*.csv)")
    if path:
        df.to_csv(path, index=index, encoding="utf-8-sig")
    return path
