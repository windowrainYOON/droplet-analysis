"""Export: tables as CSV and the current figures as PNG, into one folder."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from .common import info_label, title_label, warn


class ExportPage(QtWidgets.QWidget):
    def __init__(self, parent=None, figure_sources=None):
        super().__init__(parent)
        self.project = None
        self.result = None
        self.figure_sources = figure_sources or (lambda: {})
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(title_label("내보내기"))
        lay.addWidget(info_label("고른 항목을 한 폴더에 CSV와 PNG로 저장합니다. 엑셀이나 Prism에서 바로 열 수 있습니다 (UTF-8 BOM)."))
        self.checks = {}
        for key, label in (("particles", "입자별 표 (QC 통과 입자 + 지도 좌표 + 군집 + 신뢰도 w)"),
                           ("groups", "그룹 요약 (중앙값, 이미지 통계, n_eff, 조건)"),
                           ("score", "기준 score와 Δ"),
                           ("assoc", "조건 × 파라미터 연관 (강도, 부호, n)"),
                           ("distance", "그룹 간 거리 행렬"),
                           ("composition", "그룹별 군집 구성 비율"),
                           ("conditions", "조건 표"),
                           ("figures", "현재 화면의 그림 전부 (PNG, 200 dpi)")):
            cb = QtWidgets.QCheckBox(label); cb.setChecked(True)
            self.checks[key] = cb
            lay.addWidget(cb)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("폴더 골라서 내보내기…"); b.clicked.connect(self.export)
        row.addWidget(b); row.addStretch(1)
        lay.addLayout(row)
        self.status = QtWidgets.QLabel(""); self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.status)
        lay.addStretch(1)

    def set_project(self, project):
        self.project = project
        self.result = None

    def set_result(self, result):
        self.result = result

    def export(self):
        if self.project is None:
            warn(self, "프로젝트가 없습니다."); return
        base = QtWidgets.QFileDialog.getExistingDirectory(self, "내보낼 폴더", str(Path.home() / "Desktop"))
        if not base:
            return
        name = (self.project.path.stem if self.project.path else "droplet_map") + "_" + datetime.now().strftime("%Y%m%d-%H%M")
        out = Path(base) / name
        files = write_exports(out, self.project, self.result, {k: c.isChecked() for k, c in self.checks.items()},
                              self.figure_sources() if self.checks["figures"].isChecked() else {})
        self.status.setText(f"{len(files)}개 파일 저장: {out}")
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(out)))


def write_exports(out: Path, project, result, which: dict, figures: dict) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    files = []

    def csv(df, fname, index=True):
        p = out / fname
        df.to_csv(p, index=index, encoding="utf-8-sig")
        files.append(p)

    if which.get("conditions"):
        p = out / "conditions.csv"; p.write_text(project.conditions.to_csv(), encoding="utf-8-sig"); files.append(p)
    if result is not None:
        prep = result.prepared
        if which.get("particles"):
            d = prep.data.copy()
            d["shape_weight_w"] = prep.w
            if result.pmap is not None:
                d["umap_1"], d["umap_2"] = result.pmap.umap[:, 0], result.pmap.umap[:, 1]
                d["cluster"] = result.pmap.clusters
            csv(d, "particles_with_clusters.csv", index=False)
        if which.get("groups"):
            prof = result.profiles
            g = pd.DataFrame({"images": prof.n_images, "particles": prof.n_particles,
                              "n_eff_shape": prof.n_eff_shape, "mean_w": prof.mean_weight})
            g = g.join(prof.median_raw.add_prefix("median_")).join(prof.image_stats)
            if result.score is not None:
                g["reference_score"] = result.score["score"].reindex(g.index)
            t = project.conditions
            for c in t.schema:
                g[f"cond: {c.label}"] = t.categorical(c.name).reindex(g.index)
            csv(g, "group_summary.csv")
        if which.get("score") and result.score is not None:
            csv(result.score, "reference_score.csv")
        if which.get("assoc") and result.assoc is not None:
            a = result.assoc
            signed = a.strength * a.sign.where(a.sign != 0, 1)
            tab = signed.copy()
            tab.insert(0, "kind", a.kind); tab.insert(1, "n_groups", a.n); tab.insert(2, "hidden", a.hidden)
            csv(tab, "condition_association.csv")
        if which.get("distance"):
            csv(result.distance, "group_distance.csv")
        if which.get("composition") and result.profiles.composition is not None:
            csv(result.profiles.composition, "cluster_composition.csv")
    for fname, fig in figures.items():
        p = out / f"{fname}.png"
        try:
            fig.savefig(p, dpi=200)
            files.append(p)
        except Exception:
            pass
    return files
