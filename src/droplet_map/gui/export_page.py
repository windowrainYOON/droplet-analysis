"""Export: tables as CSV and the current figures as PNG, into one folder."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from .common import info_label


EXPORT_ITEMS = (("batch_map", "배치 지도 (배치 표 전체 항목, 거리 표, 지도 좌표, 기준 배치 score)"),
                ("particles", "입자별 표 (QC 통과 입자 + 현재 지도 좌표 + 군집 + 신뢰도 w)"),
                ("groups", "그룹 요약 (중앙값, 이미지 통계, n_eff, 조건)"),
                ("score", "기준 score와 Δ"),
                ("assoc", "조건 × 파라미터 연관 (강도, 부호, n)"),
                ("distance", "그룹 간 거리 행렬"),
                ("composition", "그룹별 군집 구성 비율 (현재 지도)"),
                ("conditions", "형성 조건 표 (배치, 배치 값, 형성 조건)"),
                ("batch", "같은 조건 기준 배치 score (배치 비교 탭에서 계산한 경우)"),
                ("measures", "조건별 측정 시트 요약 (in vitro 실험, 세포 전달 …: 반복의 중앙값)"),
                ("settings", "분석 설정값 (settings.json)"),
                ("figures", "현재 그림 전부 (PNG, 200 dpi)"))


class ExportDialog(QtWidgets.QDialog):
    """파일 › 모든 결과 통합 저장: options first, then the folder."""

    def __init__(self, parent=None, checked=None):
        super().__init__(parent)
        self.setWindowTitle("모든 결과 통합 저장")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(info_label("고른 항목을 새 폴더 하나에 CSV와 PNG로 저장하고, 프로젝트 파일(.dmap)도 함께 넣습니다. "
                                 "CSV는 엑셀·Prism에서 바로 열립니다 (UTF-8 BOM)."))
        self.checks = {}
        for key, label in EXPORT_ITEMS:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True if checked is None else checked.get(key, True))
            self.checks[key] = cb
            lay.addWidget(cb)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("폴더 고르기…")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def selection(self):
        return {k: c.isChecked() for k, c in self.checks.items()}


def export_folder_name(project) -> str:
    return (project.path.stem if project.path else "droplet_map") + "_" + datetime.now().strftime("%Y%m%d-%H%M")


def write_exports(out: Path, project, result, which: dict, figures: dict, particles=None, batch_score=None,
                  batch_map=None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    files = []

    def csv(df, fname, index=True):
        p = out / fname
        df.to_csv(p, index=index, encoding="utf-8-sig")
        files.append(p)

    if which.get("batch_map") and batch_map is not None and batch_map.desc is not None:
        d = batch_map.desc
        csv(pd.concat([d.meta.T.reindex(columns=d.values.columns), d.values.astype(object)]).T, "batch_items.csv")
        bm = batch_map.bm
        if bm is not None:
            csv(bm.distance, "batch_distance.csv")
            if bm.coords is not None:
                csv(bm.coords, "batch_map_coords.csv")
        sc = batch_map.score_frame()
        if sc is not None:
            csv(sc, "batch_map_score.csv")
    if which.get("conditions"):
        p = out / "conditions.csv"; p.write_text(_conditions_csv(project), encoding="utf-8-sig"); files.append(p)
    if result is not None:
        prep = result.prepared
        if which.get("particles"):
            d = particles
            if d is None:
                d = prep.data.copy()
                d["shape_weight_w"] = prep.w
                pm = result.pmap
                if pm is not None:
                    rows = d.index if pm.idx is None else d.index[pm.idx]
                    d.loc[rows, "umap_1"], d.loc[rows, "umap_2"] = pm.umap[:, 0], pm.umap[:, 1]
                    d.loc[rows, "cluster"] = pm.clusters
            csv(d, "particles_with_clusters.csv", index=False)
        if which.get("groups"):
            prof = result.profiles
            g = pd.DataFrame({"images": prof.n_images, "particles": prof.n_particles,
                              "n_eff_shape": prof.n_eff_shape, "mean_w": prof.mean_weight})
            g = g.join(prof.median_raw.add_prefix("median_")).join(prof.image_stats)
            if result.score is not None:
                g["reference_score"] = result.score["score"].reindex(g.index)
            t = project.view
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
        if which.get("composition"):
            if particles is not None and "cluster" in particles.columns:
                csv(pd.crosstab(particles["group"], particles["cluster"], normalize="index"), "cluster_composition.csv")
            elif result.profiles.composition is not None:
                csv(result.profiles.composition, "cluster_composition.csv")
    if which.get("batch") and batch_score is not None:
        csv(batch_score, "batch_reference_score.csv")
    if which.get("measures"):
        from ..core.batchfile import safe_file_name
        for m in project.measures:
            d = m.measures.copy()
            d.insert(0, "배치", [m.batch_of[g] for g in d.index])
            d.insert(1, "형성 그룹", [m.formation_of[g] for g in d.index])
            for s in m.subs:
                d.insert(2, s, [m.cond.value(g, s) for g in d.index])
            csv(d, f"sheet_{safe_file_name(m.name)}_summary.csv")
    if which.get("settings"):
        import json
        p = out / "settings.json"
        p.write_text(json.dumps(project.settings.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"); files.append(p)
    for fname, fig in figures.items():
        if callable(fig):            # e.g. the pyqtgraph map: a function that writes the file
            p = out / f"{fname}.png"
            try:
                fig(str(p)); files.append(p)
            except Exception:
                pass
            continue
        p = out / f"{fname}.png"
        try:
            fig.savefig(p, dpi=200)
            files.append(p)
        except Exception:
            pass
    return files


def _conditions_csv(project) -> str:
    """Formation groups with their batch and the batch's values."""
    v = project.view
    out = v.values.copy()
    out.insert(0, "기준 그룹", ["★" if g in v.references else "" for g in out.index])
    return out.to_csv()
