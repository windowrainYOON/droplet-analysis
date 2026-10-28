# Droplet Map usage (0.3)

## Layout
- **Left – batches › groups**: particle-formation and cell-delivery groups sit under their purification batch; ticking a batch ticks its groups. Checked formation groups are shown on the particle map. Right-click to set or clear reference groups (★). "조건으로 거르기" checks only the groups with a chosen condition value.
- **Center – results**: Data · Conditions · Particle map · Condition comparison · Batch comparison · Reference score · Cell delivery tabs. Close a tab with ×, reopen it from its menu (`… 탭 열기`).
- **Right – settings**: settings of the current center tab, plus the always-present measurement-weight tab.

## Batches and groups
- A **batch** is one protein purification. Batch conditions (date, concentration …) go in the batch table at the top of the Conditions tab and are inherited by every group of that batch (grey [배치] columns).
- **Particle-formation groups** carry formation conditions and particle CSVs; **cell-delivery groups** carry delivery conditions, intensity summaries and the formation group whose particles they used.
- Importing a CSV asks for its batch (blank = unassigned, fix later). Opening a 0.2 project moves its '정제 배치' condition into batches.

## Import
`File › Import (auto-detect)` (⌘O) or drag and drop. Particle CSVs, condition CSVs, cell-delivery summaries (group, region, parameter, mean …), condition templates (JSON) and projects (.dmap) are recognised by content.

## Particle map
- **Dot =** particle / image / group / condition value. Aggregated dots sit at the median map position of their particles; size shows the particle count.
- **Colour**: group, cluster, shape reliability, any condition, any measurement.
- **Cohesion**: how well the map separates the colour categories, 0 (mixed) to 1 (fully separated).
- **Hierarchical maps**: select clusters → "선택 군집 → 하위 지도", or a region → "선택 영역 → 하위 지도". Sub-maps re-standardise within the subset. Click the path at the top to go back.

- **Batches**: the full map holds every batch (colour by '조건: 배치'); '이 배치만으로 지도' recomputes the map from one batch; the '배치별 지도' tab splits the current map per batch (same coordinates) or computes one map per batch.

## Condition and batch comparison
- Formation-condition effects are computed within batches (centred ranks), so a batch difference does not look like a pH effect. Batch-level conditions are compared on batch medians and hidden below 3 batches.
- **Batch comparison** pairs groups with identical formation conditions across batches: batch distance, per-measurement difference (B − A) and a **reference-batch score** (each group against the reference batch's group with the same conditions).

## Cell delivery
Per group and channel: mean intensity inside particles and in the cytoplasm (particles excluded), and the cytoplasm/particle ratio. Linking a delivery group to the formation group whose particles it used shows that group's particle medians and reference score alongside.

## What goes into the map
In the particle-map settings, '지도에 넣을 값' lets you untick measurements (left out of the map coordinates and clusters) and condition values (their groups are left out of the map, e.g. drop pH 8). '왼쪽에서 체크한 그룹만으로 지도 계산' computes the map from the groups ticked in the left list. The reference score and comparisons still use everything. Press '지도 다시 계산' to apply; the choices are saved in the project.

## Measurement weights
0 = drop, 1 = default, 2 = double. The reference score updates immediately; the map updates with "지도 다시 계산" (or the auto re-run option).

## Saving
Per-feature saves live in each feature menu. `File › Save all results` (⇧⌘E) writes a folder with CSVs, PNGs, settings and a copy of the project. `File › Save project` (⌘S) keeps data, conditions, settings, weights, open tabs, view choices and the work log.
