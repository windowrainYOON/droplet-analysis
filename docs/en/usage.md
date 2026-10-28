# Droplet Map usage (0.2)

## Layout
- **Left – groups**: checked groups are shown on the particle map. Right-click to set or clear reference groups (★). "조건으로 거르기" checks only the groups with a chosen condition value.
- **Center – results**: Data · Conditions · Particle map · Condition comparison · Reference score tabs. Close a tab with ×, reopen it from its menu (`… 탭 열기`).
- **Right – settings**: settings of the current center tab, plus the always-present measurement-weight tab.

## Import
`File › Import (auto-detect)` (⌘O) or drag and drop. Particle CSVs, condition CSVs, condition templates (JSON) and projects (.dmap) are recognised by content.

## Particle map
- **Dot =** particle / image / group / condition value. Aggregated dots sit at the median map position of their particles; size shows the particle count.
- **Colour**: group, cluster, shape reliability, any condition, any measurement.
- **Cohesion**: how well the map separates the colour categories, 0 (mixed) to 1 (fully separated).
- **Hierarchical maps**: select clusters → "선택 군집 → 하위 지도", or a region → "선택 영역 → 하위 지도". Sub-maps re-standardise within the subset. Click the path at the top to go back.

## Measurement weights
0 = drop, 1 = default, 2 = double. The reference score updates immediately; the map updates with "지도 다시 계산" (or the auto re-run option).

## Saving
Per-feature saves live in each feature menu. `File › Save all results` (⇧⌘E) writes a folder with CSVs, PNGs, settings and a copy of the project. `File › Save project` (⌘S) keeps data, conditions, settings, weights, open tabs, view choices and the work log.
