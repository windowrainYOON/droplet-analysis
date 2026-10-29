# Droplet Map usage (0.4)

From 0.4 the app centres on the **batch map**: each purification batch is one point, placed by its distance to the other batches over everything that describes it. The per-particle cluster map is the level below (particle level).

## Layout and flow
- **Left – batches › groups**: checked batches appear on the batch map. Expand a batch to see its particle-formation and cell-delivery groups; checked formation groups appear on the particle map. Right-click to set references (★).
- **Center – main flow**: Batch map · Batch table · Batch comparison, plus the input tabs Group conditions · Data.
- **Particle level** (sub-features): Particle map · Condition comparison · Reference score · Cell delivery. Closed at first; open them from the `입자 수준` menu, or double-click a batch on the batch map to open that batch's particle map.
- **Right – settings** of the current tab, plus measurement weights (particle level).
- `File › 가상 예제 프로젝트 열기 (배치 5개)` opens a synthetic five-batch project to try the batch level (made-up data).

## What describes a batch (Batch table)
Items are grouped in **categories**: 정제 조건 (purification), in vitro 관측, 세포 전달 by default; add more in the Batch table panel.
- **Typed per batch**: one value per batch (purification date, protein concentration, column, zeta potential, antibody delivery efficiency …). Dates count as days.
- **Computed from experiments** (particles → in vitro, delivery → cell delivery):
  - **Trend (경향)**: within one batch, change of a measurement per unit of a condition (e.g. circularity per pH unit, z/unit), from a linear fit over that batch's groups with the other varying conditions in the fit. Comparable even when batches ran different conditions.
  - **Matched (같은 조건)**: the value at a condition set run by two or more batches.
  - **Batch median (대표값)**: median over the batch's groups; off by default for particles because it mixes in which conditions were run.
- Particle values are group medians of robust z over all particles (shape values size-weighted) and particles per image. Blanks are "not recorded", never zero.

## Batch map
- **Distance**: only items both batches have. Numbers: difference relative to the range over batches (z items: 2 z = full difference; other numbers: differences below half the typical size count less); categories: 0 same / 1 different. Averaged within parts (typed, trend, matched, median), parts within a category, categories with the **category weights**, so a category with many items does not swamp one with few.
- **Map**: PCoA of the distance. Each axis shows its share of variation and the items that change along it. Colour points by any item (e.g. position from observations, colour from purification).
- Category weights (0 = off), parts and single items apply at once; after data or condition changes press 'Recompute batch map' (⌘B).
- Tabs: per-category maps, distance table (n = shared items), batch dendrogram, reference-batch score (100 × (1 − distance to the reference)).
- Two batches: distance only; three or more: the map.

## Batch comparison
- **Item differences**: B − A on every item, relative to the range over batches, largest first.
- Particle level (after the particle analysis): groups paired by identical formation conditions — batch distance, per-measurement difference, matched reference-batch score.

## Import
`File › Import (auto-detect)` (⌘O) or drag and drop: particle CSVs, condition CSVs (first column = group or batch), cell-delivery summaries, condition templates, projects. Importing CSVs asks for the batch. Older projects open as before.

## Particle level
Particle map (dot unit, colouring, cohesion, hierarchical sub-maps, per-batch maps, map inputs), condition comparison (effects within batches), reference score, cell delivery — as in 0.3. `입자 수준 › 입자 분석 실행` (⌘R).

## Saving
Per-feature saves in each menu; `Batch table › save` writes typed + computed items. `File › Save all results` (⇧⌘E) writes batch-map tables (batch_items, batch_distance, batch_map_coords, batch_map_score), particle-level results, figures and the project. `File › Save project` (⌘S) keeps data, conditions, batch values and categories, batch-map settings, weights, open tabs and the work log.
