# Droplet Map

A macOS (Apple Silicon) app for overviewing phase-separated protein droplets. It reads per-particle
CSVs from an external image analysis, lets you type experimental conditions per group, and shows:

- **입자 지도** — UMAP + Leiden map of particles; shape values are down-weighted for small particles
  (w(d) = 1 / (1 + (d50/d)^k), default d50 = 1.2 µm, k = 4).
- **조건 비교** — per-image / per-group trends against conditions, condition × parameter association
  (Spearman ρ or Kruskal ε², hidden when too few groups), group similarity.
- **기준 score** — 0–100 closeness of every group to one or more reference groups (the desired batch).
- **내보내기** — CSV tables and PNG figures.

Conditions are free-form: add any number of numeric or categorical conditions, leave cells blank
when an experiment did not record them (blanks are never filled with 0).

## Input

One row per particle. `group` (one set of conditions) and `image` columns, plus numeric measurements
such as `eq_diameter_um`, `circularity`, `solidity`, `aspect_ratio`, `nnd_um`, `CH1_partition`.
Column roles are guessed and can be changed on the 데이터 page.

## Run from source

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m droplet_map
pytest
```

## Build the app

```bash
scripts/build_mac.sh   # builds with Briefcase, installs to /Applications, copies the .dmg to ~/Desktop
```

The app is ad-hoc signed. Project files (`.dmap`) and experiment CSVs are kept out of git.
