# Singlepass Studio — interface wireframe

Design study for the operator-facing side of the single-pass reconstruction system. Static HTML/CSS/JS, no build step, no dependencies — open `index.html` or serve the folder.

```bash
cd web
python -m http.server 8090
# http://localhost:8090
```

## Screens

| File | Purpose |
|---|---|
| `index.html` | Intake — footage, telemetry, capture geometry, compute budget, processing chain |
| `viewer.html` | The 3D stage — live point cloud, observability shading, measurement panel |
| `report.html` | Accuracy report — per-structure validation, method notes, declaration of limits |

## Design language

Precision survey instrument, not consumer SaaS. Brass on graphite.

| Token | Value | Role |
|---|---|---|
| `--bg` | `#0A0A0B` | warm near-black, never blue-black |
| `--surface` | `#121214` | panels |
| `--line` | `#232327` | borders, barely present |
| `--text` | `#EDEBE7` | warm white, not `#FFF` |
| `--accent` | `#C9A227` | brass — the only accent, used sparingly |

Confidence scale is deliberately muted — `#6FA98A` sage, `#C9A227` brass, `#C77D4A` terracotta, `#B4544A` clay. Safety information should read as instrument calibration, not as a warning banner.

**Type:** Instrument Serif for display, Inter for body, IBM Plex Mono for every number. All numerals tabular so figures align down a column.

**Space:** 32–56 px panel padding, 72–96 px section rhythm. Emptiness is load-bearing — it is what separates an instrument from a dashboard.

## Numbers on screen are real

Every figure comes from `run_demo.py` in this repo, measured against synthetic ground truth:

- **5.7 %** mean relative height error with IMU attitude fusion, against **41.2 %** position-only
- **0.48 %** scale residual · **2.94 m** absolute track RMSE (GNSS-bound, no GCPs)
- **26.6 %** of scene unobserved in a single pass
- **0.164 m** depth σ at stride 20, against **3.350 m** at stride 1
- **3 355 443** primitive ceiling on a 6 GB RTX 4050 before the driver spills to host RAM

Placeholder copy would have been easier. Real measurements let the interface argue for the engineering.

## `scene.js`

Dependency-free point-cloud renderer on canvas 2D — projection, depth sort, confidence shading, dashed flight path with camera stations. Drag to orbit, scroll to zoom, toggle observability against texture. Roughly mirrors the geometry in `probe/scene.py` so the wireframe shows the same scene the harness measures.
