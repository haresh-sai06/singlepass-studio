# Findings

Engineering decisions that follow from what this harness measured. Read this before writing pipeline code.

---

## 1 · Fuse IMU attitude. Position-only GPS alignment fails on a single pass.

**Measured:** 41.2% mean building-height error with position-only Sim(3) alignment, versus 5.7% once IMU attitude is fused. Same data, same scale accuracy.

**Why.** Structure-from-Motion output has arbitrary scale — the maths is identical for a real building and a dollhouse. GPS resolves it via Umeyama/Sim(3) alignment of the camera track to the GPS track.

But a straight single pass produces a **near-collinear** GPS track (measured collinearity 0.88, and it climbs toward 1.0 as flight jitter drops). Collinear points cannot constrain rotation *about* that line. Roll is free.

The trap is that **scale still looks excellent** — under 0.5% error in every method tested. A team validating only on scale will conclude the alignment works, then find every measurement wrong and not know why.

**Do this:**
- Take rotation from IMU/AHRS attitude, not from position fitting (`umeyama_with_attitude()`)
- Call `collinearity()` on the GPS track and warn above ~0.98
- Take scale from the ratio of spreads about the centroid — that part *is* rotation-invariant
- Keep RANSAC for GPS multipath outliers near tall structures

---

## 2 · Match wide frame gaps, never adjacent frames.

**Measured** at 90 m altitude, 640×480, fx ≈ 457 px:

| frame gap | baseline | intersection angle | depth σ |
|---:|---:|---:|---:|
| 1 | 2.65 m | 1.68° | **3.35 m** |
| 5 | 13.67 m | 8.69° | 0.65 m |
| 20 | 54.17 m | 33.50° | **0.16 m** |
| 40 | 108.62 m | 62.22° | 0.08 m |

`σ_Z ≈ Z²/(f·B) · σ_disparity`. Depth error grows with the **square** of distance and falls **linearly** with baseline.

Consecutive frames give a 1.7° intersection angle — numerically miserable. A 20-frame gap is 20× better.

**Do this:** build wide-baseline pairs (*i*, *i+20*) for triangulation, and keep short-baseline pairs only for tracking and correspondence. This is also why feed-forward models (MASt3R/DUSt3R) beat classical matching here — they tolerate wide baselines and low overlap that break SIFT-style pipelines.

---

## 3 · Fly oblique, not nadir.

Nadir + forward motion is **geometrically degenerate**: points near the image centre sit at the epipole and have near-zero parallax regardless of how far you fly. No software fixes that.

A 30–45° forward tilt makes motion largely perpendicular to the viewing direction, restoring parallax — and it exposes building façades, which the problem statement explicitly asks for (*"building facades and rooftops"*).

**Do this:** specify oblique capture in your recommended flight profile, and say why in the pitch. It shows you understand the geometry rather than just running a tool.

---

## 4 · Label unobserved geometry. Do not invent it.

**Measured:** 26.6% of the test scene was seen by **zero** cameras in a single pass.

The problem statement lists *"reconstruction of occluded surfaces"* as a challenge. For surfaces genuinely never observed, "reconstruction" means hallucination.

**Do this** — render an observability map with per-surface view counts:

| views | meaning | render |
|---|---|---|
| ≥3 | well triangulated | full colour |
| 2 | weak triangulation | amber |
| 1 | monocular guess | orange |
| 0 | never observed | red / hole |

For a reconnaissance or disaster analyst, a model that quietly fabricates a wall it never saw is **worse** than one with a visible hole. This also answers the hardest question a judge can ask: *"how do I know what to trust?"*

---

## 5 · Report relative accuracy separately from absolute.

They are different numbers with different limits:

- **Absolute** position is capped by GNSS — roughly 1–3 m with consumer receivers, 2–5 cm with RTK/PPK. You cannot beat your GPS.
- **Relative** measurement (building height, road width, distance between two points) is far better, because scale error is a *single global multiplier* rather than per-point noise.

"Measurement" in the problem statement is a relative task. Report both figures and explain the distinction — most teams will quote one number and not know which one they mean.

---

## 6 · GPU scoping on 6 GB

Standard 3DGS stores 59 floats per Gaussian (position 3, scale 3, quaternion 4, opacity 1, SH degree-3 colour 48) = 236 B at fp32. Adam adds gradients plus two moment buffers, so budget ≈ **944 B per Gaussian** before any rasteriser workspace.

Run `python run_demo.py` for your measured ceiling. On a 6 GB card, plan for:

- Downscale frames to **640–960 px** before reconstruction. Do not feed 4K.
- **Cap Gaussian count explicitly** — uncapped densification will OOM mid-run.
- Prefer **gsplat** over the reference 3DGS implementation; lower memory.
- Process the flight in **overlapping chunks of 20–30 frames**, then merge.
- Use **fp16/bf16 autocast** for MASt3R inference.
- Keep **Colab/Kaggle** (free T4, 16 GB) as the fallback for the final high-quality run.

---

## Pipeline order that follows from all of this

```
video
  └─ keyframe selection        blur rejection (Laplacian variance) + overlap targeting
      └─ dynamic object masks  YOLO/SAM — remove cars and people BEFORE reconstruction
          └─ pose + geometry   MASt3R / DUSt3R, wide-baseline pairs (i, i+20)
              └─ METRIC ALIGN  Sim(3) with IMU attitude + RANSAC on GPS   ← finding 1
                  └─ densify   3DGS (capped), or MASt3R pointmaps directly
                      └─ mesh  SuGaR / 2DGS
                          └─ OBSERVABILITY MAP                            ← finding 4
                              └─ viewer + measurement tools
```

The two steps in caps are where this problem is won. Everything else is integration of tools that already work.
