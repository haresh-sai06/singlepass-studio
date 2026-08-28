<div align="center">

# singlepass-probe

### Feasibility harness for single-pass drone video → georeferenced 3D reconstruction

*Answers four questions before you commit two weeks of build time.*

![Status](https://img.shields.io/badge/status-probe-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Why this exists

Building a 3D reconstruction system from a single drone pass has four make-or-break unknowns. Guessing at them costs days. This measures them in about a minute.

| # | Question | What this proves |
|---|---|---|
| 1 | Can my GPU actually train Gaussian Splatting, and at what scene size? | Allocates a real 3DGS-shaped parameter set + Adam state and grows it until OOM |
| 2 | Does metric scaling from noisy GPS genuinely recover real-world units? | Simulates GNSS error, runs Sim(3) alignment, measures buildings against known truth |
| 3 | How bad is single-pass parallax, and does camera tilt fix it? | Computes intersection angles and depth uncertainty across frame gaps |
| 4 | How much of a scene does one pass leave unobserved? | Exact per-point visibility count with z-buffer occlusion |

Everything runs against a **procedurally generated scene with exact ground truth**, so every number is a real error measurement — not a vibe.

## Quick start

```bash
git clone <your-repo-url>
cd singlepass-probe
pip install -r requirements.txt
python run_demo.py
```

No GPU or PyTorch? The geometry half still runs:

```bash
python run_demo.py --skip-gpu
```

See nadir degeneracy for yourself:

```bash
python run_demo.py --tilt 0     # nadir  — degenerate for forward motion
python run_demo.py --tilt 40    # oblique — what you actually want
```

## What it found on a 6 GB laptop GPU

Two results worth knowing before you write any pipeline code.

### 1 · A straight flight breaks position-only alignment

Structure-from-Motion gives you a reconstruction with **arbitrary scale**. GPS is how you recover metres. The standard tool is Umeyama / Sim(3) alignment of the camera track to the GPS track.

**It quietly fails on a single pass.** A straight flight produces a nearly *collinear* GPS track, and collinear points cannot constrain rotation **about** that line. Scale comes out looking excellent. Your model comes out rolled.

```
method                      recovered scale  scale err %
---------------------------------------------------------
A  positions (lsq)                 0.013712        0.090
B  positions (RANSAC)              0.013635        0.473
C  + IMU attitude                  0.013634        0.481
ground truth                       0.013700

bldg      true m   B: pos-only    err %    C: +IMU    err %
------------------------------------------------------------
1          26.88         20.55     23.5      27.69      3.0
2          17.50         10.95     37.4      18.37      5.0
3          25.53         12.26     52.0      26.36      3.3
4          21.52         17.99     16.4      22.24      3.3
5           6.40         13.41    109.4       7.29     13.9
6          16.85         18.26      8.3      17.81      5.6

mean height error   positions-only  41.2 %     with IMU   5.7 %
```

Scale error looks fine in **all three** methods. Only attitude fusion produces correct *measurements* — a **7× improvement**. Fuse IMU attitude, don't align on positions alone.

### 2 · Adjacent frames are useless for triangulation

Depth uncertainty follows `σ_Z ≈ Z²/(f·B) · σ_disparity`. At 90 m altitude:

```
 frame gap   baseline m      B/Z   angle deg   depth sigma m
------------------------------------------------------------
         1         2.65    0.029        1.68           3.350
         2         5.57    0.062        3.54           1.591
         5        13.67    0.152        8.69           0.648
        10        27.34    0.304       17.27           0.324
        20        54.17    0.602       33.50           0.164
        40       108.62    1.207       62.22           0.082
```

Consecutive frames give a **1.7° intersection angle and 3.35 m of depth uncertainty**. Match frame *i* to *i+20*, not *i+1*. Wide baselines are not an optimisation — they are the difference between a model and noise.

### 3 · One pass cannot see a quarter of the scene

```
confidence                        points     share
--------------------------------------------------
3+ views  well triangulated       21,954     71.8%
2 views   weak                       243      0.8%
1 view    monocular guess            237      0.8%
0 views   NEVER OBSERVED            8,148     26.6%
```

You cannot reconstruct geometry you never observed. The honest engineering answer is to **label** unobserved surfaces, not invent them — a model that silently fabricates a wall is worse than one with a visible hole.

## What's in here

```
singlepass-probe/
├── run_demo.py              entry point — runs all four checks
├── probe/
│   ├── gpu_probe.py         VRAM detection + real 3DGS capacity benchmark
│   ├── scene.py             procedural terrain + buildings, exact ground truth
│   ├── flight.py            single-pass path, pinhole camera, tilt geometry
│   ├── render.py            point-splat renderer + z-buffer observability
│   ├── gps.py               GNSS noise model, scale-ambiguous SfM simulator
│   └── umeyama.py           Sim(3) alignment, RANSAC, attitude fusion
└── out/                     report.json + coverage.png
```

### Modules worth reading

**`probe/umeyama.py`** — the core algorithm. Closed-form Sim(3), a `collinearity()` check that warns before the degeneracy bites, and `umeyama_with_attitude()` which fixes it.

**`probe/gpu_probe.py`** — doesn't report generic TFLOPS. Allocates real 3DGS parameters (position, scale, quaternion, opacity, 48 SH coefficients = 59 floats/Gaussian, ×4 for Adam state) and grows until OOM, so you learn your actual scene-size ceiling.

**`probe/scene.py`** — this is where ground truth comes from. NTRO supplies footage only at the hackathon; without your own truth data you cannot quote an error figure, and *"our error is 2.3 cm"* beats *"it looks good"* every time.

## Sample output

```
  singlepass-probe   SIH26158 feasibility harness

  1 / 4   GPU CAPABILITY
  2 / 4   METRIC SCALING FROM NOISY GPS
  3 / 4   SINGLE-PASS PARALLAX
  4 / 4   OBSERVABILITY

  SUMMARY
  GPU        <tier> — <verdict>
  Metric     scale recovered to 0.48%, mean height error 5.7% (with IMU)
  Coverage   26.6% of scene never observed in one pass
```

Full results land in `out/report.json`; a visual comparison in `out/coverage.png`.

## Requirements

`numpy` and `matplotlib` for the geometry half. `torch` (CUDA build) for the GPU benchmark:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Context

Built as a pre-build feasibility check for **SIH26158** — *"Single-Pass Drone Video to Accurate 3D Model Generation System"* (NTRO, Smart India Hackathon 2026).

The problem statement asks for a **georeferenced, metrically accurate** 3D model from **one** drone pass, without ground control points — while handling motion blur, dynamic objects, GPS noise, and occlusion. This harness measures how hard each of those actually is before any pipeline gets written.

## Licence

MIT
