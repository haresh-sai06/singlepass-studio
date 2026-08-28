"""
Stage 5 — cleanup, observability grading, export.

Triangulated clouds from real footage are noisy: bad matches produce points at
implausible depths, and low-parallax pairs produce points with huge uncertainty.
Filter, then grade what survives by how well it was actually resolved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def statistical_outlier_filter(pts: np.ndarray, k: int = 12, std_ratio: float = 2.0,
                               max_points: int = 60_000):
    """Drop points whose mean distance to k nearest neighbours is an outlier.

    Uses a random subsample for the distance model when the cloud is large, so
    this stays O(n * sample) instead of O(n^2).
    """
    n = len(pts)
    if n < 50:
        return np.ones(n, dtype=bool)

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=min(3000, n), replace=False)
    sample = pts[sample_idx]

    # mean distance from every point to its k nearest sampled neighbours
    means = np.empty(n, dtype=np.float32)
    chunk = 4000
    for s in range(0, n, chunk):
        blk = pts[s:s + chunk]
        d = np.linalg.norm(blk[:, None, :] - sample[None, :, :], axis=2)
        d.sort(axis=1)
        means[s:s + chunk] = d[:, 1:k + 1].mean(axis=1)

    mu, sd = float(means.mean()), float(means.std())
    keep = means < (mu + std_ratio * sd)
    return keep


def grade_observability(diagnostics: list[dict], counts: np.ndarray | None = None):
    """Map per-pair parallax onto a confidence grade for the points it produced.

    Real ray-casting observability needs a full multi-view reconstruction. In a
    pairwise setup the honest proxy is triangulation geometry: a point born from
    a 1-degree intersection is a guess, one born from 15 degrees is a measurement.

    Grades:  3 = well conditioned (>=10 deg)
             2 = usable           (5-10 deg)
             1 = weak             (2-5 deg)
             0 = unreliable       (<2 deg)  -> flagged, not deleted
    """
    grades = []
    for d in diagnostics:
        if "parallax_deg" not in d or not d.get("triangulated"):
            continue
        a = d["parallax_deg"]
        g = 3 if a >= 10 else 2 if a >= 5 else 1 if a >= 2 else 0
        grades.append((g, d["triangulated"]))
    return grades


def build_grade_array(diagnostics: list[dict]) -> np.ndarray:
    """Per-point grade array, in the same order run_sfm concatenated the cloud."""
    out = []
    for d in diagnostics:
        if "parallax_deg" not in d or not d.get("triangulated"):
            continue
        a = d["parallax_deg"]
        g = 3 if a >= 10 else 2 if a >= 5 else 1 if a >= 2 else 0
        out.append(np.full(d["triangulated"], g, dtype=np.int8))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int8)


def cloud_stats(pts: np.ndarray, grades: np.ndarray) -> dict:
    if len(pts) == 0:
        return {"points": 0}
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    dist = {int(g): int((grades == g).sum()) for g in range(4)} if len(grades) else {}
    total = max(len(grades), 1)
    return {
        "points": int(len(pts)),
        "extent": [round(float(a), 3) for a in (hi - lo)],
        "centroid": [round(float(a), 3) for a in pts.mean(axis=0)],
        "grade_counts": dist,
        "grade_pct": {k: round(v / total * 100, 1) for k, v in dist.items()},
        "well_conditioned_pct": round(
            (dist.get(3, 0) + dist.get(2, 0)) / total * 100, 1),
    }


def export_ply(path: str | Path, pts: np.ndarray, cols: np.ndarray) -> str:
    """Binary-free ASCII PLY — opens in MeshLab, CloudCompare, Blender."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(cols * 255, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, rgb):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")
    return str(path)


def export_viewer_json(path: str | Path, pts: np.ndarray, cols: np.ndarray,
                       grades: np.ndarray, max_points: int = 150_000,
                       cameras: list | None = None) -> str:
    """Cloud for the WebGL viewer, normalised into a comfortable viewing box.

    The camera trajectory is normalised with the SAME transform, so the recovered
    flight path sits correctly relative to the scene.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(pts)
    idx = (np.random.default_rng(0).choice(n, size=max_points, replace=False)
           if n > max_points else np.arange(n))

    p = pts[idx].astype(np.float32)
    centre = np.zeros(3, dtype=np.float32)
    scale = 1.0
    if len(p):
        centre = np.median(p, axis=0).astype(np.float32)
        p = p - centre
        scale = float(np.percentile(np.linalg.norm(p, axis=1), 92)) or 1.0
        p = p / scale * 40.0

    cam_out: list[float] = []
    if cameras:
        c = (np.asarray(cameras, dtype=np.float32) - centre) / scale * 40.0
        cam_out = [round(float(v), 3) for v in c.ravel()]

    data = {
        "count": int(len(p)),
        "positions": [round(float(v), 3) for v in p.ravel()],
        "colours": [round(float(v), 4) for v in cols[idx].ravel()] if len(cols) else [],
        "grades": [int(g) for g in (grades[idx] if len(grades) else [])],
        "cameras": cam_out,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)
