"""
Stage 6b — plane-sweep multi-view stereo.

WHY THIS EXISTS
---------------
Rectified SGBM (dense.py) fails on this footage, and measurably so: across 435
camera pairs the baseline-to-optical-axis ratio is 0.84-0.92, i.e. the drone
flies almost exactly along its own view direction. Rectification has to map the
epipole to infinity, and under forward motion the epipole sits INSIDE the frame,
so the rectified image collapses (roi = 0x0, output entirely black). An
empirical sweep confirmed it: only 2 of 74 attempted pairs produced any points.

Plane sweep never rectifies. Instead:

  for each candidate depth d:
      warp every neighbour view onto the reference view through the homography
      induced by a fronto-parallel plane at depth d
      measure photo-consistency between reference and warped neighbour
  per pixel, keep the depth whose warp agreed best

The warp is a plain homography, valid for ANY camera motion — forward, sideways,
rotating. That is the entire reason it works here where rectification cannot.

    H = K ( R - t nᵀ / d ) K⁻¹        n = (0,0,1) in the reference frame
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SweepConfig:
    ref_stride: int = 3          # use every Nth keyframe as a reference view
    neighbours: int = 4          # source views per reference
    neighbour_span: int = 6      # look this far either side for neighbours
    depth_planes: int = 96
    downscale: float = 0.55
    patch: int = 7               # NCC window (odd)
    min_views: int = 2
    cost_ratio: float = 0.86     # best cost must beat 2nd-best by this factor
    min_consistency: float = 0.30
    subsample: int = 2
    depth_pad: float = 1.6       # widen the sparse-derived depth range


def _depth_range(sparse_pts, R, t, pad: float):
    """Depth span of the sparse cloud as seen by this camera."""
    pc = (R @ sparse_pts.T).T + t
    z = pc[:, 2]
    z = z[(z > 1e-6) & np.isfinite(z)]
    if len(z) < 30:
        return None
    lo, hi = np.percentile(z, 4), np.percentile(z, 96)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * pad
    return max(mid - half, mid * 0.08), mid + half


def _box(img, k):
    return cv2.boxFilter(img, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)


def _ncc(ref, src, k):
    """Windowed normalised cross-correlation. Robust to exposure differences,
    which matters because drone auto-exposure drifts across a pass."""
    mu_r, mu_s = _box(ref, k), _box(src, k)
    rr = _box(ref * ref, k) - mu_r * mu_r
    ss = _box(src * src, k) - mu_s * mu_s
    rs = _box(ref * src, k) - mu_r * mu_s
    denom = np.sqrt(np.maximum(rr, 1e-6) * np.maximum(ss, 1e-6))
    return rs / denom


def sweep_reference(ref_idx, neigh_idx, keyframes, poses, K,
                    sparse_pts, cfg: SweepConfig):
    """Depth map for one reference view. Returns (points Nx3 world, colours)."""
    Rr, tr = poses[ref_idx]
    rng = _depth_range(sparse_pts, Rr, tr, cfg.depth_pad)
    if rng is None:
        return np.zeros((0, 3)), np.zeros((0, 3)), {}
    d_min, d_max = rng

    img_r = keyframes[ref_idx].image
    h, w = img_r.shape[:2]
    if cfg.downscale != 1.0:
        w2, h2 = int(w * cfg.downscale), int(h * cfg.downscale)
        img_r = cv2.resize(img_r, (w2, h2), interpolation=cv2.INTER_AREA)
        Ks = K.copy()
        Ks[:2] *= cfg.downscale
    else:
        w2, h2, Ks = w, h, K.copy()

    ref_g = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    Kinv = np.linalg.inv(Ks)
    n_vec = np.array([0.0, 0.0, 1.0])

    srcs = []
    for j in neigh_idx:
        Rs, ts = poses[j]
        img_s = keyframes[j].image
        if cfg.downscale != 1.0:
            img_s = cv2.resize(img_s, (w2, h2), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        R_rel = Rs @ Rr.T                 # reference -> source
        t_rel = ts - R_rel @ tr
        srcs.append((g, R_rel, t_rel))
    if len(srcs) < cfg.min_views:
        return np.zeros((0, 3)), np.zeros((0, 3)), {}

    # inverse-depth sampling: uniform in 1/d matches how disparity behaves
    inv = np.linspace(1.0 / d_max, 1.0 / d_min, cfg.depth_planes)
    depths = 1.0 / inv

    best = np.full((h2, w2), -2.0, np.float32)
    second = np.full((h2, w2), -2.0, np.float32)
    best_d = np.zeros((h2, w2), np.float32)

    for d in depths:
        acc = np.zeros((h2, w2), np.float32)
        cnt = np.zeros((h2, w2), np.float32)
        for g, R_rel, t_rel in srcs:
            H = Ks @ (R_rel - np.outer(t_rel, n_vec) / d) @ Kinv
            warped = cv2.warpPerspective(
                g, H, (w2, h2), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
            valid = warped >= 0
            if valid.sum() < 0.05 * w2 * h2:
                continue
            score = _ncc(ref_g, np.where(valid, warped, 0), cfg.patch)
            acc += np.where(valid, score, 0)
            cnt += valid.astype(np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt >= cfg.min_views, acc / np.maximum(cnt, 1), -2.0)

        improve = mean > best
        second = np.where(improve, best, np.maximum(second, mean))
        best_d = np.where(improve, d, best_d)
        best = np.where(improve, mean, best)

    # ── confidence gating: unique winner AND genuinely consistent ──
    ok = (best > cfg.min_consistency)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(best > 0, np.abs(second) / np.maximum(np.abs(best), 1e-6), 1.0)
    ok &= (ratio < cfg.cost_ratio) | (second < 0)
    ok &= (best_d > 0)

    ys, xs = np.nonzero(ok)
    if len(xs) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), {"kept": 0}
    if cfg.subsample > 1:
        ys, xs = ys[::cfg.subsample], xs[::cfg.subsample]

    z = best_d[ys, xs]
    rays = (Kinv @ np.stack([xs, ys, np.ones_like(xs)]).astype(np.float64)).T
    rays /= rays[:, 2:3]
    P_cam = rays * z[:, None]
    P_world = (Rr.T @ (P_cam - tr).T).T

    cols = img_r[ys, xs][:, ::-1].astype(np.float32) / 255.0
    stats = {"kept": int(len(xs)), "coverage_pct": round(ok.mean() * 100, 1),
             "d_min": round(float(d_min), 4), "d_max": round(float(d_max), 4),
             "mean_ncc": round(float(best[ok].mean()), 3)}
    return P_world, cols, stats


def run_planesweep(keyframes, poses: dict, K: np.ndarray, sparse_pts: np.ndarray,
                   cfg: SweepConfig, progress=None):
    frames = sorted(poses)
    refs = frames[::cfg.ref_stride]

    centre = np.median(sparse_pts, axis=0)
    scale = float(np.percentile(np.linalg.norm(sparse_pts - centre, axis=1), 90)) or 1.0

    clouds, cols_all, per_ref = [], [], []
    for n, rf in enumerate(refs):
        pool = [f for f in frames
                if f != rf and abs(f - rf) <= cfg.neighbour_span]
        pool.sort(key=lambda f: abs(f - rf))
        neigh = pool[:cfg.neighbours]
        if len(neigh) < cfg.min_views:
            continue

        P, C, st = sweep_reference(rf, neigh, keyframes, poses, K, sparse_pts, cfg)

        if len(P):
            d = np.linalg.norm(P - centre, axis=1)
            keep = d < scale * 3.5
            P, C = P[keep], C[keep]

        if len(P):
            clouds.append(P)
            cols_all.append(C)
        per_ref.append({"ref": rf, "neighbours": neigh, "points": int(len(P)), **st})

        if progress:
            progress((n + 1) / max(len(refs), 1),
                     f"sweep {n+1}/{len(refs)}  ref {rf}  {len(P)} pts")

    pts = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    cols = np.concatenate(cols_all) if cols_all else np.zeros((0, 3))
    worked = [r for r in per_ref if r["points"] > 0]

    return {
        "points": pts, "colours": cols, "refs": per_ref,
        "summary": {
            "references": len(refs),
            "references_used": len(worked),
            "depth_planes": cfg.depth_planes,
            "points": int(len(pts)),
            "densification_factor": round(len(pts) / max(len(sparse_pts), 1), 1),
            "mean_coverage_pct": round(
                float(np.mean([r.get("coverage_pct", 0) for r in worked])), 1) if worked else 0,
            "mean_ncc": round(
                float(np.mean([r.get("mean_ncc", 0) for r in worked])), 3) if worked else 0,
        },
    }
