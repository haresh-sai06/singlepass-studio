"""
Stage 6 — dense multi-view stereo.

Sparse SfM only triangulates points it can MATCH: corners, textured patches,
distinctive blobs. Everything smooth in between is simply missing, which is why
a sparse cloud looks like scattered confetti rather than a surface.

Dense MVS computes a depth for EVERY pixel instead. The classical recipe:

  1. pick pairs whose baseline gives real parallax (a 1-degree pair is useless
     dense or sparse — see docs/FINDINGS.md #2)
  2. RECTIFY the pair so corresponding pixels share a scanline
  3. SGBM block-matching -> per-pixel disparity
  4. reproject disparity into 3D using the known relative pose
  5. fuse every pair's cloud back into the global frame

Step 1 is the one most implementations get wrong on single-pass footage: run
this over adjacent frames and you get a wall of noise, because the geometry
cannot support it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DenseConfig:
    min_parallax_deg: float = 1.2   # skip pairs that cannot support depth
    max_pairs: int = 16             # cost control
    downscale: float = 0.6          # SGBM is O(pixels * disparities)
    num_disparities: int = 128      # must be divisible by 16
    block_size: int = 5
    uniqueness: int = 8
    speckle_window: int = 120
    speckle_range: int = 2
    max_depth_ratio: float = 12.0   # reject points far beyond the scene scale
    subsample: int = 2              # keep every Nth valid pixel
    max_forward_ratio: float = 0.96  # permissive: the empirical cover test decides
    min_rectified_cover: float = 0.12  # reject rectifications that collapse


def _pair_parallax(Ci, Cj, scene_centre) -> float:
    v1, v2 = scene_centre - Ci, scene_centre - Cj
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1))))


def _forward_ratio(Ci, Cj, Ra) -> float:
    """How much of the baseline lies ALONG the optical axis.

    This is the single most important check for dense stereo on drone footage.
    Standard rectification maps the epipole to infinity; under pure forward
    motion the epipole sits inside the image, and the rectification collapses
    to nothing (roi = 0x0, output entirely black).

    1.0 = pure forward motion  (degenerate, unusable)
    0.0 = pure sideways motion (ideal stereo geometry)
    """
    base = Cj - Ci
    n = np.linalg.norm(base)
    if n < 1e-9:
        return 1.0
    optical_axis = Ra.T @ np.array([0.0, 0.0, 1.0])   # camera +Z in world
    return float(abs(base @ optical_axis) / n)


def select_dense_pairs(poses: dict, scene_centre: np.ndarray, cfg: DenseConfig):
    """Rank stereo pairs by how well they will RECTIFY, not just parallax.

    Parallax alone is a trap here: the widest-angle pairs on a forward pass are
    exactly the ones whose baseline is most parallel to the view direction, and
    those rectify to a black frame. Score has to penalise forward motion.
    """
    frames = sorted(poses)
    centres = {f: -poses[f][0].T @ poses[f][1] for f in frames}

    cands = []
    for a_i, a in enumerate(frames):
        for b in frames[a_i + 1:]:
            ang = _pair_parallax(centres[a], centres[b], scene_centre)
            if ang < cfg.min_parallax_deg or ang > 40.0:
                continue
            fwd = _forward_ratio(centres[a], centres[b], poses[a][0])
            if fwd > cfg.max_forward_ratio:
                continue                    # would rectify to nothing
            # want: decent parallax AND sideways baseline
            score = min(ang, 15.0) * (1.0 - fwd)
            cands.append((score, ang, fwd, a, b))

    cands.sort(reverse=True)

    chosen, used = [], {}
    for score, ang, fwd, a, b in cands:
        if used.get(a, 0) >= 3 or used.get(b, 0) >= 3:
            continue                       # spread coverage across the flight
        chosen.append({"a": a, "b": b, "parallax_deg": round(ang, 2),
                       "forward_ratio": round(fwd, 3)})
        used[a] = used.get(a, 0) + 1
        used[b] = used.get(b, 0) + 1
        if len(chosen) >= cfg.max_pairs:
            break
    return chosen


def densify_pair(img_a, img_b, Ra, ta, Rb, tb, K, cfg: DenseConfig):
    """Rectified SGBM on one pair. Returns (points Nx3 world, colours Nx3)."""
    h, w = img_a.shape[:2]
    if cfg.downscale != 1.0:
        w2, h2 = int(w * cfg.downscale), int(h * cfg.downscale)
        img_a = cv2.resize(img_a, (w2, h2), interpolation=cv2.INTER_AREA)
        img_b = cv2.resize(img_b, (w2, h2), interpolation=cv2.INTER_AREA)
        Ks = K.copy()
        Ks[:2] *= cfg.downscale
    else:
        w2, h2, Ks = w, h, K.copy()

    # relative pose  b <- a
    R_rel = Rb @ Ra.T
    t_rel = tb - R_rel @ ta

    baseline = float(np.linalg.norm(t_rel))
    if baseline < 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3))

    # alpha=1 keeps every source pixel. alpha=0 crops to the "valid" rectangle,
    # which on drone footage is frequently empty (roi = 0x0) and silently
    # produces a black frame.
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        Ks, None, Ks, None, (w2, h2), R_rel, t_rel.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)

    m1x, m1y = cv2.initUndistortRectifyMap(Ks, None, R1, P1, (w2, h2), cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(Ks, None, R2, P2, (w2, h2), cv2.CV_32FC1)
    ra = cv2.remap(img_a, m1x, m1y, cv2.INTER_LINEAR)
    rb = cv2.remap(img_b, m2x, m2y, cv2.INTER_LINEAR)

    ga = cv2.cvtColor(ra, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(rb, cv2.COLOR_BGR2GRAY)

    # Verify the rectification did not collapse. Cheaper to check than to debug
    # a wall of zeros downstream.
    cover = float(min((ga > 0).mean(), (gb > 0).mean()))
    if cover < cfg.min_rectified_cover:
        return np.zeros((0, 3)), np.zeros((0, 3))

    nd = int(np.ceil(cfg.num_disparities / 16.0)) * 16
    bs = cfg.block_size
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=bs,
        P1=8 * 3 * bs * bs, P2=32 * 3 * bs * bs,
        disp12MaxDiff=1, uniquenessRatio=cfg.uniqueness,
        speckleWindowSize=cfg.speckle_window, speckleRange=cfg.speckle_range,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    disp = sgbm.compute(ga, gb).astype(np.float32) / 16.0

    pts3 = cv2.reprojectImageTo3D(disp, Q)
    valid = (disp > 0.5) & np.isfinite(pts3).all(axis=2)

    # depth sanity — reject the far plane where disparity noise explodes
    z = pts3[:, :, 2]
    med = float(np.median(np.abs(z[valid]))) if valid.any() else 0.0
    if med > 0:
        valid &= (np.abs(z) < med * cfg.max_depth_ratio) & (np.abs(z) > med / cfg.max_depth_ratio)

    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if cfg.subsample > 1:
        sel = slice(None, None, cfg.subsample)
        ys, xs = ys[sel], xs[sel]

    P = pts3[ys, xs]                       # rectified-camera-A frame
    cols = ra[ys, xs][:, ::-1].astype(np.float32) / 255.0

    # rectified A -> camera A -> world
    P_cam = (R1.T @ P.T).T
    C_a = -Ra.T @ ta
    P_world = (Ra.T @ P_cam.T).T + C_a
    return P_world, cols


def run_dense(keyframes, poses: dict, K: np.ndarray, sparse_pts: np.ndarray,
              cfg: DenseConfig, progress=None):
    """Densify across the best-conditioned pairs and fuse into one cloud."""
    if len(sparse_pts) == 0:
        return {"points": np.zeros((0, 3)), "colours": np.zeros((0, 3)),
                "summary": {"pairs_used": 0, "points": 0}}

    centre = np.median(sparse_pts, axis=0)
    scale = float(np.percentile(np.linalg.norm(sparse_pts - centre, axis=1), 90)) or 1.0

    pairs = select_dense_pairs(poses, centre, cfg)
    clouds, cols_all, used = [], [], []

    for n, pr in enumerate(pairs):
        a, b = pr["a"], pr["b"]
        Ra, ta = poses[a]
        Rb, tb = poses[b]
        try:
            P, C = densify_pair(keyframes[a].image, keyframes[b].image,
                                Ra, ta, Rb, tb, K, cfg)
        except cv2.error:
            P, C = np.zeros((0, 3)), np.zeros((0, 3))

        if len(P):
            # clip to a generous box around the sparse reconstruction
            d = np.linalg.norm(P - centre, axis=1)
            keep = d < scale * 4.0
            P, C = P[keep], C[keep]

        if len(P):
            clouds.append(P)
            cols_all.append(C)
        used.append({**pr, "points": int(len(P))})

        if progress:
            progress((n + 1) / max(len(pairs), 1),
                     f"dense pair {n+1}/{len(pairs)}  ({a}->{b})  {len(P)} pts")

    pts = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    cols = np.concatenate(cols_all) if cols_all else np.zeros((0, 3))

    return {
        "points": pts, "colours": cols,
        "summary": {
            "pairs_available": len(pairs),
            "pairs_used": sum(1 for u in used if u["points"] > 0),
            "points": int(len(pts)),
            "mean_pair_parallax": round(
                float(np.mean([u["parallax_deg"] for u in used])), 2) if used else 0,
            "densification_factor": round(len(pts) / max(len(sparse_pts), 1), 1),
        },
        "pairs": used,
    }


def voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel: float):
    """Grid-average duplicate observations — MVS pairs overlap heavily."""
    if len(pts) == 0 or voxel <= 0:
        return pts, cols
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    out_p = np.zeros((len(idx), 3), np.float64)
    out_c = np.zeros((len(idx), 3), np.float64)
    cnt = np.zeros(len(idx), np.int64)
    np.add.at(out_p, inv, pts)
    np.add.at(out_c, inv, cols)
    np.add.at(cnt, inv, 1)
    cnt = np.maximum(cnt, 1)[:, None]
    return out_p / cnt, out_c / cnt
