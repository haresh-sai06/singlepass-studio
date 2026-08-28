"""
Stages 2-4 — features, pairing, pose, triangulation.

This is CLASSICAL structure-from-motion, deliberately. It is what runs today with
no model download, and running it on real single-pass footage is the honest way to
find out how hard the low-parallax problem actually bites. Whatever it struggles
with here is exactly what a feed-forward model (MASt3R/DUSt3R) is brought in to fix.

Key design choice: WIDE-BASELINE PAIRING. Matching frame i to i+1 gives an
intersection angle around 1-2 degrees and metres of depth uncertainty. Matching
i to i+stride opens the angle and collapses the error. See docs/FINDINGS.md #2.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SfmConfig:
    detector: str = "SIFT"          # SIFT | ORB | AKAZE
    max_features: int = 4000
    ratio_test: float = 0.75        # Lowe ratio
    pair_stride: int = 6            # wide-baseline: match i -> i+stride
    also_adjacent: bool = True      # keep i -> i+1 for tracking continuity
    ransac_thresh: float = 1.5      # px, for essential matrix
    min_matches: int = 40
    fov_deg: float = 78.0           # typical consumer drone horizontal FOV


def make_detector(cfg: SfmConfig):
    name = cfg.detector.upper()
    if name == "ORB":
        return cv2.ORB_create(nfeatures=cfg.max_features), cv2.NORM_HAMMING
    if name == "AKAZE":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    return cv2.SIFT_create(nfeatures=cfg.max_features), cv2.NORM_L2


def intrinsics_from_fov(w: int, h: int, fov_deg: float) -> np.ndarray:
    fx = (w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


def detect_all(keyframes, cfg: SfmConfig, progress=None):
    det, norm = make_detector(cfg)
    feats = []
    for i, kf in enumerate(keyframes):
        gray = cv2.cvtColor(kf.image, cv2.COLOR_BGR2GRAY)
        kp, desc = det.detectAndCompute(gray, None)
        feats.append({"kp": kp, "desc": desc, "n": 0 if desc is None else len(kp)})
        if progress:
            progress((i + 1) / len(keyframes), f"features {i+1}/{len(keyframes)}")
    return feats, norm


def build_pairs(n: int, cfg: SfmConfig) -> list[tuple[int, int]]:
    pairs = set()
    for i in range(n):
        if cfg.also_adjacent and i + 1 < n:
            pairs.add((i, i + 1))
        j = i + cfg.pair_stride
        if j < n:
            pairs.add((i, j))
    return sorted(pairs)


def match_pair(fa, fb, norm, cfg: SfmConfig):
    if fa["desc"] is None or fb["desc"] is None:
        return [], []
    bf = cv2.BFMatcher(norm)
    try:
        knn = bf.knnMatch(fa["desc"], fb["desc"], k=2)
    except cv2.error:
        return [], []
    good = []
    for m in knn:
        if len(m) == 2 and m[0].distance < cfg.ratio_test * m[1].distance:
            good.append(m[0])
    if len(good) < cfg.min_matches:
        return [], []
    pa = np.float64([fa["kp"][m.queryIdx].pt for m in good])
    pb = np.float64([fb["kp"][m.trainIdx].pt for m in good])
    return pa, pb


def relative_pose(pa, pb, K, cfg: SfmConfig):
    """Essential matrix -> (R, t, inlier mask, mean parallax angle in degrees)."""
    if len(pa) < cfg.min_matches:
        return None
    E, mask = cv2.findEssentialMat(pa, pb, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=cfg.ransac_thresh)
    if E is None or E.shape != (3, 3):
        return None
    n_in, R, t, mask_pose = cv2.recoverPose(E, pa, pb, K, mask=mask)
    if n_in < cfg.min_matches // 2:
        return None

    inl = (mask_pose.ravel() > 0)
    # Parallax proxy: angle between bearing vectors of the matched rays.
    Kinv = np.linalg.inv(K)
    ha = np.hstack([pa[inl], np.ones((inl.sum(), 1))]) @ Kinv.T
    hb = np.hstack([pb[inl], np.ones((inl.sum(), 1))]) @ Kinv.T
    ha /= np.linalg.norm(ha, axis=1, keepdims=True)
    hb /= np.linalg.norm(hb, axis=1, keepdims=True)
    hb_rot = (R @ hb.T).T
    cosang = np.clip((ha * hb_rot).sum(axis=1), -1, 1)
    parallax = float(np.rad2deg(np.arccos(cosang)).mean())

    return {"R": R, "t": t, "inliers": inl, "n_inliers": int(inl.sum()),
            "parallax_deg": parallax}


def triangulate_pair(pa, pb, K, R, t, colours=None, max_depth: float = 60.0):
    """Triangulate an inlier pair in the FIRST camera's frame (unit baseline)."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, pa.T, pb.T)
    X = (X[:3] / np.where(np.abs(X[3]) < 1e-9, 1e-9, X[3])).T

    depth1 = X[:, 2]
    Xc2 = (R @ X.T + t.reshape(3, 1)).T
    depth2 = Xc2[:, 2]
    ok = (depth1 > 0.05) & (depth2 > 0.05) & (depth1 < max_depth) & np.isfinite(X).all(axis=1)

    out = X[ok]
    cols = colours[ok] if colours is not None else None
    return out, cols, ok


def sample_colours(img, pts):
    h, w = img.shape[:2]
    u = np.clip(pts[:, 0].astype(int), 0, w - 1)
    v = np.clip(pts[:, 1].astype(int), 0, h - 1)
    bgr = img[v, u]
    return bgr[:, ::-1].astype(np.float32) / 255.0     # -> RGB 0..1


def run_sfm(keyframes, cfg: SfmConfig, progress=None):
    """Pairwise SfM. Returns a point cloud plus per-pair diagnostics.

    Deliberately pairwise rather than a full incremental bundle-adjusted
    reconstruction: on a single pass, the interesting question is how each
    pair behaves, and the per-pair parallax figures are the diagnostic.
    """
    if len(keyframes) < 2:
        raise RuntimeError("need at least 2 keyframes")

    h, w = keyframes[0].image.shape[:2]
    K = intrinsics_from_fov(w, h, cfg.fov_deg)

    feats, norm = detect_all(keyframes, cfg, progress)
    pairs = build_pairs(len(keyframes), cfg)

    cloud, colours, diagnostics = [], [], []
    for n, (i, j) in enumerate(pairs):
        pa, pb = match_pair(feats[i], feats[j], norm, cfg)
        rec = {"i": i, "j": j, "stride": j - i, "matches": len(pa)}

        if len(pa) >= cfg.min_matches:
            pose = relative_pose(np.asarray(pa), np.asarray(pb), K, cfg)
            if pose:
                rec.update(n_inliers=pose["n_inliers"],
                           parallax_deg=round(pose["parallax_deg"], 3))
                inl = pose["inliers"]
                cols = sample_colours(keyframes[i].image, np.asarray(pa)[inl])
                X, C, _ = triangulate_pair(np.asarray(pa)[inl], np.asarray(pb)[inl],
                                           K, pose["R"], pose["t"], cols)
                rec["triangulated"] = int(len(X))
                if len(X):
                    cloud.append(X)
                    colours.append(C)
            else:
                rec["failed"] = "pose"
        else:
            rec["failed"] = "matches"

        diagnostics.append(rec)
        if progress:
            progress((n + 1) / len(pairs), f"pair {n+1}/{len(pairs)}  ({i}->{j})")

    pts = np.concatenate(cloud, axis=0) if cloud else np.zeros((0, 3))
    cols = np.concatenate(colours, axis=0) if colours else np.zeros((0, 3))

    ok = [d for d in diagnostics if "parallax_deg" in d]
    adj = [d for d in ok if d["stride"] == 1]
    wide = [d for d in ok if d["stride"] > 1]

    summary = {
        "keyframes": len(keyframes),
        "pairs_attempted": len(pairs),
        "pairs_solved": len(ok),
        "pairs_failed": len(diagnostics) - len(ok),
        "points": int(len(pts)),
        "K_fx": round(float(K[0, 0]), 1),
        "mean_parallax_adjacent": round(float(np.mean([d["parallax_deg"] for d in adj])), 3) if adj else None,
        "mean_parallax_wide": round(float(np.mean([d["parallax_deg"] for d in wide])), 3) if wide else None,
        "mean_matches": round(float(np.mean([d["matches"] for d in diagnostics])), 1),
        "mean_inliers": round(float(np.mean([d.get("n_inliers", 0) for d in ok])), 1) if ok else 0,
    }
    return {"points": pts, "colours": cols, "diagnostics": diagnostics,
            "summary": summary, "K": K}
