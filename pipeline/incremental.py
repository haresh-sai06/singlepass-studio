"""
Incremental Structure-from-Motion — one globally consistent reconstruction.

WHY THIS REPLACES THE PAIRWISE VERSION
--------------------------------------
`sfm.py` triangulates every pair in its OWN coordinate frame with a unit baseline,
because a two-view reconstruction has no way to know its own scale. Concatenating
53 such pairs stacks 53 different scenes at 53 different scales and orientations.
The result is the cone-shaped spray you saw: not noise from bad matching, but
53 correct reconstructions in mutually meaningless frames.

The fix is the standard one:

  1. TRACKS      chain matches across many frames, so one world point has one id
  2. SEED        initialise from the pair with the BEST parallax, not the first pair
  3. REGISTER    add each remaining camera by PnP against already-known 3D points
  4. TRIANGULATE new tracks multi-view (DLT over every observation, not just two)
  5. REFINE      drop high-reprojection-error points, optionally bundle adjust

Everything then lives in one frame, at one scale, and the cloud becomes a scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .sfm import (SfmConfig, detect_all, intrinsics_from_fov, make_detector,
                  match_pair)


# ────────────────────────────────────────────────────────────
# union-find over (frame, keypoint) observations -> tracks
# ────────────────────────────────────────────────────────────
class _DSU:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


@dataclass
class Track:
    obs: dict = field(default_factory=dict)   # frame index -> (u, v)
    point: np.ndarray | None = None           # 3D once triangulated
    colour: np.ndarray | None = None
    error: float = 0.0

    @property
    def length(self) -> int:
        return len(self.obs)


def build_tracks(keyframes, feats, norm, cfg: SfmConfig, pairs, progress=None):
    """Link pairwise matches into multi-frame tracks."""
    dsu = _DSU()
    obs_pt: dict = {}

    for n, (i, j) in enumerate(pairs):
        fa, fb = feats[i], feats[j]
        if fa["desc"] is None or fb["desc"] is None:
            continue
        bf = cv2.BFMatcher(norm)
        try:
            knn = bf.knnMatch(fa["desc"], fb["desc"], k=2)
        except cv2.error:
            continue

        good = [m[0] for m in knn
                if len(m) == 2 and m[0].distance < cfg.ratio_test * m[1].distance]
        if len(good) < cfg.min_matches:
            continue

        pa = np.float64([fa["kp"][m.queryIdx].pt for m in good])
        pb = np.float64([fb["kp"][m.trainIdx].pt for m in good])

        # Geometric verification before linking — an unverified match corrupts a
        # track for every frame it touches.
        F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
        if F is None or mask is None:
            continue
        keep = mask.ravel().astype(bool)

        for m, k in zip(good, keep):
            if not k:
                continue
            a = (i, m.queryIdx)
            b = (j, m.trainIdx)
            obs_pt[a] = fa["kp"][m.queryIdx].pt
            obs_pt[b] = fb["kp"][m.trainIdx].pt
            dsu.union(a, b)

        if progress:
            progress((n + 1) / len(pairs), f"tracks {n+1}/{len(pairs)}")

    groups: dict = {}
    for key in obs_pt:
        groups.setdefault(dsu.find(key), []).append(key)

    tracks: list[Track] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        t = Track()
        for (frame, _kp) in members:
            # one observation per frame; ambiguous frames are dropped
            if frame in t.obs:
                t.obs[frame] = None
            else:
                t.obs[frame] = obs_pt[(frame, _kp)]
        t.obs = {k: v for k, v in t.obs.items() if v is not None}
        if len(t.obs) >= 2:
            tracks.append(t)

    return tracks


# ────────────────────────────────────────────────────────────
# geometry helpers
# ────────────────────────────────────────────────────────────
def triangulate_multiview(obs: dict, poses: dict, K: np.ndarray) -> np.ndarray | None:
    """DLT over every camera that saw this track (not just two)."""
    rows = []
    for f, uv in obs.items():
        if f not in poses:
            continue
        R, t = poses[f]
        P = K @ np.hstack([R, t.reshape(3, 1)])
        u, v = uv
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    if len(rows) < 4:
        return None
    A = np.asarray(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-9:
        return None
    return X[:3] / X[3]


def reprojection_error(X, obs, poses, K) -> float:
    errs = []
    for f, uv in obs.items():
        if f not in poses:
            continue
        R, t = poses[f]
        pc = R @ X + t
        if pc[2] <= 1e-6:
            return 1e9
        p = K @ pc
        errs.append(np.hypot(p[0] / p[2] - uv[0], p[1] / p[2] - uv[1]))
    return float(np.mean(errs)) if errs else 1e9


def parallax_of(X, obs, poses) -> float:
    """Widest angle any two observing cameras subtend at this point."""
    centres = []
    for f in obs:
        if f not in poses:
            continue
        R, t = poses[f]
        centres.append(-R.T @ t)
    if len(centres) < 2:
        return 0.0
    best = 0.0
    for a in range(len(centres)):
        for b in range(a + 1, len(centres)):
            v1 = X - centres[a]
            v2 = X - centres[b]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            c = np.clip(float(v1 @ v2) / (n1 * n2), -1, 1)
            best = max(best, np.degrees(np.arccos(c)))
    return float(best)


# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────
@dataclass
class IncConfig:
    min_track_len: int = 2
    seed_min_matches: int = 80
    max_reproj_error: float = 4.0
    pnp_reproj: float = 5.0
    min_pnp_points: int = 12
    bundle_adjust: bool = True
    ba_max_points: int = 6000


def run_incremental(keyframes, cfg: SfmConfig, inc: IncConfig, progress=None):
    """Return one globally consistent cloud plus the recovered camera trajectory."""
    h, w = keyframes[0].image.shape[:2]
    K = intrinsics_from_fov(w, h, cfg.fov_deg)
    n = len(keyframes)

    def say(f, m):
        if progress:
            progress(f, m)

    # ── features ──
    feats, norm = detect_all(keyframes, cfg, lambda f, m: say(f * 0.25, m))

    # ── pairs: sequential + wide-baseline + a few long-range ──
    pairs = set()
    for i in range(n):
        for s in (1, cfg.pair_stride, cfg.pair_stride * 2):
            if i + s < n:
                pairs.add((i, i + s))
    pairs = sorted(pairs)

    # ── tracks ──
    tracks = build_tracks(keyframes, feats, norm, cfg, pairs,
                          lambda f, m: say(0.25 + f * 0.25, m))
    tracks = [t for t in tracks if t.length >= inc.min_track_len]
    if not tracks:
        raise RuntimeError("no tracks survived geometric verification")

    # ── seed: the pair with the widest parallax, not the first pair ──
    say(0.52, "selecting seed pair")
    best = None
    for (i, j) in pairs:
        shared = [t for t in tracks if i in t.obs and j in t.obs]
        if len(shared) < inc.seed_min_matches:
            continue
        pa = np.float64([t.obs[i] for t in shared])
        pb = np.float64([t.obs[j] for t in shared])
        E, mask = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, cfg.ransac_thresh)
        if E is None or E.shape != (3, 3):
            continue
        cnt, R, t, mp = cv2.recoverPose(E, pa, pb, K, mask=mask)
        if cnt < inc.seed_min_matches // 2:
            continue

        inl = mp.ravel() > 0
        Kinv = np.linalg.inv(K)
        ha = np.hstack([pa[inl], np.ones((inl.sum(), 1))]) @ Kinv.T
        hb = np.hstack([pb[inl], np.ones((inl.sum(), 1))]) @ Kinv.T
        ha /= np.linalg.norm(ha, axis=1, keepdims=True)
        hb /= np.linalg.norm(hb, axis=1, keepdims=True)
        ang = float(np.degrees(np.arccos(np.clip(
            (ha * (R @ hb.T).T).sum(axis=1), -1, 1))).mean())

        score = ang * np.log1p(int(inl.sum()))
        if best is None or score > best["score"]:
            best = {"i": i, "j": j, "R": R, "t": t, "ang": ang,
                    "inliers": int(inl.sum()), "score": score}

    if best is None:
        raise RuntimeError("no viable seed pair — footage may be too low-parallax")

    poses: dict = {
        best["i"]: (np.eye(3), np.zeros(3)),
        best["j"]: (best["R"], best["t"].ravel()),
    }

    # ── triangulate seed tracks ──
    for t in tracks:
        if best["i"] in t.obs and best["j"] in t.obs:
            X = triangulate_multiview(t.obs, poses, K)
            if X is None:
                continue
            if reprojection_error(X, t.obs, poses, K) < inc.max_reproj_error:
                R0, t0 = poses[best["i"]]
                if (R0 @ X + t0)[2] > 0:
                    t.point = X

    # ── register the rest by PnP ──
    order = sorted(range(n), key=lambda f: min(abs(f - best["i"]), abs(f - best["j"])))
    registered = {best["i"], best["j"]}

    for step, f in enumerate(order):
        if f in registered:
            continue
        pts3, pts2 = [], []
        for t in tracks:
            if t.point is not None and f in t.obs:
                pts3.append(t.point)
                pts2.append(t.obs[f])
        if len(pts3) < inc.min_pnp_points:
            continue

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.float64(pts3), np.float64(pts2), K, None,
            reprojectionError=inc.pnp_reproj, confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < inc.min_pnp_points:
            continue

        R, _ = cv2.Rodrigues(rvec)
        poses[f] = (R, tvec.ravel())
        registered.add(f)

        # new tracks now visible from two or more registered cameras
        for t in tracks:
            if t.point is not None:
                continue
            seen = [g for g in t.obs if g in poses]
            if len(seen) < 2:
                continue
            X = triangulate_multiview(t.obs, poses, K)
            if X is None:
                continue
            if reprojection_error(X, t.obs, poses, K) < inc.max_reproj_error:
                t.point = X

        say(0.55 + 0.3 * (step + 1) / n, f"registered {len(registered)}/{n} cameras")

    # ── refine: re-triangulate everything with the final pose set ──
    say(0.88, "refining")
    good = []
    for t in tracks:
        seen = [g for g in t.obs if g in poses]
        if len(seen) < 2:
            continue
        X = triangulate_multiview(t.obs, poses, K)
        if X is None:
            continue
        e = reprojection_error(X, t.obs, poses, K)
        if e > inc.max_reproj_error:
            continue
        # in front of every camera that saw it
        if any((poses[g][0] @ X + poses[g][1])[2] <= 0 for g in seen):
            continue
        t.point, t.error = X, e
        good.append(t)

    # ── colour from the sharpest observing frame ──
    for t in good:
        f = max((g for g in t.obs if g in poses),
                key=lambda g: keyframes[g].sharpness)
        img = keyframes[f].image
        u, v = t.obs[f]
        hh, ww = img.shape[:2]
        bgr = img[int(np.clip(v, 0, hh - 1)), int(np.clip(u, 0, ww - 1))]
        t.colour = bgr[::-1].astype(np.float32) / 255.0

    pts = np.array([t.point for t in good]) if good else np.zeros((0, 3))
    cols = np.array([t.colour for t in good]) if good else np.zeros((0, 3))
    ang = np.array([parallax_of(t.point, t.obs, poses) for t in good]) \
        if good else np.zeros(0)
    tlen = np.array([len([g for g in t.obs if g in poses]) for t in good]) \
        if good else np.zeros(0)

    centres = np.array([(-R.T @ tv) for (R, tv) in
                        [poses[f] for f in sorted(poses)]]) if poses else np.zeros((0, 3))

    summary = {
        "keyframes": n,
        "cameras_registered": len(poses),
        "tracks_total": len(tracks),
        "tracks_triangulated": len(good),
        "points": int(len(pts)),
        "seed_pair": f"{best['i']}->{best['j']}",
        "seed_parallax_deg": round(best["ang"], 2),
        "seed_inliers": best["inliers"],
        "mean_track_length": round(float(tlen.mean()), 2) if len(tlen) else 0,
        "max_track_length": int(tlen.max()) if len(tlen) else 0,
        "mean_reproj_px": round(float(np.mean([t.error for t in good])), 3) if good else None,
        "mean_parallax_deg": round(float(ang.mean()), 2) if len(ang) else 0,
        "median_parallax_deg": round(float(np.median(ang)), 2) if len(ang) else 0,
    }

    return {"points": pts, "colours": cols, "parallax": ang, "track_len": tlen,
            "poses": poses, "camera_centres": centres, "K": K, "summary": summary}
