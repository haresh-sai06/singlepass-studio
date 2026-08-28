"""
Umeyama / Sim(3) alignment.

This is THE core algorithm for turning a scale-ambiguous SfM reconstruction into a
metric, georeferenced model.

Structure-from-Motion gives you camera positions in an arbitrary frame with arbitrary
scale. GPS gives you where the drone actually was, in metres. Aligning the two
recovers the missing scale factor (and the rotation + translation into world frame).

Umeyama, S. (1991) "Least-squares estimation of transformation parameters between
two point patterns." IEEE PAMI 13(4).
"""
from __future__ import annotations

import numpy as np


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Find (s, R, t) minimising  || dst - (s * R @ src + t) ||^2.

    Args:
        src: (N, 3) source points  — e.g. SfM camera centres (arbitrary scale)
        dst: (N, 3) target points  — e.g. GPS positions in metres (ENU)
        with_scale: if False, forces s = 1 (rigid alignment only)

    Returns:
        s (float), R (3, 3), t (3,)
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"shape mismatch: {src.shape} vs {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("need at least 3 correspondences")

    n, dim = src.shape

    # 1. Centroids
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    # 2. Centre both point sets
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # 3. Variance of the source (needed for the scale term)
    var_src = (src_c ** 2).sum() / n

    # 4. Cross-covariance
    cov = (dst_c.T @ src_c) / n

    # 5. SVD
    U, D, Vt = np.linalg.svd(cov)

    # 6. Handle reflection: if det(U @ Vt) < 0 we'd get a mirror, not a rotation.
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1.0

    R = U @ S @ Vt

    # 7. Scale
    s = float(np.trace(np.diag(D) @ S) / var_src) if with_scale else 1.0

    # 8. Translation
    t = mu_dst - s * (R @ mu_src)

    return s, R, t


def apply_sim3(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply a Sim(3) transform to an (N, 3) point array."""
    return (s * (R @ np.asarray(points, dtype=np.float64).T)).T + t


def alignment_error(src: np.ndarray, dst: np.ndarray, s: float, R: np.ndarray, t: np.ndarray):
    """Per-point and RMSE residual after alignment, in the units of `dst` (metres)."""
    aligned = apply_sim3(src, s, R, t)
    per_point = np.linalg.norm(aligned - dst, axis=1)
    return per_point, float(np.sqrt((per_point ** 2).mean()))


def collinearity(points: np.ndarray) -> float:
    """How collinear is this point set?  1.0 = perfectly straight line, 0 = spread out.

    A straight single-pass flight produces a nearly collinear GPS track, and Umeyama
    on collinear points CANNOT determine rotation about that line. Scale and
    translation come out fine; roll is unconstrained and your model ends up tilted.
    Check this before trusting a position-only alignment.
    """
    p = np.asarray(points, dtype=np.float64)
    c = p - p.mean(axis=0)
    sv = np.linalg.svd(c, compute_uv=False)
    return float(sv[0] / max(sv.sum(), 1e-12))


def average_rotations(rots: np.ndarray) -> np.ndarray:
    """Chordal L2 mean of a stack of rotation matrices, projected back onto SO(3)."""
    M = np.asarray(rots, dtype=np.float64).sum(axis=0)
    U, _, Vt = np.linalg.svd(M)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    return U @ S @ Vt


def umeyama_with_attitude(
    src_centres: np.ndarray,
    dst_centres: np.ndarray,
    src_rots: np.ndarray,
    dst_rots: np.ndarray,
):
    """Sim(3) alignment that takes rotation from IMU attitude instead of from positions.

    This is the fix for the collinear-trajectory degeneracy. Positions along a straight
    line determine SCALE well (it is a ratio of distances) but leave rotation about the
    flight axis free. Camera attitude — which every drone IMU provides — pins it down.

    Args:
        src_centres: (N,3) reconstruction camera centres
        dst_centres: (N,3) GPS positions, metres
        src_rots:    (N,3,3) camera->recon-frame rotations (from SfM)
        dst_rots:    (N,3,3) camera->world rotations (from IMU/AHRS)

    Returns:
        s, R, t
    """
    src_centres = np.asarray(src_centres, dtype=np.float64)
    dst_centres = np.asarray(dst_centres, dtype=np.float64)

    # 1. Rotation from attitude: R_align @ R_src_i  ==  R_dst_i  for every i
    per_frame = np.einsum("nij,nkj->nik", np.asarray(dst_rots), np.asarray(src_rots))
    R = average_rotations(per_frame)

    # 2. Scale from the ratio of spreads about the centroid (rotation-invariant)
    src_c = src_centres - src_centres.mean(axis=0)
    dst_c = dst_centres - dst_centres.mean(axis=0)
    denom = np.sqrt((src_c ** 2).sum())
    s = float(np.sqrt((dst_c ** 2).sum()) / max(denom, 1e-12))

    # 3. Translation from centroids
    t = dst_centres.mean(axis=0) - s * (R @ src_centres.mean(axis=0))
    return s, R, t


def ransac_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    n_iter: int = 500,
    threshold: float = 3.0,
    rng: np.random.Generator | None = None,
):
    """Robust Umeyama. GPS has outliers (multipath near tall structures), and a single
    bad fix drags a least-squares fit badly. RANSAC ignores them.

    Args:
        threshold: inlier distance in metres.

    Returns:
        s, R, t, inlier_mask
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    rng = rng or np.random.default_rng(0)
    n = src.shape[0]

    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0

    for _ in range(n_iter):
        idx = rng.choice(n, size=min(4, n), replace=False)
        try:
            s, R, t = umeyama_alignment(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        resid = np.linalg.norm(apply_sim3(src, s, R, t) - dst, axis=1)
        inliers = resid < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_count < 3:
        # Degenerate — fall back to the plain fit over everything.
        s, R, t = umeyama_alignment(src, dst)
        return s, R, t, np.ones(n, dtype=bool)

    # Refit on the inlier set for a better estimate.
    s, R, t = umeyama_alignment(src[best_inliers], dst[best_inliers])
    return s, R, t, best_inliers
