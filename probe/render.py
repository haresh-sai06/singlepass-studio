"""
Point-splat renderer + z-buffer visibility.

Deliberately simple: project points, keep the nearest per pixel, splat a small disc.
That is enough to (a) produce plausible drone frames and (b) compute an exact
observability count — how many cameras genuinely saw each surface point.

That observability count is the honest answer to the problem statement's
"reconstruction of occluded surfaces": you cannot recover geometry you never
observed, so you label it instead of inventing it.
"""
from __future__ import annotations

import numpy as np

from .flight import Intrinsics, project


def render_frame(
    points_w: np.ndarray,
    colours: np.ndarray,
    centre: np.ndarray,
    R_c2w: np.ndarray,
    intr: Intrinsics,
    splat_radius: int = 1,
    bg: tuple[float, float, float] = (0.55, 0.68, 0.85),
):
    """Render one frame. Returns (rgb (H,W,3) float 0-1, depth (H,W) float, hit (H,W) bool)."""
    H, W = intr.height, intr.width
    uv, depth, vis = project(points_w, centre, R_c2w, intr)

    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = np.array(bg, np.float32)
    zbuf = np.full((H, W), np.inf, np.float32)
    hit = np.zeros((H, W), bool)

    idx = np.nonzero(vis)[0]
    if idx.size == 0:
        return rgb, zbuf, hit

    # Painter's algorithm via sorting far -> near, then splatting.
    order = idx[np.argsort(-depth[idx])]
    us = np.clip(uv[order, 0].astype(int), 0, W - 1)
    vs = np.clip(uv[order, 1].astype(int), 0, H - 1)
    ds = depth[order]
    cs = colours[order]

    r = splat_radius
    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            uu = np.clip(us + du, 0, W - 1)
            vv = np.clip(vs + dv, 0, H - 1)
            closer = ds < zbuf[vv, uu]
            if not closer.any():
                continue
            zbuf[vv[closer], uu[closer]] = ds[closer]
            rgb[vv[closer], uu[closer]] = cs[closer]
            hit[vv[closer], uu[closer]] = True

    return rgb, zbuf, hit


def observability(
    points_w: np.ndarray,
    centres: np.ndarray,
    rots: np.ndarray,
    intr: Intrinsics,
    occlusion_tol: float = 1.5,
):
    """How many cameras actually SAW each point (not merely had it in frustum).

    Uses a per-frame z-buffer: a point counts as observed only if its depth is within
    `occlusion_tol` metres of the nearest surface along that pixel ray.

    Returns views (N,) int.
    """
    n_pts = points_w.shape[0]
    views = np.zeros(n_pts, np.int32)
    H, W = intr.height, intr.width

    for c, R in zip(centres, rots):
        uv, depth, vis = project(points_w, c, R, intr)
        if not vis.any():
            continue
        idx = np.nonzero(vis)[0]
        us = np.clip(uv[idx, 0].astype(int), 0, W - 1)
        vs = np.clip(uv[idx, 1].astype(int), 0, H - 1)
        ds = depth[idx]

        # Build the z-buffer for this frame.
        zbuf = np.full((H, W), np.inf, np.float32)
        np.minimum.at(zbuf, (vs, us), ds)

        # A point is genuinely visible if nothing meaningfully nearer occupies its pixel.
        front = ds <= zbuf[vs, us] + occlusion_tol
        views[idx[front]] += 1

    return views


def confidence_bucket(views: np.ndarray) -> np.ndarray:
    """0 = never seen, 1 = single view (monocular guess), 2 = weak, 3 = well triangulated."""
    out = np.zeros_like(views)
    out[views == 1] = 1
    out[views == 2] = 2
    out[views >= 3] = 3
    return out


CONF_COLOURS = {
    0: (0.85, 0.15, 0.15),   # never observed  — red
    1: (0.95, 0.45, 0.10),   # 1 view          — orange
    2: (0.95, 0.85, 0.20),   # 2 views         — amber
    3: (0.20, 0.75, 0.35),   # 3+ views        — green
}
