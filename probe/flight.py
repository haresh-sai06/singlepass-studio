"""
Single-pass drone flight path + pinhole camera model.

The camera-tilt parameter here is the point of this module. Nadir (straight down)
flight is geometrically degenerate for forward motion: points near the image centre
sit at the epipole and have almost zero parallax no matter how far you fly. Tilting
the camera forward fixes that AND exposes building facades, which the problem
statement explicitly asks for.

Conventions
-----------
World:  X = east, Y = north, Z = up (metres)
Camera: OpenCV — +X right, +Y down, +Z forward along the optical axis
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    width: int = 640
    height: int = 480
    fov_deg: float = 70.0

    @property
    def fx(self) -> float:
        return self.width / (2.0 * np.tan(np.deg2rad(self.fov_deg) / 2.0))

    @property
    def fy(self) -> float:
        return self.fx  # square pixels

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ])


def camera_rotation(tilt_deg: float) -> np.ndarray:
    """Rotation matrix (camera -> world) for a drone flying along +X.

    tilt_deg = 0   -> nadir, optical axis points straight down (-Z world)
    tilt_deg = 40  -> tilted 40 deg forward, toward the direction of travel

    Returns R_c2w whose columns are the camera x, y, z axes expressed in world coords.
    """
    th = np.deg2rad(tilt_deg)
    # Derived by rotating the nadir frame about the camera's x-axis.
    x_cam = np.array([0.0, -1.0, 0.0])
    y_cam = np.array([-np.cos(th), 0.0, -np.sin(th)])
    z_cam = np.array([np.sin(th), 0.0, -np.cos(th)])
    R_c2w = np.stack([x_cam, y_cam, z_cam], axis=1)
    return R_c2w


@dataclass
class Flight:
    """A straight single pass — exactly the constraint the problem statement imposes."""
    altitude: float = 90.0        # metres above origin
    length: float = 160.0         # total ground distance covered
    n_frames: int = 60
    tilt_deg: float = 40.0        # 0 = nadir (degenerate), 30-45 = good
    y_offset: float = -35.0       # lateral offset so buildings sit in view
    jitter_m: float = 0.25        # small pose wobble (wind, vibration)
    seed: int = 11

    def poses(self):
        """Returns (centres (N,3), R_c2w (N,3,3)) — the TRUE camera trajectory."""
        rng = np.random.default_rng(self.seed)
        xs = np.linspace(-self.length / 2, self.length / 2, self.n_frames)
        centres = np.stack([
            xs,
            np.full(self.n_frames, self.y_offset),
            np.full(self.n_frames, self.altitude),
        ], axis=1)
        centres = centres + rng.normal(0.0, self.jitter_m, centres.shape)

        R = camera_rotation(self.tilt_deg)
        rots = np.repeat(R[None, :, :], self.n_frames, axis=0)

        # Tiny per-frame orientation wobble so it isn't unrealistically perfect.
        for i in range(self.n_frames):
            ang = rng.normal(0.0, np.deg2rad(0.35), 3)
            rots[i] = _small_rotation(ang) @ rots[i]

        return centres, rots

    def baseline_between(self, i: int, j: int) -> float:
        centres, _ = self.poses()
        return float(np.linalg.norm(centres[j] - centres[i]))


def _small_rotation(ang: np.ndarray) -> np.ndarray:
    """Small-angle rotation from a 3-vector of axis-angle components."""
    ax, ay, az = ang
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project(points_w: np.ndarray, centre: np.ndarray, R_c2w: np.ndarray, intr: Intrinsics):
    """Project world points into one camera.

    Returns (uv (N,2), depth (N,), visible_mask (N,))
    """
    R_w2c = R_c2w.T
    pc = (R_w2c @ (points_w - centre).T).T     # world -> camera
    depth = pc[:, 2]
    in_front = depth > 1e-6

    safe_z = np.where(in_front, depth, 1.0)
    u = intr.fx * pc[:, 0] / safe_z + intr.cx
    v = intr.fy * pc[:, 1] / safe_z + intr.cy
    uv = np.stack([u, v], axis=1)

    on_image = (
        in_front
        & (u >= 0) & (u < intr.width)
        & (v >= 0) & (v < intr.height)
    )
    return uv, depth, on_image


def parallax_angle_deg(depth: float, baseline: float) -> float:
    """Triangulation intersection angle for a point at `depth` seen from two cameras
    `baseline` apart. Small angle => ill-conditioned depth."""
    return float(np.rad2deg(2.0 * np.arctan((baseline / 2.0) / max(depth, 1e-9))))


def depth_uncertainty(depth: float, baseline: float, fx: float, sigma_px: float = 0.5) -> float:
    """sigma_Z ~= Z^2 / (f * B) * sigma_disparity  — the core error model."""
    return float(depth ** 2 / (fx * max(baseline, 1e-9)) * sigma_px)
