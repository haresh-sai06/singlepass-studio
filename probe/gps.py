"""
GPS/IMU simulation and metric recovery.

Real consumer GNSS on a drone is roughly:
  horizontal  1-3 m  (1-sigma)
  vertical    2-5 m  (1-sigma, typically ~1.5-2x worse than horizontal)
RTK/PPK brings that to 2-5 cm, but you cannot assume it.

Barometric altitude is usually BETTER than GNSS vertical for relative height changes,
which is why it is worth fusing separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GnssModel:
    sigma_h: float = 2.0          # horizontal 1-sigma, metres
    sigma_v: float = 4.0          # vertical 1-sigma, metres
    bias_h: float = 1.2           # slowly-varying bias (atmosphere) — NOT zero-mean
    drift_period: float = 40.0    # frames per bias oscillation
    outlier_rate: float = 0.04    # fraction of fixes badly wrong (multipath)
    outlier_scale: float = 12.0
    baro_sigma: float = 0.6       # barometric altitude noise, metres
    seed: int = 23

    def simulate(self, true_centres: np.ndarray):
        """Returns (gps (N,3) noisy, baro_z (N,) noisy altitude, outlier_mask (N,))."""
        rng = np.random.default_rng(self.seed)
        n = true_centres.shape[0]

        # Correlated bias — GNSS error is not independent frame to frame.
        phase = np.linspace(0, 2 * np.pi * n / self.drift_period, n)
        bias = np.stack([
            self.bias_h * np.sin(phase),
            self.bias_h * np.cos(phase * 0.7),
            np.zeros(n),
        ], axis=1)

        noise = np.stack([
            rng.normal(0, self.sigma_h, n),
            rng.normal(0, self.sigma_h, n),
            rng.normal(0, self.sigma_v, n),
        ], axis=1)

        gps = true_centres + bias + noise

        # Multipath outliers
        outliers = rng.random(n) < self.outlier_rate
        if outliers.any():
            gps[outliers] += rng.normal(0, self.outlier_scale, (int(outliers.sum()), 3))

        baro_z = true_centres[:, 2] + rng.normal(0, self.baro_sigma, n)
        return gps, baro_z, outliers


def scale_ambiguous_reconstruction(
    true_centres: np.ndarray,
    true_points: np.ndarray,
    scale: float = 0.0137,
    seed: int = 31,
    noise: float = 0.004,
):
    """Fake what SfM hands you: correct SHAPE, arbitrary scale/rotation/translation.

    This is the whole reason metric alignment is needed. An SfM reconstruction is
    identical maths for a real building and a dollhouse — only GPS resolves which.
    """
    rng = np.random.default_rng(seed)

    # Arbitrary rotation
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    t = rng.normal(0, 5, 3)

    def fwd(p):
        return (scale * (R @ p.T)).T + t

    rec_centres = fwd(true_centres)
    rec_points = fwd(true_points)

    # SfM noise, proportional to the reconstruction's own scale
    rec_centres = rec_centres + rng.normal(0, noise * scale * 40, rec_centres.shape)
    rec_points = rec_points + rng.normal(0, noise * scale * 40, rec_points.shape)

    return rec_centres, rec_points, dict(true_scale=scale, R=R, t=t)
