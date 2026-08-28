"""
Procedural synthetic scene with exact ground truth.

Why this exists: NTRO provides drone footage only at the hackathon. To develop and,
more importantly, to MEASURE accuracy beforehand, you need a scene where you already
know the true geometry. This generates one.

World frame: X = east, Y = north, Z = up. Units are metres.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Building:
    cx: float
    cy: float
    width: float
    depth: float
    height: float
    base_z: float = 0.0

    @property
    def roof_z(self) -> float:
        return self.base_z + self.height


@dataclass
class Scene:
    """Terrain heightmap + box buildings. Sampled as a dense point cloud with colours."""
    extent: float = 200.0          # scene is extent x extent metres, centred on origin
    terrain_res: int = 120         # heightmap grid resolution
    terrain_amp: float = 4.0       # terrain undulation amplitude (m)
    buildings: list[Building] = field(default_factory=list)
    seed: int = 7

    _heightmap: np.ndarray | None = None
    _grid_x: np.ndarray | None = None
    _grid_y: np.ndarray | None = None

    # ── terrain ──────────────────────────────────────────────
    def build_terrain(self) -> None:
        rng = np.random.default_rng(self.seed)
        n = self.terrain_res
        lin = np.linspace(-self.extent / 2, self.extent / 2, n)
        gx, gy = np.meshgrid(lin, lin)

        # Smooth low-frequency undulation — a couple of sine lobes plus mild noise.
        z = (
            self.terrain_amp * 0.6 * np.sin(gx / 45.0) * np.cos(gy / 55.0)
            + self.terrain_amp * 0.3 * np.sin(gx / 18.0 + 1.3)
            + self.terrain_amp * 0.1 * rng.standard_normal((n, n))
        )
        # Gentle blur so the noise term doesn't look like salt-and-pepper.
        for _ in range(3):
            z = (
                z
                + np.roll(z, 1, 0) + np.roll(z, -1, 0)
                + np.roll(z, 1, 1) + np.roll(z, -1, 1)
            ) / 5.0

        self._heightmap, self._grid_x, self._grid_y = z, gx, gy

    def terrain_height(self, x: float | np.ndarray, y: float | np.ndarray):
        """Bilinear-ish lookup (nearest cell is plenty for a synthetic target)."""
        if self._heightmap is None:
            self.build_terrain()
        n = self.terrain_res
        half = self.extent / 2
        ix = np.clip(((np.asarray(x) + half) / self.extent * (n - 1)).astype(int), 0, n - 1)
        iy = np.clip(((np.asarray(y) + half) / self.extent * (n - 1)).astype(int), 0, n - 1)
        return self._heightmap[iy, ix]

    # ── buildings ────────────────────────────────────────────
    def add_random_buildings(self, count: int = 9) -> None:
        rng = np.random.default_rng(self.seed + 1)
        margin = self.extent * 0.32
        for _ in range(count):
            cx = float(rng.uniform(-margin, margin))
            cy = float(rng.uniform(-margin, margin))
            self.buildings.append(
                Building(
                    cx=cx,
                    cy=cy,
                    width=float(rng.uniform(8, 22)),
                    depth=float(rng.uniform(8, 22)),
                    height=float(rng.uniform(6, 30)),
                    base_z=float(self.terrain_height(cx, cy)),
                )
            )

    # ── sampling ─────────────────────────────────────────────
    def sample_points(self, terrain_n: int = 60_000, per_building: int = 4_000):
        """Return (points (N,3), colours (N,3) float 0-1, labels (N,) int).

        label 0 = terrain, 1..K = building index.
        Point sampling stands in for a mesh here: it is simple, exact, and it is also
        conceptually what Gaussian Splatting operates on.
        """
        if self._heightmap is None:
            self.build_terrain()
        rng = np.random.default_rng(self.seed + 2)
        pts, cols, labs = [], [], []

        # Terrain
        half = self.extent / 2
        tx = rng.uniform(-half, half, terrain_n)
        ty = rng.uniform(-half, half, terrain_n)
        tz = self.terrain_height(tx, ty)
        terrain = np.stack([tx, ty, tz], axis=1)
        # Green-brown, shaded slightly by height so it isn't flat-looking.
        shade = (tz - tz.min()) / max(float(np.ptp(tz)), 1e-6)
        tcol = np.stack([
            0.32 + 0.18 * shade,
            0.42 + 0.20 * shade,
            0.22 + 0.10 * shade,
        ], axis=1)
        pts.append(terrain); cols.append(tcol); labs.append(np.zeros(terrain_n, int))

        # Buildings — sample the 4 walls and the roof
        for bi, b in enumerate(self.buildings, start=1):
            hw, hd = b.width / 2, b.depth / 2
            n_roof = per_building // 3
            n_wall = (per_building - n_roof) // 4

            # roof
            rx = rng.uniform(b.cx - hw, b.cx + hw, n_roof)
            ry = rng.uniform(b.cy - hd, b.cy + hd, n_roof)
            rz = np.full(n_roof, b.roof_z)
            roof = np.stack([rx, ry, rz], axis=1)

            walls = []
            for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if sx:  # east / west wall (varies in y and z)
                    wx = np.full(n_wall, b.cx + sx * hw)
                    wy = rng.uniform(b.cy - hd, b.cy + hd, n_wall)
                else:   # north / south wall (varies in x and z)
                    wx = rng.uniform(b.cx - hw, b.cx + hw, n_wall)
                    wy = np.full(n_wall, b.cy + sy * hd)
                wz = rng.uniform(b.base_z, b.roof_z, n_wall)
                walls.append(np.stack([wx, wy, wz], axis=1))

            bpts = np.concatenate([roof] + walls, axis=0)
            base_col = np.array([0.62, 0.60, 0.58]) + 0.10 * rng.standard_normal(3)
            bcol = np.clip(np.tile(base_col, (len(bpts), 1))
                           + 0.03 * rng.standard_normal((len(bpts), 3)), 0, 1)
            # Roof slightly darker so facades vs roofs are visually distinct.
            bcol[:n_roof] *= 0.78

            pts.append(bpts); cols.append(bcol)
            labs.append(np.full(len(bpts), bi, int))

        return (
            np.concatenate(pts, axis=0),
            np.clip(np.concatenate(cols, axis=0), 0, 1),
            np.concatenate(labs, axis=0),
        )

    # ── ground-truth measurements (what you validate against) ──
    def ground_truth_measurements(self) -> list[dict]:
        out = []
        for i, b in enumerate(self.buildings, start=1):
            out.append({
                "building": i,
                "height_m": round(b.height, 3),
                "width_m": round(b.width, 3),
                "depth_m": round(b.depth, 3),
                "roof_z_m": round(b.roof_z, 3),
                "centre": [round(b.cx, 3), round(b.cy, 3)],
            })
        return out


def default_scene() -> Scene:
    s = Scene()
    s.build_terrain()
    s.add_random_buildings(9)
    return s
