"""
Stage 7 — surface reconstruction and texturing.

A point cloud is a bag of dots: you cannot measure a face, cast a shadow, or
compute a volume from it. A MESH turns those dots into a watertight-ish surface
of triangles, which is what "3D model" actually means.

Pipeline:
  1. estimate per-point normals (which way each bit of surface faces)
  2. orient them consistently toward the cameras
  3. Poisson surface reconstruction -> triangle mesh
  4. trim low-density regions (Poisson happily invents surface where it saw
     nothing — exactly the fabrication this project refuses to do)
  5. transfer point colour onto vertices
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MeshConfig:
    voxel: float = 0.0          # 0 = auto from cloud extent
    normal_k: int = 30
    poisson_depth: int = 9
    density_trim_pct: float = 6.0    # drop the lowest-density N% of vertices
    target_triangles: int = 220_000
    min_cluster_frac: float = 0.02   # discard tiny disconnected fragments


def build_mesh(pts: np.ndarray, cols: np.ndarray, camera_centres: np.ndarray,
               cfg: MeshConfig, progress=None):
    """Point cloud -> coloured triangle mesh. Returns dict with mesh arrays."""
    import open3d as o3d

    def say(f, m):
        if progress:
            progress(f, m)

    if len(pts) < 500:
        return {"ok": False, "reason": "too few points to mesh"}

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    if len(cols) == len(pts):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1).astype(np.float64))

    # ── voxel downsample: MVS pairs overlap heavily, and Poisson does not
    #    benefit from 20 near-duplicate samples of the same 5 mm of surface.
    extent = float(np.linalg.norm(np.ptp(np.asarray(pcd.points), axis=0)))
    voxel = cfg.voxel if cfg.voxel > 0 else max(extent / 900.0, 1e-6)
    say(0.1, f"downsampling at {voxel:.4f}")
    pcd = pcd.voxel_down_sample(voxel)

    say(0.25, "removing outliers")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    if len(pcd.points) < 500:
        return {"ok": False, "reason": "cloud collapsed after filtering"}

    # ── normals ──
    say(0.4, "estimating normals")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 4.0, max_nn=cfg.normal_k))

    # Orient toward the flight path — Poisson needs consistent normals, and
    # "facing the camera that saw it" is the physically correct choice.
    if camera_centres is not None and len(camera_centres):
        pcd.orient_normals_towards_camera_location(
            np.asarray(camera_centres, dtype=np.float64).mean(axis=0))
    else:
        pcd.orient_normals_consistent_tangent_plane(30)

    # ── Poisson ──
    say(0.55, f"Poisson reconstruction (depth {cfg.poisson_depth})")
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=cfg.poisson_depth, width=0, scale=1.1, linear_fit=False)

    if len(mesh.triangles) == 0:
        return {"ok": False, "reason": "Poisson produced no triangles"}

    # ── trim invented surface ──
    say(0.75, "trimming low-density surface")
    d = np.asarray(density)
    thresh = np.quantile(d, cfg.density_trim_pct / 100.0)
    mesh.remove_vertices_by_mask(d < thresh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    # ── drop floating fragments ──
    say(0.85, "removing disconnected fragments")
    try:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        if len(counts):
            keep_labels = np.nonzero(counts >= counts.max() * cfg.min_cluster_frac)[0]
            mesh.remove_triangles_by_mask(~np.isin(labels, keep_labels))
            mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # ── simplify ──
    if len(mesh.triangles) > cfg.target_triangles:
        say(0.9, f"simplifying to {cfg.target_triangles} triangles")
        mesh = mesh.simplify_quadric_decimation(cfg.target_triangles)
        mesh.remove_unreferenced_vertices()

    # ── colour transfer from the dense cloud ──
    say(0.95, "transferring colour")
    if len(pcd.colors):
        tree = o3d.geometry.KDTreeFlann(pcd)
        verts = np.asarray(mesh.vertices)
        src_cols = np.asarray(pcd.colors)
        vcols = np.zeros_like(verts)
        for i, v in enumerate(verts):
            ok, idx, _ = tree.search_knn_vector_3d(v, 3)
            vcols[i] = src_cols[list(idx)].mean(axis=0) if ok else 0.7
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vcols, 0, 1))

    mesh.compute_vertex_normals()

    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    N = np.asarray(mesh.vertex_normals, dtype=np.float32)
    C = (np.asarray(mesh.vertex_colors, dtype=np.float32)
         if len(mesh.vertex_colors) else np.full_like(V, 0.72))

    return {
        "ok": True, "mesh": mesh,
        "vertices": V, "faces": F, "normals": N, "colours": C,
        "summary": {
            "input_points": int(len(pts)),
            "after_downsample": int(len(pcd.points)),
            "vertices": int(len(V)),
            "triangles": int(len(F)),
            "voxel": round(voxel, 5),
            "trimmed_pct": cfg.density_trim_pct,
        },
    }


def export_mesh_obj(path: str | Path, mesh) -> str:
    import open3d as o3d
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_colors=True)
    return str(path)


def export_mesh_json(path: str | Path, V, F, N, C, max_tris: int = 200_000) -> str:
    """Interleaved arrays for the WebGL viewer."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(F) > max_tris:
        F = F[np.random.default_rng(0).choice(len(F), max_tris, replace=False)]

    # Median centre + inner-percentile scale: Poisson can leave a few stray
    # vertices far from the body, and a mean/max normalisation lets those
    # squash the real geometry into a dot.
    centre = np.median(V, axis=0) if len(V) else np.zeros(3)
    Vc = V - centre
    radii = np.linalg.norm(Vc, axis=1)
    scale = float(np.percentile(radii, 88)) or 1.0
    Vn = (Vc / scale * 40.0).astype(np.float32)

    data = {
        "vertices": [round(float(v), 3) for v in Vn.ravel()],
        "normals": [round(float(v), 3) for v in N.ravel()],
        "colours": [round(float(v), 3) for v in C.ravel()],
        "indices": [int(v) for v in F.ravel()],
        "vertex_count": int(len(Vn)),
        "triangle_count": int(len(F)),
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)
