"""
Orchestrator — runs the enabled stages and streams progress.

Every stage is toggleable from the dashboard, because the point of the tool is
to let you see what each one contributes (or costs) on YOUR footage.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .analyze import (build_grade_array, cloud_stats, export_ply,
                      export_viewer_json, statistical_outlier_filter)
from .incremental import IncConfig, run_incremental
from .ingest import IngestConfig, probe, select_keyframes
from .mast3r_stage import Mast3rConfig, available as mast3r_available, run_mast3r
from .mesh import MeshConfig, build_mesh, export_mesh_json, export_mesh_obj
from .planesweep import SweepConfig, run_planesweep
from .sfm import SfmConfig, run_sfm


@dataclass
class RunConfig:
    video: str
    out_dir: str = "out/runs"

    # stage toggles
    do_blur_filter: bool = True
    do_outlier_filter: bool = True
    do_observability: bool = True
    do_export_ply: bool = True
    do_dense: bool = True           # plane-sweep MVS  (sparse -> dense)
    do_mesh: bool = True            # Poisson surface  (dense -> model)

    # ingest
    max_width: int = 960
    target_keyframes: int = 30
    blur_percentile: float = 25.0

    # sfm
    mode: str = "incremental"       # "incremental" | "pairwise" | "mast3r"
    mast3r_images: int = 16
    mast3r_size: int = 512
    mast3r_conf: float = 1.5
    detector: str = "SIFT"
    max_features: int = 4000
    pair_stride: int = 6
    also_adjacent: bool = True
    ratio_test: float = 0.75
    fov_deg: float = 78.0
    min_matches: int = 40
    max_reproj_error: float = 4.0

    # dense / mesh
    depth_planes: int = 96
    sweep_stride: int = 3
    sweep_neighbours: int = 4
    poisson_depth: int = 9
    target_triangles: int = 220_000

    # cleanup
    outlier_std: float = 2.0


@dataclass
class StageResult:
    name: str
    seconds: float
    detail: dict = field(default_factory=dict)


def run(cfg: RunConfig, emit=None) -> dict:
    """Execute the pipeline. `emit(event: dict)` receives progress messages."""
    def send(kind, **kw):
        if emit:
            emit({"type": kind, **kw})

    t_all = time.perf_counter()
    stages: list[StageResult] = []
    result: dict = {"config": asdict(cfg), "ok": False}

    try:
        # ── probe ──
        send("stage", stage="probe", status="running")
        info = probe(cfg.video)
        result["video"] = info
        send("stage", stage="probe", status="done", detail=info)

        # ── ingest ──
        send("stage", stage="ingest", status="running")
        t = time.perf_counter()
        icfg = IngestConfig(
            max_width=cfg.max_width,
            blur_percentile=cfg.blur_percentile if cfg.do_blur_filter else 0.0,
            target_keyframes=cfg.target_keyframes,
        )
        kfs, istats = select_keyframes(
            cfg.video, icfg,
            progress=lambda f, m: send("progress", stage="ingest", frac=f, msg=m))
        dt = time.perf_counter() - t
        stages.append(StageResult("ingest", dt, istats))
        result["ingest"] = istats
        send("stage", stage="ingest", status="done", seconds=round(dt, 2), detail=istats)

        # ── sfm ──
        send("stage", stage="sfm", status="running")
        t = time.perf_counter()
        scfg = SfmConfig(
            detector=cfg.detector, max_features=cfg.max_features,
            ratio_test=cfg.ratio_test, pair_stride=cfg.pair_stride,
            also_adjacent=cfg.also_adjacent, fov_deg=cfg.fov_deg,
            min_matches=cfg.min_matches,
        )
        if cfg.mode == "mast3r":
            ok_m, why = mast3r_available()
            if not ok_m:
                raise RuntimeError(f"MASt3R not ready: {why}")
            inc = run_mast3r(
                kfs, Mast3rConfig(max_images=cfg.mast3r_images,
                                  image_size=cfg.mast3r_size,
                                  min_conf=cfg.mast3r_conf),
                progress=lambda f, m: send("progress", stage="sfm", frac=f, msg=m))
            dt = time.perf_counter() - t
            stages.append(StageResult("sfm", dt, inc["summary"]))
            result["sfm"] = inc["summary"]
            result["cameras"] = [[round(float(v), 4) for v in c]
                                 for c in inc["camera_centres"]]
            send("stage", stage="sfm", status="done", seconds=round(dt, 2),
                 detail=inc["summary"])
            sfm_poses, sfm_K = inc["poses"], inc["K"]
            pts, cols = inc["points"], inc["colours"]
            # MASt3R returns dense pointmaps already — grade uniformly and skip
            # the plane sweep, which exists to densify a SPARSE cloud.
            grades = np.full(len(pts), 2, dtype=np.int8)
            kfs = [kfs[i] for i in np.linspace(0, len(kfs)-1,
                   min(cfg.mast3r_images, len(kfs))).round().astype(int)]
        elif cfg.mode == "incremental":
            inc = run_incremental(
                kfs, scfg, IncConfig(max_reproj_error=cfg.max_reproj_error),
                progress=lambda f, m: send("progress", stage="sfm", frac=f, msg=m))
            dt = time.perf_counter() - t
            stages.append(StageResult("sfm", dt, inc["summary"]))
            result["sfm"] = inc["summary"]
            result["cameras"] = [[round(float(v), 4) for v in c]
                                 for c in inc["camera_centres"]]
            send("stage", stage="sfm", status="done", seconds=round(dt, 2),
                 detail=inc["summary"])

            sfm_poses, sfm_K = inc["poses"], inc["K"]
            pts, cols = inc["points"], inc["colours"]
            ang = inc["parallax"]
            # Grade from the ACTUAL multi-view parallax of each point.
            grades = np.zeros(len(pts), dtype=np.int8)
            grades[ang >= 2] = 1
            grades[ang >= 5] = 2
            grades[ang >= 10] = 3
            if not cfg.do_observability:
                grades = np.full(len(pts), 3, dtype=np.int8)
        else:
            sfm = run_sfm(kfs, scfg,
                          progress=lambda f, m: send("progress", stage="sfm", frac=f, msg=m))
            dt = time.perf_counter() - t
            stages.append(StageResult("sfm", dt, sfm["summary"]))
            result["sfm"] = sfm["summary"]
            result["diagnostics"] = sfm["diagnostics"]
            send("stage", stage="sfm", status="done", seconds=round(dt, 2),
                 detail=sfm["summary"])

            pts, cols = sfm["points"], sfm["colours"]
            grades = build_grade_array(sfm["diagnostics"]) if cfg.do_observability \
                else np.full(len(pts), 3, dtype=np.int8)

        # ── dense MVS (plane sweep — works under forward motion) ──
        if cfg.do_dense and len(pts) > 500 and cfg.mode != "mast3r":
            send("stage", stage="dense", status="running")
            t = time.perf_counter()
            ps = run_planesweep(
                kfs, sfm_poses, sfm_K, pts,
                SweepConfig(depth_planes=cfg.depth_planes,
                            ref_stride=cfg.sweep_stride,
                            neighbours=cfg.sweep_neighbours),
                progress=lambda f, m: send("progress", stage="dense", frac=f, msg=m))
            dt = time.perf_counter() - t
            stages.append(StageResult("dense", dt, ps["summary"]))
            result["dense"] = ps["summary"]
            send("stage", stage="dense", status="done", seconds=round(dt, 2),
                 detail=ps["summary"])
            if len(ps["points"]) > len(pts):
                sparse_pts, sparse_cols = pts, cols
                pts, cols = ps["points"], ps["colours"]
                grades = np.full(len(pts), 2, dtype=np.int8)

        # ── cleanup ──
        if cfg.do_outlier_filter and len(pts) > 100:
            send("stage", stage="cleanup", status="running")
            t = time.perf_counter()
            keep = statistical_outlier_filter(pts, std_ratio=cfg.outlier_std)
            removed = int((~keep).sum())
            pts, cols = pts[keep], cols[keep]
            grades = grades[keep] if len(grades) == len(keep) else grades
            dt = time.perf_counter() - t
            det = {"removed": removed, "remaining": int(len(pts)),
                   "removed_pct": round(removed / max(len(keep), 1) * 100, 1)}
            stages.append(StageResult("cleanup", dt, det))
            result["cleanup"] = det
            send("stage", stage="cleanup", status="done", seconds=round(dt, 2), detail=det)

        # ── surface reconstruction ──
        mesh_res = None
        if cfg.do_mesh and len(pts) > 2000:
            send("stage", stage="mesh", status="running")
            t = time.perf_counter()
            mesh_res = build_mesh(
                pts, cols, result.get("cameras"),
                MeshConfig(poisson_depth=cfg.poisson_depth,
                           target_triangles=cfg.target_triangles),
                progress=lambda f, m: send("progress", stage="mesh", frac=f, msg=m))
            dt = time.perf_counter() - t
            det = mesh_res.get("summary", {"failed": mesh_res.get("reason")})
            stages.append(StageResult("mesh", dt, det))
            result["mesh"] = det
            send("stage", stage="mesh", status="done" if mesh_res["ok"] else "error",
                 seconds=round(dt, 2), detail=det)

        # ── stats + export ──
        send("stage", stage="export", status="running")
        t = time.perf_counter()
        stats = cloud_stats(pts, grades)
        result["cloud"] = stats

        run_dir = Path(cfg.out_dir) / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        viewer = export_viewer_json(run_dir / "cloud.json", pts, cols, grades,
                                    cameras=result.get("cameras"))
        result["viewer_json"] = viewer
        if cfg.do_export_ply:
            result["ply"] = export_ply(run_dir / "cloud.ply", pts, cols)
        if mesh_res and mesh_res.get("ok"):
            result["mesh_json"] = export_mesh_json(
                run_dir / "mesh.json", mesh_res["vertices"], mesh_res["faces"],
                mesh_res["normals"], mesh_res["colours"])
            result["mesh_obj"] = export_mesh_obj(run_dir / "mesh.obj", mesh_res["mesh"])
        dt = time.perf_counter() - t
        stages.append(StageResult("export", dt, stats))
        send("stage", stage="export", status="done", seconds=round(dt, 2), detail=stats)

        result["stages"] = [asdict(s) for s in stages]
        result["total_seconds"] = round(time.perf_counter() - t_all, 2)
        result["run_dir"] = str(run_dir)
        result["ok"] = True
        send("done", result=result)

    except Exception as e:  # noqa: BLE001
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()[-1200:]
        send("error", error=str(e))

    return result
