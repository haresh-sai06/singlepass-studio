#!/usr/bin/env python3
"""
Head-to-head: classical SfM+MVS versus MASt3R on the SAME footage.

Same clip, same keyframes, same meshing, same metrics. The only variable is how
geometry is recovered. That is the comparison the problem statement turns on.

    python compare_backends.py
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np

from pipeline.runner import RunConfig, run

OUT = Path(__file__).parent / "out"


def bar(label, value, best, width=34, unit=""):
    frac = 0 if not best else min(value / best, 1.0)
    filled = int(frac * width)
    return f"  {label:<22}{'#' * filled}{'.' * (width - filled)}  {value:,.0f}{unit}"


def main() -> None:
    vids = glob.glob(str(Path(__file__).parent / "*.mp4"))
    if not vids:
        print("no .mp4 found next to this script")
        return
    video = vids[0]
    print(f"\n  source: {Path(video).name}\n")

    runs: dict[str, dict] = {}
    for mode, label in [("incremental", "Classical  SfM + plane-sweep MVS"),
                        ("mast3r", "MASt3R     feed-forward pointmaps")]:
        print(f"  {'=' * 68}\n  {label}\n  {'=' * 68}")
        t0 = time.perf_counter()
        try:
            r = run(RunConfig(video=video, mode=mode, target_keyframes=30,
                              do_dense=(mode != "mast3r"), do_mesh=True),
                    emit=lambda e: (
                        print(f"    {e['stage']:<10} {e.get('status','')}"
                              f" {e.get('seconds','')}")
                        if e.get("type") == "stage" and e.get("status") == "done" else None))
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {e}")
            continue
        if not r.get("ok"):
            print(f"    FAILED: {r.get('error')}")
            continue
        r["_wall"] = round(time.perf_counter() - t0, 1)
        runs[mode] = r
        print()

    if len(runs) < 2:
        print("  need both backends to compare\n")
        return

    a, b = runs.get("incremental"), runs.get("mast3r")
    print(f"  {'=' * 68}\n  COMPARISON\n  {'=' * 68}\n")

    rows = [
        ("cameras", lambda r: r["sfm"].get("cameras_registered", 0), ""),
        ("points", lambda r: r["cloud"]["points"], ""),
        ("mesh vertices", lambda r: (r.get("mesh") or {}).get("vertices", 0), ""),
        ("mesh triangles", lambda r: (r.get("mesh") or {}).get("triangles", 0), ""),
        ("wall time", lambda r: r["_wall"], " s"),
    ]
    print(f"  {'metric':<22}{'classical':>14}{'MASt3R':>14}{'delta':>12}")
    print(f"  {'-' * 62}")
    for name, fn, unit in rows:
        va, vb = fn(a), fn(b)
        d = "—" if not va else f"{(vb - va) / va * 100:+.0f}%"
        print(f"  {name:<22}{va:>14,.0f}{vb:>14,.0f}{d:>12}")

    print()
    for mode, r in runs.items():
        m = r.get("mesh") or {}
        print(f"  {mode:<14}{r['cloud']['points']:>9,} pts  "
              f"{m.get('triangles', 0):>8,} tris   {r.get('mesh_obj', '—')}")

    (OUT / "comparison.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items()
                        if kk in ("sfm", "cloud", "mesh", "dense", "_wall",
                                  "total_seconds", "mesh_obj", "ply")}
                    for k, v in runs.items()}, indent=2, default=str),
        encoding="utf-8")
    print(f"\n  wrote {OUT / 'comparison.json'}\n")


if __name__ == "__main__":
    main()
