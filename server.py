#!/usr/bin/env python3
"""
Singlepass Studio — local backend.

Serves the dashboard and runs the real reconstruction pipeline against footage
in this folder, streaming stage-by-stage progress to the browser over SSE.

    pip install fastapi uvicorn opencv-python numpy
    python server.py
    -> http://localhost:8000
"""
from __future__ import annotations

import glob
import json
import queue
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.ingest import probe
from pipeline.runner import RunConfig, run

ROOT = Path(__file__).parent
WEB = ROOT / "web"

app = FastAPI(title="Singlepass Studio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

JOBS: dict[str, dict] = {}


class RunRequest(BaseModel):
    video: str
    do_blur_filter: bool = True
    do_outlier_filter: bool = True
    do_observability: bool = True
    do_export_ply: bool = True
    max_width: int = 960
    target_keyframes: int = 30
    blur_percentile: float = 25.0
    detector: str = "SIFT"
    max_features: int = 4000
    pair_stride: int = 6
    also_adjacent: bool = True
    ratio_test: float = 0.75
    fov_deg: float = 78.0
    min_matches: int = 40
    outlier_std: float = 2.0


@app.get("/api/videos")
def list_videos():
    out = []
    for pat in ("*.mp4", "*.mov", "*.MP4", "*.MOV", "*.mkv", "*.avi"):
        for f in glob.glob(str(ROOT / pat)):
            p = Path(f)
            try:
                info = probe(str(p))
                out.append({
                    "name": p.name,
                    "path": str(p),
                    "size_mb": round(p.stat().st_size / 1e6, 1),
                    "width": info["width"], "height": info["height"],
                    "fps": round(info["fps"], 2), "frames": info["frames"],
                    "duration_s": round(info["duration_s"], 2),
                })
            except Exception:
                continue
    return {"videos": out}


@app.post("/api/run")
def start_run(req: RunRequest):
    job_id = f"job-{int(time.time() * 1000)}"
    q: queue.Queue = queue.Queue()
    JOBS[job_id] = {"queue": q, "result": None, "started": time.time()}

    cfg = RunConfig(**req.model_dump())

    def worker():
        res = run(cfg, emit=lambda ev: q.put(ev))
        JOBS[job_id]["result"] = res
        q.put({"type": "eof"})

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/stream/{job_id}")
def stream(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)

    def gen():
        q = job["queue"]
        while True:
            try:
                ev = q.get(timeout=120)
            except queue.Empty:
                yield "event: ping\ndata: {}\n\n"
                continue
            if ev.get("type") == "eof":
                yield "event: eof\ndata: {}\n\n"
                break
            yield f"event: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/result/{job_id}")
def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return job["result"] or {"pending": True}


@app.get("/api/cloud")
def cloud(path: str):
    p = Path(path)
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="application/json")


@app.get("/")
def index():
    return FileResponse(WEB / "dashboard.html")


app.mount("/web", StaticFiles(directory=WEB), name="web")


if __name__ == "__main__":
    print("\n  Singlepass Studio  ->  http://localhost:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
