"""
Stage 1 — decode + keyframe selection.

Two filters, both cheap and both worth it:

  BLUR      Variance of the Laplacian. Drone footage during gusts and turns is
            full of motion blur, and a blurred frame poisons feature matching
            for every pair it participates in.

  SPACING   You do not want 375 near-identical frames. You want frames far
            enough apart to give parallax. See docs/FINDINGS.md #2 — depth
            uncertainty falls linearly with baseline, so wide spacing is the
            single cheapest accuracy win available.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Keyframe:
    index: int          # index in the original video
    order: int          # position in the selected sequence
    sharpness: float
    motion: float       # mean abs difference vs previous kept frame
    image: np.ndarray = field(repr=False, default=None)


@dataclass
class IngestConfig:
    max_width: int = 960        # downscale before anything else
    blur_percentile: float = 25 # drop the blurriest N% of candidates
    target_keyframes: int = 40
    min_motion: float = 6.0     # skip frames where the drone barely moved


def probe(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    info = {
        "path": path,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_s"] = info["frames"] / max(info["fps"], 1e-6)
    cap.release()
    return info


def select_keyframes(path: str, cfg: IngestConfig, progress=None) -> tuple[list[Keyframe], dict]:
    """Decode, score every frame, then keep a well-spaced sharp subset."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores: list[tuple[int, float, float]] = []   # (idx, sharpness, motion)
    frames: dict[int, np.ndarray] = {}

    prev_small = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        if w > cfg.max_width:
            s = cfg.max_width / w
            frame = cv2.resize(frame, (cfg.max_width, int(h * s)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        small = cv2.resize(gray, (160, 90))
        motion = 0.0 if prev_small is None else float(
            np.abs(small.astype(np.float32) - prev_small.astype(np.float32)).mean()
        )
        prev_small = small

        scores.append((idx, sharp, motion))
        frames[idx] = frame
        idx += 1

        if progress and idx % 25 == 0:
            progress(idx / max(n_total, 1), f"decoded {idx}/{n_total}")

    cap.release()

    if not scores:
        raise RuntimeError("no frames decoded")

    sharp_vals = np.array([s for _, s, _ in scores])
    cutoff = float(np.percentile(sharp_vals, cfg.blur_percentile))

    # Candidates: sharp enough, and the drone was actually moving.
    candidates = [(i, s, m) for (i, s, m) in scores if s >= cutoff and m >= cfg.min_motion]
    if len(candidates) < cfg.target_keyframes:
        candidates = [(i, s, m) for (i, s, m) in scores if s >= cutoff]
    if len(candidates) < cfg.target_keyframes:
        candidates = scores

    # Even spacing across the clip, picking the sharpest frame in each bucket.
    k = min(cfg.target_keyframes, len(candidates))
    buckets = np.array_split(np.array([c[0] for c in candidates]), k)
    lookup = {i: (s, m) for i, s, m in candidates}

    keyframes: list[Keyframe] = []
    for order, bucket in enumerate(buckets):
        if len(bucket) == 0:
            continue
        best = max(bucket, key=lambda i: lookup[i][0])
        s, m = lookup[best]
        keyframes.append(Keyframe(index=int(best), order=order, sharpness=s,
                                  motion=m, image=frames[int(best)]))

    stats = {
        "frames_decoded": len(scores),
        "blur_cutoff": round(cutoff, 1),
        "rejected_blur": int((sharp_vals < cutoff).sum()),
        "candidates": len(candidates),
        "keyframes": len(keyframes),
        "sharpness_mean": round(float(sharp_vals.mean()), 1),
        "sharpness_min": round(float(sharp_vals.min()), 1),
        "sharpness_max": round(float(sharp_vals.max()), 1),
    }
    return keyframes, stats
