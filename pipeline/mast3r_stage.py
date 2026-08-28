"""
Stage 3b — MASt3R feed-forward reconstruction.

WHY THIS EXISTS
---------------
Classical SfM + MVS is measurably crippled on this footage, and this repo has
the numbers to prove it rather than assert it:

  * median multi-view parallax  1.95 deg
  * baseline-to-optical-axis ratio  0.84-0.92 across all 435 pairs
  * rectified stereo produced points on 2 of 74 attempted pairs

All three are symptoms of one thing: the drone flies along its own optical axis,
so triangulation is ill-conditioned everywhere. Geometry alone cannot fix that
because the information is not in the images as a geometry problem.

MASt3R attacks it as a LEARNED problem instead. Give it two images and a
transformer directly regresses a "pointmap" — a 3D point per pixel, with both
views already expressed in a shared frame. No feature matching, no essential
matrix, no rectification, no bundle adjustment. It was trained on scenes where
geometry is ambiguous, so it carries a prior about what surfaces look like, and
that prior is exactly the missing information here.

Practical notes for 6 GB VRAM:
  * 512 px inference wants ~4-6 GB for a pair. We run pairs sequentially and
    fall back to 384 px automatically on OOM.
  * The global aligner is the memory-hungry part; we cap the pair graph.
"""
from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party" / "mast3r"


def _ensure_paths() -> None:
    """MASt3R expects its repo root and the dust3r/croco submodules on sys.path."""
    for p in (_THIRD_PARTY, _THIRD_PARTY / "dust3r", _THIRD_PARTY / "dust3r" / "croco"):
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)


def available() -> tuple[bool, str]:
    if not _THIRD_PARTY.exists():
        return False, "third_party/mast3r not cloned"
    _ensure_paths()
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch not installed"
    try:
        from mast3r.model import AsymmetricMASt3R  # noqa: F401
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"import failed: {str(e)[:160]}"


@dataclass
class Mast3rConfig:
    weights: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    image_size: int = 512
    fallback_size: int = 384
    max_images: int = 16        # global alignment cost grows fast
    pair_window: int = 3        # link each frame to the next N
    niter: int = 300            # global aligner iterations
    schedule: str = "cosine"
    lr: float = 0.01
    min_conf: float = 1.5       # confidence threshold on the pointmaps
    subsample: int = 2
    device: str = "cuda"


def _select(keyframes, k: int):
    """Evenly spaced subset — global alignment memory scales with pair count."""
    if len(keyframes) <= k:
        return list(range(len(keyframes)))
    return list(np.linspace(0, len(keyframes) - 1, k).round().astype(int))


def run_mast3r(keyframes, cfg: Mast3rConfig, progress=None):
    """Feed-forward reconstruction. Returns the same shape as run_incremental."""
    ok, why = available()
    if not ok:
        raise RuntimeError(f"MASt3R unavailable: {why}")

    _ensure_paths()
    import torch
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from mast3r.model import AsymmetricMASt3R

    def say(f, m):
        if progress:
            progress(f, m)

    device = cfg.device if torch.cuda.is_available() else "cpu"

    # ── write the selected keyframes out; loaders expect files ──
    idxs = _select(keyframes, cfg.max_images)
    tmp = Path(__file__).resolve().parent.parent / "out" / "_mast3r_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.png"):
        f.unlink()

    import cv2
    paths = []
    for n, i in enumerate(idxs):
        p = tmp / f"kf_{n:03d}.png"
        cv2.imwrite(str(p), keyframes[i].image)
        paths.append(str(p))

    say(0.05, f"loading MASt3R weights on {device}")
    model = AsymmetricMASt3R.from_pretrained(cfg.weights).to(device).eval()

    size = cfg.image_size
    for attempt in range(2):
        try:
            say(0.15, f"inference at {size} px over {len(paths)} frames")
            images = load_images(paths, size=size, verbose=False)
            pairs = make_pairs(images, scene_graph=f"swin-{cfg.pair_window}",
                               prefilter=None, symmetrize=True)
            output = inference(pairs, model, device, batch_size=1, verbose=False)
            break
        except torch.cuda.OutOfMemoryError:
            if attempt == 1:
                raise
            torch.cuda.empty_cache()
            gc.collect()
            size = cfg.fallback_size
            say(0.15, f"OOM at {cfg.image_size} px — retrying at {size} px")

    # ── global alignment: fuse pairwise pointmaps into one frame ──
    say(0.55, f"global alignment ({cfg.niter} iters)")
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner

    mode = (GlobalAlignerMode.PointCloudOptimizer if len(images) > 2
            else GlobalAlignerMode.PairViewer)
    scene = global_aligner(output, device=device, mode=mode, verbose=False)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init="mst", niter=cfg.niter,
                                       schedule=cfg.schedule, lr=cfg.lr)

    say(0.85, "extracting geometry")
    scene = scene.clean_pointcloud()

    pts3d = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
    imgs = scene.imgs
    confs = [c.detach().cpu().numpy() for c in scene.get_conf()]
    poses_c2w = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy().ravel()

    P, C = [], []
    for pm, im, cf in zip(pts3d, imgs, confs):
        m = cf > cfg.min_conf
        if not m.any():
            continue
        p = pm[m]
        c = im[m]
        if cfg.subsample > 1:
            p, c = p[::cfg.subsample], c[::cfg.subsample]
        P.append(p.reshape(-1, 3))
        C.append(np.asarray(c).reshape(-1, 3))

    pts = np.concatenate(P) if P else np.zeros((0, 3))
    cols = np.concatenate(C) if C else np.zeros((0, 3))
    if cols.max() > 1.01:
        cols = cols / 255.0

    centres = poses_c2w[:, :3, 3]

    # world->camera, to match the convention run_incremental returns
    poses = {}
    for n in range(len(poses_c2w)):
        Rc2w = poses_c2w[n, :3, :3]
        tc2w = poses_c2w[n, :3, 3]
        R = Rc2w.T
        poses[n] = (R, -R @ tc2w)

    h, w = imgs[0].shape[:2]
    f = float(focals[0]) if len(focals) else max(h, w)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])

    del model, output, scene
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    summary = {
        "backend": "MASt3R",
        "device": device,
        "resolution": size,
        "frames_used": len(paths),
        "pairs": len(pairs),
        "cameras_registered": len(poses),
        "keyframes": len(paths),
        "points": int(len(pts)),
        "mean_confidence": round(float(np.mean([c.mean() for c in confs])), 3),
        "focal_px": round(f, 1),
        "conf_threshold": cfg.min_conf,
    }
    return {"points": pts, "colours": cols, "poses": poses,
            "camera_centres": centres, "K": K,
            "parallax": np.full(len(pts), 6.0), "summary": summary}
