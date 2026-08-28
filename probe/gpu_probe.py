"""
GPU capability probe for a 3D-reconstruction workload.

Generic benchmarks (matmul TFLOPS) do not tell you whether you can train Gaussian
Splatting. What matters is: how many Gaussians fit in VRAM *with their gradients and
Adam optimiser state*, and how fast one optimisation step runs.

This measures exactly that, by allocating a real 3DGS-shaped parameter set and
running real optimiser steps until it either succeeds or runs out of memory.

Per-Gaussian parameters in standard 3DGS:
    position        3
    scale           3
    rotation (quat) 4
    opacity         1
    SH colour       48   (degree 3: 3 channels x 16 coeffs)
    -------------------
    total          59 floats = 236 bytes at fp32

Adam keeps 2 extra moments + gradients, so the practical cost is roughly 4x that
(~944 B/Gaussian) before any rasteriser workspace.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time

FLOATS_PER_GAUSSIAN = 59
BYTES_PER_GAUSSIAN_FP32 = FLOATS_PER_GAUSSIAN * 4
ADAM_MULTIPLIER = 4  # params + grads + exp_avg + exp_avg_sq


def nvidia_smi_info() -> dict:
    """GPU facts without needing torch installed."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode != 0:
            return {"available": False, "error": out.stderr.strip()[:200]}
        name, total, used, driver, cc = [s.strip() for s in out.stdout.strip().split(",")]
        return {
            "available": True,
            "name": name,
            "vram_total_mb": int(float(total)),
            "vram_used_mb": int(float(used)),
            "driver": driver,
            "compute_capability": cc,
        }
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)[:200]}


def theoretical_gaussian_budget(vram_mb: int, headroom_frac: float = 0.72) -> dict:
    """How many Gaussians *should* fit, before measuring."""
    usable = vram_mb * 1e6 * headroom_frac
    per_g = BYTES_PER_GAUSSIAN_FP32 * ADAM_MULTIPLIER
    return {
        "usable_bytes": int(usable),
        "bytes_per_gaussian": per_g,
        "estimated_max_gaussians": int(usable / per_g),
    }


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def benchmark_matmul(device: str = "cuda", size: int = 4096, iters: int = 30) -> dict:
    """Proxy for transformer inference throughput (MASt3R / DUSt3R / VGGT)."""
    import torch

    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    for _ in range(5):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    flops = 2 * size ** 3 * iters
    del a, b
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"size": size, "iters": iters, "seconds": round(dt, 4),
            "tflops": round(flops / dt / 1e12, 2)}


def benchmark_gaussian_capacity(
    device: str = "cuda",
    start: int = 200_000,
    step_mult: float = 1.6,
    max_try: int = 12,
    opt_steps: int = 3,
) -> dict:
    """Allocate a real 3DGS-shaped parameter set + Adam, step it, grow until OOM.

    This is the number that actually decides whether your scene will train.
    """
    import torch

    results = []
    n = start
    best = 0
    best_step_ms = None

    for _ in range(max_try):
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            means = torch.randn(n, 3, device=device, requires_grad=True)
            scales = torch.randn(n, 3, device=device, requires_grad=True)
            quats = torch.randn(n, 4, device=device, requires_grad=True)
            opac = torch.randn(n, 1, device=device, requires_grad=True)
            sh = torch.randn(n, 16, 3, device=device, requires_grad=True)

            params = [means, scales, quats, opac, sh]
            opt = torch.optim.Adam(params, lr=1e-3)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(opt_steps):
                opt.zero_grad(set_to_none=True)
                # Stand-in for the rasteriser loss: touches every parameter.
                loss = (
                    means.pow(2).mean()
                    + scales.pow(2).mean()
                    + quats.pow(2).mean()
                    + opac.pow(2).mean()
                    + sh.pow(2).mean()
                )
                loss.backward()
                opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) / opt_steps * 1000

            peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0
            results.append({"gaussians": n, "peak_mb": round(peak_mb, 1),
                            "ms_per_step": round(step_ms, 2)})
            best, best_step_ms = n, step_ms

            del means, scales, quats, opac, sh, params, opt, loss
            if device == "cuda":
                torch.cuda.empty_cache()

            n = int(n * step_mult)

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:  # type: ignore[attr-defined]
            if "out of memory" not in str(e).lower():
                raise
            results.append({"gaussians": n, "oom": True})
            if device == "cuda":
                torch.cuda.empty_cache()
            break

    return {
        "max_gaussians_measured": best,
        "ms_per_step_at_max": round(best_step_ms, 2) if best_step_ms else None,
        "ladder": results,
    }


def interpret(vram_mb: int, max_gaussians: int) -> dict:
    """Translate raw numbers into scoping decisions."""
    # Rough field guidance: a small object/building scene converges around 0.3-1M
    # Gaussians; a full city block wants several million.
    if max_gaussians >= 3_000_000:
        tier = "comfortable"
        verdict = "Full-scene 3DGS at high resolution is realistic."
    elif max_gaussians >= 1_200_000:
        tier = "workable"
        verdict = "Single building or small block at 960-1280 px. Cap Gaussian count."
    elif max_gaussians >= 500_000:
        tier = "tight"
        verdict = "Scope to ONE structure at 640-960 px. Cap densification hard."
    else:
        tier = "constrained"
        verdict = "Use feed-forward reconstruction (MASt3R) as primary; 3DGS only for a small demo patch."

    recs = []
    if vram_mb < 8000:
        recs += [
            "Downscale frames to 640-960 px before reconstruction (do NOT feed 4K).",
            "Cap Gaussians explicitly (gsplat: --max-gauss / strategy cap) to avoid mid-run OOM.",
            "Process the flight in overlapping chunks of 20-30 frames, then merge.",
            "Prefer gsplat over the reference 3DGS implementation — lower memory.",
            "Keep Colab/Kaggle (free T4, 16 GB) as the fallback for the final high-quality run.",
            "Use fp16/bf16 autocast for MASt3R inference.",
        ]
    else:
        recs += [
            "Frames at 1280 px are affordable.",
            "Standard densification settings are usable.",
        ]
    recs.append("Run MASt3R pairwise in batches; it is inference-only and cheaper than 3DGS training.")

    return {"tier": tier, "verdict": verdict, "recommendations": recs}


def run_full_probe(verbose: bool = True) -> dict:
    info = nvidia_smi_info()
    report: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": info,
    }

    if not info.get("available"):
        report["status"] = "no_gpu"
        if verbose:
            print("  No NVIDIA GPU detected via nvidia-smi.")
        return report

    vram = info["vram_total_mb"]
    report["theoretical"] = theoretical_gaussian_budget(vram)

    if not torch_available():
        report["status"] = "no_torch"
        report["note"] = "Install torch (CUDA build) to measure real capacity."
        if verbose:
            print("  torch not installed — reporting theoretical budget only.")
        est = report["theoretical"]["estimated_max_gaussians"]
        report["interpretation"] = interpret(vram, est)
        return report

    import torch

    report["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }

    if not torch.cuda.is_available():
        report["status"] = "torch_cpu_only"
        report["note"] = "torch is installed but CPU-only. Reinstall the CUDA build."
        est = report["theoretical"]["estimated_max_gaussians"]
        report["interpretation"] = interpret(vram, est)
        return report

    if verbose:
        print("  benchmarking matmul throughput ...")
    report["matmul"] = benchmark_matmul()

    if verbose:
        print("  measuring Gaussian capacity (this allocates until OOM) ...")
    cap = benchmark_gaussian_capacity()
    report["gaussian_capacity"] = cap
    report["interpretation"] = interpret(vram, cap["max_gaussians_measured"])
    report["status"] = "ok"
    return report


if __name__ == "__main__":
    print(json.dumps(run_full_probe(), indent=2))
