#!/usr/bin/env python3
"""
singlepass-probe — end-to-end feasibility demo for SIH26158.

Answers four questions before you commit two weeks of build time:

  1. Can this GPU actually train Gaussian Splatting, and at what scene size?
  2. Does metric scaling from noisy GPS genuinely recover real-world units?
  3. How bad is single-pass parallax, and does camera tilt fix it?
  4. How much of the scene does one pass leave unobserved?

Usage:
    python run_demo.py                # everything
    python run_demo.py --skip-gpu     # geometry only (no torch needed)
    python run_demo.py --tilt 0       # see nadir degeneracy for yourself
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from probe.flight import (Flight, Intrinsics, depth_uncertainty,
                          parallax_angle_deg, project)
from probe.gps import GnssModel, scale_ambiguous_reconstruction
from probe.gpu_probe import run_full_probe
from probe.render import (CONF_COLOURS, confidence_bucket, observability,
                          render_frame)
from probe.scene import default_scene
from probe.umeyama import (alignment_error, apply_sim3, collinearity,
                           ransac_umeyama, umeyama_alignment,
                           umeyama_with_attitude)

OUT = Path(__file__).parent / "out"


def hr(title: str) -> None:
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


# ─────────────────────────────────────────────────────────────
def step_gpu(report: dict) -> None:
    hr("1 / 4   GPU CAPABILITY")
    r = run_full_probe(verbose=True)
    report["gpu_probe"] = r

    g = r.get("gpu", {})
    if not g.get("available"):
        print(f"  No GPU detected: {g.get('error')}")
        return

    print(f"\n  Device        {g['name']}")
    print(f"  VRAM          {g['vram_total_mb']} MB")
    print(f"  Driver        {g['driver']}   compute {g.get('compute_capability')}")

    th = r.get("theoretical", {})
    if th:
        print(f"\n  Theoretical Gaussian budget   ~{th['estimated_max_gaussians']:,}")
        print(f"  (at {th['bytes_per_gaussian']} B/Gaussian incl. Adam state)")

    if "matmul" in r:
        m = r["matmul"]
        print(f"\n  Matmul {m['size']}^3       {m['tflops']} TFLOPS fp32")

    if "gaussian_capacity" in r:
        c = r["gaussian_capacity"]
        print(f"\n  MEASURED capacity   {c['max_gaussians_measured']:,} Gaussians")
        if c.get("ms_per_step_at_max"):
            print(f"  Optimiser step      {c['ms_per_step_at_max']} ms at that size")
        print("\n  ladder:")
        for row in c["ladder"]:
            if row.get("oom"):
                print(f"    {row['gaussians']:>10,}  OOM")
            else:
                print(f"    {row['gaussians']:>10,}  {row['peak_mb']:>8.1f} MB   {row['ms_per_step']:>6.2f} ms/step")

    it = r.get("interpretation", {})
    if it:
        print(f"\n  VERDICT [{it['tier'].upper()}]  {it['verdict']}")
        print("\n  Recommendations:")
        for rec in it["recommendations"]:
            print(f"    - {rec}")


# ─────────────────────────────────────────────────────────────
def step_metric(report: dict, scene, flight) -> None:
    hr("2 / 4   METRIC SCALING FROM NOISY GPS")

    true_centres, rots = flight.poses()
    points, colours, labels = scene.sample_points(terrain_n=25_000, per_building=1_800)

    # What SfM actually hands you: right shape, meaningless scale.
    rec_centres, rec_points, gt = scale_ambiguous_reconstruction(true_centres, points)
    print(f"  SfM reconstruction scale factor (hidden truth):  {gt['true_scale']:.6f}")
    print(f"  -> a {scene.buildings[0].height:.1f} m building measures "
          f"{scene.buildings[0].height * gt['true_scale']:.4f} 'units' in the raw reconstruction")

    gnss = GnssModel()
    gps, baro_z, outliers = gnss.simulate(true_centres)
    print(f"\n  Simulated GNSS: sigma_h={gnss.sigma_h} m  sigma_v={gnss.sigma_v} m  "
          f"outliers={int(outliers.sum())}/{len(gps)}")

    # ── THE TRAP: a straight flight gives a nearly collinear GPS track ──
    collin = collinearity(gps)
    print(f"\n  GPS track collinearity: {collin:.4f}   (1.000 = perfectly straight line)")
    if collin > 0.98:
        print("  WARNING  near-collinear. Position-only Sim(3) cannot resolve rotation")
        print("           ABOUT the flight axis. Scale will look great and your model")
        print("           will still come out rolled. IMU attitude is required.")

    # Method A — positions only, least squares
    s_ls, R_ls, t_ls = umeyama_alignment(rec_centres, gps)
    _, rmse_ls = alignment_error(rec_centres, gps, s_ls, R_ls, t_ls)

    # Method B — positions only, RANSAC (robust to GPS multipath)
    s_rs, R_rs, t_rs, inliers = ransac_umeyama(rec_centres, gps, threshold=4.0)
    _, rmse_rs = alignment_error(rec_centres[inliers], gps[inliers], s_rs, R_rs, t_rs)

    # Method C — scale from positions, ROTATION FROM IMU ATTITUDE
    rec_rots = np.einsum("ij,njk->nik", gt["R"], rots)   # SfM-frame camera attitudes
    s_at, R_at, t_at = umeyama_with_attitude(rec_centres, gps, rec_rots, rots)
    _, rmse_at = alignment_error(rec_centres, gps, s_at, R_at, t_at)

    def scale_err(s):
        return abs((1.0 / s) - gt["true_scale"]) / gt["true_scale"] * 100

    print(f"\n  {'method':<26}{'recovered scale':>17}{'scale err %':>13}{'traj RMSE m':>13}")
    print(f"  {'-' * 69}")
    print(f"  {'A  positions (lsq)':<26}{1/s_ls:>17.6f}{scale_err(s_ls):>13.3f}{rmse_ls:>13.2f}")
    print(f"  {'B  positions (RANSAC)':<26}{1/s_rs:>17.6f}{scale_err(s_rs):>13.3f}{rmse_rs:>13.2f}")
    print(f"  {'C  + IMU attitude':<26}{1/s_at:>17.6f}{scale_err(s_at):>13.3f}{rmse_at:>13.2f}")
    print(f"  {'ground truth':<26}{gt['true_scale']:>17.6f}")

    # Measure real buildings under B and C — this is what the PS actually asks for.
    print(f"\n  Building heights (the deliverable the PS calls 'measurement'):\n")
    print(f"  {'bldg':<7}{'true m':>9}{'B: pos-only':>14}{'err %':>9}{'C: +IMU':>11}{'err %':>9}")
    print(f"  {'-' * 60}")

    pts_b = apply_sim3(rec_points, s_rs, R_rs, t_rs)
    pts_c = apply_sim3(rec_points, s_at, R_at, t_at)
    errs_b, errs_c = [], []
    for bi, b in enumerate(scene.buildings[:6], start=1):
        m = labels == bi
        if m.sum() < 50:
            continue
        hb = float(np.ptp(pts_b[m][:, 2]))
        hc = float(np.ptp(pts_c[m][:, 2]))
        eb = abs(hb - b.height) / b.height * 100
        ec = abs(hc - b.height) / b.height * 100
        errs_b.append(eb); errs_c.append(ec)
        print(f"  {bi:<7}{b.height:>9.2f}{hb:>14.2f}{eb:>9.1f}{hc:>11.2f}{ec:>9.1f}")

    mb = float(np.mean(errs_b)) if errs_b else float("nan")
    mc = float(np.mean(errs_c)) if errs_c else float("nan")
    print(f"\n  mean height error   positions-only {mb:5.1f} %     with IMU {mc:5.1f} %")
    print(f"\n  THE LESSON: scale looked fine in every method — but only attitude fusion")
    print(f"  gives correct MEASUREMENTS. On a straight flight, GPS alone is not enough.")
    print("  Absolute position stays GPS-limited (metres); RELATIVE measurement is far")
    print("  better because scale error is a single global multiplier. Report both.")

    report["metric"] = {
        "true_scale": gt["true_scale"],
        "gps_collinearity": collin,
        "scale_error_pct_lsq": scale_err(s_ls),
        "scale_error_pct_ransac": scale_err(s_rs),
        "scale_error_pct_attitude": scale_err(s_at),
        "traj_rmse_m_ransac": rmse_rs,
        "traj_rmse_m_attitude": rmse_at,
        "mean_height_error_pct_positions_only": mb,
        "mean_height_error_pct_with_imu": mc,
        "gnss_outliers": int(outliers.sum()),
    }


# ─────────────────────────────────────────────────────────────
def step_parallax(report: dict, scene, flight) -> None:
    hr("3 / 4   SINGLE-PASS PARALLAX  (why camera tilt decides everything)")

    intr = Intrinsics()
    centres, _ = flight.poses()
    depth = flight.altitude

    print(f"  Altitude {flight.altitude:.0f} m   fx = {intr.fx:.1f} px   "
          f"disparity precision 0.5 px\n")
    print(f"  {'frame gap':>10}{'baseline m':>13}{'B/Z':>9}{'angle deg':>12}{'depth sigma m':>16}")
    print(f"  {'-' * 60}")

    rows = []
    for gap in (1, 2, 5, 10, 20, 40):
        if gap >= len(centres):
            continue
        B = float(np.linalg.norm(centres[gap] - centres[0]))
        ang = parallax_angle_deg(depth, B)
        sig = depth_uncertainty(depth, B, intr.fx)
        rows.append({"gap": gap, "baseline_m": B, "angle_deg": ang, "sigma_z_m": sig})
        print(f"  {gap:>10}{B:>13.2f}{B / depth:>9.3f}{ang:>12.2f}{sig:>16.3f}")

    print("\n  READ THIS: adjacent frames give a ~1 deg intersection angle and metres of")
    print("  depth uncertainty. Wide frame gaps are not optional — they are the")
    print("  difference between a model and noise. Match frame i to i+20, not i+1.")

    # Tilt comparison — how much of the scene is even in view
    points, _, labels = scene.sample_points(terrain_n=20_000, per_building=1_500)
    facade = labels > 0
    print(f"\n  {'tilt deg':>10}{'pts in view':>14}{'facade pts':>13}{'facade %':>11}")
    print(f"  {'-' * 48}")
    tilt_rows = []
    for tilt in (0, 15, 30, 40, 55):
        f = Flight(altitude=flight.altitude, length=flight.length,
                   n_frames=flight.n_frames, tilt_deg=tilt, y_offset=flight.y_offset)
        cs, rs = f.poses()
        seen = np.zeros(len(points), bool)
        for c, R in zip(cs, rs):
            _, _, vis = project(points, c, R, intr)
            seen |= vis
        nf = int((seen & facade).sum())
        pct = nf / max(int(seen.sum()), 1) * 100
        tilt_rows.append({"tilt": tilt, "in_view": int(seen.sum()),
                          "facade_pts": nf, "facade_pct": pct})
        print(f"  {tilt:>10}{int(seen.sum()):>14,}{nf:>13,}{pct:>11.1f}")

    print("\n  Nadir (0 deg) is geometrically degenerate for forward motion AND sees")
    print("  almost no facades. The PS explicitly asks for 'building facades and")
    print("  rooftops' — so oblique capture is a requirement, not a preference.")

    report["parallax"] = {"baseline_table": rows, "tilt_table": tilt_rows}


# ─────────────────────────────────────────────────────────────
def step_coverage(report: dict, scene, flight) -> None:
    hr("4 / 4   OBSERVABILITY  (what one pass can never see)")

    intr = Intrinsics()
    centres, rots = flight.poses()
    points, colours, labels = scene.sample_points(terrain_n=18_000, per_building=1_400)

    t0 = time.perf_counter()
    views = observability(points, centres, rots, intr)
    dt = time.perf_counter() - t0
    buckets = confidence_bucket(views)

    n = len(points)
    counts = {int(b): int((buckets == b).sum()) for b in range(4)}
    print(f"  {n:,} surface points, {len(centres)} frames  ({dt:.1f}s)\n")
    print(f"  {'confidence':<28}{'points':>12}{'share':>10}")
    print(f"  {'-' * 50}")
    for b, lbl in [(3, "3+ views  well triangulated"),
                   (2, "2 views   weak"),
                   (1, "1 view    monocular guess"),
                   (0, "0 views   NEVER OBSERVED")]:
        print(f"  {lbl:<28}{counts[b]:>12,}{counts[b] / n * 100:>9.1f}%")

    unobserved_pct = counts[0] / n * 100
    print(f"\n  {unobserved_pct:.1f}% of this scene was never seen by any camera.")
    print("  You cannot reconstruct that. The PS asks for 'reconstruction of occluded")
    print("  surfaces' — the honest engineering answer is to LABEL it, not invent it.")
    print("  A model that silently fabricates a wall it never saw is worse than one")
    print("  with a visible hole, for a recon or disaster analyst.")

    report["coverage"] = {
        "n_points": n, "buckets": counts,
        "unobserved_pct": unobserved_pct,
        "seconds": round(dt, 2),
    }

    # Save artefacts
    OUT.mkdir(exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rgb, _, _ = render_frame(points, colours, centres[len(centres) // 2],
                                 rots[len(rots) // 2], intr)
        conf_cols = np.array([CONF_COLOURS[int(b)] for b in buckets])
        rgb_conf, _, _ = render_frame(points, conf_cols, centres[len(centres) // 2],
                                      rots[len(rots) // 2], intr)

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].imshow(np.clip(rgb, 0, 1)); ax[0].set_title("simulated drone frame"); ax[0].axis("off")
        ax[1].imshow(np.clip(rgb_conf, 0, 1))
        ax[1].set_title("observability  (green=3+  amber=2  orange=1  red=never)")
        ax[1].axis("off")
        fig.tight_layout()
        p = OUT / "coverage.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        print(f"\n  wrote {p}")
        report["coverage"]["figure"] = str(p)
    except ImportError:
        print("\n  (matplotlib not installed — skipping figure)")


# ─────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="SIH26158 feasibility probe")
    ap.add_argument("--skip-gpu", action="store_true", help="geometry only, no torch")
    ap.add_argument("--tilt", type=float, default=40.0, help="camera tilt deg (0 = nadir)")
    ap.add_argument("--altitude", type=float, default=90.0)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    print("\n  singlepass-probe   SIH26158 feasibility harness")
    print("  single-pass drone video -> georeferenced metric 3D\n")

    report: dict = {"config": vars(args)}

    scene = default_scene()
    flight = Flight(altitude=args.altitude, n_frames=args.frames, tilt_deg=args.tilt)

    if not args.skip_gpu:
        step_gpu(report)
    step_metric(report, scene, flight)
    step_parallax(report, scene, flight)
    step_coverage(report, scene, flight)

    OUT.mkdir(exist_ok=True)
    rp = OUT / "report.json"
    rp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    hr("SUMMARY")
    gp = report.get("gpu_probe", {})
    if gp.get("interpretation"):
        print(f"  GPU        {gp['interpretation']['tier'].upper()} — {gp['interpretation']['verdict']}")
    m = report.get("metric", {})
    if m:
        print(f"  Metric     scale recovered to {m['scale_error_pct_ransac']:.2f}% "
              f"(RANSAC), mean height error {m['mean_relative_height_error_pct']:.2f}%")
    c = report.get("coverage", {})
    if c:
        print(f"  Coverage   {c['unobserved_pct']:.1f}% of scene never observed in one pass")
    print(f"\n  full report -> {rp}\n")


if __name__ == "__main__":
    main()
