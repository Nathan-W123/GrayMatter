"""Experiments. Every number reported in the README is produced here and saved in results/.

Worlds: every multi-draw experiment runs the hidden process in a named world
(see configs/default.yaml). ``matched`` is the tracker's own model; ``realistic``
adds a foam pad, force-control errors (calibration gain and line-to-line
ripple), two-stage abrasive wear and scanner artefacts, none of which the
estimators model; ``only_*`` worlds add one group of those at a time. Truth
parameters, wear fluctuations, force errors and scan noise come from the same
random streams in every world, so worlds are compared on paired draws.

Random streams (seed, stream, draw): 1 truth parameters, 2 wear fluctuations,
3 scans, 4 particle filter, 5 filter rollouts, 6 re-seeded filter (control),
7 force-calibration gain, 8 force ripple, 9 model-description checks.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import __version__
from .config import Context, config_hash, rng_for
from .estimators import CalibrateOnce, Nominal, ParticleFilter, RefitEachPass
from .geometry import Panel
from .pad import ContactGeometry, contact_fraction_sweep
from .process import HiddenTruth, WearLaw, line_effectiveness, simulate_truth
from .scan import Scanner
from .tracker2 import C2Config, DiscrepancyTracker

UM = 1e3  # mm -> micrometres
PARAMS = (("k_pad", "true_k_pad"), ("lam", "true_lam"), ("K", "true_K"), ("K0", "true_K0"))
ESTS = ("A", "B", "C", "D", "oracle")


# --------------------------------------------------------------------------- helpers


def draw_truth(ctx: Context, draw: int, world: str = "matched") -> HiddenTruth:
    """Hidden parameters of a draw. The force-calibration gain uses its own stream
    (7), so (k_pad, K0, lambda) are identical in every world."""
    p = ctx.priors.sample(rng_for(ctx.cfg["seed"], 1, draw))
    w, _ = ctx.world(world)
    gain = float(np.exp(w.force_gain_sigma * rng_for(ctx.cfg["seed"], 7, draw).standard_normal()))
    return HiddenTruth(float(p["k_pad"]), float(p["K0"]), float(p["lam"]), gain)


def make_truth(ctx: Context, draw: int, schedule_kind: str = "alternating", world: str = "matched"):
    """Hidden-truth trajectory. The same wear-noise stream is used for a draw under
    every schedule and world, so those comparisons are paired."""
    w, _ = ctx.world(world)
    return simulate_truth(ctx.model, draw_truth(ctx, draw, world), ctx.schedule(schedule_kind), ctx.wear_sigma,
                          rng_for(ctx.cfg["seed"], 2, draw), ctx.threshold, world=w,
                          ripple_rng=rng_for(ctx.cfg["seed"], 8, draw))


def rmse_um(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)) * UM)


def rel_width(row: dict, name: str) -> float:
    return (row[f"C_{name}_hi"] - row[f"C_{name}_lo"]) / row[f"C_{name}_median"]


def inside(row: dict, name: str, true_key: str) -> bool:
    return row[f"C_{name}_lo"] <= row[true_key] <= row[f"C_{name}_hi"]


def _outside(row: dict, name: str, true_key: str) -> float:
    """Relative distance of the truth outside C's 90% interval (0 if inside)."""
    t, lo, hi = row[true_key], row[f"C_{name}_lo"], row[f"C_{name}_hi"]
    return float(max(lo - t, t - hi, 0.0) / t)


def q(x, p) -> float:
    return float(np.percentile(np.asarray(x, dtype=float), p))


def iqr_summary(x) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"median": None, "q25": None, "q75": None, "n": 0, "min": None, "max": None}
    return {"median": q(x, 50), "q25": q(x, 25), "q75": q(x, 75), "n": int(x.size),
            "min": float(x.min()), "max": float(x.max())}


def _gmean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x)))) if x.size and np.all(x > 0) else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")   # nothing to report (e.g. a tiny configuration); keep the file
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def by_draw(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(int(r["draw"]), []).append(r)
    return {d: sorted(rr, key=lambda r: r["pass"]) for d, rr in sorted(out.items())}


# --------------------------------------------------------------------------- one episode


def regions(ctx: Context, blocks: tuple[int, int] = (4, 4)):
    """Scan-point indices of a blocks[0] x blocks[1] grid of regions over the scan, and the
    mean of the surrogate tables over each region (for map-level calibration checks)."""
    if "_regions" not in ctx.__dict__:
        ny, nx = ctx.obs_shape
        iy = np.minimum(np.arange(ny) * blocks[0] // ny, blocks[0] - 1)
        ix = np.minimum(np.arange(nx) * blocks[1] // nx, blocks[1] - 1)
        lab = (iy[:, None] * blocks[1] + ix[None, :]).ravel()
        index = [np.flatnonzero(lab == r) for r in range(blocks[0] * blocks[1])]
        tables = np.array([ctx.table.X[ix_].mean(axis=0) for ix_ in index])
        ctx.__dict__["_regions"] = (index, tables)
    return ctx.__dict__["_regions"]


def c2_config(cfg: dict) -> C2Config:
    """Tracker C2's settings from the ``c2`` section of the config (defaults otherwise)."""
    c = cfg.get("c2", {}) or {}
    return C2Config(**{k: (int(v) if k in ("draws", "learn_from_pass") else float(v)) for k, v in c.items()})


def d_obs_sd(pf) -> float:
    """Noise of D's log K_i in its wear fit: C's prior median wear-fluctuation and transient levels."""
    gm = lambda r: float(np.sqrt(r[0] * r[1])) if r[0] > 0 else float(r[1]) / 2
    return max(float(np.hypot(gm(pf.wear_noise_range), gm(pf.transient_range))), 1e-3)


def run_episode(ctx: Context, draw: int, world: str = "matched", schedule_kind: str = "alternating",
                noise_um: float | None = None, experiment: str = "", record_maps: tuple[int, ...] = (),
                traj=None, full: bool = True, pf_stream: int = 4, with_D: bool | None = None,
                risk: bool = True, with_B: bool | None = None, with_C2: bool = False) -> tuple[list[dict], dict]:
    """Run all estimators through one sequence of passes on one hidden truth.

    Each pass: (1) every estimator predicts the removal map of the coming pass
    at the scan points from the commanded action (C also gives a 90%
    predictive interval of its mean), (2) the hidden process executes it, (3)
    the scanner observes it, (4) estimators update on the scan.

    ``full=False`` (used for the extra noise levels, the ablation and the
    tuning runs, which only report on C) skips baselines B and D and the
    abrasive-change predictions.
    ``pf_stream`` selects the particle filter's random stream (a different value
    re-runs the identical problem with a different Monte Carlo seed).
    ``with_D`` runs baseline D even when ``full`` is False (tuning of its decision margin);
    ``risk=False`` skips C's second crossing quantile (only the decision rules use it);
    ``with_B=False`` skips baseline B (not reported for the single-effect and stress worlds);
    ``with_C2`` adds tracker C2 (C plus a discrepancy layer; C itself is unaffected).
    """
    cfg = ctx.cfg
    seed = cfg["seed"]
    noise = float(cfg["scan"]["noise_um"] if noise_um is None else noise_um)
    schedule = ctx.schedule(schedule_kind)
    if traj is None:
        traj = make_truth(ctx, draw, schedule_kind, world)
    truth = traj.truth
    _, artefacts = ctx.world(world)
    scanner = Scanner(ctx.panel, noise, cfg["scan"]["stride"], artefacts)
    scan_rng = rng_for(seed, 3, draw)
    oi = ctx.obs_index
    nominal_cache = ctx.__dict__.setdefault("_nominal_cache", {})
    A = Nominal(ctx.model, ctx.priors, oi, nominal_cache)
    use_B = full if with_B is None else with_B
    B = CalibrateOnce(ctx.model, ctx.priors, ctx.table, oi, nominal_cache) if use_B else None
    use_D = full if with_D is None else with_D
    D = RefitEachPass(ctx.priors, ctx.table, force_ref=float(cfg["pad"]["reference_force_N"]),
                      force_exponent_sd=ctx.pf.force_exponent_sd, obs_sd=d_obs_sd(ctx.pf)) if use_D else None
    C = ParticleFilter(ctx.table, ctx.priors, ctx.pf, rng_for(seed, pf_stream, draw),
                       rollout_rng=rng_for(seed, 5, draw), obs_shape=ctx.obs_shape)
    cross_A = A.crossing_pass(schedule, ctx.threshold)
    region_index, region_tables = regions(ctx)
    C2 = DiscrepancyTracker(C, region_index, region_tables, ctx.obs_shape, c2_config(cfg),
                            rng_for(seed, 12, draw)) if with_C2 else None
    rows, maps = [], {}
    nan = float("nan")
    for n, action in enumerate(schedule):
        if n > 0:
            prev = schedule[n - 1]
            for est in (A, B, C, D):
                if est is not None:
                    est.advance(prev)
        true_obs = traj.removal[n][oi]
        # oracle: true physics and the true state after the previous pass, but not the
        # wear fluctuation drawn after it (pass 1: the true initial state)
        preds = {"A": A.predict(action), "B": B.predict(action) if use_B else None, "C": C.predict(action),
                 "D": D.predict(action) if use_D else None, "oracle": traj.oracle[n][oi]}
        p_lo, p_med, p_hi = C.predict_mean_removal(action)
        true_mean = float(true_obs.mean())
        if C2 is not None:
            C2.prepare(action)
            preds["C2"] = C2.predict_map(preds["C"])
            c2_lo, _, c2_hi = C2.predict_mean_removal()
            c2_reg = C2.predict_region_removal()
        reg_q = C.predict_region_removal(action, region_tables)
        reg_true = np.array([true_obs[ix].mean() for ix in region_index])
        reg_inside = float(np.mean((reg_q[:, 0] <= reg_true) & (reg_true <= reg_q[:, 2])))
        scan = scanner.observe(traj.removal[n], scan_rng)
        A.update(scan, action)
        if use_B:
            B.update(scan, action)
        if use_D:
            D.update(scan, action)
        C.update(scan, action, noise)
        if C2 is not None:
            C2.update(scan, noise)
        s = C.summary()
        cr = (C.predict_crossing(n + 1, schedule, ctx.threshold) if full
              else {"median": nan, "lo": nan, "hi": nan, "band_mass": nan})
        q_risk = ctx.cfg["abrasive"].get("risk_quantile", 0.25)
        cr_risk = (C.predict_crossing(n + 1, schedule, ctx.threshold, q=(q_risk, 1 - q_risk))["lo"]
                   if full and risk else nan)
        row = {
            "experiment": experiment, "world": world, "schedule": schedule_kind, "noise_um": noise, "draw": draw,
            "pass": n + 1, "force_N": action.force, "rpm": action.rpm,
            "true_k_pad": truth.k_pad, "true_K0": truth.K0, "true_lam": truth.lam, "true_K": float(traj.K[n]),
            "true_force_gain": truth.force_gain,
            "true_mean_removal_um": true_mean * UM, "scan_mean_um": float(np.nanmean(scan)) * UM,
            "rmse_A_um": rmse_um(preds["A"], true_obs),
            "rmse_B_um": rmse_um(preds["B"], true_obs) if use_B else nan,
            "rmse_C_um": rmse_um(preds["C"], true_obs),
            "rmse_D_um": rmse_um(preds["D"], true_obs) if use_D else nan,
            "rmse_oracle_um": rmse_um(preds["oracle"], true_obs),
            "C_pred_mean_lo_um": float(p_lo) * UM, "C_pred_mean_median_um": float(p_med) * UM,
            "C_pred_mean_hi_um": float(p_hi) * UM, "C_pred_mean_inside": bool(p_lo <= true_mean <= p_hi),
            "C_region_inside_fraction": reg_inside,
        }
        for name in ("k_pad", "K0", "lam", "K"):
            for stat in ("mean", "median", "lo", "hi"):
                row[f"C_{name}_{stat}"] = s[name][stat]
        row.update({
            "C_corr_logk_logK": s["corr_logk_logK"], "C_stages": s["stages"],
            "C_sigma_um": s["sigma_um"], "C_profile_sd_um": s["profile_sd_um"], "C_wear_noise": s["wear_noise"], "C_transient_noise": s["transient_noise"], "C_min_path_diversity": s["min_path_diversity"], "C_outliers": s["outliers"], "C_redone": bool(s["redone"]),
            "B_k_pad": B.k_pad if use_B else nan, "B_K": B.K if use_B else nan,
            "D_k_pad": D.k_pad if use_D else nan, "D_K": D.K if use_D else nan, "D_lam": D.lam if use_D else nan,
            "D_next_ratio": D.next_ratio() if use_D else nan,
            "cross_C_median": cr["median"], "cross_C_lo": cr["lo"], "cross_C_hi": cr["hi"],
            "cross_C_band_mass": cr["band_mass"], "cross_C_risk": cr_risk,
            "cross_D": D.crossing_pass(n + 1, schedule, ctx.threshold) if use_D else nan,
            "cross_A": cross_A, "cross_true": traj.crossing_pass,
        })
        if C2 is not None:
            reg_true_c2 = reg_true
            row.update({
                "rmse_C2_um": rmse_um(preds["C2"], true_obs),
                "C2_pred_mean_lo_um": float(c2_lo) * UM, "C2_pred_mean_hi_um": float(c2_hi) * UM,
                "C2_pred_mean_inside": bool(c2_lo <= true_mean <= c2_hi),
                "C2_region_inside_fraction": float(np.mean((c2_reg[:, 0] <= reg_true_c2)
                                                           & (reg_true_c2 <= c2_reg[:, 2]))),
                **{f"C2_{k}": v for k, v in C2.summary().items()},
            })
        rows.append(row)
        if n + 1 in record_maps:
            maps[n + 1] = {"true": true_obs, "scan": scan,
                           **{k: v for k, v in preds.items() if v is not None}}
    return rows, {"maps": maps, "traj": traj}


# --------------------------------------------------------------------------- model description


def experiment_contact_sweep(ctx: Context, out: Path, verbose: bool = True) -> dict:
    cfg = ctx.cfg
    kmin, kmax, nk = cfg["experiments"]["contact_sweep_k"]
    ks = np.logspace(np.log10(kmin), np.log10(kmax), int(nk))
    F_ref = cfg["pad"]["reference_force_N"]
    forces = sorted(set(cfg["schedule"]["forces_N"]) | {F_ref})
    rows = []
    pr = ctx.priors
    for F in forces:
        cf = contact_fraction_sweep(ctx.panel, ctx.pad_radius, F, ks)
        rows += [{"force_N": F, "k_pad": float(k), "contact_fraction": float(c)} for k, c in zip(ks, cf)]
    # grid-convergence reference: the same sweep on a 4x finer grid at the reference force
    fine_grid = ctx.panel.grid / 4
    fine = Panel(ctx.panel.length_x, ctx.panel.width_y, ctx.panel.radius, fine_grid)
    cf_fine = contact_fraction_sweep(fine, ctx.pad_radius, F_ref, ks)
    rows += [{"force_N": F_ref, "k_pad": float(k), "contact_fraction": float(c), "grid_mm": fine_grid}
             for k, c in zip(ks, cf_fine)]
    # the realistic world's foam pad (same small-strain stiffness, stiffens as it densifies)
    foam = next((w.pad_law for w, _ in ctx.worlds.values() if w.pad_law.kind == "foam"), None)
    foam_bounds = None
    if foam is not None:
        cf_foam = contact_fraction_sweep(ctx.panel, ctx.pad_radius, F_ref, ks, foam)
        rows += [{"force_N": F_ref, "k_pad": float(k), "contact_fraction": float(c), "pad_law": "foam"}
                 for k, c in zip(ks, cf_foam)]
        lo, hi = contact_fraction_sweep(ctx.panel, ctx.pad_radius, F_ref, np.array([pr.k_low, pr.k_high]), foam)
        foam_bounds = {"thickness_mm": foam.thickness, "soft_k_low": float(lo), "stiff_k_high": float(hi),
                       "force_N": F_ref}
    for r in rows:
        r.setdefault("grid_mm", ctx.panel.grid)
        r.setdefault("pad_law", "linear")
    fine_bounds = contact_fraction_sweep(fine, ctx.pad_radius, F_ref, np.array([pr.k_low, pr.k_high]))
    at_bounds = {}
    for F in forces:
        lo, hi = contact_fraction_sweep(ctx.panel, ctx.pad_radius, F, np.array([pr.k_low, pr.k_high]))
        at_bounds[f"{F:g}N"] = {"soft_k_low": float(lo), "stiff_k_high": float(hi)}
    summary = {"k_low": pr.k_low, "k_high": pr.k_high, "reference_force_N": F_ref,
               "grid_mm": ctx.panel.grid, "radius_mm": ctx.panel.radius,
               "contact_fraction_at_prior_bounds": at_bounds,
               "fine_grid_reference": {"grid_mm": fine_grid, "soft_k_low": float(fine_bounds[0]),
                                       "stiff_k_high": float(fine_bounds[1]), "force_N": F_ref},
               "foam_pad_at_prior_bounds": foam_bounds,
               "pad_face_area_mm2": float(np.pi * ctx.pad_radius**2)}
    # pressure profile across the cylinder (through the pad centre) for stiff / mid / soft pads
    centre = np.array([[round(ctx.panel.length_x / 2 / ctx.panel.grid) * ctx.panel.grid,
                        round(ctx.panel.width_y / 2 / ctx.panel.grid) * ctx.panel.grid]])
    cg = ContactGeometry(ctx.panel, centre, ctx.pad_radius)
    xs = ctx.panel.X.ravel()[cg.idx]
    ys = ctx.panel.Y.ravel()[cg.idx]
    on_line = np.isclose(xs, centre[0, 0])
    prof = []
    k_mid = float(np.sqrt(pr.k_low * pr.k_high))
    for label, k in (("stiff", pr.k_high), ("mid", k_mid), ("soft", pr.k_low)):
        p = cg.pressure(F_ref, k)
        cf = float(cg.contact_fraction(F_ref, k)[0])
        order = np.argsort(ys[on_line])
        for yy, pp in zip(ys[on_line][order], p[on_line][order]):
            prof.append({"pad": label, "k_pad": float(k), "contact_fraction": cf,
                         "offset_mm": float(yy - centre[0, 1]), "pressure_kPa": float(pp * 1e3)})
    write_csv(out / "contact_profiles.csv", prof)
    write_csv(out / "contact_sweep.csv", rows)
    write_json(out / "contact_sweep.json", summary)
    if verbose:
        print("\nContact fraction vs pad stiffness (pad centred on the panel)")
        print("  k_pad [N/mm^3] " + "".join(f"  F={F:>4g} N" for F in forces))
        for k in np.logspace(np.log10(kmin), np.log10(kmax), 15):
            vals = [contact_fraction_sweep(ctx.panel, ctx.pad_radius, F, np.array([k]))[0] for F in forces]
            print(f"  {k:12.4g}   " + "".join(f"  {v:9.1%}" for v in vals))
        b = at_bounds[f"{F_ref:g}N"]
        print(f"  prior range k in [{pr.k_low:g}, {pr.k_high:g}] -> contact {b['stiff_k_high']:.1%} "
              f"(stiff) to {b['soft_k_low']:.1%} (soft) at {F_ref:g} N on the {ctx.panel.grid:g} mm grid; "
              f"{fine_bounds[1]:.1%} to {fine_bounds[0]:.1%} on a {fine_grid:g} mm grid")
    return summary



def experiment_model_info(ctx: Context, out: Path, verbose: bool = True) -> dict:
    """Derived constants of the default model quoted in the README (geometry, kinematics,
    the near-independence of removed volume from stiffness, surrogate accuracy, and the
    size of each mismatch of the realistic world)."""
    cfg, pr, m = ctx.cfg, ctx.priors, ctx.model
    a = ctx.pad_radius
    R = ctx.panel.radius
    F_ref = float(cfg["pad"]["reference_force_N"])
    forces_all = sorted(set(cfg["schedule"]["forces_N"]) | {cfg["schedule"]["constant_force_N"]})
    ks = np.geomspace(pr.k_low, pr.k_high, 9)
    vol = np.array([m.volume(m.exposure(F_ref, k)) for k in ks])          # per unit K
    # surrogate vs the exact forward model (linear pad, within-pass wear), at the scan points
    rng = rng_for(cfg["seed"], 9)
    oi = ctx.obs_index
    errs, zmax = [], 0.0
    for _ in range(30):
        k = float(np.exp(rng.uniform(np.log(pr.k_low), np.log(pr.k_high))))
        K = float(pr.K0_median * np.exp(pr.K0_sigma_log * rng.standard_normal()) * rng.uniform(0.4, 1.0))
        lam = float(pr.lam_median * np.exp(pr.lam_sigma_log * rng.standard_normal()))
        for F in forces_all:
            E, u = m.line_exposures(F, k)
            exact = line_effectiveness(K, lam, u) @ E[:, oi]
            approx = ctx.table.map_obs(F / k, K * F, lam * K * F)
            errs.append(float(np.sqrt(np.mean((exact - approx) ** 2)) / np.sqrt(np.mean(exact**2))))
            zmax = max(zmax, lam * K * float(u.sum()))
    nom = pr.log_mean()
    centre = np.array([[round(ctx.panel.length_x / 2 / ctx.panel.grid) * ctx.panel.grid,
                        round(ctx.panel.width_y / 2 / ctx.panel.grid) * ctx.panel.grid]])
    cg = ContactGeometry(ctx.panel, centre, a)
    penetration = {f"{F:g}N": {"soft_k_low_mm": float(cg.penetration(F / pr.k_low)[0]),
                               "stiff_k_high_mm": float(cg.penetration(F / pr.k_high)[0])}
                   for F in forces_all}
    fresh = {f"{F:g}N": float(nom["K0"] * m.exposure(F, nom["k_pad"])[oi].mean() * UM) for F in forces_all}
    info = {
        "grid_nodes": int(ctx.panel.n_pix), "scan_points": int(len(oi)),
        "scan_grid_mm": float(ctx.panel.grid * cfg["scan"]["stride"]),
        "panel_sag_mm": float(ctx.panel.Z.max() - ctx.panel.Z.min()),
        "pad_rim_gap_mm": float(R - np.sqrt(R**2 - a**2)) if R else 0.0,
        "raster_lines": int(ctx.path.line.max() + 1), "stations_per_pass": int(ctx.path.n_stations),
        "actual_stepover_mm": ctx.path.stepover, "pass_duration_s": ctx.path.duration,
        "sliding_speed_centre_mm_s": float(ctx.sander.speed(0.0)),
        "sliding_speed_rim_mm_s": float(ctx.sander.speed(a)),
        "nominal_parameters": nom,
        "nominal_fresh_mean_removal_um": fresh,
        "pad_penetration_at_prior_bounds": penetration,
        "volume_per_unit_K_vs_k": {"k_pad": ks.tolist(), "volume_mm3_per_mm2N": vol.tolist(),
                                   "relative_spread": float((vol.max() - vol.min()) / vol.mean()),
                                   "force_N": F_ref},
        "surrogate": {"n_eta": int(ctx.table.n_eta), "eta_min_mm3": float(ctx.table.eta[0]),
                      "eta_max_mm3": float(ctx.table.eta[-1]), "wear_terms": 4,
                      "max_relative_rms_error": float(max(errs)),
                      "median_relative_rms_error": float(np.median(errs)), "n_checks": len(errs),
                      "max_lambda_times_pass_volume": zmax},
        "worlds": _world_info(ctx, cg, forces_all, nom),
    }
    write_json(out / "model_info.json", info)
    if verbose:
        print(f"\nModel: {info['grid_nodes']} grid nodes, {info['scan_points']} scan points, "
              f"{info['stations_per_pass']} stations/pass ({info['pass_duration_s']:.1f} s), sliding speed "
              f"{info['sliding_speed_centre_mm_s']:.0f}-{info['sliding_speed_rim_mm_s']:.0f} mm/s; removed volume "
              f"per unit K varies by {info['volume_per_unit_K_vs_k']['relative_spread']:.2%} across the k_pad "
              f"prior; surrogate max relative RMS error {info['surrogate']['max_relative_rms_error']:.2e}")
    return info


def _world_info(ctx: Context, cg: ContactGeometry, forces: list[float], nom: dict) -> dict:
    """How large each mismatch of every configured world is, in numbers."""
    from scipy.optimize import brentq
    pr = ctx.priors
    out = {}
    for name, (w, art) in ctx.worlds.items():
        d = {"pad_law": w.pad_law.kind, "force_gain_sigma_log": w.force_gain_sigma,
             "force_ripple": w.force_ripple, "ripple_corr": w.ripple_corr, "wear_law": w.wear.kind,
             "wear_rings": w.wear_rings, "preston_exponent": w.preston_exponent,
             "scan_artefacts": {k: getattr(art, k) for k in art.__dataclass_fields__}}
        if w.pad_law.kind == "foam":
            h = w.pad_law.thickness
            pk = {}
            for label, k in (("soft_k_low", pr.k_low), ("mid", nom["k_pad"]), ("stiff_k_high", pr.k_high)):
                F = max(forces)
                _, p_lin = cg.contact_points(F, k)
                _, p_foam = cg.contact_points(F, k, w.pad_law)
                strain = float(p_foam.max() / (k + p_foam.max() / h) / h)     # invert p = k d / (1 - d/h)
                pk[label] = {"k_pad": float(k), "force_N": F, "max_strain_foam": strain,
                             "peak_pressure_ratio_foam_over_linear": float(p_foam.max() / p_lin.max()),
                             "contact_fraction_linear": float(cg.contact_fraction(F, k)[0]),
                             "contact_fraction_foam": float(cg.contact_fraction(F, k, w.pad_law)[0])}
            d["foam"] = {"thickness_mm": h, "at_max_force": pk}
        if w.wear.kind != "single":
            lam = nom["lam"]
            single = WearLaw()
            v50 = {law: float(brentq(lambda W: wl.phi(lam, W) - 0.5, 0.0, 50.0 / lam))
                   for law, wl in (("single", single), ("two_stage", w.wear))}
            # K/K0 after the volume removed by the first nominal pass at the low force
            V1 = float(nom["K0"] * ctx.model.volume(ctx.model.exposure(min(forces), nom["k_pad"])))
            d["two_stage"] = {"fast_fraction": w.wear.fast_fraction, "fast_ratio": w.wear.fast_ratio,
                              "volume_to_half_K0_mm3": v50, "nominal_first_pass_volume_mm3": V1,
                              "K_over_K0_after_first_pass": {"single": single.phi(lam, V1),
                                                             "two_stage": w.wear.phi(lam, V1)}}
        if w.preston_exponent != 1.0:
            F = float(ctx.cfg["pad"]["reference_force_N"])
            vol = {lab: float(ctx.model.line_exposures(F, k, alpha=w.preston_exponent)[1].sum()
                              / ctx.model.line_exposures(F, k)[1].sum())
                   for lab, k in (("soft_k_low", pr.k_low), ("mid", nom["k_pad"]), ("stiff_k_high", pr.k_high))}
            d["preston"] = {"exponent": w.preston_exponent, "p_ref_kPa": 10.0, "force_N": F,
                            "volume_ratio_vs_linear": vol,
                            "soft_over_stiff_volume_ratio": vol["soft_k_low"] / vol["stiff_k_high"]}
        if w.wear_rings > 1:
            from .process import World
            sched = ctx.schedule()
            k20 = {}
            for lab, k in (("soft_k_low", pr.k_low), ("mid", nom["k_pad"]), ("stiff_k_high", pr.k_high)):
                truth = HiddenTruth(float(k), nom["K0"], nom["lam"])
                rings = simulate_truth(ctx.model, truth, sched, 0.0, rng_for(0, 0), max_extend=0,
                                       world=World("r", wear=w.wear, wear_rings=w.wear_rings))
                uni = simulate_truth(ctx.model, truth, sched, 0.0, rng_for(0, 0), max_extend=0,
                                     world=World("u", wear=w.wear))
                k20[lab] = {"rings": float(rings.K[-1] / nom["K0"]), "uniform": float(uni.K[-1] / nom["K0"])}
            d["radial_wear"] = {"rings": w.wear_rings, "K_over_K0_at_last_pass_nominal": k20}
        if w.loading_max > 0:
            from .process import World
            sched = ctx.schedule()
            truth = HiddenTruth(nom["k_pad"], nom["K0"], nom["lam"])
            base = World("b", wear=w.wear, wear_rings=w.wear_rings)
            load = World("l", wear=w.wear, wear_rings=w.wear_rings, loading_max=w.loading_max,
                         loading_volume=w.loading_volume, loading_clean=w.loading_clean)
            v0 = simulate_truth(ctx.model, truth, sched, 0.0, rng_for(0, 0), max_extend=0, world=base).volume
            v1 = simulate_truth(ctx.model, truth, sched, 0.0, rng_for(0, 0), max_extend=0, world=load).volume
            d["loading"] = {"max": w.loading_max, "volume_mm3": w.loading_volume, "clean": w.loading_clean,
                            "volume_loss_nominal": {"pass_1": float(1 - v1[0] / v0[0]),
                                                    "last_pass": float(1 - v1[-1] / v0[-1])}}
        if w.tilt_stiffness > 0:
            F = max(forces)
            tilt = {}
            for lab, k in (("soft_k_low", pr.k_low), ("mid", nom["k_pad"]), ("stiff_k_high", pr.k_high)):
                _, a, b = ctx.model.contact.tilt_solution(F, k, w.tilt_stiffness)   # every station of a pass
                E_t = ctx.model.line_exposures(F, k, tilt_stiffness=w.tilt_stiffness)[0].sum(axis=0)
                E_r = ctx.model.line_exposures(F, k)[0].sum(axis=0)
                tilt[lab] = {"max_tilt_deg": float(np.degrees(np.max(np.hypot(a, b)))),
                             "map_change_rel_rms": float(np.sqrt(np.mean((E_t - E_r) ** 2)) / np.mean(E_r))}
            d["tilt"] = {"stiffness_Nmm_per_rad": w.tilt_stiffness, "force_N": F, "at_max_force": tilt}
        out[name] = d
    return out


def _main_summary(rows: list[dict], traj, e: dict) -> dict:
    last = rows[-1]
    n_passes = len(rows)
    ests = ESTS
    return {
        "truth": {"k_pad": traj.truth.k_pad, "K0": traj.truth.K0, "lam": traj.truth.lam,
                  "force_gain": traj.truth.force_gain},
        "true_crossing_pass": traj.crossing_pass,
        "final_pass": n_passes,
        "final_rmse_um": {k: last[f"rmse_{k}_um"] for k in ests},
        "first_pass_rmse_um": {k: rows[0][f"rmse_{k}_um"] for k in ests},
        "early_pass": e["early_pass"],
        "early_rmse_um": {k: rows[e["early_pass"] - 1][f"rmse_{k}_um"] for k in ests},
        "final_true_mean_removal_um": last["true_mean_removal_um"],
        "B_calibration": {"k_pad": last["B_k_pad"], "K": last["B_K"]},
        "D_final": {"k_pad": last["D_k_pad"], "K": last["D_K"], "lam": last["D_lam"]},
        "C_final": {name: {"mean": last[f"C_{name}_mean"], "median": last[f"C_{name}_median"],
                           "lo": last[f"C_{name}_lo"], "hi": last[f"C_{name}_hi"]}
                    for name in ("k_pad", "K0", "lam", "K")},
        "true_final_K": last["true_K"],
        "C_final_inside_90": {name: inside(last, name, tk) for name, tk in PARAMS},
        "C_final_relative_distance_outside_90": {name: _outside(last, name, tk) for name, tk in PARAMS},
        "C_predictive_mean_inside_90_passes_2_on": int(sum(r["C_pred_mean_inside"] for r in rows[1:])),
        "C_noise_level_um": iqr_summary([r["C_sigma_um"] for r in rows]),
        "C_rejected_points": iqr_summary([r["C_outliers"] for r in rows]),
        "A_crossing_pass": last["cross_A"],
        "C_crossing_prediction": [{"after_pass": r["pass"], "median": r["cross_C_median"], "lo": r["cross_C_lo"],
                                   "hi": r["cross_C_hi"], "D": r["cross_D"]} for r in rows],
        "passes_C_better_than_B": [r["pass"] for r in rows if r["rmse_C_um"] < r["rmse_B_um"]],
    }


def experiment_main(ctx: Context, out: Path, verbose: bool = True) -> dict:
    """The main run: one hidden truth, in the headline world and in the matched world."""
    e = ctx.cfg["experiments"]
    n_passes = ctx.cfg["schedule"]["n_passes"]
    main_draw = select_main_draw(ctx)
    main_world = e.get("main_world", "realistic")
    worlds = [main_world] + (["matched"] if main_world != "matched" else [])
    summary = {"draw": main_draw, "world": main_world,
               "draw_selection": "config" if str(e.get("main_draw", "typical")) != "typical"
               else "robustness draw at the median standardised distance from the prior centre",
               "worlds": {}}
    for w in worlds:
        record = (e["early_pass"], n_passes) if w == main_world else ()
        rows, extra = run_episode(ctx, main_draw, w, experiment="main", record_maps=record)
        write_csv(out / ("main_run.csv" if w == main_world else f"main_run_{w}.csv"), rows)
        summary["worlds"][w] = _main_summary(rows, extra["traj"], e)
        if w == main_world:
            stride = ctx.cfg["scan"]["stride"]
            np.savez_compressed(out / "main_run_maps.npz",
                                **{f"pass{p}_{k}": v.reshape(ctx.obs_shape).astype(np.float32) * UM
                                   for p, d in extra["maps"].items() for k, v in d.items()},
                                x=ctx.panel.x[::stride], y=ctx.panel.y[::stride])
        if verbose:
            t = extra["traj"].truth
            print(f"\nMain run, {w} world (draw {main_draw}): truth k={t.k_pad:.4g} N/mm^3, K0={t.K0:.3g} mm^2/N, "
                  f"lambda={t.lam:.3g} 1/mm^3, force gain {t.force_gain:.3f}, crossing pass "
                  f"{extra['traj'].crossing_pass}")
            print("  pass  F[N]  true mean[um]  RMSE A    RMSE B    RMSE C   oracle [um]  C 90% pred. mean")
            for r in rows:
                print(f"  {r['pass']:4d} {r['force_N']:5.0f} {r['true_mean_removal_um']:10.2f} "
                      f"{r['rmse_A_um']:9.3f} {r['rmse_B_um']:9.3f} {r['rmse_C_um']:9.3f} {r['rmse_oracle_um']:9.3f}"
                      f"   [{r['C_pred_mean_lo_um']:.2f}, {r['C_pred_mean_hi_um']:.2f}]"
                      f"{'' if r['C_pred_mean_inside'] else '  miss'}")
    write_json(out / "main_run.json", summary)
    return summary


def select_main_draw(ctx: Context) -> int:
    """The main run's hidden truth, chosen by a rule that looks only at the priors and
    the sampled truths (never at any estimator result): among the robustness draws,
    the one at the median standardised distance from the prior centre over
    (log k_pad, log K0, log lambda) -- a member of the prior's typical set, neither
    a tail draw nor one that happens to match the nominal parameters. An integer
    ``experiments.main_draw`` in the config overrides the rule."""
    e = ctx.cfg["experiments"]
    if str(e.get("main_draw", "typical")) != "typical":
        return int(e["main_draw"])
    pr = ctx.priors
    lo, hi = np.log(pr.k_low), np.log(pr.k_high)

    def dist2(d):
        t = draw_truth(ctx, d)
        z = ((np.log(t.k_pad) - 0.5 * (lo + hi)) / ((hi - lo) / np.sqrt(12.0)),   # log-uniform s.d.
             (np.log(t.K0) - np.log(pr.K0_median)) / pr.K0_sigma_log,
             (np.log(t.lam) - np.log(pr.lam_median)) / pr.lam_sigma_log)
        return float(sum(v * v for v in z))

    n = int(e["robustness_draws"])
    ranked = sorted(range(n), key=dist2)
    return int(ranked[(n - 1) // 2])


STEADY_FROM = 4   # first pass of the 'steady state' comparison (D has three scans for its wear fit)


def _final_draw_summary(rows: list[dict]) -> dict:
    last = rows[-1]
    later = [r for r in rows if r["pass"] >= 2]
    return {
        "world": last["world"], "draw": last["draw"], "schedule": last["schedule"], "noise_um": last["noise_um"],
        "true_k_pad": last["true_k_pad"], "true_K0": last["true_K0"], "true_lam": last["true_lam"],
        "true_force_gain": last["true_force_gain"],
        "true_final_K": last["true_K"], "cross_true": last["cross_true"], "cross_A": last["cross_A"],
        **{f"final_rmse_{k}_um": last[f"rmse_{k}_um"] for k in ESTS},
        "final_true_mean_removal_um": last["true_mean_removal_um"],
        # over passes 2..N (pass 1 is a prior prediction for every estimator), geometric mean
        **{f"mean_rmse_{k}_um": _gmean([r[f"rmse_{k}_um"] for r in later]) for k in ("A", "B", "C", "D", "oracle")},
        # start-up (passes 2-3: D has at most two scans for its wear fit), steady state (passes
        # STEADY_FROM..N) and the last two passes (one at each force of the alternating schedule)
        **{f"startup_rmse_{k}_um": _gmean([r[f"rmse_{k}_um"] for r in later if r["pass"] < STEADY_FROM])
           for k in ("C", "D", "oracle")},
        **{f"steady_rmse_{k}_um": _gmean([r[f"rmse_{k}_um"] for r in rows if r["pass"] >= STEADY_FROM])
           for k in ("C", "D", "oracle")},
        **{f"last_two_rmse_{k}_um": _gmean([r[f"rmse_{k}_um"] for r in rows[-2:]]) for k in ("C", "D", "oracle")},
        **({"final_rmse_C2_um": last["rmse_C2_um"],
            "startup_rmse_C2_um": _gmean([r["rmse_C2_um"] for r in later if r["pass"] < STEADY_FROM]),
            "steady_rmse_C2_um": _gmean([r["rmse_C2_um"] for r in rows if r["pass"] >= STEADY_FROM])}
           if "rmse_C2_um" in last else {}),
        "n_passes_C_better_than_B": sum(r["rmse_C_um"] < r["rmse_B_um"] for r in rows),
        "n_passes_C_better_than_D": sum(r["rmse_C_um"] < r["rmse_D_um"] for r in rows),
        "predictive_inside_passes_2_on": int(sum(r["C_pred_mean_inside"] for r in later)),
        "predictive_passes_2_on": len(later),
        **{f"in90_{name}": inside(last, name, tk) for name, tk in PARAMS},
        "relw_k_pad": rel_width(last, "k_pad"), "relw_lam": rel_width(last, "lam"), "relw_K": rel_width(last, "K"),
        "relerr_k_pad": last["C_k_pad_mean"] / last["true_k_pad"] - 1,
        "relerr_lam": last["C_lam_mean"] / last["true_lam"] - 1,
        "relerr_K": last["C_K_mean"] / last["true_K"] - 1,
        "corr_logk_logK": last["C_corr_logk_logK"],
        "B_relerr_k_pad": last["B_k_pad"] / last["true_k_pad"] - 1,
        "median_sigma_C_um": float(np.median([r["C_sigma_um"] for r in rows])),
        "median_rejected_points_C": float(np.median([r["C_outliers"] for r in rows])),
        "passes_redone_C": int(sum(r["C_redone"] for r in rows)),
    }


def _coverage(finals: list[dict]) -> dict:
    n = len(finals)
    out = {}
    for name in ("k_pad", "lam", "K", "K0"):
        hits = sum(f[f"in90_{name}"] for f in finals)
        out[name] = {"inside": int(hits), "n": n, "fraction": hits / n}
    joint = sum(f["in90_k_pad"] and f["in90_lam"] and f["in90_K"] for f in finals)
    # not a joint 90% region: with calibrated, independent marginals expect 0.9^3
    out["all_three_marginals_inside"] = {"inside": int(joint), "n": n, "fraction": joint / n,
                                         "expected_if_calibrated_independent": 0.9**3}
    out["binomial_se_if_calibrated"] = float(np.sqrt(0.9 * 0.1 / n))
    return out


def bootstrap_ci(x, stat=np.median, n_boot: int = 2000, level: float = 0.9, seed: int = 10) -> list[float]:
    """Percentile bootstrap interval of a statistic over draws (resampling draws)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[rng.integers(0, x.size, x.size)]) for _ in range(n_boot)])
    a = (1 - level) / 2
    return [float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))]


def _mean_or_none(x) -> float | None:
    return float(np.mean(x)) if len(x) else None


def _c2_summary(rows: list[dict], finals: list[dict]) -> dict | None:
    """Tracker C2 against C on the same draws: accuracy ratios (C/C2 > 1: C2 more accurate,
    medians over draws with 90% bootstrap intervals and sign-flip p-values on the log ratio)
    and the calibration of C2's 90% intervals (mean removal and blocks)."""
    if not rows or "rmse_C2_um" not in rows[0]:
        return None
    later = [r for r in rows if r["pass"] >= 2]
    out = {"final_rmse_um": iqr_summary([f["final_rmse_C2_um"] for f in finals])}
    for label in ("startup", "steady", "final"):
        ratio = np.array([f[f"{label}_rmse_C_um"] / f[f"{label}_rmse_C2_um"] for f in finals])
        out[f"C_over_C2_{label}"] = {"median": float(np.median(ratio)), "ci90": bootstrap_ci(ratio),
                                     "C2_better": int(np.sum(ratio > 1)), "n": int(ratio.size),
                                     "p_perm": paired_permutation_p(np.log(ratio))}
    inside = [r["C2_pred_mean_inside"] for r in later]
    out["predictive_90"] = {
        "passes_2_on": {"inside": int(sum(inside)), "n": len(inside), "fraction": float(np.mean(inside)),
                        "below": int(sum(r["true_mean_removal_um"] < r["C2_pred_mean_lo_um"] for r in later)),
                        "above": int(sum(r["true_mean_removal_um"] > r["C2_pred_mean_hi_um"] for r in later))},
        "passes_2_to_5": _mean_or_none([r["C2_pred_mean_inside"] for r in later if r["pass"] <= 5]),
        "passes_11_on": _mean_or_none([r["C2_pred_mean_inside"] for r in later if r["pass"] >= 11]),
        "regional_inside_fraction_passes_2_on": float(np.mean([r["C2_region_inside_fraction"] for r in later])),
        "relative_width_passes_2_on": iqr_summary([(r["C2_pred_mean_hi_um"] - r["C2_pred_mean_lo_um"])
                                                   / r["true_mean_removal_um"] for r in later]),
    }
    out["final_level"] = iqr_summary([rr[-1]["C2_level"] for rr in by_draw(rows).values()])
    out["final_shape_rms"] = iqr_summary([rr[-1]["C2_shape_rms"] for rr in by_draw(rows).values()])
    return out


def paired_permutation_p(diff, n_perm: int = 20000, seed: int = 11) -> float:
    """Two-sided p-value for a zero mean of paired differences, by random sign flips
    (exact under the null hypothesis that each difference is symmetric about zero)."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or not np.any(d):
        return 1.0
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n_perm, 5000):
        k = min(5000, n_perm - start)
        signs = rng.integers(0, 2, size=(k, d.size)) * 2 - 1
        hits += int(np.sum(np.abs(signs @ d) / d.size >= obs - 1e-12))
    return float((1 + hits) / (n_perm + 1))


def holm_adjust(pvals: dict) -> dict:
    """Holm step-down adjusted p-values (family-wise error control) for a dict of p-values."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def apply_holm(worlds: dict, family: str, key: str = "p_holm") -> int:
    """Holm-adjust the paired-comparison p-values of every world's decision rules together
    (one family: C_r against the tuned D_r and R_r for every cost ratio and world); the
    adjusted value is stored next to each p-value as ``p_holm``. Returns the family size."""
    entries = {}
    for w, s in worlds.items():
        for tag, e in s["decision_rules"]["by_cost_ratio"].items():
            for other in (f"D_{tag}", f"R_{tag}"):
                entries[(w, tag, other)] = e[f"C_minus_{other}"]
    adj = holm_adjust({k: v["p_perm"] for k, v in entries.items()})
    for k, v in entries.items():
        v[key] = adj[k]
        v[f"{key}_family"] = family
    return len(entries)


def _c_vs_d(finals: list[dict]) -> dict:
    """C against D over draws: error ratios and win fractions with 90% bootstrap intervals."""
    out = {}
    for label, c_key, d_key in (("final", "final_rmse_C_um", "final_rmse_D_um"),
                                ("mean_over_passes", "mean_rmse_C_um", "mean_rmse_D_um"),
                                ("startup", "startup_rmse_C_um", "startup_rmse_D_um"),
                                ("steady", "steady_rmse_C_um", "steady_rmse_D_um"),
                                ("last_two", "last_two_rmse_C_um", "last_two_rmse_D_um")):
        ratio = np.array([f[d_key] / f[c_key] for f in finals])
        wins = (ratio > 1).astype(float)
        out[label] = {"D_over_C_median": float(np.median(ratio)), "D_over_C_median_ci90": bootstrap_ci(ratio),
                      "C_better": int(wins.sum()), "n": int(wins.size),
                      "C_better_fraction_ci90": bootstrap_ci(wins, np.mean)}
    return out


def interval_score(r: dict, alpha: float = 0.1) -> float:
    """Interval score (Gneiting & Raftery 2007) of C's central 90% interval for the
    mean removal, relative to the true value: width plus 2/alpha times any miss.
    Lower is better; it rewards narrow intervals only if they keep covering."""
    lo, hi, x = r["C_pred_mean_lo_um"], r["C_pred_mean_hi_um"], r["true_mean_removal_um"]
    return ((hi - lo) + (2 / alpha) * max(lo - x, 0.0) + (2 / alpha) * max(x - hi, 0.0)) / x


def _predictive(rows: list[dict], n_passes: int) -> dict:
    """Calibration of C's 90% predictive interval for the next pass's mean removal
    (made before the pass, checked against the true mean removal at the scan points)."""
    later = [r for r in rows if r["pass"] >= 2]
    first = [r for r in rows if r["pass"] == 1]
    hits = int(sum(r["C_pred_mean_inside"] for r in later))

    def frac(lo, hi):
        sub = [r["C_pred_mean_inside"] for r in rows if lo <= r["pass"] <= hi]
        return {"inside": int(sum(sub)), "n": len(sub), "fraction": float(np.mean(sub)) if sub else None}

    per_pass = [float(np.mean([r["C_pred_mean_inside"] for r in rows if r["pass"] == p]))
                for p in range(1, n_passes + 1)]
    # where the truth falls relative to the interval: below, inside, above
    below = int(sum(r["true_mean_removal_um"] < r["C_pred_mean_lo_um"] for r in later))
    above = int(sum(r["true_mean_removal_um"] > r["C_pred_mean_hi_um"] for r in later))
    n_draws = len({r["draw"] for r in rows})
    return {
        "passes_2_on": {"inside": hits, "n": len(later), "fraction": hits / max(len(later), 1),
                        "below": below, "above": above},
        "relative_interval_score_passes_2_on": float(np.mean([interval_score(r) for r in later])) if later else None,
        "regional_inside_fraction_passes_2_on": float(np.mean([r["C_region_inside_fraction"] for r in later]))
        if later else None,
        "pass_1": {"inside": int(sum(r["C_pred_mean_inside"] for r in first)), "n": len(first)},
        "passes_2_to_5": frac(2, 5),
        "passes_11_on": frac(11, n_passes),
        "per_pass_fraction": per_pass,
        "relative_width_passes_2_on": iqr_summary([(r["C_pred_mean_hi_um"] - r["C_pred_mean_lo_um"])
                                                   / r["C_pred_mean_median_um"] for r in later]),
        "draws": n_draws,
        "binomial_se_if_calibrated_per_pass": float(np.sqrt(0.9 * 0.1 / max(n_draws, 1))),
    }


# --------------------------------------------------------------------------- multi-draw runs


_WORKER_CTX: Context | None = None


def _init_worker(cfg: dict) -> None:
    """Pool initializer: reuse a fork-inherited context, else rebuild it (spawn)."""
    global _WORKER_CTX
    if _WORKER_CTX is None or config_hash(_WORKER_CTX.cfg) != config_hash(cfg):
        from .config import build_context
        _WORKER_CTX = build_context(cfg)


def task_list(ctx: Context) -> list[tuple]:
    """Every (kind, world, draw) task of the multi-draw experiments."""
    e = ctx.cfg["experiments"]
    n = int(e["robustness_draws"])
    tasks = [("robustness", w, d) for w in e["robustness_worlds"] for d in range(n)]
    for w in validation_worlds(ctx):
        tasks += [("robustness", w, d) for d in range(int(e.get("validation_draws", n)))]
    nb = int(e.get("breakdown_draws", 0))
    tasks += [("breakdown", w, d) for d in range(nb) for w in breakdown_extra_worlds(ctx)]
    return tasks


def validation_worlds(ctx: Context) -> list[str]:
    """The pre-registered mismatch worlds (robustness runs only, never tuned on)."""
    return [w for w in ctx.cfg["experiments"].get("validation_worlds", []) if w in ctx.worlds]


def breakdown_extra_worlds(ctx: Context) -> list[str]:
    """Worlds run only for the breakdown: one per mismatch group, plus stress tests
    (``experiments.breakdown_worlds`` if given, else every other configured world)."""
    e = ctx.cfg["experiments"]
    if e.get("breakdown_worlds") is not None:
        return [w for w in e["breakdown_worlds"] if w in ctx.worlds]
    skip = set(e["robustness_worlds"]) | set(validation_worlds(ctx)) | set(extended_only_worlds(ctx))
    return [w for w in ctx.worlds if w not in skip and w != "matched"]


def extended_only_worlds(ctx: Context) -> list[str]:
    """Worlds evaluated only in the extended study (the pre-registered worlds)."""
    return [w for w in (ctx.cfg.get("extended", {}) or {}).get("worlds", []) if w in ctx.worlds]


def tuning_tasks(ctx: Context) -> list[tuple]:
    t = ctx.cfg["tuning"]
    lo, hi = (int(x) for x in t["draws"])
    values = tuple(float(v) for v in t["rate_drift"])
    return [("tuning", w, d, values) for d in range(lo, hi) for w in t["worlds"]]


def run_task(ctx: Context, task: tuple) -> dict:
    """All episodes of one task, sharing the hidden-truth trajectory where possible.

    robustness: the alternating schedule at the default scan noise, plus (in the
    ablation world, first ``ablation_draws`` draws) the constant-force schedule
    and a re-seeded filter control, plus (in the noise world, first
    ``noise_draws`` draws) every other noise level.
    breakdown: the alternating schedule in a single-mismatch world.
    tuning: C alone with a candidate wear-rate drift (held-out draws).
    """
    kind, world, d = task[:3]
    e = ctx.cfg["experiments"]
    default = float(ctx.cfg["scan"]["noise_um"])
    out: dict[tuple, list[dict]] = {}
    traj = make_truth(ctx, d, "alternating", world)
    if kind == "tuning":   # every candidate drift on the same hidden-truth trajectory
        import dataclasses
        for i, v in enumerate(task[3]):     # D (independent of the drift) once, for its decision margins
            tctx = dataclasses.replace(ctx, pf=dataclasses.replace(ctx.pf, rate_drift=v))
            out[(kind, world, v)], _ = run_episode(tctx, d, world, "alternating", default, experiment=kind,
                                                   traj=traj, full=False, with_D=(i == 0))
        return out
    out[(kind, world, default)], _ = run_episode(ctx, d, world, "alternating", default, experiment=kind, traj=traj,
                                                 risk=(kind == "robustness"), with_B=(kind == "robustness"),
                                                 with_C2=bool(e.get("with_C2", False)))
    if kind != "robustness":
        return out
    if world == e.get("ablation_world", "matched") and d < int(e.get("ablation_draws", e["robustness_draws"])):
        out[("ablation_control", world, default)], _ = run_episode(
            ctx, d, world, "alternating", default, experiment="ablation_control", traj=traj, full=False, pf_stream=6)
        out[("ablation_constant", world, default)], _ = run_episode(
            ctx, d, world, "constant", default, experiment="ablation",
            traj=make_truth(ctx, d, "constant", world), full=False)
    if world == e.get("noise_world", "realistic") and d < int(e.get("noise_draws", 0)):
        for sigma in sorted({float(s) for s in e["noise_levels_um"]} - {default}):
            out[("noise", world, sigma)], _ = run_episode(ctx, d, world, "alternating", sigma, experiment="noise",
                                                          traj=traj, full=False)
    return out


def _run_task_worker(task) -> dict:
    return run_task(_WORKER_CTX, task)


def resolve_workers(workers) -> int:
    if workers in (None, "auto"):
        return max(1, min(4, (os.cpu_count() or 1)))
    return max(1, int(workers))


def run_draws(ctx: Context, verbose: bool = True, workers=1, tasks: list[tuple] | None = None) -> dict:
    """All multi-draw runs. Tasks are independent and individually seeded, so the
    results do not depend on the number of worker processes."""
    global _WORKER_CTX
    tasks = task_list(ctx) if tasks is None else tasks
    workers = resolve_workers(workers)
    t0 = time.time()
    results: list[dict] = []

    def progress(i):
        if verbose and ((i + 1) % 25 == 0 or i + 1 == len(tasks)):
            print(f"  ... {i + 1}/{len(tasks)} tasks ({time.time() - t0:.0f} s, {workers} worker(s))")

    if workers == 1:
        for i, t in enumerate(tasks):
            results.append(run_task(ctx, t))
            progress(i)
    else:
        _WORKER_CTX = ctx  # inherited by forked workers; spawned workers rebuild it from ctx.cfg
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(ctx.cfg,)) as pool:
            for i, res in enumerate(pool.map(_run_task_worker, tasks)):
                results.append(res)
                progress(i)
    runs: dict[tuple, list[dict]] = {}
    for res in results:
        for key, rows in res.items():
            runs.setdefault(key, []).extend(rows)
    return runs


# --------------------------------------------------------------------------- tuning


def experiment_tuning(ctx: Context, out: Path, verbose: bool = True, workers=1) -> dict:
    """Choose the tracker's wear-rate drift on held-out draws (never used elsewhere).

    Every candidate is run in each tuning world on the same held-out draws; the
    chosen value minimises the mean relative interval score of C's 90%
    predictive interval of the next pass's mean removal (passes 2 on), averaged
    over the worlds with equal weight. The context is updated in place.
    """
    t = ctx.cfg["tuning"]
    runs = run_draws(ctx, verbose, workers, tuning_tasks(ctx))
    n_passes = ctx.cfg["schedule"]["n_passes"]
    table, by_value = [], []
    for v in (float(x) for x in t["rate_drift"]):
        per_world = {}
        for w in t["worlds"]:
            rows = runs[("tuning", w, v)]
            finals = [_final_draw_summary(rr) for rr in by_draw(rows).values()]
            pred = _predictive(rows, n_passes)
            per_world[w] = {
                "interval_score": pred["relative_interval_score_passes_2_on"],
                "coverage": pred["passes_2_on"]["fraction"],
                "relative_width": pred["relative_width_passes_2_on"]["median"],
                "mean_rmse_C_over_oracle": float(np.median([f["mean_rmse_C_um"] / f["mean_rmse_oracle_um"]
                                                            for f in finals])),
                "coverage_k_pad": _coverage(finals)["k_pad"]["fraction"],
            }
            table.append({"rate_drift": v, "world": w, **per_world[w]})
        score = float(np.mean([per_world[w]["interval_score"] for w in t["worlds"]]))
        by_value.append({"rate_drift": v, "mean_interval_score": score, "worlds": per_world})
    best = min(by_value, key=lambda x: (x["mean_interval_score"], x["rate_drift"]))
    chosen = best["rate_drift"]
    first = float(t["rate_drift"][0])
    d_rows = [r for w in t["worlds"] for r in runs[("tuning", w, first)]]
    margins = _tune_margins(d_rows, n_passes, ctx.threshold, cost_ratios(ctx))
    summary = {"draws": [int(x) for x in t["draws"]], "first_draw": int(t["draws"][0]),
               "last_draw": int(t["draws"][1]) - 1, "n_draws": int(t["draws"][1]) - int(t["draws"][0]),
               "worlds": list(t["worlds"]), "criterion":
               "mean relative interval score of the 90% predictive interval of next-pass mean removal",
               "by_value": by_value, "chosen_rate_drift": chosen, "chosen_index": by_value.index(best),
               "chosen": best, "constant_rate": next((v for v in by_value if v["rate_drift"] == 0.0), None),
               "decision_margins": margins}
    summary["config_hash"] = _tuning_hash(ctx.cfg)
    write_csv(out / "tuning.csv", table)
    write_json(out / "tuning.json", summary)
    set_rate_drift(ctx, chosen)
    ctx.__dict__["decision_margins"] = margins
    if verbose:
        print(f"\nTuning of the wear-rate drift on held-out draws {t['draws'][0]}-{t['draws'][1] - 1} "
              f"({', '.join(t['worlds'])}):")
        for v in by_value:
            cells = "  ".join(f"{w}: cov {x['coverage']:.1%} width {x['relative_width']:.2%}"
                              for w, x in v["worlds"].items())
            print(f"  drift {v['rate_drift']:5.3f}: interval score {v['mean_interval_score']:.4f}  {cells}")
        print(f"  chosen rate_drift = {chosen:g}")
        nan = float("nan")
        for tag, m in margins.items():
            print(f"  cost ratio {tag[1:]}: margin D {m['D']:+.2f} (held-out cost {m.get('D_held_out_cost', nan):.2f}, "
                  f"{m.get('D_held_out_cost_no_margin', nan):.2f} without), R {m['R']:+.2f} "
                  f"({m.get('R_held_out_cost', nan):.2f}, {m.get('R_held_out_cost_no_margin', nan):.2f})")
    return summary


def _tuning_hash(cfg: dict) -> str:
    """Hash of everything the tuning result depends on: the config without the tuned value
    and the source of the simulation and estimation modules."""
    import copy
    c = copy.deepcopy(cfg)
    c["filter"].pop("rate_drift", None)
    skip = {"plots.py", "report.py", "cli.py", "__main__.py"}
    src = b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")) if p.name not in skip)
    return hashlib.sha256(config_hash(c).encode() + src).hexdigest()[:12]


def set_rate_drift(ctx: Context, value: float) -> None:
    """Use ``value`` for the tracker's wear-rate drift from now on (config and filter)."""
    import dataclasses
    ctx.cfg["filter"]["rate_drift"] = float(value)
    ctx.pf = dataclasses.replace(ctx.pf, rate_drift=float(value))


def cost_ratios(ctx: Context) -> list[float]:
    return [float(c) for c in ctx.cfg["abrasive"].get("cost_ratios", [1, 3, 19])]


def risk_quantile(ctx: Context) -> float:
    return float(ctx.cfg["abrasive"].get("risk_quantile", 0.25))


def decision_margins(ctx: Context) -> dict:
    """Safety margins of D and R per cost ratio, tuned on the held-out draws (empty: none)."""
    return ctx.__dict__.get("decision_margins", {})


def tuned_rate_drift(ctx: Context, out: Path, verbose: bool = True, workers=1) -> float:
    """The tuned drift: from ``out``/tuning.json if it was made with this configuration, else tune now."""
    if str(ctx.cfg["filter"].get("rate_drift", "auto")) != "auto":
        return float(ctx.cfg["filter"]["rate_drift"])
    path = out / "tuning.json"
    if path.exists():
        info = json.loads(path.read_text())
        if info.get("config_hash") == _tuning_hash(ctx.cfg):
            set_rate_drift(ctx, info["chosen_rate_drift"])
            ctx.__dict__["decision_margins"] = info.get("decision_margins", {})
            return float(info["chosen_rate_drift"])
        if verbose:
            print(f"[swt] {path} was made with another configuration: tuning again")
    return experiment_tuning(ctx, out, verbose, workers)["chosen_rate_drift"]


# --------------------------------------------------------------------------- summaries


def _per_pass_stats(rows: list[dict], n_passes: int) -> dict:
    """Per-pass medians of every estimator's RMSE, C-vs-B win counts and coverage counts."""
    out = {"pass": list(range(1, n_passes + 1)), "median_rmse_um": {}, "q25_rmse_um": {}, "q75_rmse_um": {},
           "draws_C_better_than_B": [], "draws_C_better_than_D": [], "coverage_90": {}}
    by_pass = [[r for r in rows if r["pass"] == p] for p in range(1, n_passes + 1)]
    for k in ESTS:
        out["median_rmse_um"][k] = [q([r[f"rmse_{k}_um"] for r in rr], 50) for rr in by_pass]
        out["q25_rmse_um"][k] = [q([r[f"rmse_{k}_um"] for r in rr], 25) for rr in by_pass]
        out["q75_rmse_um"][k] = [q([r[f"rmse_{k}_um"] for r in rr], 75) for rr in by_pass]
    out["draws_C_better_than_B"] = [int(sum(r["rmse_C_um"] < r["rmse_B_um"] for r in rr)) for rr in by_pass]
    out["draws_C_better_than_D"] = [int(sum(r["rmse_C_um"] < r["rmse_D_um"] for r in rr)) for rr in by_pass]
    for name, tk in PARAMS:
        out["coverage_90"][name] = [int(sum(inside(r, name, tk) for r in rr)) for rr in by_pass]
    return out


def _k_drop_stats(rows: list[dict], ctx: Context) -> dict:
    """Fractional drop of the true K over the first low- and high-force pass."""
    out = {}
    sched = ctx.schedule("alternating")
    for p in (1, 2):
        drops = [1 - rr[p]["true_K"] / rr[p - 1]["true_K"] for rr in by_draw(rows).values() if len(rr) > p]
        out[f"pass{p}_{sched[p - 1].force:g}N"] = iqr_summary(drops)
    return out


def _world_robustness(ctx: Context, rows: list[dict]) -> dict:
    n_passes = ctx.cfg["schedule"]["n_passes"]
    draws = by_draw(rows)
    finals = [_final_draw_summary(rr) for rr in draws.values()]
    first_better = []
    for rr in draws.values():
        better = {r["pass"] for r in rr if r["rmse_C_um"] < r["rmse_B_um"]}
        # first pass from which C stays better than B for every later pass
        # (from pass 2: at pass 1 B is still the uncalibrated nominal model)
        first_better.append(next((p for p in range(2, n_passes + 1)
                                  if all(x in better for x in range(p, n_passes + 1))), None))
    return {
        "n_draws": len(finals),
        "final_pass": n_passes,
        "final_rmse_um": {k: iqr_summary([f[f"final_rmse_{k}_um"] for f in finals]) for k in ESTS},
        "final_rmse_C_over_oracle": iqr_summary([f["final_rmse_C_um"] / f["final_rmse_oracle_um"] for f in finals]),
        "final_rmse_C_relative_to_true_removal": iqr_summary([f["final_rmse_C_um"] / f["final_true_mean_removal_um"]
                                                              for f in finals]),
        "final_true_mean_removal_um": iqr_summary([f["final_true_mean_removal_um"] for f in finals]),
        "first_true_mean_removal_um": iqr_summary([rr[0]["true_mean_removal_um"] for rr in draws.values()]),
        "per_pass_K_drop_fraction": _k_drop_stats(rows, ctx),
        "per_pass": _per_pass_stats(rows, n_passes),
        "mean_over_passes_rmse_um": {k: iqr_summary([f[f"mean_rmse_{k}_um"] for f in finals])
                                     for k in ESTS},
        "draws_C_better_than_B_at_final_pass": int(sum(f["final_rmse_C_um"] < f["final_rmse_B_um"] for f in finals)),
        "draws_C_better_than_A_at_final_pass": int(sum(f["final_rmse_C_um"] < f["final_rmse_A_um"] for f in finals)),
        "draws_C_better_than_D_at_final_pass": int(sum(f["final_rmse_C_um"] < f["final_rmse_D_um"] for f in finals)),
        "draws_C_better_than_D_mean_over_passes": int(sum(f["mean_rmse_C_um"] < f["mean_rmse_D_um"] for f in finals)),
        "final_rmse_D_over_C": iqr_summary([f["final_rmse_D_um"] / f["final_rmse_C_um"] for f in finals]),
        "C_vs_D": _c_vs_d(finals),
        "mean_rmse_D_over_C": iqr_summary([f["mean_rmse_D_um"] / f["mean_rmse_C_um"] for f in finals]),
        "draws_B_worse_than_A_at_final_pass": int(sum(f["final_rmse_B_um"] > f["final_rmse_A_um"] for f in finals)),
        "draws_C_within_2x_oracle_at_final_pass": int(sum(f["final_rmse_C_um"] <= 2 * f["final_rmse_oracle_um"]
                                                          for f in finals)),
        "pass_from_which_C_stays_better_than_B": iqr_summary([p for p in first_better if p is not None]),
        "draws_where_C_never_stays_better_than_B": int(sum(p is None for p in first_better)),
        "coverage_90": _coverage(finals),
        "coverage_90_after_pass1": {
            name: int(sum(inside(rr[0], name, tk) for rr in draws.values()))
            for name, tk in (("k_pad", "true_k_pad"), ("K0", "true_K0"), ("K", "true_K"))},
        "predictive_90": _predictive(rows, n_passes),
        "final_relative_error_C": {k: iqr_summary([abs(f[f"relerr_{k}"]) for f in finals]) for k in ("k_pad", "lam", "K")},
        "final_relative_width_C": {k: iqr_summary([f[f"relw_{k}"] for f in finals]) for k in ("k_pad", "lam", "K")},
        "B_abs_relative_error_k_pad": iqr_summary([abs(f["B_relerr_k_pad"]) for f in finals]),
        "noise_level_C_um": iqr_summary([f["median_sigma_C_um"] for f in finals]),
        "rejected_points_C": iqr_summary([f["median_rejected_points_C"] for f in finals]),
        "passes_redone_C": int(sum(f["passes_redone_C"] for f in finals)),
        **({"C2": c2} if (c2 := _c2_summary(rows, finals)) is not None else {}),
    }, finals


def robust_worlds(ctx: Context) -> list[str]:
    """Worlds with full robustness runs: the configured ones plus the validation world."""
    worlds = list(ctx.cfg["experiments"]["robustness_worlds"])
    return worlds + [w for w in validation_worlds(ctx) if w not in worlds]


def experiment_robustness(ctx: Context, out: Path, runs: dict, verbose: bool = True) -> dict:
    default = float(ctx.cfg["scan"]["noise_um"])
    worlds = robust_worlds(ctx)
    summary = {"n_draws": int(ctx.cfg["experiments"]["robustness_draws"]), "worlds": {}}
    all_rows, all_finals = [], []
    finals_by_world = {}
    for w in worlds:
        rows = runs[("robustness", w, default)]
        s, finals = _world_robustness(ctx, rows)
        summary["worlds"][w] = s
        finals_by_world[w] = finals
        all_rows += rows
        all_finals += finals
    if "matched" in finals_by_world:
        base = {f["draw"]: f for f in finals_by_world["matched"]}
        for w in worlds:
            if w == "matched":
                continue
            pairs = [(f, base[f["draw"]]) for f in finals_by_world[w] if f["draw"] in base]
            summary["worlds"][w]["paired_vs_matched"] = {
                "final_rmse_C_ratio": iqr_summary([f["final_rmse_C_um"] / b["final_rmse_C_um"] for f, b in pairs]),
                "final_rmse_oracle_ratio": iqr_summary([f["final_rmse_oracle_um"] / b["final_rmse_oracle_um"]
                                                        for f, b in pairs]),
            }
    write_csv(out / "robustness_per_pass.csv", all_rows)
    write_csv(out / "robustness_per_draw.csv", all_finals)
    write_json(out / "robustness_summary.json", summary)
    if verbose:
        for w, s in summary["worlds"].items():
            print(f"\nRobustness, {w} world, {s['n_draws']} hidden truths (final pass {s['final_pass']}):")
            for k in ESTS:
                v = s["final_rmse_um"][k]
                print(f"  final RMSE {k:6s}: median {v['median']:.3f} um  IQR [{v['q25']:.3f}, {v['q75']:.3f}]")
            pv = s["predictive_90"]["passes_2_on"]
            print(f"  C 90% predictive interval of next-pass mean removal: {pv['inside']}/{pv['n']} inside "
                  f"({pv['below']} below, {pv['above']} above)")
            for name, c in s["coverage_90"].items():
                if isinstance(c, dict) and "inside" in c:
                    print(f"  90% interval coverage {name}: {c['inside']}/{c['n']}")
    return summary


def experiment_ablation(ctx: Context, out: Path, runs: dict, verbose: bool = True) -> dict:
    """Constant force on every pass vs. the alternating low/high schedule (same draws).

    A null control re-runs the alternating schedule with a different particle-filter
    seed: its paired ratios show how much of any difference is Monte Carlo noise.
    """
    default = float(ctx.cfg["scan"]["noise_um"])
    world = ctx.cfg["experiments"].get("ablation_world", "matched")
    const_rows = runs[("ablation_constant", world, default)]
    control_rows = runs[("ablation_control", world, default)]
    n_ab = len({r["draw"] for r in const_rows})
    robust_rows = [r for r in runs[("robustness", world, default)] if r["draw"] < n_ab]
    finals_a = [_final_draw_summary(rr) for rr in by_draw(robust_rows).values()]
    finals_c = [_final_draw_summary(rr) for rr in by_draw(const_rows).values()]
    finals_r = [_final_draw_summary(rr) for rr in by_draw(control_rows).values()]
    n_draws = len(finals_a)
    write_csv(out / "ablation_constant_force_per_pass.csv", const_rows)
    write_csv(out / "ablation_control_reseeded_per_pass.csv", control_rows)
    n_passes = ctx.cfg["schedule"]["n_passes"]

    def per_pass(rows, key_fn):
        return [iqr_summary([key_fn(r) for r in rows if r["pass"] == p]) for p in range(1, n_passes + 1)]

    def first_pass_below(rows, name, thr):
        """First pass whose interval width is below thr; draws that never get there are
        returned separately (censored), not pooled into the median."""
        res, never = [], 0
        for rr in by_draw(rows).values():
            p = next((r["pass"] for r in rr if rel_width(r, name) < thr), None)
            if p is None:
                never += 1
            else:
                res.append(p)
        return {**iqr_summary(res), "n_never_reached": never}

    sched = {"alternating": (robust_rows, finals_a), "constant": (const_rows, finals_c),
             "alternating_reseeded": (control_rows, finals_r)}
    summary = {"world": world, "n_draws": n_draws, "forces_alternating_N": ctx.cfg["schedule"]["forces_N"],
               "force_constant_N": ctx.cfg["schedule"]["constant_force_N"], "by_schedule": {}}
    for name, (rows, finals) in sched.items():
        summary["by_schedule"][name] = {
            "final_relative_width": {k: iqr_summary([f[f"relw_{k}"] for f in finals]) for k in ("k_pad", "lam", "K")},
            "final_abs_relative_error": {k: iqr_summary([abs(f[f"relerr_{k}"]) for f in finals])
                                         for k in ("k_pad", "lam", "K")},
            "final_abs_corr_logk_logK": iqr_summary([abs(f["corr_logk_logK"]) for f in finals]),
            "final_rmse_C_um": iqr_summary([f["final_rmse_C_um"] for f in finals]),
            "coverage_90": _coverage(finals),
            "final_rmse_C_relative_to_true_removal": iqr_summary(
                [f["final_rmse_C_um"] / f["final_true_mean_removal_um"] for f in finals]),
            "final_rmse_C_over_oracle": iqr_summary([f["final_rmse_C_um"] / f["final_rmse_oracle_um"] for f in finals]),
            "pass_k_width_below_2pct": first_pass_below(rows, "k_pad", 0.02),
            "pass_lam_width_below_20pct": first_pass_below(rows, "lam", 0.20),
            "per_pass_relw_k_pad": per_pass(rows, lambda r: rel_width(r, "k_pad")),
            "per_pass_relw_K": per_pass(rows, lambda r: rel_width(r, "K")),
            "per_pass_relw_lam": per_pass(rows, lambda r: rel_width(r, "lam")),
            "per_pass_abs_corr": per_pass(rows, lambda r: abs(r["C_corr_logk_logK"])),
        }
    # paired comparison per draw, against the alternating run; the reseeded control
    # gives the spread expected from particle-filter Monte Carlo noise alone
    for other, finals_o, base, base_name in (("constant", finals_c, finals_a, "alternating"),
                                             ("control", finals_r, finals_a, "alternating"),
                                             ("constant", finals_c, finals_r, "control")):
        ratio_k = [fo["relw_k_pad"] / fb["relw_k_pad"] for fb, fo in zip(base, finals_o)]
        ratio_lam = [fo["relw_lam"] / fb["relw_lam"] for fb, fo in zip(base, finals_o)]
        summary[f"paired_width_ratio_{other}_over_{base_name}"] = {
            "k_pad": iqr_summary(ratio_k), "lam": iqr_summary(ratio_lam),
            f"draws_{other}_wider_k_pad": int(sum(r > 1 for r in ratio_k)),
            f"draws_{other}_wider_lam": int(sum(r > 1 for r in ratio_lam)),
        }
    write_json(out / "ablation_summary.json", summary)
    if verbose:
        print(f"\nIdentifiability ablation, {world} world (median over draws, final pass):")
        for name, s in summary["by_schedule"].items():
            print(f"  {name:20s}: 90% width k {s['final_relative_width']['k_pad']['median']:.2%}, "
                  f"lambda {s['final_relative_width']['lam']['median']:.2%}, K {s['final_relative_width']['K']['median']:.2%}; "
                  f"|corr(log k, log K)| {s['final_abs_corr_logk_logK']['median']:.2f}; "
                  f"RMSE C / true removal {s['final_rmse_C_relative_to_true_removal']['median']:.2%}")
        for other in ("constant", "control"):
            p = summary[f"paired_width_ratio_{other}_over_alternating"]
            print(f"  {other:8s} vs alternating, paired: k width ratio {p['k_pad']['median']:.3f} "
                  f"(wider in {p[f'draws_{other}_wider_k_pad']}/{n_draws}), lambda ratio {p['lam']['median']:.3f} "
                  f"(wider in {p[f'draws_{other}_wider_lam']}/{n_draws})")
    return summary


def experiment_noise(ctx: Context, out: Path, runs: dict, verbose: bool = True) -> dict:
    """Scan-noise sensitivity on the same draws at every noise level."""
    e = ctx.cfg["experiments"]
    world = e.get("noise_world", "realistic")
    n_draws = int(e.get("noise_draws", 0))
    default = float(ctx.cfg["scan"]["noise_um"])
    n_passes = ctx.cfg["schedule"]["n_passes"]
    levels = [float(s) for s in e["noise_levels_um"]]
    rows_out, by_level = [], {}
    for sigma in levels:
        if sigma == default:   # the default level is the robustness run of the same draws
            rows = [r for r in runs[("robustness", world, default)] if r["draw"] < n_draws]
        else:
            rows = runs[("noise", world, sigma)]
            rows_out += rows
        finals = [_final_draw_summary(rr) for rr in by_draw(rows).values()]
        by_level[f"{sigma:g}"] = {
            "noise_um": sigma,
            "final_rmse_C_um": iqr_summary([f["final_rmse_C_um"] for f in finals]),
            "final_rmse_oracle_um": iqr_summary([f["final_rmse_oracle_um"] for f in finals]),
            "final_rmse_C_over_oracle": iqr_summary([f["final_rmse_C_um"] / f["final_rmse_oracle_um"] for f in finals]),
            "final_relative_width": {k: iqr_summary([f[f"relw_{k}"] for f in finals]) for k in ("k_pad", "lam", "K")},
            "coverage_90": _coverage(finals),
            "predictive_90": _predictive(rows, n_passes),
            "noise_level_C_um": iqr_summary([f["median_sigma_C_um"] for f in finals]),
        }
    write_csv(out / "noise_sensitivity_per_pass.csv", rows_out)
    write_csv(out / "noise_sensitivity.csv", [
        {"noise_um": v["noise_um"], "rmse_C_median_um": v["final_rmse_C_um"]["median"],
         "rmse_C_q25_um": v["final_rmse_C_um"]["q25"], "rmse_C_q75_um": v["final_rmse_C_um"]["q75"],
         "rmse_oracle_median_um": v["final_rmse_oracle_um"]["median"],
         "relw_k_pad_median": v["final_relative_width"]["k_pad"]["median"],
         "relw_lam_median": v["final_relative_width"]["lam"]["median"],
         "predictive_coverage": v["predictive_90"]["passes_2_on"]["fraction"],
         "sigma_C_median_um": v["noise_level_C_um"]["median"]} for v in by_level.values()])
    summary = {"world": world, "n_draws": n_draws, "levels": by_level}
    write_json(out / "noise_sensitivity.json", summary)
    if verbose:
        print(f"\nScan-noise sensitivity, {world} world, {n_draws} draws (final-pass RMSE of C, median [IQR]):")
        for v in by_level.values():
            s = v["final_rmse_C_um"]
            print(f"  sigma {v['noise_um']:4g} um: {s['median']:.3f} [{s['q25']:.3f}, {s['q75']:.3f}] um; "
                  f"oracle {v['final_rmse_oracle_um']['median']:.3f}; predictive coverage "
                  f"{v['predictive_90']['passes_2_on']['fraction']:.1%}")
    return summary


def _crossing_records(rows: list[dict], n_passes: int, fixed: list[int]) -> list[dict]:
    recs = []
    for d, rr in by_draw(rows).items():
        rr = {r["pass"]: r for r in rr}
        true_cross = rr[1]["cross_true"]
        decisions = [(f"after_pass_{p}", p) for p in fixed]
        decisions.append(("last_pass_before_crossing", true_cross - 1))
        for label, p in decisions:
            if p < 1 or p > n_passes or p >= true_cross:
                continue  # only predictions made before the crossing count
            r = rr[p]
            recs.append({
                "world": r["world"], "draw": d, "decision": label, "after_pass": p, "true_crossing": true_cross,
                "pred_C_median": r["cross_C_median"], "pred_C_lo": r["cross_C_lo"], "pred_C_hi": r["cross_C_hi"],
                "err_C": r["cross_C_median"] - true_cross, "abs_err_C": abs(r["cross_C_median"] - true_cross),
                "in_band_C": bool(r["cross_C_lo"] <= true_cross <= r["cross_C_hi"]),
                "band_mass_C": r["cross_C_band_mass"],
                "pred_A": r["cross_A"], "err_A": r["cross_A"] - true_cross, "abs_err_A": abs(r["cross_A"] - true_cross),
                "pred_D": r["cross_D"], "err_D": r["cross_D"] - true_cross, "abs_err_D": abs(r["cross_D"] - true_cross),
            })
    return recs


def _model_free_ratios(rr: list[dict]) -> np.ndarray:
    """Model-free forecast (rule R) of K / K0 at the start of the next pass, after each scan.

    A least-squares fit of log(scanned mean removal per newton) against the cumulative
    scanned removal (at mid-pass), with a log-force term once two forces were seen,
    extrapolated to the start of the next pass; NaN after the first scan (no slope yet).
    """
    m = np.array([r["scan_mean_um"] for r in rr])
    F = np.array([r["force_N"] for r in rr])
    S_start = np.concatenate([[0.0], np.cumsum(m)[:-1]])
    y = np.log(m / F)
    out = np.full(len(rr), np.nan)
    for n in range(2, len(rr) + 1):            # forecast after scan n (1-based)
        cols = [np.ones(n), S_start[:n] + 0.5 * m[:n]]
        if n >= 3 and np.ptp(F[:n]) > 0:
            cols.append(np.log(F[:n]))
        coef = np.linalg.lstsq(np.column_stack(cols), y[:n], rcond=None)[0]
        out[n - 1] = np.exp(coef[1] * (S_start[n - 1] + m[n - 1]))      # slope = -lambda
    return out


def _change_from_ratio(passes, ratio, threshold: float, margin: float) -> int | None:
    """First change pass of a rule that changes when its forecast K/K0 for the next pass
    is below threshold * (1 + margin)."""
    for p, x in zip(passes, ratio):
        if np.isfinite(x) and x < threshold * (1.0 + margin):
            return int(p) + 1
    return None


def _c_quantile_key(q: float, risk_q: float) -> str:
    """Row column holding the q-quantile of C's crossing distribution."""
    for key, value in (("cross_C_median", 0.5), ("cross_C_risk", risk_q), ("cross_C_lo", 0.05)):
        if abs(q - value) < 1e-9:
            return key
    raise ValueError(f"no stored quantile {q} of C's crossing distribution")


def _change_passes(rr: list[dict], threshold: float, margins: dict, cost_ratios, risk_q: float) -> dict:
    """Change pass (None: not before the end of the run) of every decision rule on one draw."""
    passes = [r["pass"] for r in rr]
    d_ratio = np.array([r.get("D_next_ratio", np.nan) for r in rr], dtype=float)
    r_ratio = _model_free_ratios(rr)
    out = {"A": int(rr[0]["cross_A"]),
           "D": _change_from_ratio(passes, d_ratio, threshold, 0.0),
           "R": _change_from_ratio(passes, r_ratio, threshold, 0.0)}
    for c in cost_ratios:
        key = _c_quantile_key(1.0 / (1.0 + c), risk_q)
        out[f"C_r{c:g}"] = next((int(r["pass"]) + 1 for r in rr if r[key] <= r["pass"] + 1), None)
        m = margins.get(f"r{c:g}", {})
        out[f"D_r{c:g}"] = _change_from_ratio(passes, d_ratio, threshold, m.get("D", 0.0))
        out[f"R_r{c:g}"] = _change_from_ratio(passes, r_ratio, threshold, m.get("R", 0.0))
    return out


def _cost(err: np.ndarray, ratio: float) -> np.ndarray:
    """Cost of changing err passes late (err > 0, each pass on worn abrasive costs ``ratio``)
    or early (err < 0, each pass of abrasive life thrown away costs 1)."""
    return ratio * np.maximum(err, 0) + np.maximum(-err, 0)


def _decision_rules(rows: list[dict], n_passes: int, threshold: float, margins: dict | None = None,
                    cost_ratios=(1, 3, 19), risk_q: float = 0.25) -> dict:
    """Sequential abrasive-change decisions, scored against the true crossing.

    After each scanned pass n a rule decides whether pass n+1 runs on fresh
    abrasive. The ideal change pass is the true crossing c (first pass whose
    starting K < threshold * K0). Error = change pass - c (negative: early, abrasive
    life thrown away; positive: late, passes run on worn abrasive). For a cost
    ratio r (a late pass costs r times an early one) the cost of a draw is
    r * late passes + early passes.

    Rules: A (nominal crossing, fixed in advance); D and R (change when the
    forecast K/K0 of the next pass is below the threshold: D's MAP forecast, R the
    model-free extrapolation of the scanned means); for every cost ratio r,
    C_r (change when C's P(crossed by the next pass) >= 1 / (1 + r), the Bayes
    decision for that cost; no tuning) and D_r, R_r (D and R with a safety
    margin on the threshold tuned for that r on held-out draws, ``margins``).
    A rule that has not fired by the end of the run counts as changing at pass
    n_passes + 1; only draws with c <= n_passes + 1 are scored, and changes before
    pass n_passes + 1 on the other draws are counted as premature.
    """
    margins = margins or {}
    last = n_passes + 1
    per_rule: dict[str, list[tuple[int, int, bool]]] = {}
    premature: dict[str, int] = {}
    later_draws = 0
    for rr in by_draw(rows).values():
        c = int(rr[0]["cross_true"])
        change = _change_passes(rr, threshold, margins, cost_ratios, risk_q)
        if c > last:
            later_draws += 1
            for k, v in change.items():
                premature[k] = premature.get(k, 0) + int(v is not None and v < last)
            continue
        for k, v in change.items():
            per_rule.setdefault(k, []).append((c, last if v is None else v, v is None))
    out: dict = {"rules": {}, "cost_ratios": [float(c) for c in cost_ratios], "by_cost_ratio": {}}
    errs = {}
    for k, recs in per_rule.items():
        e = np.array([ch - c for c, ch, _ in recs], dtype=float)
        errs[k] = e
        out["rules"][k] = {"n": int(e.size), "mean_abs_err": float(np.mean(np.abs(e))), "exact": int(np.sum(e == 0)),
                           "within_1": int(np.sum(np.abs(e) <= 1)), "late": int(np.sum(e > 0)),
                           "early": int(np.sum(e < 0)), "mean_passes_late": float(np.mean(np.maximum(e, 0))),
                           "mean_passes_early": float(np.mean(np.maximum(-e, 0))),
                           "not_fired": int(sum(nf for _, _, nf in recs)),
                           "premature_on_later_draws": int(premature.get(k, 0)),
                           **{f"cost_r{c:g}": float(np.mean(_cost(e, c))) for c in cost_ratios}}
    for c in cost_ratios:
        tag = f"r{c:g}"
        if f"C_{tag}" not in errs:
            continue
        cC = _cost(errs[f"C_{tag}"], c)
        entry = {"C_threshold_probability": 1.0 / (1.0 + c), "margins": margins.get(tag, {"D": 0.0, "R": 0.0})}
        for other in (f"D_{tag}", f"R_{tag}", "D", "R"):
            diff = cC - _cost(errs[other], c)
            entry[f"C_minus_{other}"] = {"mean": float(np.mean(diff)), "ci90": bootstrap_ci(diff, np.mean),
                                         "p_perm": paired_permutation_p(diff)}
        costs = {k: out["rules"][k][f"cost_{tag}"] for k in out["rules"]}
        entry["lowest_cost_rule"] = min(costs, key=costs.get)
        out["by_cost_ratio"][tag] = entry
    out["draws_crossing_after_run"] = later_draws
    return out


def _tune_margins(rows: list[dict], n_passes: int, threshold: float, cost_ratios,
                  grid=tuple(np.round(np.arange(-0.30, 0.501, 0.01), 2))) -> dict:
    """Safety margins of rules D and R for each cost ratio, minimising the mean cost on
    (held-out) draws whose crossing falls inside the run; ties go to the smaller |margin|."""
    last = n_passes + 1
    worlds = sorted({r["world"] for r in rows})       # draws are paired across worlds: group by both
    draws = [rr for w in worlds for rr in by_draw([r for r in rows if r["world"] == w]).values()
             if int(rr[0]["cross_true"]) <= last]
    fc = {"D": lambda rr: np.array([r["D_next_ratio"] for r in rr], dtype=float),
          "R": _model_free_ratios}
    errs = {}
    for name, f in fc.items():
        ratios = [(rr, [r["pass"] for r in rr], f(rr)) for rr in draws]
        errs[name] = {m: np.array([(_change_from_ratio(p, x, threshold, m) or last) - int(rr[0]["cross_true"])
                                   for rr, p, x in ratios], dtype=float) for m in grid}
    out = {}
    for c in cost_ratios:
        entry = {"n_draws": len(draws)}
        if not draws:                    # nothing to tune on: no margin
            out[f"r{c:g}"] = {**entry, **{k: 0.0 for k in fc}}
            continue
        for name in fc:
            cost = {m: float(np.mean(_cost(e, c))) for m, e in errs[name].items()}
            best = min(grid, key=lambda m: (round(cost[m], 12), abs(m)))
            entry[name] = float(best)
            entry[f"{name}_held_out_cost"] = cost[best]
            entry[f"{name}_held_out_cost_no_margin"] = cost[0.0]
        out[f"r{c:g}"] = entry
    return out


def _crossing_by_decision(recs: list[dict], labels: list[str]) -> dict:
    out = {}
    for label in labels:
        sub = [r for r in recs if r["decision"] == label]
        if not sub:
            continue
        out[label] = {
            "n": len(sub),
            "abs_err_C_passes": iqr_summary([r["abs_err_C"] for r in sub]),
            "mean_abs_err_C_passes": float(np.mean([r["abs_err_C"] for r in sub])),
            "mean_err_C_passes": float(np.mean([r["err_C"] for r in sub])),
            "abs_err_A_passes": iqr_summary([r["abs_err_A"] for r in sub]),
            "mean_abs_err_A_passes": float(np.mean([r["abs_err_A"] for r in sub])),
            "mean_abs_err_D_passes": float(np.mean([r["abs_err_D"] for r in sub])),
            "within_1_pass_D": int(sum(r["abs_err_D"] <= 1 for r in sub)),
            "band_coverage_C": float(np.mean([r["in_band_C"] for r in sub])),
            "band_hits_C": int(sum(r["in_band_C"] for r in sub)),
            "exact_C": int(sum(r["abs_err_C"] == 0 for r in sub)),
            "within_1_pass_C": int(sum(r["abs_err_C"] <= 1 for r in sub)),
            "late_C": int(sum(r["err_C"] > 0 for r in sub)),
            "band_width_passes": iqr_summary([r["pred_C_hi"] - r["pred_C_lo"] for r in sub]),
            "band_predictive_mass": iqr_summary([r["band_mass_C"] for r in sub]),
        }
    return out


def _crossing_by_lead(rows: list[dict], n_passes: int) -> dict:
    """Crossing-prediction error versus lead time (passes before the true crossing),
    on a fixed population: the draws for which every lead 1..L_max is observable."""
    L_max = min(12, max(1, n_passes // 2))
    by = {d: {r["pass"]: r for r in rr} for d, rr in by_draw(rows).items()}
    pop = [d for d in by if L_max + 1 <= by[d][1]["cross_true"] <= n_passes + 1]
    out = []
    for L in range(1, L_max + 1):
        recs = [by[d][by[d][1]["cross_true"] - L] for d in pop]
        if not recs:
            continue
        tc = [r["cross_true"] for r in recs]
        out.append({
            "world": recs[0]["world"], "lead_passes": L, "n_draws": len(recs),
            "mean_abs_err_C": float(np.mean([abs(r["cross_C_median"] - t) for r, t in zip(recs, tc)])),
            "mean_abs_err_A": float(np.mean([abs(r["cross_A"] - t) for r, t in zip(recs, tc)])),
            "mean_abs_err_D": float(np.mean([abs(r["cross_D"] - t) for r, t in zip(recs, tc)])),
            "band_coverage_C": float(np.mean([r["cross_C_lo"] <= t <= r["cross_C_hi"] for r, t in zip(recs, tc)])),
        })
    cov = [r["band_coverage_C"] for r in out]
    return {"L_max": L_max, "population": len(pop),
            "crossing_range": [L_max + 1, n_passes + 1], "rows": out,
            "band_coverage_min": float(min(cov)) if cov else None,
            "band_coverage_max": float(max(cov)) if cov else None}


def experiment_abrasive_change(ctx: Context, out: Path, runs: dict, verbose: bool = True) -> dict:
    """How well C predicts the pass at which K drops below threshold * K0, in every robustness world."""
    default = float(ctx.cfg["scan"]["noise_um"])
    n_passes = ctx.cfg["schedule"]["n_passes"]
    fixed = [int(p) for p in ctx.cfg["experiments"]["crossing_decision_passes"]]
    labels = [f"after_pass_{p}" for p in fixed] + ["last_pass_before_crossing"]
    summary = {"threshold_fraction": ctx.threshold, "worlds": {}}
    all_recs, all_lead = [], []
    for w in robust_worlds(ctx):
        rows = runs[("robustness", w, default)]
        recs = _crossing_records(rows, n_passes, fixed)
        crosses = [rr[0]["cross_true"] for rr in by_draw(rows).values()]
        lead = _crossing_by_lead(rows, n_passes)
        summary["worlds"][w] = {
            "n_draws": len(crosses),
            "true_crossing_pass": iqr_summary(crosses),
            "true_crossings_within_experiment": int(sum(c <= n_passes for c in crosses)),
            "by_decision": _crossing_by_decision(recs, labels),
            "by_lead": lead,
            "decision_rules": _decision_rules(rows, n_passes, ctx.threshold, decision_margins(ctx),
                                              cost_ratios(ctx), risk_quantile(ctx)),
        }
        all_recs += recs
        all_lead += lead["rows"]
    summary["holm_family_size"] = apply_holm(summary["worlds"], "core")
    write_csv(out / "abrasive_change.csv", all_recs)
    write_csv(out / "abrasive_change_by_lead.csv", all_lead)
    write_json(out / "abrasive_change.json", summary)
    if verbose:
        for w, s in summary["worlds"].items():
            dr = s["decision_rules"]
            print(f"\nAbrasive-change decision rules, {w} world (change pass - true crossing):")
            for k, v in dr["rules"].items():
                costs = ", ".join(f"r={c:g}: {v[f'cost_r{c:g}']:.2f}" for c in dr["cost_ratios"])
                print(f"  {k:6s} n={v['n']} mean |err| {v['mean_abs_err']:.2f}, within 1 {v['within_1']}, "
                      f"late {v['late']}, early {v['early']}; cost {costs}")
            for tag, e in dr["by_cost_ratio"].items():
                cd, cr = e[f"C_minus_D_{tag}"], e[f"C_minus_R_{tag}"]
                print(f"  cost ratio {tag[1:]}: C - tuned D {cd['mean']:+.2f} [{cd['ci90'][0]:+.2f}, {cd['ci90'][1]:+.2f}], "
                      f"C - tuned R {cr['mean']:+.2f} [{cr['ci90'][0]:+.2f}, {cr['ci90'][1]:+.2f}]; "
                      f"lowest cost: {e['lowest_cost_rule']}")
            print(f"Abrasive-change prediction, {w} world (|predicted - true| crossing pass):")
            for label, v in s["by_decision"].items():
                print(f"  {label:26s} n={v['n']:3d}  C median {v['abs_err_C_passes']['median']:.1f} "
                      f"(mean {v['mean_abs_err_C_passes']:.2f}, within 1: {v['within_1_pass_C']})  band coverage "
                      f"{v['band_coverage_C']:.0%}  |  nominal A mean {v['mean_abs_err_A_passes']:.2f}")
    return summary


def experiment_breakdown(ctx: Context, out: Path, runs: dict, verbose: bool = True) -> dict:
    """Each mismatch of the realistic world on its own, on the same draws."""
    e = ctx.cfg["experiments"]
    default = float(ctx.cfg["scan"]["noise_um"])
    n_draws = int(e.get("breakdown_draws", 0))
    n_passes = ctx.cfg["schedule"]["n_passes"]
    fixed = [int(p) for p in e["crossing_decision_passes"]]
    order = ["matched"] + [w for w in ctx.worlds if w.startswith("only_")] + ["realistic"] + \
        [w for w in breakdown_extra_worlds(ctx) if not w.startswith("only_")]
    summary = {"n_draws": n_draws, "worlds": {}}
    table = []
    for w in order:
        key = ("robustness", w, default) if w in ("matched", "realistic") else ("breakdown", w, default)
        if key not in runs:
            continue
        rows = [r for r in runs[key] if r["draw"] < n_draws]
        if not rows:
            continue
        finals = [_final_draw_summary(rr) for rr in by_draw(rows).values()]
        recs = _crossing_records(rows, n_passes, fixed)
        cross = _crossing_by_decision(recs, ["last_pass_before_crossing", f"after_pass_{max(fixed)}"])
        last = cross.get("last_pass_before_crossing", {})
        pred = _predictive(rows, n_passes)
        cov = _coverage(finals)
        s = {
            "final_rmse_um": {k: iqr_summary([f[f"final_rmse_{k}_um"] for f in finals]) for k in ESTS},
            "final_rmse_C_over_oracle": iqr_summary([f["final_rmse_C_um"] / f["final_rmse_oracle_um"] for f in finals]),
            "mean_over_passes_rmse_um": {k: iqr_summary([f[f"mean_rmse_{k}_um"] for f in finals])
                                         for k in ESTS},
            "predictive_90": pred,
            "coverage_90": cov,
            "crossing": cross,
            "noise_level_C_um": iqr_summary([f["median_sigma_C_um"] for f in finals]),
            "rejected_points_C": iqr_summary([f["median_rejected_points_C"] for f in finals]),
            **({"C2": c2} if (c2 := _c2_summary(rows, finals)) is not None else {}),
        }
        summary["worlds"][w] = s
        table.append({
            "world": w, "n_draws": len(finals),
            "final_rmse_A_um": s["final_rmse_um"]["A"]["median"],
            "final_rmse_C_um": s["final_rmse_um"]["C"]["median"],
            "final_rmse_oracle_um": s["final_rmse_um"]["oracle"]["median"],
            "C_over_oracle": s["final_rmse_C_over_oracle"]["median"],
            "predictive_coverage": pred["passes_2_on"]["fraction"],
            "coverage_k_pad": cov["k_pad"]["fraction"], "coverage_lam": cov["lam"]["fraction"],
            "coverage_K": cov["K"]["fraction"],
            "crossing_mean_abs_err_last": last.get("mean_abs_err_C_passes", float("nan")),
            "crossing_band_coverage_last": last.get("band_coverage_C", float("nan")),
            "sigma_C_um": s["noise_level_C_um"]["median"],
        })
    write_csv(out / "world_breakdown.csv", table)
    write_json(out / "world_breakdown.json", summary)
    if verbose:
        print(f"\nMismatch breakdown ({n_draws} paired draws; medians at the final pass):")
        print("  world             RMSE C   oracle  C/oracle  pred.cov  cov k  cov lam  cov K  crossing |err| (last)")
        for t in table:
            print(f"  {t['world']:16s} {t['final_rmse_C_um']:7.3f} {t['final_rmse_oracle_um']:8.3f} "
                  f"{t['C_over_oracle']:8.2f} {t['predictive_coverage']:9.1%} {t['coverage_k_pad']:6.0%} "
                  f"{t['coverage_lam']:8.0%} {t['coverage_K']:6.0%} {t['crossing_mean_abs_err_last']:8.2f}")
    return summary


# --------------------------------------------------------------------------- driver


def run_all(ctx: Context, out_dir: str | Path = "results", verbose: bool = True, workers=1) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    contact = experiment_contact_sweep(ctx, out, verbose)
    model_info = experiment_model_info(ctx, out, verbose)
    tuning = None
    if str(ctx.cfg["filter"].get("rate_drift", "auto")) == "auto":
        if verbose:
            t = ctx.cfg["tuning"]
            print(f"\nTuning the wear-rate drift and the decision margins (held-out draws {t['draws'][0]}-"
                  f"{t['draws'][1] - 1} in {len(t['worlds'])} worlds)...")
        tuning = experiment_tuning(ctx, out, verbose, workers)
    main = experiment_main(ctx, out, verbose)
    if verbose:
        print(f"\nRunning {len(task_list(ctx))} multi-draw tasks (robustness worlds, ablation, noise levels, "
              f"single-mismatch worlds)...")
    runs = run_draws(ctx, verbose, workers)
    robust = experiment_robustness(ctx, out, runs, verbose)
    ablation = experiment_ablation(ctx, out, runs, verbose)
    noise = experiment_noise(ctx, out, runs, verbose)
    change = experiment_abrasive_change(ctx, out, runs, verbose)
    breakdown = experiment_breakdown(ctx, out, runs, verbose)
    summary = {"contact_sweep": contact, "model_info": model_info, "tuning": tuning, "main_run": main,
               "robustness": robust,
               "ablation": ablation, "noise_sensitivity": noise, "abrasive_change": change,
               "world_breakdown": breakdown}
    write_json(out / "summary.json", summary)
    src_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))))
    info = {"swt_version": __version__, "python": platform.python_version(), "numpy": np.__version__,
            "config_hash": config_hash(ctx.cfg), "source_hash": src_hash.hexdigest()[:12],
            "experiment_seconds": round(time.time() - t0, 1),
            "context_build_seconds": round(float(getattr(ctx, "build_seconds", float("nan"))), 1),
            "workers": resolve_workers(workers), "cpu_count": os.cpu_count(),
            "episodes": int(sum(len(v) for v in runs.values()) // ctx.cfg["schedule"]["n_passes"]),
            "config": ctx.cfg}
    write_json(out / "run_info.json", info)
    return summary


def extended_context(ctx: Context) -> Context:
    """The context of the extended study: the pre-registered worlds at ``extended.draws`` draws and
    the stress tests at ``extended.stress_draws``, nothing else; same tuned drift and margins."""
    import copy
    import dataclasses
    ext = ctx.cfg["extended"]
    cfg = copy.deepcopy(ctx.cfg)
    e = cfg["experiments"]
    e.update({"robustness_worlds": list(ext["worlds"]), "robustness_draws": int(ext["draws"]),
              "validation_worlds": [], "breakdown_worlds": list(ext.get("stress_worlds", [])),
              "breakdown_draws": int(ext.get("stress_draws", 0)), "ablation_draws": 0, "noise_draws": 0})
    out = dataclasses.replace(ctx, cfg=cfg)
    out.__dict__.update({k: v for k, v in ctx.__dict__.items() if k not in out.__dict__})
    return out


def run_extended(ctx: Context, out_dir: str | Path = "results/extended", core_dir: str | Path = "results",
                 verbose: bool = True, workers=1) -> dict:
    """The extended study (``run-all --extended``): tracker C2 against C, D and R in the pre-registered
    worlds at 100 draws, the stress tests, and Holm-adjusted decision comparisons over the core and
    extended worlds together. Needs the tuned drift and margins of the core run (in ``ctx``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ectx = extended_context(ctx)
    if verbose:
        print(f"\n[swt] extended study: {len(task_list(ectx))} multi-draw tasks "
              f"({', '.join(ectx.cfg['experiments']['robustness_worlds'])}; stress tests "
              f"{', '.join(breakdown_extra_worlds(ectx))})...")
    runs = run_draws(ectx, verbose, workers)
    robust = experiment_robustness(ectx, out, runs, verbose)
    change = experiment_abrasive_change(ectx, out, runs, verbose)
    stress = experiment_breakdown(ectx, out, runs, verbose)
    core = json.loads((Path(core_dir) / "abrasive_change.json").read_text())
    every = {**{f"core:{w}": v for w, v in core["worlds"].items()},
             **{f"extended:{w}": v for w, v in change["worlds"].items()}}
    n_all = apply_holm(every, "core and extended worlds", key="p_holm_all")
    change["holm_all_family_size"] = n_all
    write_json(out / "abrasive_change.json", change)
    summary = {"robustness": robust, "abrasive_change": change, "stress_tests": stress,
               "core_decision_p_holm_all": {w: v["decision_rules"]["by_cost_ratio"] for w, v in core["worlds"].items()}}
    write_json(out / "summary.json", summary)
    src_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))))
    write_json(out / "run_info.json", {
        "source_hash": src_hash.hexdigest()[:12], "config_hash": config_hash(ectx.cfg),
        "experiment_seconds": round(time.time() - t0, 1), "workers": resolve_workers(workers),
        "cpu_count": os.cpu_count(),
        "episodes": int(sum(len(v) for v in runs.values()) // ctx.cfg["schedule"]["n_passes"])})
    return summary
