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


def run_episode(ctx: Context, draw: int, world: str = "matched", schedule_kind: str = "alternating",
                noise_um: float | None = None, experiment: str = "", record_maps: tuple[int, ...] = (),
                traj=None, full: bool = True, pf_stream: int = 4) -> tuple[list[dict], dict]:
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
    B = CalibrateOnce(ctx.model, ctx.priors, ctx.table, oi, nominal_cache) if full else None
    D = RefitEachPass(ctx.priors, ctx.table, force_ref=float(cfg["pad"]["reference_force_N"])) if full else None
    C = ParticleFilter(ctx.table, ctx.priors, ctx.pf, rng_for(seed, pf_stream, draw),
                       rollout_rng=rng_for(seed, 5, draw), obs_shape=ctx.obs_shape)
    cross_A = A.crossing_pass(schedule, ctx.threshold)
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
        preds = {"A": A.predict(action), "B": B.predict(action) if full else None, "C": C.predict(action),
                 "D": D.predict(action) if full else None, "oracle": traj.oracle[n][oi]}
        p_lo, p_med, p_hi = C.predict_mean_removal(action)
        true_mean = float(true_obs.mean())
        scan = scanner.observe(traj.removal[n], scan_rng)
        A.update(scan, action)
        if full:
            B.update(scan, action)
            D.update(scan, action)
        C.update(scan, action, noise)
        s = C.summary()
        cr = (C.predict_crossing(n + 1, schedule, ctx.threshold) if full
              else {"median": nan, "lo": nan, "hi": nan, "band_mass": nan})
        row = {
            "experiment": experiment, "world": world, "schedule": schedule_kind, "noise_um": noise, "draw": draw,
            "pass": n + 1, "force_N": action.force, "rpm": action.rpm,
            "true_k_pad": truth.k_pad, "true_K0": truth.K0, "true_lam": truth.lam, "true_K": float(traj.K[n]),
            "true_force_gain": truth.force_gain,
            "true_mean_removal_um": true_mean * UM, "scan_mean_um": float(np.nanmean(scan)) * UM,
            "rmse_A_um": rmse_um(preds["A"], true_obs),
            "rmse_B_um": rmse_um(preds["B"], true_obs) if full else nan,
            "rmse_C_um": rmse_um(preds["C"], true_obs),
            "rmse_D_um": rmse_um(preds["D"], true_obs) if full else nan,
            "rmse_oracle_um": rmse_um(preds["oracle"], true_obs),
            "C_pred_mean_lo_um": float(p_lo) * UM, "C_pred_mean_median_um": float(p_med) * UM,
            "C_pred_mean_hi_um": float(p_hi) * UM, "C_pred_mean_inside": bool(p_lo <= true_mean <= p_hi),
        }
        for name in ("k_pad", "K0", "lam", "K"):
            for stat in ("mean", "median", "lo", "hi"):
                row[f"C_{name}_{stat}"] = s[name][stat]
        row.update({
            "C_corr_logk_logK": s["corr_logk_logK"], "C_stages": s["stages"],
            "C_sigma_um": s["sigma_um"], "C_profile_sd_um": s["profile_sd_um"], "C_wear_noise": s["wear_noise"], "C_transient_noise": s["transient_noise"], "C_min_path_diversity": s["min_path_diversity"], "C_outliers": s["outliers"], "C_redone": bool(s["redone"]),
            "B_k_pad": B.k_pad if full else nan, "B_K": B.K if full else nan,
            "D_k_pad": D.k_pad if full else nan, "D_K": D.K if full else nan, "D_lam": D.lam if full else nan,
            "cross_C_median": cr["median"], "cross_C_lo": cr["lo"], "cross_C_hi": cr["hi"],
            "cross_C_band_mass": cr["band_mass"],
            "cross_D": D.crossing_pass(n + 1, schedule, ctx.threshold) if full else nan,
            "cross_A": cross_A, "cross_true": traj.crossing_pass,
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
        "mean_rmse_C_um": float(np.mean([r["rmse_C_um"] for r in rows])),
        "mean_rmse_B_um": float(np.mean([r["rmse_B_um"] for r in rows])),
        "mean_rmse_A_um": float(np.mean([r["rmse_A_um"] for r in rows])),
        "mean_rmse_D_um": float(np.mean([r["rmse_D_um"] for r in rows])),
        "mean_rmse_oracle_um": float(np.mean([r["rmse_oracle_um"] for r in later])) if later else float("nan"),
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


def _c_vs_d(finals: list[dict]) -> dict:
    """C against D over draws: error ratios and win fractions with 90% bootstrap intervals."""
    out = {}
    for label, c_key, d_key in (("final", "final_rmse_C_um", "final_rmse_D_um"),
                                ("mean_over_passes", "mean_rmse_C_um", "mean_rmse_D_um")):
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
    """Worlds run only for the breakdown: one per mismatch group, plus stress tests."""
    skip = set(ctx.cfg["experiments"]["robustness_worlds"]) | set(validation_worlds(ctx))
    return [w for w in ctx.worlds if w not in skip and w != "matched"]


def tuning_tasks(ctx: Context) -> list[tuple]:
    t = ctx.cfg["tuning"]
    lo, hi = (int(x) for x in t["draws"])
    return [("tuning", w, d, float(v)) for v in t["rate_drift"] for w in t["worlds"] for d in range(lo, hi)]


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
    if kind == "tuning":
        import dataclasses
        tctx = dataclasses.replace(ctx, pf=dataclasses.replace(ctx.pf, rate_drift=task[3]))
        out[(kind, world, task[3])], _ = run_episode(tctx, d, world, "alternating", default, experiment=kind,
                                                     traj=traj, full=False)
        return out
    out[(kind, world, default)], _ = run_episode(ctx, d, world, "alternating", default, experiment=kind, traj=traj)
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
    summary = {"draws": [int(x) for x in t["draws"]], "first_draw": int(t["draws"][0]),
               "last_draw": int(t["draws"][1]) - 1, "n_draws": int(t["draws"][1]) - int(t["draws"][0]),
               "worlds": list(t["worlds"]), "criterion":
               "mean relative interval score of the 90% predictive interval of next-pass mean removal",
               "by_value": by_value, "chosen_rate_drift": chosen, "chosen_index": by_value.index(best),
               "chosen": best, "constant_rate": next((v for v in by_value if v["rate_drift"] == 0.0), None)}
    summary["config_hash"] = _tuning_hash(ctx.cfg)
    write_csv(out / "tuning.csv", table)
    write_json(out / "tuning.json", summary)
    set_rate_drift(ctx, chosen)
    if verbose:
        print(f"\nTuning of the wear-rate drift on held-out draws {t['draws'][0]}-{t['draws'][1] - 1} "
              f"({', '.join(t['worlds'])}):")
        for v in by_value:
            cells = "  ".join(f"{w}: cov {x['coverage']:.1%} width {x['relative_width']:.2%}"
                              for w, x in v["worlds"].items())
            print(f"  drift {v['rate_drift']:5.3f}: interval score {v['mean_interval_score']:.4f}  {cells}")
        print(f"  chosen rate_drift = {chosen:g}")
    return summary


def _tuning_hash(cfg: dict) -> str:
    """Hash of everything the tuning result depends on (the config without the tuned value)."""
    import copy
    c = copy.deepcopy(cfg)
    c["filter"].pop("rate_drift", None)
    return config_hash(c)


def set_rate_drift(ctx: Context, value: float) -> None:
    """Use ``value`` for the tracker's wear-rate drift from now on (config and filter)."""
    import dataclasses
    ctx.cfg["filter"]["rate_drift"] = float(value)
    ctx.pf = dataclasses.replace(ctx.pf, rate_drift=float(value))


def tuned_rate_drift(ctx: Context, out: Path, verbose: bool = True, workers=1) -> float:
    """The tuned drift: from ``out``/tuning.json if it was made with this configuration, else tune now."""
    if str(ctx.cfg["filter"].get("rate_drift", "auto")) != "auto":
        return float(ctx.cfg["filter"]["rate_drift"])
    path = out / "tuning.json"
    if path.exists():
        info = json.loads(path.read_text())
        if info.get("config_hash") == _tuning_hash(ctx.cfg):
            set_rate_drift(ctx, info["chosen_rate_drift"])
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


def _decision_rules(rows: list[dict], n_passes: int, threshold: float) -> dict:
    """Sequential abrasive-change decisions, scored against the true crossing.

    After each scanned pass n a rule decides whether pass n+1 runs on fresh
    abrasive. C: when its predicted probability that the crossing happens by
    pass n+1 is at least 50% (median of its crossing distribution <= n+1).
    D: when its point forecast is <= n+1. A: at its nominal crossing pass,
    fixed in advance. R (reactive, no model): when the latest scan's mean
    removal per newton falls below ``threshold`` times the first scan's.
    The ideal change pass is the true crossing c (first pass whose starting
    K < threshold * K0). Error = change pass - c (negative: early, abrasive
    life wasted; positive: late, passes run on worn abrasive). A rule that has
    not fired by the end of the run counts as changing at pass n_passes + 1
    or later; only draws with c <= n_passes + 1 are scored, and changes
    before pass n_passes + 1 on the other draws are counted as premature.
    """
    last = n_passes + 1
    out = {}
    per_rule: dict[str, list[tuple[int, int, bool]]] = {k: [] for k in ("C", "D", "A", "R")}
    premature = {k: 0 for k in per_rule}
    later_draws = 0
    for rr in by_draw(rows).values():
        c = int(rr[0]["cross_true"])
        ref = rr[0]["scan_mean_um"] / rr[0]["force_N"]
        change = {"A": int(rr[0]["cross_A"])}
        for key, fired in (("C", lambda r: r["cross_C_median"] <= r["pass"] + 1),
                           ("D", lambda r: r["cross_D"] <= r["pass"] + 1),
                           ("R", lambda r: r["scan_mean_um"] / r["force_N"] < threshold * ref)):
            change[key] = next((int(r["pass"]) + 1 for r in rr if fired(r)), None)
        if c > last:
            later_draws += 1
            for k, v in change.items():
                premature[k] += int(v is not None and v < last)
            continue
        for k, v in change.items():
            per_rule[k].append((c, last if v is None else v, v is None))
    for k, recs in per_rule.items():
        if not recs:
            continue
        e = np.array([ch - c for c, ch, _ in recs], dtype=float)
        out[k] = {"n": int(e.size), "mean_abs_err": float(np.mean(np.abs(e))), "exact": int(np.sum(e == 0)),
                  "within_1": int(np.sum(np.abs(e) <= 1)), "late": int(np.sum(e > 0)), "early": int(np.sum(e < 0)),
                  "mean_passes_late": float(np.mean(np.maximum(e, 0))),
                  "mean_passes_early": float(np.mean(np.maximum(-e, 0))),
                  "not_fired": int(sum(nf for _, _, nf in recs)),
                  "premature_on_later_draws": int(premature[k])}
    out["draws_crossing_after_run"] = later_draws
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
            "decision_rules": _decision_rules(rows, n_passes, ctx.threshold),
        }
        all_recs += recs
        all_lead += lead["rows"]
    write_csv(out / "abrasive_change.csv", all_recs)
    write_csv(out / "abrasive_change_by_lead.csv", all_lead)
    write_json(out / "abrasive_change.json", summary)
    if verbose:
        for w, s in summary["worlds"].items():
            dr = s["decision_rules"]
            print(f"\nAbrasive-change decision rules, {w} world (change pass - true crossing):")
            for k in ("C", "D", "A", "R"):
                if k in dr:
                    v = dr[k]
                    print(f"  {k}: n={v['n']} mean |err| {v['mean_abs_err']:.2f}, exact {v['exact']}, within 1 {v['within_1']}, "
                          f"late {v['late']} (mean {v['mean_passes_late']:.2f} passes), early {v['early']} "
                          f"(mean {v['mean_passes_early']:.2f}); premature on later draws {v['premature_on_later_draws']}")
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
        }
        summary["worlds"][w] = s
        table.append({
            "world": w, "n_draws": len(finals),
            "final_rmse_A_um": s["final_rmse_um"]["A"]["median"], "final_rmse_B_um": s["final_rmse_um"]["B"]["median"],
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
            print(f"\nTuning the wear-rate drift ({len(tuning_tasks(ctx))} held-out episodes)...")
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
