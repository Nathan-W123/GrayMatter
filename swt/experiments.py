"""Experiments. Every number reported in the README is produced here and saved in results/."""
from __future__ import annotations

import csv
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import __version__
from .config import Context, config_hash, rng_for
from .estimators import CalibrateOnce, Nominal, ParticleFilter
from .pad import ContactGeometry, contact_fraction_sweep
from .geometry import Panel
from .process import HiddenTruth, simulate_truth
from .scan import Scanner

UM = 1e3  # mm -> micrometres

_SCHEDULE_ID = {"alternating": 0, "constant": 1}


# --------------------------------------------------------------------------- helpers


def draw_truth(ctx: Context, draw: int) -> HiddenTruth:
    p = ctx.priors.sample(rng_for(ctx.cfg["seed"], 1, draw))
    return HiddenTruth(float(p["k_pad"]), float(p["K0"]), float(p["lam"]))


def make_truth(ctx: Context, draw: int, schedule_kind: str):
    """Hidden-truth trajectory for a draw. The same wear-noise stream is used for a
    draw under either schedule, so schedule comparisons are paired."""
    truth = draw_truth(ctx, draw)
    return simulate_truth(ctx.model, truth, ctx.schedule(schedule_kind), ctx.wear_sigma,
                          rng_for(ctx.cfg["seed"], 2, draw), ctx.threshold)


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


# --------------------------------------------------------------------------- one episode


def run_episode(ctx: Context, draw: int, schedule_kind: str = "alternating", noise_um: float | None = None,
                experiment: str = "", record_maps: tuple[int, ...] = (), traj=None,
                full: bool = True, pf_stream: int = 4) -> tuple[list[dict], dict]:
    """Run all estimators through one sequence of passes on one hidden truth.

    Each pass: (1) every estimator predicts the removal map of the coming pass
    from the commanded action, (2) the hidden process executes it, (3) the
    scanner observes it, (4) estimators update on the scan.

    ``full=False`` (used for the extra noise levels and the ablation, which
    only report on C) skips baseline B and the abrasive-change rollouts.
    ``pf_stream`` selects the particle filter's random stream (a different value
    re-runs the identical problem with a different Monte Carlo seed).
    """
    cfg = ctx.cfg
    seed = cfg["seed"]
    noise = float(cfg["scan"]["noise_um"] if noise_um is None else noise_um)
    schedule = ctx.schedule(schedule_kind)
    if traj is None:
        traj = make_truth(ctx, draw, schedule_kind)
    truth = traj.truth
    scanner = Scanner(ctx.panel, noise, cfg["scan"]["stride"])
    scan_rng = rng_for(seed, 3, draw)
    nominal_cache = ctx.__dict__.setdefault("_nominal_cache", {})
    A = Nominal(ctx.model, ctx.priors, cache=nominal_cache)
    B = CalibrateOnce(ctx.model, ctx.priors, ctx.table, cache=nominal_cache) if full else None
    C = ParticleFilter(ctx.table, ctx.priors, ctx.pf, rng_for(seed, pf_stream, draw),
                       rollout_rng=rng_for(seed, 5, draw))
    cross_A = A.crossing_pass(schedule, ctx.threshold)
    rows, maps = [], {}
    K_oracle = truth.K0
    for n, action in enumerate(schedule):
        if n > 0:
            prev = schedule[n - 1]
            for est in (A, B, C):
                if est is not None:
                    est.advance(prev)
            # oracle: true k and lambda, true K of the previous pass, no knowledge of the fluctuation
            K_oracle = traj.K[n - 1] * np.exp(-truth.lam * traj.volume[n - 1])
        true_map = traj.removal[n]
        exposure_true = true_map / traj.K[n]
        preds = {"A": A.predict(action), "B": B.predict(action) if full else None, "C": C.predict(action),
                 "oracle": K_oracle * exposure_true}
        scan = scanner.observe(true_map, scan_rng)
        A.update(scan, action)
        if full:
            B.update(scan, action)
        C.update(scan, action, noise)
        s = C.summary()
        nan = float("nan")
        cr = (C.predict_crossing(n + 1, schedule, ctx.threshold) if full
              else {"median": nan, "lo": nan, "hi": nan, "band_mass": nan})
        row = {
            "experiment": experiment, "schedule": schedule_kind, "noise_um": noise, "draw": draw,
            "pass": n + 1, "force_N": action.force, "rpm": action.rpm,
            "true_k_pad": truth.k_pad, "true_K0": truth.K0, "true_lam": truth.lam, "true_K": float(traj.K[n]),
            "true_mean_removal_um": float(true_map.mean() * UM),
            "rmse_A_um": rmse_um(preds["A"], true_map),
            "rmse_B_um": rmse_um(preds["B"], true_map) if full else nan,
            "rmse_C_um": rmse_um(preds["C"], true_map), "rmse_oracle_um": rmse_um(preds["oracle"], true_map),
        }
        for name in ("k_pad", "K0", "lam", "K"):
            row[f"C_{name}_mean"] = s[name]["mean"]
            row[f"C_{name}_median"] = s[name]["median"]
            row[f"C_{name}_lo"] = s[name]["lo"]
            row[f"C_{name}_hi"] = s[name]["hi"]
        row.update({
            "C_corr_logk_logK": s["corr_logk_logK"], "C_stages": s["stages"],
            "B_k_pad": B.k_pad if full else nan, "B_K": B.K if full else nan,
            "cross_C_median": cr["median"], "cross_C_lo": cr["lo"], "cross_C_hi": cr["hi"],
            "cross_C_band_mass": cr["band_mass"],
            "cross_A": cross_A, "cross_true": traj.crossing_pass,
        })
        rows.append(row)
        if n + 1 in record_maps:
            maps[n + 1] = {"true": true_map, **{k: v for k, v in preds.items() if k != "oracle" and v is not None}}
    return rows, {"maps": maps, "traj": traj}


# --------------------------------------------------------------------------- experiments


def experiment_contact_sweep(ctx: Context, out: Path, verbose: bool = True) -> dict:
    cfg = ctx.cfg
    kmin, kmax, nk = cfg["experiments"]["contact_sweep_k"]
    ks = np.logspace(np.log10(kmin), np.log10(kmax), int(nk))
    F_ref = cfg["pad"]["reference_force_N"]
    forces = sorted(set(cfg["schedule"]["forces_N"]) | {F_ref})
    rows = []
    for F in forces:
        cf = contact_fraction_sweep(ctx.panel, ctx.pad_radius, F, ks)
        rows += [{"force_N": F, "k_pad": float(k), "contact_fraction": float(c)} for k, c in zip(ks, cf)]
    pr = ctx.priors
    # grid-convergence reference: the same sweep on a 4x finer grid at the reference force
    fine_grid = ctx.panel.grid / 4
    fine = Panel(ctx.panel.length_x, ctx.panel.width_y, ctx.panel.radius, fine_grid)
    cf_fine = contact_fraction_sweep(fine, ctx.pad_radius, F_ref, ks)
    rows += [{"force_N": F_ref, "k_pad": float(k), "contact_fraction": float(c), "grid_mm": fine_grid}
             for k, c in zip(ks, cf_fine)]
    for r in rows:
        r.setdefault("grid_mm", ctx.panel.grid)
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
    the near-independence of removed volume from stiffness, surrogate accuracy)."""
    cfg, pr, m = ctx.cfg, ctx.priors, ctx.model
    a = ctx.pad_radius
    R = ctx.panel.radius
    F_ref = float(cfg["pad"]["reference_force_N"])
    ks = np.geomspace(pr.k_low, pr.k_high, 9)
    vol = np.array([m.volume(m.exposure(F_ref, k)) for k in ks])          # per unit K
    rng = rng_for(cfg["seed"], 9)
    errs = []
    for k in np.exp(rng.uniform(np.log(pr.k_low), np.log(pr.k_high), 30)):
        for F in sorted(set(cfg["schedule"]["forces_N"]) | {cfg["schedule"]["constant_force_N"]}):
            exact = m.exposure(F, k)
            approx = F * ctx.table.map_full(F / k)
            errs.append(float(np.sqrt(np.mean((exact - approx) ** 2)) / np.sqrt(np.mean(exact**2))))
    nom = pr.log_mean()
    centre = np.array([[round(ctx.panel.length_x / 2 / ctx.panel.grid) * ctx.panel.grid,
                        round(ctx.panel.width_y / 2 / ctx.panel.grid) * ctx.panel.grid]])
    cg = ContactGeometry(ctx.panel, centre, a)
    forces_all = sorted(set(cfg["schedule"]["forces_N"]) | {cfg["schedule"]["constant_force_N"]})
    penetration = {f"{F:g}N": {"soft_k_low_mm": float(cg.penetration(F / pr.k_low)[0]),
                               "stiff_k_high_mm": float(cg.penetration(F / pr.k_high)[0])}
                   for F in forces_all}
    fresh = {f"{F:g}N": float(nom["K0"] * m.exposure(F, nom["k_pad"]).mean() * UM)
             for F in sorted(set(cfg["schedule"]["forces_N"]) | {cfg["schedule"]["constant_force_N"]})}
    info = {
        "grid_nodes": int(ctx.panel.n_pix), "scan_points": int(len(ctx.obs_index)),
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
                      "eta_max_mm3": float(ctx.table.eta[-1]),
                      "max_relative_rms_error": float(max(errs)),
                      "median_relative_rms_error": float(np.median(errs)), "n_checks": len(errs)},
    }
    write_json(out / "model_info.json", info)
    if verbose:
        print(f"\nModel: {info['grid_nodes']} grid nodes, {info['stations_per_pass']} stations/pass "
              f"({info['pass_duration_s']:.1f} s), sliding speed {info['sliding_speed_centre_mm_s']:.0f}-"
              f"{info['sliding_speed_rim_mm_s']:.0f} mm/s; removed volume per unit K varies by "
              f"{info['volume_per_unit_K_vs_k']['relative_spread']:.2%} across the k_pad prior; "
              f"surrogate max relative RMS error {info['surrogate']['max_relative_rms_error']:.2e}")
    return info


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


def experiment_main(ctx: Context, out: Path, verbose: bool = True) -> dict:
    e = ctx.cfg["experiments"]
    n_passes = ctx.cfg["schedule"]["n_passes"]
    main_draw = select_main_draw(ctx)
    rows, extra = run_episode(ctx, main_draw, "alternating", experiment="main",
                              record_maps=(e["early_pass"], n_passes))
    write_csv(out / "main_run.csv", rows)
    maps = extra["maps"]
    np.savez_compressed(out / "main_run_maps.npz",
                        **{f"pass{p}_{k}": v.reshape(ctx.panel.shape).astype(np.float32) * UM
                           for p, d in maps.items() for k, v in d.items()},
                        x=ctx.panel.x, y=ctx.panel.y)
    traj = extra["traj"]
    last = rows[-1]
    summary = {
        "draw": main_draw,
        "draw_selection": "config" if str(e.get("main_draw", "typical")) != "typical"
        else "robustness draw at the median standardised distance from the prior centre",
        "truth": {"k_pad": traj.truth.k_pad, "K0": traj.truth.K0, "lam": traj.truth.lam},
        "true_crossing_pass": traj.crossing_pass,
        "final_pass": n_passes,
        "final_rmse_um": {k: last[f"rmse_{k}_um"] for k in ("A", "B", "C", "oracle")},
        "first_pass_rmse_um": {k: rows[0][f"rmse_{k}_um"] for k in ("A", "B", "C", "oracle")},
        "early_pass": e["early_pass"],
        "early_rmse_um": {k: rows[e["early_pass"] - 1][f"rmse_{k}_um"] for k in ("A", "B", "C", "oracle")},
        "B_calibration": {"k_pad": last["B_k_pad"], "K": last["B_K"]},
        "C_final": {name: {"mean": last[f"C_{name}_mean"], "median": last[f"C_{name}_median"],
                           "lo": last[f"C_{name}_lo"], "hi": last[f"C_{name}_hi"]}
                    for name in ("k_pad", "K0", "lam", "K")},
        "true_final_K": last["true_K"],
        "C_final_inside_90": {"k_pad": inside(last, "k_pad", "true_k_pad"), "lam": inside(last, "lam", "true_lam"),
                              "K": inside(last, "K", "true_K"), "K0": inside(last, "K0", "true_K0")},
        "C_final_relative_distance_outside_90": {
            name: _outside(last, name, tk) for name, tk in
            (("k_pad", "true_k_pad"), ("lam", "true_lam"), ("K", "true_K"), ("K0", "true_K0"))},
        "predicted_mean_removal_um": {
            f"pass{p}": {k: float(np.mean(v) * UM) for k, v in maps[p].items()} for p in sorted(maps)},
        "A_crossing_pass": last["cross_A"],
        "C_crossing_prediction": [{"after_pass": r["pass"], "median": r["cross_C_median"], "lo": r["cross_C_lo"],
                                   "hi": r["cross_C_hi"]} for r in rows],
        "passes_C_better_than_B": [r["pass"] for r in rows if r["rmse_C_um"] < r["rmse_B_um"]],
    }
    write_json(out / "main_run.json", summary)
    if verbose:
        print("\nMain run (draw %d): truth k=%.4g N/mm^3, K0=%.3g mm^2/N, lambda=%.3g 1/mm^3, crossing pass %s"
              % (main_draw, traj.truth.k_pad, traj.truth.K0, traj.truth.lam, traj.crossing_pass))
        print("  pass  F[N]  true mean[um]  RMSE A    RMSE B    RMSE C   oracle [um]")
        for r in rows:
            print(f"  {r['pass']:4d} {r['force_N']:5.0f} {r['true_mean_removal_um']:10.2f} "
                  f"{r['rmse_A_um']:9.3f} {r['rmse_B_um']:9.3f} {r['rmse_C_um']:9.3f} {r['rmse_oracle_um']:9.3f}")
    return summary


def _final_draw_summary(rows: list[dict]) -> dict:
    last = rows[-1]
    return {
        "draw": last["draw"], "schedule": last["schedule"], "noise_um": last["noise_um"],
        "true_k_pad": last["true_k_pad"], "true_K0": last["true_K0"], "true_lam": last["true_lam"],
        "true_final_K": last["true_K"], "cross_true": last["cross_true"], "cross_A": last["cross_A"],
        **{f"final_rmse_{k}_um": last[f"rmse_{k}_um"] for k in ("A", "B", "C", "oracle")},
        "mean_rmse_C_um": float(np.mean([r["rmse_C_um"] for r in rows])),
        "mean_rmse_B_um": float(np.mean([r["rmse_B_um"] for r in rows])),
        "mean_rmse_A_um": float(np.mean([r["rmse_A_um"] for r in rows])),
        "n_passes_C_better_than_B": sum(r["rmse_C_um"] < r["rmse_B_um"] for r in rows),
        "in90_k_pad": inside(last, "k_pad", "true_k_pad"), "in90_lam": inside(last, "lam", "true_lam"),
        "in90_K": inside(last, "K", "true_K"), "in90_K0": inside(last, "K0", "true_K0"),
        "relw_k_pad": rel_width(last, "k_pad"), "relw_lam": rel_width(last, "lam"), "relw_K": rel_width(last, "K"),
        "relerr_k_pad": last["C_k_pad_mean"] / last["true_k_pad"] - 1,
        "relerr_lam": last["C_lam_mean"] / last["true_lam"] - 1,
        "relerr_K": last["C_K_mean"] / last["true_K"] - 1,
        "corr_logk_logK": last["C_corr_logk_logK"],
        "B_relerr_k_pad": last["B_k_pad"] / last["true_k_pad"] - 1,
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


_WORKER_CTX: Context | None = None


def _init_worker(cfg: dict) -> None:
    """Pool initializer: reuse a fork-inherited context, else rebuild it (spawn)."""
    global _WORKER_CTX
    if _WORKER_CTX is None or config_hash(_WORKER_CTX.cfg) != config_hash(cfg):
        from .config import build_context
        _WORKER_CTX = build_context(cfg)


def _draw_runs(ctx: Context, d: int) -> dict:
    """Every run for one hidden-truth draw.

    The alternating schedule at every scan-noise level (the default level is the
    robustness experiment) and the constant-force schedule at the default noise.
    """
    default = float(ctx.cfg["scan"]["noise_um"])
    levels = sorted({float(s) for s in ctx.cfg["experiments"]["noise_levels_um"]} | {default})
    out: dict[tuple[str, float], list[dict]] = {}
    traj = make_truth(ctx, d, "alternating")
    for sigma in levels:
        primary = sigma == default
        rows, _ = run_episode(ctx, d, "alternating", sigma, experiment="robustness" if primary else "noise",
                              traj=traj, full=primary)
        out[("alternating", sigma)] = rows
        if primary:
            # null control for the ablation: same problem, different filter seed
            rows, _ = run_episode(ctx, d, "alternating", sigma, experiment="ablation_control",
                                  traj=traj, full=False, pf_stream=6)
            out[("alternating_reseeded", sigma)] = rows
    traj = make_truth(ctx, d, "constant")
    rows, _ = run_episode(ctx, d, "constant", default, experiment="ablation", traj=traj, full=False)
    out[("constant", default)] = rows
    return out


def _draw_runs_worker(d: int) -> dict:
    return _draw_runs(_WORKER_CTX, d)


def resolve_workers(workers) -> int:
    if workers in (None, "auto"):
        return max(1, min(4, (os.cpu_count() or 1)))
    return max(1, int(workers))


def run_draws(ctx: Context, verbose: bool = True, workers=1) -> dict:
    """All multi-draw runs. Draws are independent and individually seeded, so the
    results do not depend on the number of worker processes."""
    global _WORKER_CTX
    n_draws = int(ctx.cfg["experiments"]["robustness_draws"])
    workers = resolve_workers(workers)
    t0 = time.time()
    per_draw: list[dict] = []
    if workers == 1:
        for d in range(n_draws):
            per_draw.append(_draw_runs(ctx, d))
            if verbose and (d + 1) % 10 == 0:
                print(f"  ... {d + 1}/{n_draws} draws ({time.time() - t0:.0f} s)")
    else:
        _WORKER_CTX = ctx  # inherited by forked workers; spawned workers rebuild it
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(ctx.cfg,)) as pool:
            for i, res in enumerate(pool.map(_draw_runs_worker, range(n_draws))):
                per_draw.append(res)
                if verbose and (i + 1) % 10 == 0:
                    print(f"  ... {i + 1}/{n_draws} draws ({time.time() - t0:.0f} s, {workers} workers)")
    runs: dict[tuple[str, float], list[dict]] = {}
    for res in per_draw:
        for key, rows in res.items():
            runs.setdefault(key, []).extend(rows)
    return runs


def _per_pass_stats(rows: list[dict], n_passes: int) -> dict:
    """Per-pass medians of every estimator's RMSE, C-vs-B win counts and coverage counts."""
    out = {"pass": list(range(1, n_passes + 1)), "median_rmse_um": {}, "draws_C_better_than_B": [],
           "coverage_90": {}}
    by_pass = [[r for r in rows if r["pass"] == p] for p in range(1, n_passes + 1)]
    for k in ("A", "B", "C", "oracle"):
        out["median_rmse_um"][k] = [q([r[f"rmse_{k}_um"] for r in rr], 50) for rr in by_pass]
    out["draws_C_better_than_B"] = [int(sum(r["rmse_C_um"] < r["rmse_B_um"] for r in rr)) for rr in by_pass]
    for name, tk in (("k_pad", "true_k_pad"), ("lam", "true_lam"), ("K", "true_K"), ("K0", "true_K0")):
        out["coverage_90"][name] = [int(sum(inside(r, name, tk) for r in rr)) for rr in by_pass]
    return out


def _k_drop_stats(rows: list[dict], ctx: Context) -> dict:
    """Fractional drop of the true K over single passes (wear is applied per pass)."""
    out = {}
    n_draws = int(ctx.cfg["experiments"]["robustness_draws"])
    sched = ctx.schedule("alternating")
    for p in (1, 2):   # first low-force and first high-force pass
        drops = []
        for d in range(n_draws):
            rr = {r["pass"]: r for r in rows if r["draw"] == d}
            drops.append(1 - rr[p + 1]["true_K"] / rr[p]["true_K"])
        out[f"pass{p}_{sched[p - 1].force:g}N"] = {**iqr_summary(drops), "max": float(max(drops))}
    return out


def experiment_robustness(ctx: Context, out: Path, all_rows: list[dict], verbose: bool = True) -> dict:
    n_draws = int(ctx.cfg["experiments"]["robustness_draws"])
    finals = [_final_draw_summary([r for r in all_rows if r["draw"] == d]) for d in range(n_draws)]
    write_csv(out / "robustness_per_pass.csv", all_rows)
    write_csv(out / "robustness_per_draw.csv", finals)
    n_passes = ctx.cfg["schedule"]["n_passes"]
    first_better = []
    for d in range(n_draws):
        rr = [r for r in all_rows if r["draw"] == d]
        better = [r["pass"] for r in rr if r["rmse_C_um"] < r["rmse_B_um"]]
        # first pass from which C stays better than B for every later pass
        # from pass 2: at pass 1 B is still the uncalibrated nominal model
        stay = next((p for p in range(2, n_passes + 1) if all(x in better for x in range(p, n_passes + 1))), None)
        first_better.append(stay)
    summary = {
        "n_draws": n_draws,
        "final_pass": n_passes,
        "final_rmse_um": {k: iqr_summary([f[f"final_rmse_{k}_um"] for f in finals]) for k in ("A", "B", "C", "oracle")},
        "final_true_mean_removal_um": iqr_summary([r["true_mean_removal_um"] for r in all_rows
                                                   if r["pass"] == n_passes]),
        "first_true_mean_removal_um": iqr_summary([r["true_mean_removal_um"] for r in all_rows if r["pass"] == 1]),
        "per_pass_K_drop_fraction": _k_drop_stats(all_rows, ctx),
        "per_pass": _per_pass_stats(all_rows, n_passes),
        "draws_B_error_exceeds_true_removal_at_final_pass": int(sum(
            r["rmse_B_um"] > r["true_mean_removal_um"] for r in all_rows if r["pass"] == n_passes)),
        "mean_over_passes_rmse_um": {k: iqr_summary([f[f"mean_rmse_{k}_um"] for f in finals]) for k in ("A", "B", "C")},
        "draws_C_better_than_B_at_final_pass": int(sum(f["final_rmse_C_um"] < f["final_rmse_B_um"] for f in finals)),
        "draws_C_better_than_A_at_final_pass": int(sum(f["final_rmse_C_um"] < f["final_rmse_A_um"] for f in finals)),
        "draws_B_worse_than_A_at_final_pass": int(sum(f["final_rmse_B_um"] > f["final_rmse_A_um"] for f in finals)),
        "pass_from_which_C_stays_better_than_B": iqr_summary([p for p in first_better if p is not None]),
        "draws_where_C_never_stays_better_than_B": int(sum(p is None for p in first_better)),
        "coverage_90": _coverage(finals),
        "coverage_90_after_pass1": {
            name: int(sum(inside(r, name, tk) for r in all_rows if r["pass"] == 1))
            for name, tk in (("k_pad", "true_k_pad"), ("K0", "true_K0"), ("K", "true_K"))},
        "final_relative_error_C": {k: iqr_summary([abs(f[f"relerr_{k}"]) for f in finals]) for k in ("k_pad", "lam", "K")},
        "final_relative_width_C": {k: iqr_summary([f[f"relw_{k}"] for f in finals]) for k in ("k_pad", "lam", "K")},
        "B_abs_relative_error_k_pad": iqr_summary([abs(f["B_relerr_k_pad"]) for f in finals]),
    }
    write_json(out / "robustness_summary.json", summary)
    if verbose:
        print(f"\nRobustness over {n_draws} hidden-truth draws (final pass {n_passes}):")
        for k in ("A", "B", "C", "oracle"):
            s = summary["final_rmse_um"][k]
            print(f"  final RMSE {k:6s}: median {s['median']:.3f} um  IQR [{s['q25']:.3f}, {s['q75']:.3f}]")
        for name, c in summary["coverage_90"].items():
            if isinstance(c, dict):
                print(f"  90% interval coverage {name}: {c['inside']}/{c['n']}")
    return summary


def experiment_ablation(ctx: Context, out: Path, robust_rows: list[dict], const_rows: list[dict],
                        control_rows: list[dict], verbose: bool = True) -> dict:
    """Constant force on every pass vs. the alternating low/high schedule (same draws).

    A null control re-runs the alternating schedule with a different particle-filter
    seed: its paired ratios show how much of any difference is Monte Carlo noise.
    """
    n_draws = int(ctx.cfg["experiments"]["robustness_draws"])
    finals_c = [_final_draw_summary([r for r in const_rows if r["draw"] == d]) for d in range(n_draws)]
    finals_r = [_final_draw_summary([r for r in control_rows if r["draw"] == d]) for d in range(n_draws)]
    write_csv(out / "ablation_constant_force_per_pass.csv", const_rows)
    write_csv(out / "ablation_control_reseeded_per_pass.csv", control_rows)
    finals_a = [_final_draw_summary([r for r in robust_rows if r["draw"] == d]) for d in range(n_draws)]
    n_passes = ctx.cfg["schedule"]["n_passes"]

    def per_pass(rows, key_fn):
        return [iqr_summary([key_fn(r) for r in rows if r["pass"] == p]) for p in range(1, n_passes + 1)]

    def first_pass_below(rows, name, thr):
        """First pass whose interval width is below thr; draws that never get there are
        returned separately (censored), not pooled into the median."""
        res, never = [], 0
        for d in range(n_draws):
            rr = sorted((r for r in rows if r["draw"] == d), key=lambda r: r["pass"])
            p = next((r["pass"] for r in rr if rel_width(r, name) < thr), None)
            if p is None:
                never += 1
            else:
                res.append(p)
        return {**iqr_summary(res), "n_never_reached": never}

    def rel_rmse(finals, rows):
        last = {r["draw"]: r for r in rows if r["pass"] == n_passes}
        return iqr_summary([f["final_rmse_C_um"] / last[f["draw"]]["true_mean_removal_um"] for f in finals])

    def over_oracle(finals):
        return iqr_summary([f["final_rmse_C_um"] / f["final_rmse_oracle_um"] for f in finals])

    sched = {"alternating": (robust_rows, finals_a), "constant": (const_rows, finals_c),
             "alternating_reseeded": (control_rows, finals_r)}
    summary = {"n_draws": n_draws, "forces_alternating_N": ctx.cfg["schedule"]["forces_N"],
               "force_constant_N": ctx.cfg["schedule"]["constant_force_N"], "by_schedule": {}}
    for name, (rows, finals) in sched.items():
        summary["by_schedule"][name] = {
            "final_relative_width": {k: iqr_summary([f[f"relw_{k}"] for f in finals]) for k in ("k_pad", "lam", "K")},
            "final_abs_relative_error": {k: iqr_summary([abs(f[f"relerr_{k}"]) for f in finals])
                                         for k in ("k_pad", "lam", "K")},
            "final_abs_corr_logk_logK": iqr_summary([abs(f["corr_logk_logK"]) for f in finals]),
            "final_rmse_C_um": iqr_summary([f["final_rmse_C_um"] for f in finals]),
            "coverage_90": _coverage(finals),
            "final_rmse_C_relative_to_true_removal": rel_rmse(finals, rows),
            "final_rmse_C_over_oracle": over_oracle(finals),
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
        print("\nIdentifiability ablation (median over draws, final pass):")
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
    n_draws = int(ctx.cfg["experiments"]["robustness_draws"])
    default = float(ctx.cfg["scan"]["noise_um"])
    levels = [float(s) for s in ctx.cfg["experiments"]["noise_levels_um"]]
    rows_out, by_level = [], {}
    for sigma in levels:
        rows = runs[("alternating", sigma)]
        if sigma != default:  # the default level is the robustness experiment
            rows_out += rows
        finals = [_final_draw_summary([r for r in rows if r["draw"] == d]) for d in range(n_draws)]
        by_level[f"{sigma:g}"] = {
            "noise_um": sigma,
            "final_rmse_C_um": iqr_summary([f["final_rmse_C_um"] for f in finals]),
            "final_rmse_oracle_um": iqr_summary([f["final_rmse_oracle_um"] for f in finals]),
            "final_relative_width": {k: iqr_summary([f[f"relw_{k}"] for f in finals]) for k in ("k_pad", "lam", "K")},
            "coverage_90": _coverage(finals),
        }
    write_csv(out / "noise_sensitivity_per_pass.csv", rows_out)
    write_csv(out / "noise_sensitivity.csv", [
        {"noise_um": v["noise_um"], "rmse_C_median_um": v["final_rmse_C_um"]["median"],
         "rmse_C_q25_um": v["final_rmse_C_um"]["q25"], "rmse_C_q75_um": v["final_rmse_C_um"]["q75"],
         "rmse_oracle_median_um": v["final_rmse_oracle_um"]["median"],
         "relw_k_pad_median": v["final_relative_width"]["k_pad"]["median"],
         "relw_lam_median": v["final_relative_width"]["lam"]["median"],
         "coverage_k_pad": v["coverage_90"]["k_pad"]["fraction"], "coverage_lam": v["coverage_90"]["lam"]["fraction"],
         "coverage_K": v["coverage_90"]["K"]["fraction"]} for v in by_level.values()])
    summary = {"n_draws": n_draws, "levels": by_level}
    write_json(out / "noise_sensitivity.json", summary)
    if verbose:
        print("\nScan-noise sensitivity (final-pass RMSE of C, median [IQR] over draws):")
        for v in by_level.values():
            s = v["final_rmse_C_um"]
            print(f"  sigma {v['noise_um']:4g} um: {s['median']:.3f} [{s['q25']:.3f}, {s['q75']:.3f}] um; "
                  f"k width {v['final_relative_width']['k_pad']['median']:.2%}, "
                  f"lambda width {v['final_relative_width']['lam']['median']:.2%}")
    return summary


def _crossing_by_lead(rows: list[dict], n_passes: int) -> dict:
    """Crossing-prediction error versus lead time (passes before the true crossing),
    on a fixed population: the draws for which every lead 1..L_max is observable."""
    L_max = min(12, max(1, n_passes // 2))
    draws = sorted({r["draw"] for r in rows})
    by = {d: {r["pass"]: r for r in rows if r["draw"] == d} for d in draws}
    pop = [d for d in draws if L_max + 1 <= by[d][1]["cross_true"] <= n_passes + 1]
    out = []
    for L in range(1, L_max + 1):
        recs = [by[d][by[d][1]["cross_true"] - L] for d in pop]
        if not recs:
            continue
        tc = [r["cross_true"] for r in recs]
        out.append({
            "lead_passes": L, "n_draws": len(recs),
            "mean_abs_err_C": float(np.mean([abs(r["cross_C_median"] - t) for r, t in zip(recs, tc)])),
            "mean_abs_err_A": float(np.mean([abs(r["cross_A"] - t) for r, t in zip(recs, tc)])),
            "band_coverage_C": float(np.mean([r["cross_C_lo"] <= t <= r["cross_C_hi"] for r, t in zip(recs, tc)])),
        })
    cov = [r["band_coverage_C"] for r in out]
    return {"L_max": L_max, "population": len(pop),
            "crossing_range": [L_max + 1, n_passes + 1], "rows": out,
            "band_coverage_min": float(min(cov)) if cov else None,
            "band_coverage_max": float(max(cov)) if cov else None}


def experiment_abrasive_change(ctx: Context, out: Path, robust_rows: list[dict], verbose: bool = True) -> dict:
    """How well C predicts the pass at which K drops below threshold * K0."""
    n_draws = int(ctx.cfg["experiments"]["robustness_draws"])
    n_passes = ctx.cfg["schedule"]["n_passes"]
    fixed = [int(p) for p in ctx.cfg["experiments"]["crossing_decision_passes"]]
    recs = []
    for d in range(n_draws):
        rr = {r["pass"]: r for r in robust_rows if r["draw"] == d}
        true_cross = rr[1]["cross_true"]
        decisions = [(f"after_pass_{p}", p) for p in fixed]
        decisions.append(("last_pass_before_crossing", true_cross - 1))
        for label, p in decisions:
            if p < 1 or p > n_passes or p >= true_cross:
                continue  # only predictions made before the crossing count
            r = rr[p]
            recs.append({
                "draw": d, "decision": label, "after_pass": p, "true_crossing": true_cross,
                "pred_C_median": r["cross_C_median"], "pred_C_lo": r["cross_C_lo"], "pred_C_hi": r["cross_C_hi"],
                "err_C": r["cross_C_median"] - true_cross, "abs_err_C": abs(r["cross_C_median"] - true_cross),
                "in_band_C": bool(r["cross_C_lo"] <= true_cross <= r["cross_C_hi"]),
                "band_mass_C": r["cross_C_band_mass"],
                "pred_A": r["cross_A"], "err_A": r["cross_A"] - true_cross, "abs_err_A": abs(r["cross_A"] - true_cross),
            })
    write_csv(out / "abrasive_change.csv", recs)
    summary = {"threshold_fraction": ctx.threshold,
               "true_crossing_pass": iqr_summary([robust_rows[i]["cross_true"] for i in range(0, len(robust_rows), n_passes)]),
               "true_crossings_within_experiment": int(sum(robust_rows[i]["cross_true"] <= n_passes
                                                           for i in range(0, len(robust_rows), n_passes))),
               "n_draws": n_draws, "by_decision": {}}
    for label in [f"after_pass_{p}" for p in fixed] + ["last_pass_before_crossing"]:
        sub = [r for r in recs if r["decision"] == label]
        if not sub:
            continue
        summary["by_decision"][label] = {
            "n": len(sub),
            "abs_err_C_passes": iqr_summary([r["abs_err_C"] for r in sub]),
            "mean_abs_err_C_passes": float(np.mean([r["abs_err_C"] for r in sub])),
            "abs_err_A_passes": iqr_summary([r["abs_err_A"] for r in sub]),
            "mean_abs_err_A_passes": float(np.mean([r["abs_err_A"] for r in sub])),
            "band_coverage_C": float(np.mean([r["in_band_C"] for r in sub])),
            "band_hits_C": int(sum(r["in_band_C"] for r in sub)),
            "exact_C": int(sum(r["abs_err_C"] == 0 for r in sub)),
            "within_1_pass_C": int(sum(r["abs_err_C"] <= 1 for r in sub)),
            "band_width_passes": iqr_summary([r["pred_C_hi"] - r["pred_C_lo"] for r in sub]),
            "band_predictive_mass": iqr_summary([r["band_mass_C"] for r in sub]),
        }
    summary["by_lead"] = _crossing_by_lead(robust_rows, n_passes)
    write_csv(out / "abrasive_change_by_lead.csv", summary["by_lead"]["rows"])
    write_json(out / "abrasive_change.json", summary)
    if verbose:
        print("\nAbrasive-change prediction (|predicted - true| crossing pass):")
        for label, s in summary["by_decision"].items():
            print(f"  {label:26s} n={s['n']:2d}  C median {s['abs_err_C_passes']['median']:.1f} "
                  f"(mean {s['mean_abs_err_C_passes']:.2f})  band coverage {s['band_coverage_C']:.0%}  |  "
                  f"nominal A median {s['abs_err_A_passes']['median']:.1f} (mean {s['mean_abs_err_A_passes']:.2f})")
    return summary


# --------------------------------------------------------------------------- driver


def run_all(ctx: Context, out_dir: str | Path = "results", verbose: bool = True, workers=1) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    contact = experiment_contact_sweep(ctx, out, verbose)
    model_info = experiment_model_info(ctx, out, verbose)
    main = experiment_main(ctx, out, verbose)
    if verbose:
        print(f"\nRunning {ctx.cfg['experiments']['robustness_draws']} hidden-truth draws "
              f"(alternating schedule at every noise level + constant-force ablation)...")
    runs = run_draws(ctx, verbose, workers)
    default = float(ctx.cfg["scan"]["noise_um"])
    robust_rows = runs[("alternating", default)]
    robust = experiment_robustness(ctx, out, robust_rows, verbose)
    ablation = experiment_ablation(ctx, out, robust_rows, runs[("constant", default)],
                                   runs[("alternating_reseeded", default)], verbose)
    noise = experiment_noise(ctx, out, runs, verbose)
    change = experiment_abrasive_change(ctx, out, robust_rows, verbose)
    summary = {"contact_sweep": contact, "model_info": model_info, "main_run": main, "robustness": robust, "ablation": ablation,
               "noise_sensitivity": noise, "abrasive_change": change}
    write_json(out / "summary.json", summary)
    import hashlib
    src_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))))
    info = {"swt_version": __version__, "python": platform.python_version(), "numpy": np.__version__,
            "config_hash": config_hash(ctx.cfg), "source_hash": src_hash.hexdigest()[:12], "experiment_seconds": round(time.time() - t0, 1),
            "context_build_seconds": round(float(getattr(ctx, "build_seconds", float("nan"))), 1),
            "workers": resolve_workers(workers), "cpu_count": os.cpu_count(),
            "config": ctx.cfg}
    write_json(out / "run_info.json", info)
    return summary
