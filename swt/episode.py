"""One episode: every estimator through one sequence of passes on one hidden truth.

Worlds: every multi-draw experiment runs the hidden process in a named world
(see configs/default.yaml). ``matched`` is the tracker's own model; ``realistic``
adds a foam pad, force-control errors (calibration gain and line-to-line
ripple), two-stage abrasive wear and scanner artefacts, none of which the
estimators model; ``only_*`` worlds add one group of those at a time. Truth
parameters, wear fluctuations, force errors and scan noise come from the same
random streams in every world, so worlds are compared on paired draws.

Random streams (seed, stream, draw): 1 truth parameters, 2 wear fluctuations,
3 scans, 4 particle filter, 5 filter rollouts, 6 re-seeded filter (control),
7 force-calibration gain, 8 force ripple, 9 model-description checks, 12 tracker C2.
"""
from __future__ import annotations

import numpy as np

from .config import Context, rng_for
from .estimators import CalibrateOnce, Nominal, ParticleFilter, RefitEachPass
from .process import HiddenTruth, simulate_truth
from .scan import Scanner
from .stats import UM, rmse_um
from .tracker2 import C2Config, DiscrepancyTracker


# --------------------------------------------------------------------------- hidden truth


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


# --------------------------------------------------------------------------- estimator settings


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


# --------------------------------------------------------------------------- the episode


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
