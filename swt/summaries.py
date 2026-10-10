"""Per-draw and per-world summaries of episode rows (accuracy, calibration, C against D, C2 and C3)."""
from __future__ import annotations

import numpy as np

from .config import Context
from .stats import bootstrap_ci, gmean, iqr_summary, mean_or_none, paired_permutation_p, q

PARAMS = (("k_pad", "true_k_pad"), ("lam", "true_lam"), ("K", "true_K"), ("K0", "true_K0"))
ESTS = ("A", "B", "C", "D", "oracle")
STEADY_FROM = 4   # first pass of the 'steady state' comparison (D has three scans for its wear fit)


# --------------------------------------------------------------------------- rows


def by_draw(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(int(r["draw"]), []).append(r)
    return {d: sorted(rr, key=lambda r: r["pass"]) for d, rr in sorted(out.items())}


def rel_width(row: dict, name: str) -> float:
    return (row[f"C_{name}_hi"] - row[f"C_{name}_lo"]) / row[f"C_{name}_median"]


def inside(row: dict, name: str, true_key: str) -> bool:
    return row[f"C_{name}_lo"] <= row[true_key] <= row[f"C_{name}_hi"]


def outside_distance(row: dict, name: str, true_key: str) -> float:
    """Relative distance of the truth outside C's 90% interval (0 if inside)."""
    t, lo, hi = row[true_key], row[f"C_{name}_lo"], row[f"C_{name}_hi"]
    return float(max(lo - t, t - hi, 0.0) / t)


def interval_score(r: dict, alpha: float = 0.1) -> float:
    """Interval score (Gneiting & Raftery 2007) of C's central 90% interval for the
    mean removal, relative to the true value: width plus 2/alpha times any miss.
    Lower is better; it rewards narrow intervals only if they keep covering."""
    lo, hi, x = r["C_pred_mean_lo_um"], r["C_pred_mean_hi_um"], r["true_mean_removal_um"]
    return ((hi - lo) + (2 / alpha) * max(lo - x, 0.0) + (2 / alpha) * max(x - hi, 0.0)) / x


# --------------------------------------------------------------------------- one draw


def final_draw_summary(rows: list[dict]) -> dict:
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
        **{f"mean_rmse_{k}_um": gmean([r[f"rmse_{k}_um"] for r in later]) for k in ("A", "B", "C", "D", "oracle")},
        # start-up (passes 2-3: D has at most two scans for its wear fit), steady state (passes
        # STEADY_FROM..N) and the last two passes (one at each force of the alternating schedule)
        **{f"startup_rmse_{k}_um": gmean([r[f"rmse_{k}_um"] for r in later if r["pass"] < STEADY_FROM])
           for k in ("C", "D", "oracle")},
        **{f"steady_rmse_{k}_um": gmean([r[f"rmse_{k}_um"] for r in rows if r["pass"] >= STEADY_FROM])
           for k in ("C", "D", "oracle")},
        **{f"last_two_rmse_{k}_um": gmean([r[f"rmse_{k}_um"] for r in rows[-2:]]) for k in ("C", "D", "oracle")},
        **({"final_rmse_C2_um": last["rmse_C2_um"],
            "startup_rmse_C2_um": gmean([r["rmse_C2_um"] for r in later if r["pass"] < STEADY_FROM]),
            "steady_rmse_C2_um": gmean([r["rmse_C2_um"] for r in rows if r["pass"] >= STEADY_FROM])}
           if "rmse_C2_um" in last else {}),
        **({"final_rmse_C3_um": last["rmse_C3_um"],
            "startup_rmse_C3_um": gmean([r["rmse_C3_um"] for r in later if r["pass"] < STEADY_FROM]),
            "steady_rmse_C3_um": gmean([r["rmse_C3_um"] for r in rows if r["pass"] >= STEADY_FROM]),
            "final_C3_L_inf": last["C3_L_inf"], "final_C3_lam_ratio": last["C3_lam_median"] / last["true_lam"],
            "final_C_lam_ratio": last["C_lam_median"] / last["true_lam"]}
           if "rmse_C3_um" in last else {}),
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


# --------------------------------------------------------------------------- many draws


def parameter_coverage(finals: list[dict]) -> dict:
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


def c2_summary(rows: list[dict], finals: list[dict]) -> dict | None:
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
        "passes_2_to_5": mean_or_none([r["C2_pred_mean_inside"] for r in later if r["pass"] <= 5]),
        "passes_11_on": mean_or_none([r["C2_pred_mean_inside"] for r in later if r["pass"] >= 11]),
        "regional_inside_fraction_passes_2_on": float(np.mean([r["C2_region_inside_fraction"] for r in later])),
        "relative_width_passes_2_on": iqr_summary([(r["C2_pred_mean_hi_um"] - r["C2_pred_mean_lo_um"])
                                                   / r["true_mean_removal_um"] for r in later]),
    }
    out["final_level"] = iqr_summary([rr[-1]["C2_level"] for rr in by_draw(rows).values()])
    out["final_shape_rms"] = iqr_summary([rr[-1]["C2_shape_rms"] for rr in by_draw(rows).values()])
    return out


def c3_summary(rows: list[dict], finals: list[dict]) -> dict | None:
    """Tracker C3 against C on the same draws: accuracy ratios (C/C3 > 1: C3 more accurate, medians
    over draws with 90% bootstrap intervals and sign-flip p-values on the log ratio), the calibration
    of C3's 90% intervals, its loading estimate and the wear rate of both against the truth."""
    if not rows or "rmse_C3_um" not in rows[0]:
        return None
    later = [r for r in rows if r["pass"] >= 2]
    out = {"final_rmse_um": iqr_summary([f["final_rmse_C3_um"] for f in finals])}
    for label in ("startup", "steady", "final"):
        ratio = np.array([f[f"{label}_rmse_C_um"] / f[f"{label}_rmse_C3_um"] for f in finals])
        out[f"C_over_C3_{label}"] = {"median": float(np.median(ratio)), "ci90": bootstrap_ci(ratio),
                                     "C3_better": int(np.sum(ratio > 1)), "n": int(ratio.size),
                                     "p_perm": paired_permutation_p(np.log(ratio))}
    inside_ = [r["C3_pred_mean_inside"] for r in later]
    out["predictive_90"] = {
        "passes_2_on": {"inside": int(sum(inside_)), "n": len(inside_), "fraction": float(np.mean(inside_)),
                        "below": int(sum(r["true_mean_removal_um"] < r["C3_pred_mean_lo_um"] for r in later)),
                        "above": int(sum(r["true_mean_removal_um"] > r["C3_pred_mean_hi_um"] for r in later))},
        "passes_2_to_5": mean_or_none([r["C3_pred_mean_inside"] for r in later if r["pass"] <= 5]),
        "passes_11_on": mean_or_none([r["C3_pred_mean_inside"] for r in later if r["pass"] >= 11]),
        "regional_inside_fraction_passes_2_on": float(np.mean([r["C3_region_inside_fraction"] for r in later])),
    }
    out["final_L_inf"] = iqr_summary([f["final_C3_L_inf"] for f in finals])
    out["final_lam_over_true"] = {"C3": iqr_summary([f["final_C3_lam_ratio"] for f in finals]),
                                  "C": iqr_summary([f["final_C_lam_ratio"] for f in finals])}
    return out


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


def predictive_calibration(rows: list[dict], n_passes: int) -> dict:
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


def world_robustness(ctx: Context, rows: list[dict]) -> dict:
    n_passes = ctx.cfg["schedule"]["n_passes"]
    draws = by_draw(rows)
    finals = [final_draw_summary(rr) for rr in draws.values()]
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
        "coverage_90": parameter_coverage(finals),
        "coverage_90_after_pass1": {
            name: int(sum(inside(rr[0], name, tk) for rr in draws.values()))
            for name, tk in (("k_pad", "true_k_pad"), ("K0", "true_K0"), ("K", "true_K"))},
        "predictive_90": predictive_calibration(rows, n_passes),
        "final_relative_error_C": {k: iqr_summary([abs(f[f"relerr_{k}"]) for f in finals]) for k in ("k_pad", "lam", "K")},
        "final_relative_width_C": {k: iqr_summary([f[f"relw_{k}"] for f in finals]) for k in ("k_pad", "lam", "K")},
        "B_abs_relative_error_k_pad": iqr_summary([abs(f["B_relerr_k_pad"]) for f in finals]),
        "noise_level_C_um": iqr_summary([f["median_sigma_C_um"] for f in finals]),
        "rejected_points_C": iqr_summary([f["median_rejected_points_C"] for f in finals]),
        "passes_redone_C": int(sum(f["passes_redone_C"] for f in finals)),
        **({"C2": c2} if (c2 := c2_summary(rows, finals)) is not None else {}),
        **({"C3": c3} if (c3 := c3_summary(rows, finals)) is not None else {}),
    }, finals
