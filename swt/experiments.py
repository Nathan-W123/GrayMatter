"""Experiments. Every number reported in the README is produced here and saved in results/.

``run_all`` is the core study (``python -m swt run-all``); ``run_extended`` the extended one
(``run-all --extended``). The pieces live in their own modules: one episode in ``episode``,
the multi-draw tasks in ``runner``, summaries of episode rows in ``summaries``, the
abrasive-change decisions in ``decisions``, the held-out tuning in ``tuning`` and the
model-description experiments in ``describe``.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from . import __version__
from .config import Context, config_hash
from .decisions import (apply_holm, cost_ratios, crossing_by_decision, crossing_by_lead, crossing_records,
                        decision_margins, decision_rules, risk_quantile)
from .describe import experiment_contact_sweep, experiment_model_info
from .episode import draw_truth, run_episode
from .runner import breakdown_extra_worlds, resolve_workers, robust_worlds, run_draws, source_hash, task_list
from .stats import UM, iqr_summary, write_csv, write_json
from .summaries import (ESTS, PARAMS, by_draw, c2_summary, c3_summary, final_draw_summary, inside, outside_distance,
                        parameter_coverage, predictive_calibration, rel_width, world_robustness)
from .tuning import experiment_tuning, tuned_rate_drift

__all__ = ["run_all", "run_extended", "experiment_main", "experiment_contact_sweep", "experiment_model_info",
           "experiment_tuning", "tuned_rate_drift"]


# --------------------------------------------------------------------------- main run


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
        "C_final_relative_distance_outside_90": {name: outside_distance(last, name, tk) for name, tk in PARAMS},
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


# --------------------------------------------------------------------------- multi-draw experiments


def experiment_robustness(ctx: Context, out: Path, runs: dict, verbose: bool = True) -> dict:
    default = float(ctx.cfg["scan"]["noise_um"])
    worlds = robust_worlds(ctx)
    summary = {"n_draws": int(ctx.cfg["experiments"]["robustness_draws"]), "worlds": {}}
    all_rows, all_finals = [], []
    finals_by_world = {}
    for w in worlds:
        rows = runs[("robustness", w, default)]
        s, finals = world_robustness(ctx, rows)
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
    finals_a = [final_draw_summary(rr) for rr in by_draw(robust_rows).values()]
    finals_c = [final_draw_summary(rr) for rr in by_draw(const_rows).values()]
    finals_r = [final_draw_summary(rr) for rr in by_draw(control_rows).values()]
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
            "coverage_90": parameter_coverage(finals),
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
        finals = [final_draw_summary(rr) for rr in by_draw(rows).values()]
        by_level[f"{sigma:g}"] = {
            "noise_um": sigma,
            "final_rmse_C_um": iqr_summary([f["final_rmse_C_um"] for f in finals]),
            "final_rmse_oracle_um": iqr_summary([f["final_rmse_oracle_um"] for f in finals]),
            "final_rmse_C_over_oracle": iqr_summary([f["final_rmse_C_um"] / f["final_rmse_oracle_um"] for f in finals]),
            "final_relative_width": {k: iqr_summary([f[f"relw_{k}"] for f in finals]) for k in ("k_pad", "lam", "K")},
            "coverage_90": parameter_coverage(finals),
            "predictive_90": predictive_calibration(rows, n_passes),
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
        recs = crossing_records(rows, n_passes, fixed)
        crosses = [rr[0]["cross_true"] for rr in by_draw(rows).values()]
        lead = crossing_by_lead(rows, n_passes)
        summary["worlds"][w] = {
            "n_draws": len(crosses),
            "true_crossing_pass": iqr_summary(crosses),
            "true_crossings_within_experiment": int(sum(c <= n_passes for c in crosses)),
            "by_decision": crossing_by_decision(recs, labels),
            "by_lead": lead,
            "decision_rules": decision_rules(rows, n_passes, ctx.threshold, decision_margins(ctx),
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
        finals = [final_draw_summary(rr) for rr in by_draw(rows).values()]
        recs = crossing_records(rows, n_passes, fixed)
        cross = crossing_by_decision(recs, ["last_pass_before_crossing", f"after_pass_{max(fixed)}"])
        last = cross.get("last_pass_before_crossing", {})
        pred = predictive_calibration(rows, n_passes)
        cov = parameter_coverage(finals)
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
            **({"C2": c2} if (c2 := c2_summary(rows, finals)) is not None else {}),
            **({"C3": c3} if (c3 := c3_summary(rows, finals)) is not None else {}),
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


# --------------------------------------------------------------------------- drivers


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
    info = {"swt_version": __version__, "python": platform.python_version(), "numpy": np.__version__,
            "config_hash": config_hash(ctx.cfg), "source_hash": source_hash(),
            "experiment_seconds": round(time.time() - t0, 1),
            "context_build_seconds": round(float(getattr(ctx, "build_seconds", float("nan"))), 1),
            "workers": resolve_workers(workers), "cpu_count": os.cpu_count(),
            "episodes": int(sum(len(v) for v in runs.values()) // ctx.cfg["schedule"]["n_passes"]),
            "config": ctx.cfg}
    write_json(out / "run_info.json", info)
    return summary


def extended_context(ctx: Context) -> Context:
    """The context of the extended study: the pre-registered worlds at ``extended.draws`` draws and
    the stress tests at ``extended.stress_draws``, nothing else; same tuned drift and margins. With
    ``extended.with_C3``, tracker C3 runs everywhere and the core worlds ``extended.repeat_core_worlds``
    are run again (same draws, so C's results there equal the core run's) to compare C3 with C."""
    import copy
    import dataclasses
    ext = ctx.cfg["extended"]
    cfg = copy.deepcopy(ctx.cfg)
    e = cfg["experiments"]
    repeat = list(ext.get("repeat_core_worlds", [])) if ext.get("with_C3", False) else []
    e.update({"robustness_worlds": repeat + list(ext["worlds"]), "robustness_draws": int(ext["draws"]),
              "with_C3": bool(ext.get("with_C3", False)),
              "validation_worlds": [], "breakdown_worlds": list(ext.get("stress_worlds", [])),
              "breakdown_draws": int(ext.get("stress_draws", 0)), "ablation_draws": 0, "noise_draws": 0})
    out = dataclasses.replace(ctx, cfg=cfg)
    out.__dict__.update({k: v for k, v in ctx.__dict__.items() if k not in out.__dict__})
    return out


def run_extended(ctx: Context, out_dir: str | Path = "results/extended", core_dir: str | Path = "results",
                 verbose: bool = True, workers=1) -> dict:
    """The extended study (``run-all --extended``): trackers C2 and C3 against C, D and R in the
    pre-registered worlds at 100 draws (C3 also in the repeated core worlds), the stress tests, and
    Holm-adjusted decision comparisons: C against D and R over the core and extended worlds together
    (each world once), and C3 against C, D and R over the extended worlds. Needs the tuned drift and
    margins of the core run (in ``ctx``)."""
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
             **{f"extended:{w}": v for w, v in change["worlds"].items() if w not in core["worlds"]}}
    n_all = apply_holm(every, "core and extended worlds", key="p_holm_all")
    change["holm_all_family_size"] = n_all
    if ectx.cfg["experiments"].get("with_C3", False):
        change["holm_C3_family_size"] = apply_holm(change["worlds"], "C3 in the extended worlds", key="p_holm_C3",
                                                   tracker="C3")
    write_json(out / "abrasive_change.json", change)
    summary = {"robustness": robust, "abrasive_change": change, "stress_tests": stress,
               "core_decision_p_holm_all": {w: v["decision_rules"]["by_cost_ratio"] for w, v in core["worlds"].items()}}
    write_json(out / "summary.json", summary)
    write_json(out / "run_info.json", {
        "source_hash": source_hash(), "config_hash": config_hash(ectx.cfg),
        "experiment_seconds": round(time.time() - t0, 1), "workers": resolve_workers(workers),
        "cpu_count": os.cpu_count(),
        "episodes": int(sum(len(v) for v in runs.values()) // ctx.cfg["schedule"]["n_passes"])})
    return summary
