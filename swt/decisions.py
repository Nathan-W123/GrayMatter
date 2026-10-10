"""Abrasive-change forecasts and sequential decision rules, scored against the true crossing."""
from __future__ import annotations

import numpy as np

from .config import Context
from .stats import bootstrap_ci, holm_adjust, iqr_summary, paired_permutation_p
from .summaries import by_draw


# --------------------------------------------------------------------------- settings


def cost_ratios(ctx: Context) -> list[float]:
    return [float(c) for c in ctx.cfg["abrasive"].get("cost_ratios", [1, 3, 19])]


def risk_quantile(ctx: Context) -> float:
    return float(ctx.cfg["abrasive"].get("risk_quantile", 0.25))


def decision_margins(ctx: Context) -> dict:
    """Safety margins of D and R per cost ratio, tuned on the held-out draws (empty: none)."""
    return ctx.__dict__.get("decision_margins", {})


# --------------------------------------------------------------------------- crossing forecasts


def crossing_records(rows: list[dict], n_passes: int, fixed: list[int]) -> list[dict]:
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


def crossing_by_decision(recs: list[dict], labels: list[str]) -> dict:
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


def crossing_by_lead(rows: list[dict], n_passes: int) -> dict:
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


# --------------------------------------------------------------------------- decision rules


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
        if "cross_C3_median" in rr[0]:      # tracker C3: the same Bayes rule on its own crossing distribution
            k3 = key.replace("cross_C_", "cross_C3_")
            out[f"C3_r{c:g}"] = next((int(r["pass"]) + 1 for r in rr if r[k3] <= r["pass"] + 1), None)
    return out


def _cost(err: np.ndarray, ratio: float) -> np.ndarray:
    """Cost of changing err passes late (err > 0, each pass on worn abrasive costs ``ratio``)
    or early (err < 0, each pass of abrasive life thrown away costs 1)."""
    return ratio * np.maximum(err, 0) + np.maximum(-err, 0)


def decision_rules(rows: list[dict], n_passes: int, threshold: float, margins: dict | None = None,
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
    decision for that cost; no tuning), C3_r (the same rule for tracker C3, when
    it was run) and D_r, R_r (D and R with a safety margin on the threshold tuned
    for that r on held-out draws, ``margins``).
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
        if f"C3_{tag}" in errs:
            c3 = _cost(errs[f"C3_{tag}"], c)
            for other in (f"C_{tag}", f"D_{tag}", f"R_{tag}"):
                diff = c3 - _cost(errs[other], c)
                entry[f"C3_minus_{other}"] = {"mean": float(np.mean(diff)), "ci90": bootstrap_ci(diff, np.mean),
                                              "p_perm": paired_permutation_p(diff)}
        costs = {k: out["rules"][k][f"cost_{tag}"] for k in out["rules"]}
        entry["lowest_cost_rule"] = min(costs, key=costs.get)
        out["by_cost_ratio"][tag] = entry
    out["draws_crossing_after_run"] = later_draws
    return out


def tune_margins(rows: list[dict], n_passes: int, threshold: float, cost_ratios,
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


def apply_holm(worlds: dict, family: str, key: str = "p_holm", tracker: str = "C") -> int:
    """Holm-adjust the paired-comparison p-values of every world's decision rules together
    (one family: ``tracker``'s rule against the tuned D_r and R_r, and for C3 also against C_r,
    for every cost ratio and world); the adjusted value is stored next to each p-value as
    ``key``. Returns the family size."""
    entries = {}
    others = ("D", "R") if tracker == "C" else ("C", "D", "R")
    for w, s in worlds.items():
        for tag, e in s["decision_rules"]["by_cost_ratio"].items():
            for o in others:
                k = f"{tracker}_minus_{o}_{tag}"
                if k in e:
                    entries[(w, tag, o)] = e[k]
    adj = holm_adjust({k: v["p_perm"] for k, v in entries.items()})
    for k, v in entries.items():
        v[key] = adj[k]
        v[f"{key}_family"] = family
    return len(entries)
