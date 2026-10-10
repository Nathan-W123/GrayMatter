"""Tuning on held-out draws: the tracker's wear-rate drift and the decision margins of D and R."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import Context, config_hash
from .decisions import cost_ratios, tune_margins
from .runner import run_draws, tuning_tasks
from .stats import write_csv, write_json
from .summaries import by_draw, final_draw_summary, parameter_coverage, predictive_calibration


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
            finals = [final_draw_summary(rr) for rr in by_draw(rows).values()]
            pred = predictive_calibration(rows, n_passes)
            per_world[w] = {
                "interval_score": pred["relative_interval_score_passes_2_on"],
                "coverage": pred["passes_2_on"]["fraction"],
                "relative_width": pred["relative_width_passes_2_on"]["median"],
                "mean_rmse_C_over_oracle": float(np.median([f["mean_rmse_C_um"] / f["mean_rmse_oracle_um"]
                                                            for f in finals])),
                "coverage_k_pad": parameter_coverage(finals)["k_pad"]["fraction"],
            }
            table.append({"rate_drift": v, "world": w, **per_world[w]})
        score = float(np.mean([per_world[w]["interval_score"] for w in t["worlds"]]))
        by_value.append({"rate_drift": v, "mean_interval_score": score, "worlds": per_world})
    best = min(by_value, key=lambda x: (x["mean_interval_score"], x["rate_drift"]))
    chosen = best["rate_drift"]
    first = float(t["rate_drift"][0])
    d_rows = [r for w in t["worlds"] for r in runs[("tuning", w, first)]]
    margins = tune_margins(d_rows, n_passes, ctx.threshold, cost_ratios(ctx))
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
