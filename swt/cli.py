"""Command-line interface.

    python -m swt run-all [--config configs/default.yaml] [--results results] [--figures figures] [--workers N] [--full]
    python -m swt run --config configs/default.yaml
    python -m swt contact-sweep [--config ...]
    python -m swt figures [--results results] [--figures figures]
    python -m swt report [--results results]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import build_context, load_config


# --full: the secondary experiments at the sizes used before they were cut to keep run-all
# under 10 minutes (about 15 minutes with 4 workers)
FULL = {"experiments": {"breakdown_draws": 40, "noise_draws": 40, "ablation_draws": 40,
                        "noise_levels_um": [1.0, 2.0, 5.0, 10.0]},
        "tuning": {"draws": [200, 220]}}


def _ctx(args):
    t = time.time()
    cfg = load_config(args.config, FULL if getattr(args, "full", False) else None)
    ctx = build_context(cfg)
    print(f"[swt] context built in {time.time() - t:.1f} s "
          f"({ctx.model.contact.n_stations} stations, {ctx.panel.n_pix} grid nodes, "
          f"{ctx.table.n_eta}-node surrogate, {len(ctx.obs_index)} scan points)")
    return ctx


def cmd_run_all(args) -> int:
    from . import experiments, plots
    t = time.time()
    ctx = _ctx(args)
    experiments.run_all(ctx, args.results, workers=args.workers)
    plots.make_all(args.results, args.figures, ctx)
    print(f"[swt] run-all finished in {time.time() - t:.1f} s; results in {args.results}/, figures in {args.figures}/")
    return 0


def cmd_run(args) -> int:
    from . import experiments, plots
    ctx = _ctx(args)
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    drift = experiments.tuned_rate_drift(ctx, out, workers=args.workers)
    print(f"[swt] wear-rate drift of the tracker: {drift:g}")
    experiments.experiment_main(ctx, out)
    plots.make_main(args.results, args.figures, ctx)
    return 0


def cmd_contact(args) -> int:
    from . import experiments
    ctx = _ctx(args)
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    experiments.experiment_contact_sweep(ctx, out)
    return 0


def cmd_report(args) -> int:
    from .report import render_reports
    for path, n in render_reports(args.results).items():
        print(f"[swt] rendered {n} numbers from {args.results}/ into {path}")
    return 0


def cmd_figures(args) -> int:
    from . import plots
    plots.make_all(args.results, args.figures, None)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swt", description="sanding-wear-tracker experiments")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in [
        ("run-all", cmd_run_all, "run every experiment and regenerate all figures"),
        ("run", cmd_run, "run the main 20-pass experiment"),
        ("contact-sweep", cmd_contact, "print and save the contact-fraction vs stiffness sweep"),
        ("figures", cmd_figures, "redraw figures from existing results"),
        ("report", cmd_report, "render README.md and SUMMARY.md from docs/*.template.md and results/"),
    ]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
        sp.add_argument("--results", default="results", help="output directory for JSON/CSV")
        sp.add_argument("--figures", default="figures", help="output directory for PNG figures")
        sp.add_argument("--workers", default="auto",
                        help="worker processes for the multi-draw experiments ('auto' = min(4, CPUs); 1 = serial)")
        sp.add_argument("--full", action="store_true",
                        help="larger single-effect, stress, noise, ablation and tuning experiments (slower)")
        sp.set_defaults(func=fn)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
