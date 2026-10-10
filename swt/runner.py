"""The multi-draw tasks of a study and their (parallel) execution."""
from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import Context, config_hash
from .episode import loading_table, make_truth, run_episode

_WORKER_CTX: Context | None = None


def source_hash() -> str:
    """Hash of the package's source files, recorded with every run."""
    return hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()[:12]


# --------------------------------------------------------------------------- which tasks


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


def robust_worlds(ctx: Context) -> list[str]:
    """Worlds with full robustness runs: the configured ones plus the validation world."""
    worlds = list(ctx.cfg["experiments"]["robustness_worlds"])
    return worlds + [w for w in validation_worlds(ctx) if w not in worlds]


def tuning_tasks(ctx: Context) -> list[tuple]:
    t = ctx.cfg["tuning"]
    lo, hi = (int(x) for x in t["draws"])
    values = tuple(float(v) for v in t["rate_drift"])
    return [("tuning", w, d, values) for d in range(lo, hi) for w in t["worlds"]]


# --------------------------------------------------------------------------- running them


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
                                                 with_C2=bool(e.get("with_C2", False)),
                                                 with_C3=bool(e.get("with_C3", False)))
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


def _init_worker(cfg: dict) -> None:
    """Pool initializer: reuse a fork-inherited context, else rebuild it (spawn)."""
    global _WORKER_CTX
    if _WORKER_CTX is None or config_hash(_WORKER_CTX.cfg) != config_hash(cfg):
        from .config import build_context
        _WORKER_CTX = build_context(cfg)


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
    if ctx.cfg["experiments"].get("with_C3", False):
        loading_table(ctx)     # build C3's tables once, before the workers fork
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
