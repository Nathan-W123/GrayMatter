import numpy as np

from swt import episode
from swt.process import HiddenTruth, simulate_truth


def test_run_episode_nominal_truth_makes_A_and_oracle_exact(small_ctx):
    """If the hidden truth equals A's nominal parameters and there is no wear fluctuation,
    A and the oracle predict every pass exactly; this pins the driver's pass indexing
    (which action drives each wear step) and the oracle definition."""
    nom = small_ctx.priors.log_mean()
    truth = HiddenTruth(nom["k_pad"], nom["K0"], nom["lam"])
    traj = simulate_truth(small_ctx.model, truth, small_ctx.schedule(), 0.0, np.random.default_rng(0),
                          small_ctx.threshold)
    rows, _ = episode.run_episode(small_ctx, 0, traj=traj, full=True)
    assert max(r["rmse_A_um"] for r in rows) < 1e-9
    assert max(r["rmse_oracle_um"] for r in rows) < 1e-9
    assert all(r["cross_A"] == traj.crossing_pass for r in rows)
    assert [r["pass"] for r in rows] == list(range(1, len(rows) + 1))


def test_worlds_share_the_hidden_parameters(small_ctx):
    """Paired comparison: every world draws the same (k_pad, K0, lambda) for a draw id;
    only the force-error world has a non-unit gain."""
    m = episode.draw_truth(small_ctx, 5, "matched")
    r = episode.draw_truth(small_ctx, 5, "realistic")
    assert (m.k_pad, m.K0, m.lam) == (r.k_pad, r.K0, r.lam)
    assert m.force_gain == 1.0 and r.force_gain != 1.0
    assert episode.draw_truth(small_ctx, 5, "only_wear").force_gain == 1.0


def test_task_list_covers_every_world_and_draw(small_ctx):
    from swt import runner
    e = small_ctx.cfg["experiments"]
    tasks = runner.task_list(small_ctx)
    n = int(e["robustness_draws"])
    robust = [t for t in tasks if t[0] == "robustness"]
    assert len(robust) == n * len(e["robustness_worlds"]) + sum(
        int(e.get("validation_draws", n)) for _ in runner.validation_worlds(small_ctx))
    assert len(set(tasks)) == len(tasks)
    assert runner.resolve_workers(3) == 3 and 1 <= runner.resolve_workers("auto") <= 4
