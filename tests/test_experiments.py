import numpy as np

from swt import experiments
from swt.process import HiddenTruth, simulate_truth


def test_run_episode_nominal_truth_makes_A_and_oracle_exact(small_ctx):
    """If the hidden truth equals A's nominal parameters and there is no wear fluctuation,
    A and the oracle predict every pass exactly; this pins the driver's pass indexing
    (which action drives each wear step) and the oracle definition."""
    nom = small_ctx.priors.log_mean()
    truth = HiddenTruth(nom["k_pad"], nom["K0"], nom["lam"])
    traj = simulate_truth(small_ctx.model, truth, small_ctx.schedule(), 0.0, np.random.default_rng(0),
                          small_ctx.threshold)
    rows, _ = experiments.run_episode(small_ctx, 0, traj=traj, full=True)
    assert max(r["rmse_A_um"] for r in rows) < 1e-9
    assert max(r["rmse_oracle_um"] for r in rows) < 1e-9
    assert all(r["cross_A"] == traj.crossing_pass for r in rows)
    assert [r["pass"] for r in rows] == list(range(1, len(rows) + 1))


def test_coverage_counts_use_the_interval_bounds():
    row = {"C_k_pad_lo": 1.0, "C_k_pad_hi": 2.0, "C_k_pad_median": 1.5, "true_k_pad": 1.9}
    assert experiments.inside(row, "k_pad", "true_k_pad")
    row["true_k_pad"] = 2.1
    assert not experiments.inside(row, "k_pad", "true_k_pad")
    assert experiments._outside(row, "k_pad", "true_k_pad") == np.float64((2.1 - 2.0) / 2.1)
