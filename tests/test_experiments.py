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


def test_worlds_share_the_hidden_parameters(small_ctx):
    """Paired comparison: every world draws the same (k_pad, K0, lambda) for a draw id;
    only the force-error world has a non-unit gain."""
    m = experiments.draw_truth(small_ctx, 5, "matched")
    r = experiments.draw_truth(small_ctx, 5, "realistic")
    assert (m.k_pad, m.K0, m.lam) == (r.k_pad, r.K0, r.lam)
    assert m.force_gain == 1.0 and r.force_gain != 1.0
    assert experiments.draw_truth(small_ctx, 5, "only_wear").force_gain == 1.0


def test_interval_score_rewards_coverage_and_narrowness():
    row = {"C_pred_mean_lo_um": 9.0, "C_pred_mean_hi_um": 11.0, "true_mean_removal_um": 10.0}
    assert experiments.interval_score(row) == 0.2
    row["true_mean_removal_um"] = 12.0                 # 1 um above: width 2 + 20 * 1
    assert experiments.interval_score(row) == (2 + 20 * 1.0) / 12.0


def test_coverage_counts_use_the_interval_bounds():
    row = {"C_k_pad_lo": 1.0, "C_k_pad_hi": 2.0, "C_k_pad_median": 1.5, "true_k_pad": 1.9}
    assert experiments.inside(row, "k_pad", "true_k_pad")
    row["true_k_pad"] = 2.1
    assert not experiments.inside(row, "k_pad", "true_k_pad")
    assert experiments._outside(row, "k_pad", "true_k_pad") == np.float64((2.1 - 2.0) / 2.1)


def test_decision_rules_score_change_passes():
    """Two draws crossing at pass 4. C's median rule changes on time, its 25% rule one pass
    early, its 5% rule two passes early; D's forecast K/K0 (0.55 after pass 3, 0.45 after
    pass 4) makes it one pass late without a margin and on time with a 20% margin."""
    rows = []
    d_ratio = [0.9, 0.7, 0.55, 0.45, 0.4, 0.35]
    for d in range(2):
        for p in range(1, 7):
            rows.append({"world": "w", "draw": d, "pass": p, "cross_true": 4, "cross_A": 6, "force_N": 20.0,
                         "scan_mean_um": 10.0 * 0.85 ** (p - 1), "D_next_ratio": d_ratio[p - 1],
                         "cross_C_median": 4 if p >= 3 else 9, "cross_C_risk": 3 if p >= 2 else 9,
                         "cross_C_lo": 2})
    dr = experiments._decision_rules(rows, 6, 0.5, {"r3": {"D": 0.2, "R": 0.0}}, cost_ratios=(1, 3, 19))
    rules = dr["rules"]
    assert rules["C_r1"]["exact"] == 2 and rules["C_r1"]["mean_abs_err"] == 0.0
    assert rules["C_r3"]["early"] == 2 and rules["C_r3"]["cost_r3"] == 1.0
    assert rules["C_r19"]["mean_passes_early"] == 2.0
    assert rules["D"]["late"] == 2 and rules["D"]["cost_r3"] == 3.0
    assert rules["D_r3"]["exact"] == 2
    assert rules["A"]["mean_abs_err"] == 2.0
    assert rules["R"]["n"] == 2
    assert dr["by_cost_ratio"]["r3"]["C_minus_D_r3"]["mean"] == 1.0
    assert dr["by_cost_ratio"]["r3"]["C_minus_D"]["mean"] == -2.0
    margins = experiments._tune_margins(rows, 6, 0.5, (3,))
    assert margins["r3"]["D"] == 0.11 and margins["r3"]["D_held_out_cost"] == 0.0
    # the same draw ids in two worlds are two episodes, not one
    rows2 = [dict(r, world="matched") for r in rows] + [dict(r, world="realistic") for r in rows]
    m2 = experiments._tune_margins(rows2, 6, 0.5, (3,))
    assert m2["r3"]["n_draws"] == 4 and m2["r3"]["D"] == 0.11
