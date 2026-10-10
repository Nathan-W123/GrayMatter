import numpy as np

from swt import summaries


def test_interval_score_rewards_coverage_and_narrowness():
    row = {"C_pred_mean_lo_um": 9.0, "C_pred_mean_hi_um": 11.0, "true_mean_removal_um": 10.0}
    assert summaries.interval_score(row) == 0.2
    row["true_mean_removal_um"] = 12.0                 # 1 um above: width 2 + 20 * 1
    assert summaries.interval_score(row) == (2 + 20 * 1.0) / 12.0


def test_coverage_counts_use_the_interval_bounds():
    row = {"C_k_pad_lo": 1.0, "C_k_pad_hi": 2.0, "C_k_pad_median": 1.5, "true_k_pad": 1.9}
    assert summaries.inside(row, "k_pad", "true_k_pad")
    row["true_k_pad"] = 2.1
    assert not summaries.inside(row, "k_pad", "true_k_pad")
    assert summaries.outside_distance(row, "k_pad", "true_k_pad") == np.float64((2.1 - 2.0) / 2.1)
