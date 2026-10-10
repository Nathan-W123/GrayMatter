import numpy as np

from swt import stats


def test_permutation_p_and_holm():
    """Sign-flip p-values: a clear shift is significant, symmetric noise is not; Holm keeps the order
    of the p-values and multiplies the smallest by the family size."""
    rng = np.random.default_rng(0)
    assert stats.paired_permutation_p(rng.normal(1.0, 1.0, 50)) < 0.001
    assert stats.paired_permutation_p(rng.normal(0.0, 1.0, 50)) > 0.05
    assert stats.paired_permutation_p(np.zeros(10)) == 1.0
    adj = stats.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06


def test_iqr_summary_and_gmean():
    s = stats.iqr_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (s["median"], s["q25"], s["q75"], s["n"], s["min"], s["max"]) == (3.0, 2.0, 4.0, 5, 1.0, 5.0)
    assert stats.iqr_summary([])["median"] is None and stats.iqr_summary([])["n"] == 0
    assert np.isclose(stats.gmean([1.0, 4.0, 16.0]), 4.0)
    assert np.isnan(stats.gmean([1.0, 0.0])) and np.isnan(stats.gmean([]))
    assert stats.mean_or_none([]) is None and stats.mean_or_none([True, False]) == 0.5


def test_bootstrap_ci_is_ordered_reproducible_and_brackets_the_statistic():
    x = np.random.default_rng(1).normal(5.0, 1.0, 200)
    lo, hi = stats.bootstrap_ci(x)
    assert lo < np.median(x) < hi and hi - lo < 0.5
    assert stats.bootstrap_ci(x) == [lo, hi]                       # fixed seed
    assert all(np.isnan(stats.bootstrap_ci([1.0])))


def test_rmse_um_and_result_files(tmp_path):
    assert np.isclose(stats.rmse_um(np.array([0.001, 0.003]), np.array([0.0, 0.0])), np.sqrt(5.0))
    path = tmp_path / "rows.csv"
    stats.write_csv(path, [{"a": 1 / 3, "b": "x", "c": 2}])
    assert path.read_text().splitlines() == ["a,b,c", "0.333333,x,2"]
    stats.write_csv(path, [])
    assert path.read_text() == ""
    stats.write_json(tmp_path / "x.json", {"v": np.float64(0.5)})
    assert (tmp_path / "x.json").read_text().replace(" ", "").replace("\n", "") == '{"v":0.5}'
