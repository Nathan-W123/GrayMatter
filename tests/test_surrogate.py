import numpy as np
import pytest

from swt.process import line_effectiveness


def _random_params(tab, rng, n):
    eta = np.exp(rng.uniform(tab.log_eta[0], tab.log_eta[-1], n))
    a = rng.uniform(1e-4, 3e-3, n)
    z = rng.uniform(0.0, 0.3, n)
    return eta, a, z


def test_surrogate_matches_exact_model_with_within_pass_wear(small_ctx, rng):
    m, tab, pr, oi = small_ctx.model, small_ctx.table, small_ctx.priors, small_ctx.obs_index
    for k in np.exp(rng.uniform(np.log(pr.k_low), np.log(pr.k_high), 10)):
        for F in (20.0, 40.0):
            K, lam = 6e-5 * rng.uniform(0.4, 1.0), 2.5e-4 * np.exp(0.35 * rng.standard_normal())
            E, u = m.line_exposures(F, k)
            exact = line_effectiveness(K, lam, u) @ E[:, oi]
            approx = tab.map_obs(F / k, K * F, lam * K * F)
            assert np.sqrt(np.mean((exact - approx) ** 2)) / np.sqrt(np.mean(exact**2)) < 2e-3
            assert tab.volume(np.array([F / k]))[0] * F == pytest.approx(u.sum(), rel=2e-3)


@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_sse_matches_brute_force_with_missing_points_and_profile_offsets(small_ctx, rng, rho):
    """The O(1) likelihood equals the direct (Woodbury) quadratic form over the valid points."""
    tab, shape = small_ctx.table, small_ctx.obs_shape
    y = rng.normal(0.01, 0.002, tab.X.shape[0])
    y[rng.random(y.size) < 0.05] = np.nan
    valid = np.isfinite(y)
    st = tab.scan_stats(y, valid, shape, rho)
    eta, a, z = _random_params(tab, rng, 20)
    fast = tab.sse(eta, a, z, st)
    n_c = valid.reshape(shape).sum(axis=0)
    for i in range(20):
        r = np.where(valid, y - tab.map_obs(eta[i], a[i], z[i]), 0.0).reshape(shape)
        brute = (r**2).sum() - np.sum(rho / (1 + n_c * rho) * r.sum(axis=0) ** 2)
        assert fast[i] == pytest.approx(brute, rel=1e-9)


def test_predictive_mixtures_match_single_maps(small_ctx, rng):
    tab = small_ctx.table
    eta, a, z = _random_params(tab, rng, 40)
    w = rng.random(40)
    w /= w.sum()
    b = tab.coefficients(a, z)
    maps = np.array([tab.map_obs(eta[i], a[i], z[i]) for i in range(40)])
    assert np.allclose(tab.mixture_obs(eta, b, w), w @ maps, rtol=1e-10, atol=1e-15)
    assert np.allclose(tab.mean_removal(eta, b), maps.mean(axis=1), rtol=1e-10)
