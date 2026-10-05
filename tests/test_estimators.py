import dataclasses

import numpy as np
import pytest

from swt.config import rng_for
from swt.estimators import (CalibrateOnce, Nominal, ParticleFilter, PFConfig, least_squares_fit,
                            systematic_resample, weighted_quantiles)
from swt.process import HiddenTruth, PassAction, simulate_truth
from swt.scan import Scanner


def test_systematic_resampling_counts(rng):
    w = rng.random(1000)
    w /= w.sum()
    counts = np.bincount(systematic_resample(w, rng), minlength=w.size)
    assert counts.sum() == w.size
    assert np.all(np.abs(counts - w.size * w) < 1.0 + 1e-9)


def test_weighted_quantiles(rng):
    x = rng.normal(size=20000)
    w = np.ones_like(x)
    assert np.allclose(weighted_quantiles(x, w, [0.05, 0.5, 0.95]), np.quantile(x, [0.05, 0.5, 0.95]), atol=0.02)
    # doubling the weight of the upper half moves the median to Phi^-1(0.625)
    from scipy.stats import norm
    xd = norm.ppf((np.arange(20000) + 0.5) / 20000)          # deterministic standard-normal sample
    w2 = np.where(xd > 0, 2.0, 1.0)
    assert weighted_quantiles(xd, w2, [0.5])[0] == pytest.approx(norm.ppf(0.625), abs=0.01)


def test_least_squares_recovers_noiseless(small_ctx):
    m, tab = small_ctx.model, small_ctx.table
    sched = small_ctx.schedule()
    sc = Scanner(small_ctx.panel, 0.0, small_ctx.cfg["scan"]["stride"])
    for k, K, rpm in ((2e-3, 5e-5, 8000.0), (0.3, 8e-5, 8000.0), (0.05, 6e-5, 4000.0)):
        a = PassAction(sched[0].force, rpm)
        y = sc.observe(m.removal(a.force, k, K, rpm), None)
        k_hat, K_hat = least_squares_fit(tab, y, a, (small_ctx.priors.k_low, small_ctx.priors.k_high))
        assert k_hat == pytest.approx(k, rel=0.01)
        assert K_hat == pytest.approx(K, rel=0.005)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_particle_filter_recovers_truth_without_noise(small_ctx, seed):
    """Required test 4: zero scan noise and a correct prior (the truth is drawn from
    the same priors the filter uses; no wear fluctuation in the truth) ->
    C recovers k_pad, lambda, K0 and the current K within 5%."""
    p = small_ctx.priors.sample(rng_for(1234, seed))
    truth = HiddenTruth(float(p["k_pad"]), float(p["K0"]), float(p["lam"]))
    sched = small_ctx.schedule()
    traj = simulate_truth(small_ctx.model, truth, sched, 0.0, np.random.default_rng(0))
    sc = Scanner(small_ctx.panel, 0.0, small_ctx.cfg["scan"]["stride"])
    pf = ParticleFilter(small_ctx.table, small_ctx.priors, small_ctx.pf, rng_for(7, seed))
    for n, a in enumerate(sched):
        if n:
            pf.advance(sched[n - 1])
        pf.update(sc.observe(traj.removal[n], None), a, 0.0)
    s = pf.summary()
    for name, true in (("k_pad", truth.k_pad), ("lam", truth.lam), ("K0", truth.K0), ("K", traj.K[-1])):
        assert s[name]["mean"] == pytest.approx(true, rel=0.05), name
        assert s[name]["median"] == pytest.approx(true, rel=0.05), name


def test_baselines_behave_as_specified(small_ctx):
    """A never changes its parameters; B freezes after the first scan."""
    m, tab, pr = small_ctx.model, small_ctx.table, small_ctx.priors
    sched = small_ctx.schedule()
    A, B = Nominal(m, pr), CalibrateOnce(m, pr, tab)
    sc = Scanner(small_ctx.panel, 2.0, 2)
    traj = simulate_truth(m, HiddenTruth(0.05, 6e-5, 2.5e-4), sched, 0.01, np.random.default_rng(3))
    rng = np.random.default_rng(4)
    k_after_first = None
    for n, a in enumerate(sched[:6]):
        if n:
            A.advance(sched[n - 1])
            B.advance(sched[n - 1])
        y = sc.observe(traj.removal[n], rng)
        A.update(y, a)
        B.update(y, a)
        if n == 0:
            k_after_first = (B.k_pad, B.K)
    assert (B.k_pad, B.K) == k_after_first
    assert A.k_pad == pr.log_mean()["k_pad"] and A.lam == pr.lam_median
    assert A.K < A.K0   # nominal wear model is propagated


def test_surrogate_sse_matches_brute_force(small_ctx, rng):
    """The O(1) Gram-matrix likelihood equals the direct sum of squares."""
    tab = small_ctx.table
    y = rng.normal(0.01, 0.002, tab.obs.shape[0])
    ym, yy = tab.scan_terms(y)
    eta = np.exp(rng.uniform(np.log(tab.eta[0]), np.log(tab.eta[-1]), 25))
    a = rng.uniform(0.5, 2.0, 25) * 1e-3
    fast = tab.sse(eta, a, ym, yy)
    for i in range(25):
        m = tab.map_full(eta[i])[tab.obs_index]
        assert fast[i] == pytest.approx(np.sum((y - a[i] * m) ** 2), rel=1e-9)


def test_crossing_prediction_before_and_after_threshold(small_ctx):
    """Noise-free truth: C predicts the exact crossing pass ahead of time and, once past
    it, reports the pass at which it happened (not the current pass). The truth sits
    clearly on either side of the threshold around the crossing, so no slack is needed."""
    truth = HiddenTruth(k_pad=0.05, K0=6e-5, lam=2.8e-4)
    sched = small_ctx.schedule()
    traj = simulate_truth(small_ctx.model, truth, sched, 0.0, np.random.default_rng(0))
    tc = traj.crossing_pass
    assert 4 <= tc <= 18
    sc = Scanner(small_ctx.panel, 0.0, small_ctx.cfg["scan"]["stride"])
    pf = ParticleFilter(small_ctx.table, small_ctx.priors, small_ctx.pf, rng_for(11, 0))
    preds = {}
    for n, a in enumerate(sched):
        if n:
            pf.advance(sched[n - 1])
        pf.update(sc.observe(traj.removal[n], None), a, 0.0)
        preds[n + 1] = pf.predict_crossing(n + 1, sched, small_ctx.threshold)
    k_ratio = traj.K / truth.K0
    assert k_ratio[tc - 2] > 0.51 and k_ratio[tc - 1] < 0.49  # margin around the threshold
    for p in range(3, tc):                      # predictions made before the crossing
        assert preds[p]["median"] == tc
        assert preds[p]["lo"] <= tc <= preds[p]["hi"]
    for p in range(tc, len(sched) + 1):         # after it: the crossing is remembered
        assert preds[p]["median"] == tc


def test_particle_filter_wear_step_uses_each_particles_own_removed_volume(small_ctx):
    """advance(): log K drops by lambda times the exact removed volume of that particle."""
    m, pr = small_ctx.model, small_ctx.priors
    cfg = dataclasses.replace(small_ctx.pf, wear_sigma=0.0, n_particles=4)
    pf = ParticleFilter(small_ctx.table, pr, cfg, rng_for(0, 0))
    ks = np.array([pr.k_low * 1.01, 0.01, 0.3, pr.k_high * 0.99])
    Ks = np.array([6e-5, 5e-5, 4e-5, 7e-5])
    lams = np.array([2e-4, 3e-4, 2.5e-4, 4e-4])
    pf.X = np.column_stack([np.log(ks), np.log(Ks * 1.2), np.log(lams), np.log(Ks)])
    a = PassAction(40.0, 8000.0)
    pf.advance(a)
    exact = np.array([m.volume(m.removal(a.force, k, K)) for k, K in zip(ks, Ks)])
    assert np.allclose(np.exp(pf.X[:, 3]), Ks * np.exp(-lams * exact), rtol=1e-4)


def test_credible_interval_width_matches_analytic_posterior(small_cfg):
    """With stiffness pinned by a very narrow prior, the posterior of K after one scan is
    Gaussian with sd = sigma / ||m_obs||: the reported 90% interval must match it."""
    from swt.config import build_context, deep_update
    ctx = build_context(deep_update(small_cfg, {"priors": {"k_low": 0.05, "k_high": 0.050001}}))
    sc = Scanner(ctx.panel, 2.0, ctx.cfg["scan"]["stride"])
    a, K_true = PassAction(30.0, 8000.0), 6e-5
    m_obs = ctx.model.exposure(a.force, 0.05)[sc.obs_index]
    sd_K = 2e-3 / np.sqrt(m_obs @ m_obs)        # removal = K * exposure, noise sigma = 2 um
    ratios, inside = [], 0
    for seed in range(8):
        pf = ParticleFilter(ctx.table, ctx.priors, ctx.pf, rng_for(50, seed))
        pf.update(sc.observe(ctx.model.removal(a.force, 0.05, K_true), np.random.default_rng(seed)), a, 2.0)
        s = pf.summary()["K"]
        ratios.append((s["hi"] - s["lo"]) / (2 * 1.6449 * sd_K))
        inside += s["lo"] <= K_true <= s["hi"]
    assert 0.85 < np.median(ratios) < 1.15
    assert inside >= 5
