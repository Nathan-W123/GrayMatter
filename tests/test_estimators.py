import dataclasses

import numpy as np
import pytest

from swt.config import rng_for
from swt.estimators import (CalibrateOnce, Nominal, ParticleFilter, RefitEachPass, least_squares_fit,
                            systematic_resample, weighted_quantiles)
from swt.process import HiddenTruth, PassAction, simulate_truth
from swt.scan import ScanArtefacts, Scanner


def _pf(ctx, seed, **changes):
    cfg = dataclasses.replace(ctx.pf, **changes)
    return ParticleFilter(ctx.table, ctx.priors, cfg, rng_for(7, seed), obs_shape=ctx.obs_shape)


def _track(ctx, pf, traj, scanner, rng, noise_um, passes=None):
    sched = ctx.schedule()
    for n, a in enumerate(sched[:passes]):
        if n:
            pf.advance(sched[n - 1])
        pf.update(scanner.observe(traj.removal[n], rng), a, noise_um)
    return pf


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
    traj = simulate_truth(small_ctx.model, truth, small_ctx.schedule(), 0.0, np.random.default_rng(0))
    sc = Scanner(small_ctx.panel, 0.0, small_ctx.cfg["scan"]["stride"])
    s = _track(small_ctx, _pf(small_ctx, seed), traj, sc, None, 0.0).summary()
    for name, true in (("k_pad", truth.k_pad), ("lam", truth.lam), ("K0", truth.K0), ("K", traj.K[-1])):
        assert s[name]["mean"] == pytest.approx(true, rel=0.05), name
        assert s[name]["median"] == pytest.approx(true, rel=0.05), name


def test_baselines_behave_as_specified(small_ctx):
    """A never changes its parameters; B freezes after the first scan; D refits every scan."""
    m, tab, pr, oi = small_ctx.model, small_ctx.table, small_ctx.priors, small_ctx.obs_index
    sched = small_ctx.schedule()
    A, B, D = Nominal(m, pr, oi), CalibrateOnce(m, pr, tab, oi), RefitEachPass(pr, tab)
    sc = Scanner(small_ctx.panel, 2.0, 2)
    traj = simulate_truth(m, HiddenTruth(0.05, 6e-5, 2.5e-4), sched, 0.01, np.random.default_rng(3))
    rng = np.random.default_rng(4)
    after_first, D_K = None, []
    for n, a in enumerate(sched[:6]):
        if n:
            for e in (A, B, D):
                e.advance(sched[n - 1])
        y = sc.observe(traj.removal[n], rng)
        for e in (A, B, D):
            e.update(y, a)
        if n == 0:
            after_first = (B.k_pad, B.K)
        D_K.append(D.K)
    assert (B.k_pad, B.K) == after_first
    assert A.k_pad == pr.log_mean()["k_pad"] and A.lam == pr.lam_median
    assert A.K < A.K0   # nominal wear model is propagated
    assert len(set(D_K)) == 6 and D_K[-1] < D_K[0]
    for e in (A, B, D):
        assert e.predict(sched[0]).shape == oi.shape


def test_refit_baseline_recovers_the_wear_rate(small_ctx):
    """Noise-free, single-stage truth: D's log-linear fit of the per-scan K against removed
    volume recovers lambda, and its crossing forecast is close to the truth."""
    truth = HiddenTruth(0.05, 6e-5, 3e-4)
    sched = small_ctx.schedule()
    traj = simulate_truth(small_ctx.model, truth, sched, 0.0, np.random.default_rng(0), small_ctx.threshold)
    D = RefitEachPass(small_ctx.priors, small_ctx.table)
    sc = Scanner(small_ctx.panel, 0.0, small_ctx.cfg["scan"]["stride"])
    for n, a in enumerate(sched[:6]):
        D.update(sc.observe(traj.removal[n], None), a)
    assert D.lam == pytest.approx(truth.lam, rel=0.03)
    assert abs(D.crossing_pass(6, sched, small_ctx.threshold) - traj.crossing_pass) <= 1


def test_mh_moves_leave_the_prior_invariant(small_ctx):
    """With no scan information (tempering exponent 0) the rejuvenation moves must keep
    the prior: log k uniform on its bounds, log lambda and log K0 Gaussian."""
    pf = _pf(small_ctx, 3, n_particles=6000)
    pr = small_ctx.priors
    pf.F[0], pf.s[0], pf.sigma2[0] = 30.0, 1.0, 1.0
    pf.stats[0] = small_ctx.table.scan_stats(np.zeros(small_ctx.obs_index.size))
    pf.LL[:, 0] = pf._loglik(0, pf.lk, pf.lpath[:, 0], pf.path[:, 0])
    for _ in range(25):
        pf._mh_sweep(0, 0.0)
    lo, hi = np.log(pr.k_low), np.log(pr.k_high)
    assert np.mean(pf.lk) == pytest.approx(0.5 * (lo + hi), abs=0.08)
    assert np.std(pf.lk) == pytest.approx((hi - lo) / np.sqrt(12), rel=0.05)
    assert np.mean(pf.lpath[:, 0]) == pytest.approx(np.log(pr.lam_median), abs=0.02)
    assert np.std(pf.lpath[:, 0]) == pytest.approx(pr.lam_sigma_log, rel=0.06)
    assert np.mean(pf.path[:, 0]) == pytest.approx(np.log(pr.K0_median), abs=0.02)
    assert np.std(pf.path[:, 0]) == pytest.approx(pr.K0_sigma_log, rel=0.06)
    lo, hi = (np.log(v) for v in small_ctx.pf.transient_range)
    assert np.mean(pf.lsg) == pytest.approx(0.5 * (lo + hi), abs=0.08)
    assert np.std(pf.lsg) == pytest.approx((hi - lo) / np.sqrt(12), rel=0.06)


def test_crossing_prediction_before_and_after_threshold(small_ctx):
    """Noise-free truth: once the wear rate is learned, C predicts the exact crossing pass
    ahead of time (inside its band) and, once past it, reports the pass at which it
    happened (not the current pass)."""
    truth = HiddenTruth(k_pad=0.05, K0=6e-5, lam=2.9e-4)
    sched = small_ctx.schedule()
    traj = simulate_truth(small_ctx.model, truth, sched, 0.0, np.random.default_rng(0))
    tc = traj.crossing_pass
    assert 6 <= tc <= 18
    sc = Scanner(small_ctx.panel, 0.0, small_ctx.cfg["scan"]["stride"])
    pf = _pf(small_ctx, 11)
    preds = {}
    for n, a in enumerate(sched):
        if n:
            pf.advance(sched[n - 1])
        pf.update(sc.observe(traj.removal[n], None), a, 0.0)
        preds[n + 1] = pf.predict_crossing(n + 1, sched, small_ctx.threshold)
    k_ratio = traj.K / truth.K0
    assert k_ratio[tc - 2] > 0.51 and k_ratio[tc - 1] < 0.49  # margin around the threshold
    for p in range(3, tc):                      # predictions made before the crossing
        assert preds[p]["lo"] <= tc <= preds[p]["hi"]
        assert abs(preds[p]["median"] - tc) <= (1 if p < tc - 2 else 0)
    for p in range(tc, len(sched) + 1):         # after it: the crossing is remembered
        assert preds[p]["median"] == tc


def test_particle_filter_wear_step_uses_each_particles_own_removed_volume(small_ctx):
    """advance(): K at the next pass = K / (1 + lambda K U), U = that particle's removed
    volume per unit effectiveness (exact within-pass wear of the single-stage law)."""
    m, pr = small_ctx.model, small_ctx.priors
    pf = _pf(small_ctx, 0, wear_noise_range=(1e-12, 1e-12), rate_drift=0.0, force_exponent_sd=0.0,
             transient_range=(0.0, 0.0), n_particles=4)
    ks = np.array([pr.k_low * 1.01, 0.01, 0.3, pr.k_high * 0.99])
    Ks = np.array([6e-5, 5e-5, 4e-5, 7e-5])
    lams = np.array([2e-4, 3e-4, 2.5e-4, 4e-4])
    pf.lk, pf.path[:, 0], pf.lpath[:, 0] = np.log(ks), np.log(Ks), np.log(lams)
    a = PassAction(40.0, 8000.0)
    pf.advance(a)
    U = np.array([m.volume(m.removal(a.force, k, 1.0)) for k in ks])
    assert np.allclose(np.exp(pf.path[:, 1]), Ks / (1 + lams * Ks * U), rtol=1e-4)
    assert np.allclose(pf.lpath[:, 1], pf.lpath[:, 0])          # no drift when rate_drift = 0


def test_credible_interval_width_matches_analytic_posterior(small_cfg):
    """With stiffness and wear rate pinned by very narrow priors, the posterior of K after
    one scan is Gaussian with sd = sigma / ||d map / d K||: the 90% interval must match."""
    from swt.config import build_context, deep_update
    ctx = build_context(deep_update(small_cfg, {"priors": {"k_low": 0.05, "k_high": 0.050001,
                                                           "lam_sigma_log": 1e-6}}))
    sc = Scanner(ctx.panel, 2.0, ctx.cfg["scan"]["stride"])
    a, K_true, lam = PassAction(30.0, 8000.0), 6e-5, ctx.priors.lam_median
    from swt.process import line_effectiveness
    E, u = ctx.model.line_exposures(a.force, 0.05)

    def removal(K):
        return line_effectiveness(K, lam, u) @ E

    h = 1e-4
    dm = (removal(K_true * (1 + h)) - removal(K_true * (1 - h)))[sc.obs_index] / (2 * h * K_true)
    sd_K = 2e-3 / np.sqrt(dm @ dm)                 # noise sigma = 2 um
    ratios, inside = [], 0
    cfg = dataclasses.replace(ctx.pf, transient_range=(0.0, 0.0))   # no per-pass gain to confound K
    for seed in range(8):
        pf = ParticleFilter(ctx.table, ctx.priors, cfg, rng_for(50, seed), obs_shape=ctx.obs_shape)
        pf.update(sc.observe(removal(K_true), np.random.default_rng(seed)), a, 2.0)
        s = pf.summary()["K"]
        ratios.append((s["hi"] - s["lo"]) / (2 * 1.6449 * sd_K))
        inside += s["lo"] <= K_true <= s["hi"]
    assert 0.85 < np.median(ratios) < 1.15
    assert inside >= 5


def test_innovation_gating_rejects_spikes_but_not_sharp_maps(small_ctx):
    """A stiff pad gives sharp removal stripes: no genuine point may be rejected. Spikes
    injected into a later scan are rejected and do not move the estimate of K."""
    truth = HiddenTruth(0.5, 6e-5, 2.5e-4)
    sched = small_ctx.schedule()
    traj = simulate_truth(small_ctx.model, truth, sched, 0.0, np.random.default_rng(0))
    sc = Scanner(small_ctx.panel, 2.0, small_ctx.cfg["scan"]["stride"])
    clean = _track(small_ctx, _pf(small_ctx, 1), traj, sc, np.random.default_rng(5), 2.0, passes=3)
    assert clean.last_outliers <= 2
    y_rng = np.random.default_rng(5)
    spiky = _pf(small_ctx, 1)
    spikes = np.random.default_rng(9).choice(small_ctx.obs_index.size, 25, replace=False)
    for n, a in enumerate(sched[:3]):
        if n:
            spiky.advance(sched[n - 1])
        y = sc.observe(traj.removal[n], y_rng)
        if n == 2:
            y[spikes] += 0.03                       # 30 um spikes
        spiky.update(y, a, 2.0)
    assert spiky.last_outliers >= 25
    assert spiky.summary()["K"]["median"] == pytest.approx(clean.summary()["K"]["median"], rel=2e-3)


def test_profile_offsets_are_estimated_only_when_present(small_ctx):
    truth = HiddenTruth(0.02, 6e-5, 2.5e-4)
    traj = simulate_truth(small_ctx.model, truth, small_ctx.schedule(), 0.0, np.random.default_rng(0))
    plain = Scanner(small_ctx.panel, 2.0, small_ctx.cfg["scan"]["stride"])
    biased = Scanner(small_ctx.panel, 2.0, small_ctx.cfg["scan"]["stride"], ScanArtefacts(profile_bias_um=1.0))
    pf = _track(small_ctx, _pf(small_ctx, 2), traj, plain, np.random.default_rng(1), 2.0, passes=4)
    assert pf.summary()["profile_sd_um"] == 0.0
    pf = _track(small_ctx, _pf(small_ctx, 2), traj, biased, np.random.default_rng(1), 2.0, passes=4)
    assert pf.summary()["profile_sd_um"] == pytest.approx(1.0, rel=0.35)
    assert pf.summary()["sigma_um"] == pytest.approx(2.0, rel=0.1)


def test_wear_noise_level_is_learned(small_ctx):
    """The wear-fluctuation s.d. is a parameter with a log-uniform prior on [0.5%, 5%]: for a
    truth fluctuating 1% per pass its posterior median ends near 1%, for 4% well above 2.5%."""
    sched = small_ctx.schedule()
    sc = Scanner(small_ctx.panel, 2.0, small_ctx.cfg["scan"]["stride"])
    out = {}
    for sw in (0.01, 0.04):
        traj = simulate_truth(small_ctx.model, HiddenTruth(0.05, 6e-5, 2.0e-4), sched, sw,
                              np.random.default_rng(2))
        out[sw] = _track(small_ctx, _pf(small_ctx, 4), traj, sc, np.random.default_rng(3), 2.0).wear_noise()
    assert 0.006 < out[0.01] < 0.016
    assert out[0.04] > 0.025


def test_full_path_sweeps_keep_the_path_diverse(small_ctx):
    """After 12 passes, every stored log K value (not only the current one) still has many
    distinct particles, because the final sweep moves the whole path."""
    traj = simulate_truth(small_ctx.model, HiddenTruth(0.05, 6e-5, 2.5e-4), small_ctx.schedule(), 0.01,
                          np.random.default_rng(2))
    sc = Scanner(small_ctx.panel, 2.0, small_ctx.cfg["scan"]["stride"])
    pf = _track(small_ctx, _pf(small_ctx, 5), traj, sc, np.random.default_rng(3), 2.0, passes=12)
    assert min(pf.ancestral_diversity(i) for i in range(12)) > 0.2 * pf.lk.size


def test_force_exponent_is_learned(small_ctx):
    """A truth whose removal scales as F^0.8 at fixed contact shape (simulated by scaling each
    pass's removal by (F/F_ref)^-0.2): C's force exponent moves from its prior (1) towards 0.8,
    and D's log-force regression finds it too."""
    sched = small_ctx.schedule()
    traj = simulate_truth(small_ctx.model, HiddenTruth(0.05, 6e-5, 2.0e-4), sched, 0.0, np.random.default_rng(1))
    f = np.array([(a.force / 30.0) ** -0.2 for a in sched])
    traj.removal = traj.removal * f[:, None]
    sc = Scanner(small_ctx.panel, 2.0, small_ctx.cfg["scan"]["stride"])
    pf = _track(small_ctx, _pf(small_ctx, 6), traj, sc, np.random.default_rng(2), 2.0, passes=10)
    s = pf.summary()["beta"]
    assert s["lo"] < 0.8 < s["hi"] and s["hi"] < 0.95
    D = RefitEachPass(small_ctx.priors, small_ctx.table)
    rng = np.random.default_rng(2)
    for n, a in enumerate(sched[:10]):
        D.update(sc.observe(traj.removal[n], rng), a)
    assert D.gamma == pytest.approx(-0.2, abs=0.05)


def test_transient_gain_is_told_apart_from_wear(small_ctx):
    """A truth whose removal has an independent 4% gain per pass (and no lasting fluctuation):
    C attributes it to the transient gain (sigma_g well above its matched-world value) and
    keeps the wear-fluctuation estimate small."""
    sched = small_ctx.schedule()
    traj = simulate_truth(small_ctx.model, HiddenTruth(0.05, 6e-5, 2.0e-4), sched, 0.0, np.random.default_rng(1))
    traj.removal = traj.removal * np.exp(0.04 * np.random.default_rng(7).standard_normal(len(sched)))[:, None]
    sc = Scanner(small_ctx.panel, 2.0, small_ctx.cfg["scan"]["stride"])
    s = _track(small_ctx, _pf(small_ctx, 8), traj, sc, np.random.default_rng(2), 2.0).summary()
    assert s["transient_noise"] > 0.02
    assert s["wear_noise"] < s["transient_noise"]


def test_refit_baseline_uses_the_priors_with_few_scans(small_ctx):
    """With removal proportional to F^0.8, two scans at 20 and 40 N differ by a force effect
    that an unpenalised fit would read as wear (a wear rate several times too high); D's
    MAP fit with C's priors keeps the wear rate near its prior and splits the difference."""
    sched = small_ctx.schedule()
    traj = simulate_truth(small_ctx.model, HiddenTruth(0.05, 6e-5, 2.0e-4), sched, 0.0, np.random.default_rng(1))
    f = np.array([(a.force / 30.0) ** -0.2 for a in sched])
    traj.removal = traj.removal * f[:, None]
    sc = Scanner(small_ctx.panel, 2.0, small_ctx.cfg["scan"]["stride"])
    D = RefitEachPass(small_ctx.priors, small_ctx.table)
    rng = np.random.default_rng(2)
    D.update(sc.observe(traj.removal[0], rng), sched[0])
    assert D.lam == pytest.approx(small_ctx.priors.lam_median) and abs(D.gamma) < 1e-9
    D.update(sc.observe(traj.removal[1], rng), sched[1])
    assert 0.5 * small_ctx.priors.lam_median < D.lam < 2.0 * small_ctx.priors.lam_median
    assert -0.2 < D.gamma < 0.0
