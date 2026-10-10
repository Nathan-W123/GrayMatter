import dataclasses

import numpy as np
import pytest

from swt.config import rng_for
from swt.episode import loading_table, regions
from swt.estimators import ParticleFilter
from swt.process import HiddenTruth, World, simulate_truth
from swt.scan import Scanner
from swt.tracker3 import C3Config, LoadingTracker


def test_loading_likelihood_matches_brute_force(small_ctx):
    """The collapsed sufficient statistics give the exact SSE of A sum (-z)^k X_k + B Psi_kappa,
    with masked points and with a random offset per scan profile (Woodbury)."""
    lt = loading_table(small_ctx)
    rng = np.random.default_rng(0)
    n = 6
    eta = np.exp(rng.uniform(np.log(lt.base.eta[3]), np.log(lt.base.eta[-4]), n))
    A, B = rng.uniform(0.5, 2, n) * 1e-3, rng.uniform(0, 0.5, n) * 1e-3
    z, k = rng.uniform(0, 0.2, n), rng.integers(0, lt.n_k, n)
    y = rng.normal(0, 1e-3, lt.P.shape[0])
    valid = rng.random(y.size) > 0.05
    ny, nx = small_ctx.obs_shape
    for rho in (0.0, 0.3):
        fast = lt.sse(eta, A, B, z, k, lt.scan_stats(y, valid, small_ctx.obs_shape, rho))
        for i in range(n):
            e, t = lt.locate(eta[i:i + 1])
            e, t = int(e[0]), float(t[0])
            X = (1 - t) * lt.base.X[:, :, e] + t * lt.base.X[:, :, e + 1]
            P = (1 - t) * lt.P[:, k[i], e] + t * lt.P[:, k[i], e + 1]
            r = np.where(valid, y - A[i] * X @ (-z[i]) ** np.arange(4) - B[i] * P, 0.0)
            s = r @ r
            if rho:
                n_c = valid.reshape(ny, nx).sum(axis=0)
                s -= np.sum(rho / (1 + n_c * rho) * r.reshape(ny, nx).sum(axis=0) ** 2)
            assert fast[i] == pytest.approx(s, rel=1e-10)


def test_loading_paths_follow_the_recursion(small_ctx):
    """Clean abrasive at pass 1, then S_{j+1} = (1 - c) (L_inf - (L_inf - S_j) exp(-kappa))."""
    pf = LoadingTracker(loading_table(small_ctx), small_ctx.priors, dataclasses.replace(small_ctx.pf, n_particles=50),
                        rng_for(1, 1), obs_shape=small_ctx.obs_shape)
    Linf, c, kidx = np.array([0.0, 0.2, 0.4]), np.array([0.2, 0.5, 1.0]), np.array([0, 1, 2])
    S, Lb = pf._paths(Linf, c, kidx)
    kap = pf.kap[kidx]
    s = np.zeros(3)
    for i in range(30):
        np.testing.assert_allclose(S[:, i], s, atol=1e-12)
        np.testing.assert_allclose(Lb[:, i], Linf - (Linf - s) * (1 - np.exp(-kap)) / kap, atol=1e-12)
        s = (1 - c) * (Linf - (Linf - s) * np.exp(-kap))
    assert np.all(S[0] == 0.0) and np.all(Lb[0] == 0.0)


def _run(ctx, world, passes=10, seed=5):
    """C and C3 through ``passes`` passes of one hidden truth in ``world``."""
    sched = ctx.schedule()
    truth = HiddenTruth(0.05, 6e-5, 2.5e-4)
    traj = simulate_truth(ctx.model, truth, sched, 0.0, np.random.default_rng(1), world=world)
    cfg = dataclasses.replace(ctx.pf, n_particles=1500)
    C = ParticleFilter(ctx.table, ctx.priors, cfg, rng_for(7, seed), rollout_rng=rng_for(7, seed + 1),
                       obs_shape=ctx.obs_shape)
    C3 = LoadingTracker(loading_table(ctx), ctx.priors, cfg, rng_for(8, seed), rollout_rng=rng_for(8, seed + 1),
                        obs_shape=ctx.obs_shape, c3=C3Config(), region_index=regions(ctx)[0])
    sc = Scanner(ctx.panel, 2.0, ctx.cfg["scan"]["stride"])
    rng = np.random.default_rng(2)
    for n, a in enumerate(sched[:passes]):
        if n:
            C.advance(sched[n - 1])
            C3.advance(sched[n - 1])
        scan = sc.observe(traj.removal[n], rng)
        C.update(scan, a, 2.0)
        C3.update(scan, a, 2.0)
    return C, C3, truth


def test_c3_learns_loading_and_keeps_the_wear_rate(small_ctx):
    """With 20% loading, half shed between passes, C3 finds the loading and its wear rate stays
    near the truth, while C's is inflated by the fall of removal within each pass."""
    C, C3, truth = _run(small_ctx, World("l", loading_max=0.2, loading_volume=50.0, loading_clean=0.5))
    s3, s = C3.summary(), C.summary()
    assert 0.1 < s3["L_inf"]["median"] < 0.3
    assert abs(np.log(s3["lam"]["median"] / truth.lam)) < 0.5 * abs(np.log(s["lam"]["median"] / truth.lam))


def test_c3_finds_no_loading_in_the_trackers_own_world(small_ctx):
    _, C3, truth = _run(small_ctx, World())
    s3 = C3.summary()
    assert s3["L_inf"]["median"] < 0.03
    assert abs(np.log(s3["lam"]["median"] / truth.lam)) < 0.3
