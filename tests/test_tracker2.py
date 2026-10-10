import dataclasses

import numpy as np

from swt.config import rng_for
from swt.estimators import ParticleFilter
from swt.experiments import regions
from swt.process import HiddenTruth, simulate_truth
from swt.scan import Scanner
from swt.tracker2 import C2Config, DiscrepancyTracker, _bilinear_weights


def _run(ctx, shape_factor=None, passes=12, with_c2=True, seed=3):
    """C (and optionally C2 around it) through ``passes`` passes of a matched truth whose removal
    maps are multiplied by ``shape_factor`` (a map-shape error C's model cannot represent).
    Returns C, C2 and per-pass (true, C map, C2 map) at the scan points."""
    sched = ctx.schedule()
    traj = simulate_truth(ctx.model, HiddenTruth(0.05, 6e-5, 2.5e-4), sched, 0.0, np.random.default_rng(1))
    if shape_factor is not None:
        traj.removal = traj.removal * shape_factor[None, :]
    pf = ParticleFilter(ctx.table, ctx.priors, dataclasses.replace(ctx.pf, n_particles=1000), rng_for(7, seed),
                        rollout_rng=rng_for(7, seed + 1), obs_shape=ctx.obs_shape)
    idx, tab = regions(ctx)
    c2 = DiscrepancyTracker(pf, idx, tab, ctx.obs_shape, C2Config(), rng_for(7, 99)) if with_c2 else None
    sc = Scanner(ctx.panel, 2.0, ctx.cfg["scan"]["stride"])
    rng = np.random.default_rng(2)
    out = []
    for n, a in enumerate(sched[:passes]):
        if n:
            pf.advance(sched[n - 1])
        c_map = pf.predict(a)
        if c2 is not None:
            c2.prepare(a)
            out.append((traj.removal[n][ctx.obs_index], c_map, c2.predict_map(c_map)))
        else:
            out.append((traj.removal[n][ctx.obs_index], c_map, None))
        scan = sc.observe(traj.removal[n], rng)
        pf.update(scan, a, 2.0)
        if c2 is not None:
            c2.update(scan, 2.0)
    return pf, c2, out


def test_bilinear_weights_are_a_partition_of_unity():
    W = _bilinear_weights((37, 51), (4, 4))
    assert W.shape == (37 * 51, 16)
    assert np.all(W >= 0)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)


def test_c2_leaves_c_unchanged(small_ctx):
    """C2 only reads C's state: C's predictions are identical with and without C2."""
    _, _, with_c2 = _run(small_ctx, passes=6)
    _, _, without = _run(small_ctx, passes=6, with_c2=False)
    for (_, a, _), (_, b, _) in zip(with_c2, without):
        np.testing.assert_array_equal(a, b)


def test_c2_stays_out_of_the_way_without_a_discrepancy(small_ctx):
    """In the tracker's own world the shape layer stays at zero and the level near zero."""
    _, c2, out = _run(small_ctx)
    assert c2.summary()["shape_rms"] < 0.01
    assert abs(c2.summary()["level"]) < 0.02
    err_c = np.sqrt(np.mean([(c - t) ** 2 for t, c, _ in out[4:]]))
    err_c2 = np.sqrt(np.mean([(c2m - t) ** 2 for t, _, c2m in out[4:]]))
    assert err_c2 < 1.05 * err_c


def test_c2_learns_a_map_shape_error(small_ctx):
    """A smooth 30% left-to-right tilt of every removal map: C cannot represent it, C2's shape
    layer learns it and its later predictions are clearly more accurate."""
    x = small_ctx.panel.X.ravel() / small_ctx.panel.length_x
    factor = 0.85 + 0.30 * x
    _, c2, out = _run(small_ctx, shape_factor=factor)
    assert c2.summary()["shape_rms"] > 0.03
    err_c = np.sqrt(np.mean([(c - t) ** 2 for t, c, _ in out[6:]]))
    err_c2 = np.sqrt(np.mean([(c2m - t) ** 2 for t, _, c2m in out[6:]]))
    assert err_c2 < 0.7 * err_c
