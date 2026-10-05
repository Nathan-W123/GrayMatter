import numpy as np
import pytest

from swt.scan import ScanArtefacts, Scanner


def test_artefacts_have_the_configured_statistics(small_ctx):
    panel = small_ctx.panel
    clean = np.full(panel.n_pix, 0.01)
    art = ScanArtefacts(profile_bias_um=1.0, outlier_fraction=0.01, outlier_um=15.0,
                        dropout_fraction=0.04, dropout_gap_points=8)
    sc = Scanner(panel, 0.0, 2, art)
    rng = np.random.default_rng(0)
    ys = [sc.observe(clean, rng).reshape(sc.obs_shape) for _ in range(30)]
    drop = np.mean([np.isnan(y).mean() for y in ys])
    assert drop == pytest.approx(0.04, rel=0.3)
    # without noise, a profile's points share one offset unless hit by an outlier
    offs = [np.nanmedian(y - 0.01, axis=0) for y in ys]
    assert np.std(np.concatenate(offs)) * 1e3 == pytest.approx(1.0, rel=0.15)
    dev = np.concatenate([(y - 0.01 - o[None, :])[np.isfinite(y)] for y, o in zip(ys, offs)])
    assert np.mean(np.abs(dev) > 1e-9) == pytest.approx(0.01, rel=0.3)


def test_registration_shift_moves_the_map(small_ctx):
    panel = small_ctx.panel
    ramp = (panel.X * 1e-4).ravel()                       # removal growing along x
    sc = Scanner(panel, 0.0, 2, ScanArtefacts(registration_sigma_mm=0.5))
    rng = np.random.default_rng(1)
    shifts = []
    for _ in range(40):
        y = sc.observe(ramp, rng).reshape(sc.obs_shape)
        inner = y[:, 5:-5] - ramp[sc.obs_index].reshape(sc.obs_shape)[:, 5:-5]
        shifts.append(-np.mean(inner) / 1e-4)             # shift in mm (sign: map moved by +off)
    assert np.std(shifts) == pytest.approx(0.5, rel=0.3)


def test_no_artefacts_is_plain_gaussian_noise(small_ctx):
    sc = Scanner(small_ctx.panel, 2.0, 2)
    y = sc.observe(np.zeros(small_ctx.panel.n_pix), np.random.default_rng(2))
    assert np.all(np.isfinite(y))
    assert np.std(y) == pytest.approx(2e-3, rel=0.05)
