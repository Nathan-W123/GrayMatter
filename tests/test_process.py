import numpy as np
import pytest

from swt.process import (HiddenTruth, PassAction, make_schedule, preston_removal, simulate_truth,
                         wear_step)
from swt.scan import Scanner


@pytest.mark.parametrize("k_pad", [6.3e-4, 0.05, 5.0])
def test_zero_force_gives_zero_removal(small_ctx, k_pad):
    """Required test 3."""
    m = small_ctx.model
    assert np.all(m.removal(0.0, k_pad, 6e-5) == 0.0)
    traj = simulate_truth(m, HiddenTruth(k_pad, 6e-5, 2.5e-4), make_schedule([0.0], 3, 8000.0),
                          0.0, np.random.default_rng(0), max_extend=0)
    assert np.all(traj.removal == 0.0)
    assert np.all(traj.K == 6e-5)          # no work done -> no wear


@pytest.mark.parametrize("k_pad", [1e-3, 0.05, 2.0])
def test_removal_linear_in_K_at_fixed_pressure(small_ctx, k_pad):
    """Required test 5: removal scales linearly with K at fixed pressure."""
    m = small_ctx.model
    base = m.removal(30.0, k_pad, 1e-5)
    for c in (0.5, 2.0, 7.0):
        assert np.allclose(m.removal(30.0, k_pad, c * 1e-5), c * base, rtol=1e-12, atol=0)
    p = np.array([0.0, 0.001, 0.003])
    v = np.array([2000.0, 2500.0, 3000.0])
    assert np.allclose(preston_removal(3e-5, p, v, 0.2), 3 * preston_removal(1e-5, p, v, 0.2))


def test_removed_volume_is_almost_independent_of_stiffness(small_ctx):
    """Force balance => volume = K * F * mean speed * time; stiffness only enters
    through the radial speed profile (a percent-level effect)."""
    m = small_ctx.model
    pr = small_ctx.priors
    vols = [m.volume(m.removal(30.0, k, 1.0)) for k in np.geomspace(pr.k_low, pr.k_high, 7)]
    assert (max(vols) - min(vols)) / np.mean(vols) < 0.03


def test_wear_decreases_effectiveness(small_ctx):
    assert wear_step(1.0, 2.5e-4, 0.0) == 1.0
    assert wear_step(1.0, 2.5e-4, 400.0) == pytest.approx(np.exp(-0.1))
    traj = simulate_truth(small_ctx.model, HiddenTruth(0.05, 6e-5, 2.5e-4),
                          make_schedule([20.0, 40.0], 20, 8000.0), 0.0, np.random.default_rng(1))
    assert np.all(np.diff(traj.K) < 0)
    assert traj.crossing_pass is not None
    assert traj.K_extended[traj.crossing_pass - 1] < 0.5 * 6e-5 <= traj.K_extended[traj.crossing_pass - 2]


def test_surrogate_matches_exact_model(small_ctx, rng):
    m, tab = small_ctx.model, small_ctx.table
    pr = small_ctx.priors
    for k in np.exp(rng.uniform(np.log(pr.k_low), np.log(pr.k_high), 12)):
        for F in (20.0, 40.0):
            exact = m.exposure(F, k)
            approx = F * tab.map_full(F / k)
            assert np.sqrt(np.mean((exact - approx) ** 2)) / np.sqrt(np.mean(exact**2)) < 2e-3


def test_scan_noise_and_downsampling(small_ctx, rng):
    sc = Scanner(small_ctx.panel, noise_um=2.0, stride=2)
    ny, nx = small_ctx.panel.shape
    assert sc.n_obs == ((ny + 1) // 2) * ((nx + 1) // 2)
    clean = np.full(small_ctx.panel.n_pix, 0.01)
    y = sc.observe(clean, rng)
    assert np.std(y - 0.01) == pytest.approx(0.002, rel=0.1)
    assert np.all(Scanner(small_ctx.panel, 0.0, 1).observe(clean, rng) == clean)


def test_schedule_cycles_forces():
    s = make_schedule([20.0, 40.0], 5, 8000.0)
    assert [a.force for a in s] == [20.0, 40.0, 20.0, 40.0, 20.0]
    assert s[0] == PassAction(20.0, 8000.0)


@pytest.mark.parametrize("spacing", [5.0, 15.0, 30.0, 35.0])
def test_raster_covers_the_panel_for_any_station_spacing(small_ctx, spacing):
    from swt.geometry import raster_path
    from swt.pad import ContactGeometry
    panel = small_ctx.panel
    path = raster_path(panel, stepover=15.0, station_spacing=spacing, feed=25.0, dwell=0.25)
    xs = path.stations[path.line == 0, 0]
    assert xs.min() == 0.0 and xs.max() == panel.length_x
    # time per line = line length / feed + two dwells
    t_line = path.dt[path.line == 0].sum()
    assert t_line == pytest.approx(panel.length_x / 25.0 + 2 * 0.25)
    ContactGeometry(panel, path.stations, 62.5)   # every station lies on the panel


def test_truth_trajectory_continues_past_the_scanned_passes(small_ctx):
    traj = simulate_truth(small_ctx.model, HiddenTruth(0.05, 6e-5, 2.5e-4),
                          make_schedule([20.0, 40.0], 5, 8000.0), 0.0, np.random.default_rng(0))
    assert traj.crossing_pass > 5 and len(traj.K) == 5
    assert traj.K_extended[traj.crossing_pass - 1] < 0.5 * 6e-5 <= traj.K_extended[traj.crossing_pass - 2]


def test_preston_removal_matches_brute_force_over_the_whole_pass(small_ctx):
    """Independent assembly of dh = K p v dt: Brent root per station, Winkler pressure,
    sliding speed at each point's radius, and each station's own dwell time."""
    from swt.pad import solve_penetration_brentq, winkler_pressure
    m, cg = small_ctx.model, small_ctx.model.contact
    F, k, K = 30.0, 0.05, 6e-5
    brute = np.zeros(small_ctx.panel.n_pix)
    for s in range(cg.n_stations):
        seg = cg.segment(s)
        d = solve_penetration_brentq(cg.gap[seg], cg.area[seg], F, k)
        p = winkler_pressure(cg.gap[seg], d, k)
        np.add.at(brute, cg.idx[seg], K * p * small_ctx.sander.speed(cg.r[seg]) * small_ctx.path.dt[s])
    assert np.max(np.abs(m.removal(F, k, K) - brute)) / brute.max() < 1e-8


def test_removal_scales_with_spindle_speed(small_ctx):
    m = small_ctx.model
    assert np.allclose(m.removal(30.0, 0.05, 6e-5, rpm=4000.0), 0.5 * m.removal(30.0, 0.05, 6e-5), rtol=1e-12)


def test_truth_follows_the_documented_wear_recursion(small_ctx):
    truth = HiddenTruth(0.05, 6e-5, 2.5e-4)
    traj = simulate_truth(small_ctx.model, truth, small_ctx.schedule(), 0.0, np.random.default_rng(0))
    assert np.allclose(traj.K[1:], traj.K[:-1] * np.exp(-truth.lam * traj.volume[:-1]), rtol=1e-12)
    assert np.allclose(traj.volume, [small_ctx.model.volume(r) for r in traj.removal], rtol=1e-12)
