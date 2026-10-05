import numpy as np
import pytest

from swt.pad import PadLaw
from swt.process import (HiddenTruth, PassAction, WearLaw, World, line_effectiveness, make_schedule,
                         preston_removal, simulate_truth, wear_step)
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


@pytest.mark.parametrize("wear", [WearLaw(), WearLaw("two_stage", 0.25, 6.0)])
def test_within_pass_wear_conserves_volume(small_ctx, wear):
    """K wears during the pass: the removal map's volume equals the volume that drives the
    wear, and the K at the next pass is K0 * phi(W) with W the cumulative removed volume."""
    world = World("w", wear=wear)
    truth = HiddenTruth(0.05, 6e-5, 3e-4)
    traj = simulate_truth(small_ctx.model, truth, small_ctx.schedule(), 0.0, np.random.default_rng(0), world=world)
    W = np.concatenate([[0.0], np.cumsum(traj.volume)[:-1]])
    assert np.allclose(traj.volume, [small_ctx.model.volume(r) for r in traj.removal], rtol=1e-9)
    assert np.allclose(traj.K, [truth.K0 * wear.phi(truth.lam, w) for w in W], rtol=1e-6)


def test_within_pass_wear_closed_form_is_the_ode_solution():
    """Single-stage law: the closed form equals the RK4 integration used for other laws, and
    with lambda = 0 every line has the starting effectiveness (removal linear in K)."""
    u = np.array([3.0, 5.0, 2.0, 7.0]) * 1e5
    K, lam = 6e-5, 3e-4
    closed = line_effectiveness(K, lam, u)
    ode = line_effectiveness(K, lam, u, WearLaw("two_stage", 0.0, 1.0), K0=K)   # f = 0: single stage via RK4
    assert np.allclose(closed, ode, rtol=1e-8)
    assert np.all(np.diff(closed) < 0)
    assert np.all(line_effectiveness(K, 0.0, u) == K)


def test_two_stage_wear_law():
    w = WearLaw("two_stage", 0.25, 6.0)
    assert w.phi(3e-4, 0.0) == 1.0
    # faster than the single-stage law at first, the same rate in the long run
    assert w.phi(3e-4, 500.0) < WearLaw().phi(3e-4, 500.0)
    assert w.phi(3e-4, 2e4) / WearLaw().phi(3e-4, 2e4) == pytest.approx(0.75, rel=1e-3)


def test_force_gain_scales_the_actual_force(small_ctx):
    sched = make_schedule([30.0], 2, 8000.0)
    t1 = simulate_truth(small_ctx.model, HiddenTruth(0.05, 6e-5, 0.0, force_gain=1.1), sched, 0.0,
                        np.random.default_rng(0), max_extend=0)
    t2 = simulate_truth(small_ctx.model, HiddenTruth(0.05, 6e-5, 0.0), make_schedule([33.0], 2, 8000.0), 0.0,
                        np.random.default_rng(0), max_extend=0)
    assert np.allclose(t1.removal, t2.removal, rtol=1e-12)


def test_force_ripple_first_order_model(small_ctx):
    """Ripple of 3%: the first-order line exposures match the exact ones to < 0.1%, the
    oracle ignores the ripple, and without ripple nothing changes."""
    m = small_ctx.model
    sched = make_schedule([30.0], 1, 8000.0)
    truth = HiddenTruth(0.05, 6e-5, 0.0)
    world = World("r", force_ripple=0.03, ripple_corr=0.5)
    traj = simulate_truth(m, truth, sched, 0.0, np.random.default_rng(0), max_extend=0, world=world,
                          ripple_rng=np.random.default_rng(1))
    # rebuild the ripple sequence and the exact removal line by line
    r = np.random.default_rng(1).standard_normal(m.n_lines)
    e = np.empty(m.n_lines)
    e[0] = r[0]
    for i in range(1, m.n_lines):
        e[i] = 0.5 * e[i - 1] + np.sqrt(0.75) * r[i]
    e *= 0.03
    exact = sum(6e-5 * m.line_exposures(30.0 * (1 + e[i]), 0.05)[0][i] for i in range(m.n_lines))
    assert np.linalg.norm(traj.removal[0] - exact) / np.linalg.norm(exact) < 1e-3
    assert np.allclose(traj.oracle[0], m.removal(30.0, 0.05, 6e-5), rtol=1e-12)
    plain = simulate_truth(m, truth, sched, 0.0, np.random.default_rng(0), max_extend=0)
    assert np.allclose(plain.removal[0], m.removal(30.0, 0.05, 6e-5), rtol=1e-12)


def test_foam_pad_changes_the_map_but_not_the_load(small_ctx):
    m = small_ctx.model
    foam = PadLaw("foam", 15.0)
    for k in (1e-3, 0.05):
        lin, _ = m.line_exposures(30.0, k)
        fo, _ = m.line_exposures(30.0, k, law=foam)
        # the same total load and sliding speed field -> almost the same volume; a different map
        assert fo.sum() == pytest.approx(lin.sum(), rel=0.02)
        assert np.linalg.norm(fo - lin) / np.linalg.norm(lin) > 1e-3


def test_preston_exponent_keeps_the_load_and_changes_the_volume(small_ctx):
    """alpha = 1 is the default law; with alpha < 1 the pressure-weighted removal favours
    spread-out (soft) contact over concentrated (stiff) contact."""
    m = small_ctx.model
    for k in (1e-3, 2.0):
        E1, u1 = m.line_exposures(30.0, k)
        assert np.allclose(m.line_exposures(30.0, k, alpha=1.0)[0], E1)
    soft = m.line_exposures(30.0, 1e-3, alpha=0.8)[1].sum() / m.line_exposures(30.0, 1e-3)[1].sum()
    stiff = m.line_exposures(30.0, 2.0, alpha=0.8)[1].sum() / m.line_exposures(30.0, 2.0)[1].sum()
    assert soft > stiff


def test_ring_resolved_exposures_add_up(small_ctx):
    m = small_ctx.model
    E, u = m.line_exposures(30.0, 0.05)
    Er, ur = m.line_exposures(30.0, 0.05, rings=5)
    assert Er.shape == (5,) + E.shape and np.allclose(Er.sum(axis=0), E) and np.allclose(ur.sum(axis=0), u)


def test_radial_wear_dulls_the_pad_centre_of_a_stiff_pad(small_ctx):
    """With ring-resolved wear, a stiff pad (contact near its centre) loses effectiveness
    faster than with uniform wear, and one ring reproduces the uniform law exactly."""
    sched = small_ctx.schedule()
    truth = HiddenTruth(2.0, 6e-5, 2.5e-4)
    uni = simulate_truth(small_ctx.model, truth, sched, 0.0, np.random.default_rng(0))
    one = simulate_truth(small_ctx.model, truth, sched, 0.0, np.random.default_rng(0), world=World("1", wear_rings=1))
    rings = simulate_truth(small_ctx.model, truth, sched, 0.0, np.random.default_rng(0), world=World("r", wear_rings=6))
    assert np.allclose(one.removal, uni.removal, rtol=1e-12)
    assert rings.K[-1] < uni.K[-1]
    assert np.allclose(rings.volume, [small_ctx.model.volume(r) for r in rings.removal], rtol=1e-9)
