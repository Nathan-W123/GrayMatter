import numpy as np
import pytest

from swt.geometry import Panel
from swt.pad import (ContactGeometry, contact_fraction_sweep, solve_penetration,
                     solve_penetration_brentq, winkler_pressure)


@pytest.mark.parametrize("force", [5.0, 20.0, 40.0])
@pytest.mark.parametrize("k_pad", [6.3e-4, 0.01, 0.2, 5.0])
def test_integrated_pressure_equals_force(small_ctx, force, k_pad):
    """Required test 1: integral of p over the contact = applied force (within 0.5%),
    at every station of the raster, including stations half off the panel edge."""
    cg = small_ctx.model.contact
    per_station = cg.station_force(force, k_pad)
    assert np.all(np.abs(per_station - force) / force < 0.005)
    # same check by rasterising the pressure onto the grid and integrating there
    p = cg.pressure(force, k_pad)
    for s in (0, cg.n_stations // 2, cg.n_stations - 1):
        seg = cg.segment(s)
        grid = np.zeros(small_ctx.panel.n_pix)
        np.add.at(grid, cg.idx[seg], p[seg] * cg.area[seg])
        assert abs(grid.sum() - force) / force < 0.005


@pytest.mark.parametrize("k_pad", [1e-3, 0.05, 3.0])
def test_exact_root_matches_brentq(small_ctx, k_pad):
    cg = small_ctx.model.contact
    for s in (0, 7, cg.n_stations // 2):
        seg = cg.segment(s)
        d_exact = solve_penetration(cg.gap[seg], cg.area[seg], 30.0, k_pad)
        d_brent = solve_penetration_brentq(cg.gap[seg], cg.area[seg], 30.0, k_pad)
        assert d_exact == pytest.approx(d_brent, rel=1e-9, abs=1e-12)
        assert d_exact == pytest.approx(cg.penetration(30.0 / k_pad)[s], rel=1e-9)
        F = np.sum(winkler_pressure(cg.gap[seg], d_brent, k_pad) * cg.area[seg])
        assert F == pytest.approx(30.0, rel=1e-6)


def test_contact_fraction_increases_as_stiffness_decreases():
    """Required test 2: contact fraction rises monotonically as k_pad decreases."""
    panel = Panel(200.0, 150.0, 300.0, 1.0)
    ks = np.logspace(1, -4, 80)            # stiff -> soft
    cf = contact_fraction_sweep(panel, 62.5, 30.0, ks)
    assert np.all(np.diff(cf) >= -1e-12)
    assert cf[0] < 0.06 and cf[-1] > 0.95


def test_default_prior_range_spans_5_to_97_percent(small_cfg):
    from swt.config import load_config
    cfg = load_config()      # the shipped default geometry, pad and reference force
    g, pr = cfg["geometry"], cfg["priors"]
    panel = Panel(g["length_x_mm"], g["width_y_mm"], g["radius_mm"], g["grid_mm"])
    soft, stiff = contact_fraction_sweep(panel, 0.5 * cfg["pad"]["diameter_mm"], cfg["pad"]["reference_force_N"],
                                         np.array([pr["k_low"], pr["k_high"]]))
    assert 0.04 <= stiff <= 0.07
    assert 0.95 <= soft <= 0.99


def test_flat_panel_full_contact_is_uniform():
    """On a flat panel every pad point has zero gap: pressure is F / area everywhere."""
    panel = Panel(200.0, 150.0, None, 1.0)
    cg = ContactGeometry(panel, np.array([[100.0, 75.0]]), 62.5)
    p = cg.pressure(30.0, 0.05)
    assert np.allclose(p, 30.0 / cg.area.sum())


@pytest.mark.parametrize("radius", [300.0, 100.0])
def test_footprint_contains_every_node_within_pad_radius(radius):
    """Brute force over the whole grid: every node whose in-plane radius is <= a must be
    in the footprint, including tight radii where plan distance exceeds in-plane distance."""
    panel = Panel(200.0, 150.0, radius, 1.0)
    a = 62.5
    for y0 in (0.0, 39.0, 75.0, 120.0):
        cg = ContactGeometry(panel, np.array([[100.0, y0]]), a)
        iy, ix = panel.node_index(100.0, y0)
        c = np.array([panel.X[iy, ix], panel.Y[iy, ix], panel.Z[iy, ix]])
        nc = panel.normals[iy, ix]
        P = np.stack([panel.X, panel.Y, panel.Z], axis=-1).reshape(-1, 3) - c
        dn = P @ nc
        r = np.linalg.norm(P - dn[:, None] * nc, axis=1)
        assert cg.counts[0] == int(np.sum(r <= a + 1e-9))


@pytest.mark.parametrize("radius", [300.0, 100.0])
def test_footprint_area_equals_the_pad_face(radius):
    """Independent check of the integration weights: a fully supported pad's footprint
    (surface area projected onto the pad plane) must equal pi a^2."""
    cg = ContactGeometry(Panel(200.0, 150.0, radius, 1.0), np.array([[100.0, 75.0]]), 62.5)
    assert cg.footprint_area()[0] == pytest.approx(np.pi * 62.5**2, rel=0.005)


@pytest.mark.parametrize("k_pad", [6.3e-4, 0.05, 5.0])
@pytest.mark.parametrize("force", [20.0, 40.0])
def test_foam_pad_force_balance_and_contact(small_ctx, force, k_pad):
    """Foam law (stiffens as it densifies): the integrated pressure still equals the force,
    and less of the pad face touches than for a linear pad of the same small-strain k."""
    from swt.pad import PadLaw
    cg = small_ctx.model.contact
    foam = PadLaw("foam", 15.0)
    assert np.allclose(cg.station_force(force, k_pad, foam), force, rtol=1e-9)
    assert np.all(cg.contact_fraction(force, k_pad, foam) <= cg.contact_fraction(force, k_pad) + 1e-12)
