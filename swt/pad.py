"""Compliant sanding pad: Winkler (elastic foundation) contact model.

The pad face is a disc of radius ``a``. At every station the pad is held with
its face tangent to the surface at the pad centre (the robot keeps the tool
normal to the part). Each surface point under the disc sits a distance
``gap`` below the pad plane (zero at the centre, growing towards the rim on a
convex part). If the pad is pushed a depth ``d`` into the part, the local
Winkler pressure is

    p = k_pad * max(0, d - gap)        [MPa = N/mm^2]

and ``d`` is fixed by the force balance  sum_i p_i * A_i = F  where ``A_i`` is
the pad-face area that maps onto surface node ``i``. The left-hand side is a
monotone, piecewise-linear function of ``d`` with breakpoints at the sorted
gaps, so the 1-D root is found exactly: bracket it between two breakpoints and
solve the linear piece. ``solve_penetration_brentq`` is a generic Brent
root-finder used as an independent reference in the tests.

The force balance only depends on ``F / k_pad`` (call it eta, units mm^3):
penetration and contact patch are functions of eta, and the pressure is
``k_pad * (d(eta) - gap) = F * (d(eta) - gap) / eta``. The surrogate table in
:mod:`swt.surrogate` exploits this.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .geometry import Panel


def solve_penetration(gap: np.ndarray, area: np.ndarray, force: float, k_pad: float) -> float:
    """Exact pad penetration ``d`` such that k * sum(A * max(0, d - gap)) = F."""
    if force <= 0:
        return float(np.min(gap))
    order = np.argsort(gap, kind="stable")
    g = gap[order]
    a = area[order]
    C = np.cumsum(a)
    S = np.cumsum(a * g)
    B = np.empty_like(g)
    B[0] = 0.0
    B[1:] = g[1:] * C[:-1] - S[:-1]
    target = force / k_pad
    j = int(np.searchsorted(B, target, side="right")) - 1
    return float((target + S[j]) / C[j])


def solve_penetration_brentq(gap: np.ndarray, area: np.ndarray, force: float, k_pad: float,
                             xtol: float = 1e-14) -> float:
    """Same root as :func:`solve_penetration`, found with Brent's method."""
    if force <= 0:
        return float(np.min(gap))

    def residual(d):
        return k_pad * np.sum(area * np.maximum(0.0, d - gap)) - force

    lo = float(np.min(gap))
    hi = float(np.max(gap)) + force / (k_pad * float(np.sum(area))) + 1e-12
    return float(brentq(residual, lo, hi, xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=500))


def winkler_pressure(gap: np.ndarray, depth: float, k_pad: float) -> np.ndarray:
    """Winkler pressure p = k * max(0, d - gap) [MPa]."""
    return k_pad * np.maximum(0.0, depth - gap)


class ContactGeometry:
    """Pad footprints for every station of a toolpath, precomputed once.

    For all stations the footprint points are stored in one flat array
    (CSR-style, ``ptr`` gives each station's slice), sorted by gap within each
    station so that the force balance for all stations is solved in a few
    vectorised operations.
    """

    def __init__(self, panel: Panel, stations: np.ndarray, pad_radius: float):
        self.panel = panel
        self.stations = np.asarray(stations, dtype=float).reshape(-1, 2)
        self.pad_radius = float(pad_radius)
        self.pad_area = np.pi * self.pad_radius**2
        a = self.pad_radius
        g = panel.grid
        ny, nx = panel.shape
        # plan-view distance of an in-pad node can exceed its in-plane radius on
        # curved parts; widen the search window by the maximum surface slope
        h = int(np.ceil(a * np.sqrt(1.0 + panel.max_slope**2) / g)) + 2
        idx_l, gap_l, area_l, r_l, cnt = [], [], [], [], []
        for x0, y0 in self.stations:
            iy0, ix0 = panel.node_index(x0, y0)
            ys = slice(max(0, iy0 - h), min(ny, iy0 + h + 1))
            xs = slice(max(0, ix0 - h), min(nx, ix0 + h + 1))
            c = np.array([panel.X[iy0, ix0], panel.Y[iy0, ix0], panel.Z[iy0, ix0]])
            nc = panel.normals[iy0, ix0]
            P = np.stack([panel.X[ys, xs], panel.Y[ys, xs], panel.Z[ys, xs]], axis=-1) - c
            dn = P @ nc
            gap = -dn                                   # depth below the pad plane [mm]
            rvec = P - dn[..., None] * nc
            r = np.linalg.norm(rvec, axis=-1)           # radius in the pad plane [mm]
            inside = r <= a + 1e-9
            # surface area of the node projected onto the pad plane
            proj = panel.node_area[ys, xs] * (panel.normals[ys, xs] @ nc)
            flat = (np.arange(ny)[ys][:, None] * nx + np.arange(nx)[xs][None, :])
            gi, ai, ri, fi = gap[inside], proj[inside], r[inside], flat[inside]
            order = np.argsort(gi, kind="stable")
            idx_l.append(fi[order].astype(np.int32))
            gap_l.append(gi[order])
            area_l.append(ai[order])
            r_l.append(ri[order])
            cnt.append(gi.size)
        self.counts = np.asarray(cnt, dtype=np.int64)
        self.ptr = np.concatenate([[0], np.cumsum(self.counts)])
        self.idx = np.concatenate(idx_l)
        self.gap = np.concatenate(gap_l)
        self.area = np.concatenate(area_l)
        self.r = np.concatenate(r_l)
        self.station_of_point = np.repeat(np.arange(self.n_stations), self.counts)
        # Segment-wise cumulative sums for the piecewise-linear force balance.
        start = self.ptr[:-1]
        cA = np.cumsum(self.area)
        cS = np.cumsum(self.area * self.gap)
        offA = np.repeat(np.concatenate([[0.0], cA[start[1:] - 1]]), self.counts)
        offS = np.repeat(np.concatenate([[0.0], cS[start[1:] - 1]]), self.counts)
        self.C = cA - offA            # sum of A over points with gap <= this gap
        self.S = cS - offS            # sum of A * gap over the same points
        Cprev = np.concatenate([[0.0], self.C[:-1]])
        Sprev = np.concatenate([[0.0], self.S[:-1]])
        Cprev[start] = 0.0
        Sprev[start] = 0.0
        # Breakpoints: value of sum(A * max(0, d - gap)) at d = gap_j.
        self.B = self.gap * Cprev - Sprev

    @property
    def n_stations(self) -> int:
        return self.stations.shape[0]

    def segment(self, s: int) -> slice:
        return slice(int(self.ptr[s]), int(self.ptr[s + 1]))

    def _solve(self, eta: float) -> tuple[np.ndarray, np.ndarray]:
        """Penetration per station and the number of leading (sorted) points in contact."""
        count = np.add.reduceat((self.B <= eta).astype(np.int64), self.ptr[:-1])
        q = self.ptr[:-1] + count - 1
        return (eta + self.S[q]) / self.C[q], count

    def penetration(self, eta: float) -> np.ndarray:
        """Penetration depth d at every station for eta = F / k_pad [mm^3]."""
        if eta <= 0:
            return self.gap[self.ptr[:-1]].copy()
        return self._solve(eta)[0]

    def contact_points(self, force: float, k_pad: float) -> tuple[np.ndarray, np.ndarray]:
        """Flat indices of footprint points in contact and their pressure [MPa].

        Points are sorted by gap within each station, so the contact set of a
        station is a prefix of its segment.
        """
        if force <= 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        d, count = self._solve(force / k_pad)
        start = np.repeat(self.ptr[:-1] - (np.cumsum(count) - count), count)
        pts = np.arange(int(count.sum())) + start
        p = k_pad * np.maximum(0.0, np.repeat(d, count) - self.gap[pts])
        return pts, p

    def pressure(self, force: float, k_pad: float) -> np.ndarray:
        """Winkler pressure at every footprint point of every station [MPa]."""
        if force <= 0:
            return np.zeros_like(self.gap)
        d = self.penetration(force / k_pad)
        return k_pad * np.maximum(0.0, d[self.station_of_point] - self.gap)

    def station_force(self, force: float, k_pad: float) -> np.ndarray:
        """Integrated pressure per station [N] (should equal ``force``)."""
        p = self.pressure(force, k_pad)
        return np.add.reduceat(p * self.area, self.ptr[:-1])

    def footprint_area(self) -> np.ndarray:
        """Discretised pad-face area over the part at every station [mm^2]."""
        return np.add.reduceat(self.area, self.ptr[:-1])

    def contact_fraction(self, force: float, k_pad: float) -> np.ndarray:
        """Fraction of the pad face (over the part) that is in contact, per station."""
        p = self.pressure(force, k_pad)
        return np.add.reduceat(self.area * (p > 0), self.ptr[:-1]) / self.footprint_area()

    def accumulate(self, values: np.ndarray) -> np.ndarray:
        """Sum per-point ``values`` onto the panel grid (flat, n_pix)."""
        return np.bincount(self.idx, weights=values, minlength=self.panel.n_pix)


def contact_fraction_sweep(panel: Panel, pad_radius: float, force: float,
                           k_values: np.ndarray) -> np.ndarray:
    """Contact fraction of a pad centred on the panel for a range of k_pad.

    The station is the panel centre, where the default panel fully supports
    the pad face, so the fraction is a property of pad vs. curvature only.
    """
    centre = np.array([[round(panel.length_x / 2 / panel.grid) * panel.grid,
                        round(panel.width_y / 2 / panel.grid) * panel.grid]])
    cg = ContactGeometry(panel, centre, pad_radius)
    return np.array([cg.contact_fraction(force, k)[0] for k in np.atleast_1d(k_values)])
