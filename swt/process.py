"""Material removal (Preston's law), abrasive wear, and the hidden-truth simulator.

Removal
-------
Preston's law gives the removal depth rate at a surface point as

    dh/dt = K * p * v                              [mm/s]

with K the abrasive effectiveness (Preston coefficient) [mm^2/N], p the Winkler
contact pressure [MPa] and v the local sliding speed [mm/s]. Summing over the
stations of a pass, the removal map is

    h(x) = K * E(x),   E(x) = sum_stations p(x) * v(x) * dt     [N/mm]

where E is the "exposure" map. K is held constant within a pass.

Sliding speed
-------------
A random-orbital sander moves every point of the pad on a small orbit
(diameter D_orb at the spindle frequency f) while the pad also turns slowly
about its own axis (a fraction ``spin_ratio`` of the spindle speed). The two
motions are not phase-locked, so we use the RMS sliding speed

    v(r) = sqrt( (pi * D_orb * f)^2 + (2 * pi * f * spin_ratio * r)^2 )

which grows from the pad centre to the rim. Soft pads that contact near the
rim therefore remove slightly more volume per newton than stiff pads. Stiffness
changes the *scale* of removal only through this contact-weighted sliding
speed and the surface-slope factor of the contact area (together about 1.7%
across the default prior); the force balance fixes the total load.

Wear
----
Abrasive effectiveness decays with the cumulative volume of material the
abrasive has cut (the logic of the grinding "G-ratio"):

    K(W) = K0 * exp(-lambda * W),   W = cumulative removed volume [mm^3]

Applied once per pass with a small multiplicative fluctuation (grain fracture,
loading and batch variation are not deterministic):

    K_{n+1} = K_n * exp(-lambda * dV_n + sigma_w * xi_n),   xi_n ~ N(0, 1)

Because dV_n is itself proportional to K_n, dull paper both cuts and wears
more slowly: without noise K_n follows a hyperbolic decay in pass count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Panel, RasterPath
from .pad import ContactGeometry


@dataclass
class Sander:
    """Random-orbital sander kinematics."""

    spindle_rpm: float = 8000.0
    orbit_diameter: float = 5.0      # mm
    pad_spin_ratio: float = 0.02     # pad rotation / spindle rotation

    def speed(self, r: np.ndarray, rpm: float | None = None) -> np.ndarray:
        """RMS sliding speed [mm/s] at radius ``r`` [mm] from the pad centre."""
        f = (self.spindle_rpm if rpm is None else rpm) / 60.0
        v_orb = np.pi * self.orbit_diameter * f
        v_spin = 2.0 * np.pi * f * self.pad_spin_ratio * np.asarray(r)
        return np.sqrt(v_orb**2 + v_spin**2)


def preston_removal(K: float | np.ndarray, pressure: np.ndarray, speed: np.ndarray,
                    dt: float | np.ndarray) -> np.ndarray:
    """Preston's law: removal depth dh = K * p * v * dt [mm]."""
    return K * pressure * speed * dt


def wear_step(K: float | np.ndarray, lam: float | np.ndarray, removed_volume: float | np.ndarray,
              noise: float | np.ndarray = 0.0) -> np.ndarray:
    """K after one pass: K * exp(-lambda * dV + noise)."""
    return K * np.exp(-lam * removed_volume + noise)


@dataclass
class PassAction:
    """Commanded action for one pass (path parameters are shared by all passes)."""

    force: float          # N
    rpm: float            # spindle speed


def make_schedule(forces: list[float], n_passes: int, rpm: float) -> list[PassAction]:
    """Cycle through ``forces`` (e.g. [20, 40] alternates low/high)."""
    return [PassAction(float(forces[i % len(forces)]), float(rpm)) for i in range(n_passes)]


class ProcessModel:
    """Exact forward model for one pass: exposure, removal map, removed volume."""

    def __init__(self, panel: Panel, path: RasterPath, pad_radius: float, sander: Sander):
        self.panel = panel
        self.path = path
        self.sander = sander
        self.contact = ContactGeometry(panel, path.stations, pad_radius)
        # v * dt for every footprint point at the reference spindle speed.
        self.vdt = sander.speed(self.contact.r) * path.dt[self.contact.station_of_point]
        self.node_area_flat = panel.node_area.ravel()

    def speed_scale(self, rpm: float) -> float:
        """Both sliding-speed components scale linearly with spindle speed."""
        return rpm / self.sander.spindle_rpm

    def exposure(self, force: float, k_pad: float, rpm: float | None = None) -> np.ndarray:
        """Exposure map E = sum p v dt [N/mm] (flat, n_pix)."""
        if force <= 0:
            return np.zeros(self.panel.n_pix)
        pts, p = self.contact.contact_points(force, k_pad)
        scale = 1.0 if rpm is None else self.speed_scale(rpm)
        return scale * np.bincount(self.contact.idx[pts], weights=p * self.vdt[pts],
                                   minlength=self.panel.n_pix)

    def removal(self, force: float, k_pad: float, K: float, rpm: float | None = None) -> np.ndarray:
        """Removal depth map for one pass [mm] (flat, n_pix)."""
        return K * self.exposure(force, k_pad, rpm)

    def volume(self, removal_flat: np.ndarray) -> float:
        """Removed volume [mm^3]."""
        return float(removal_flat @ self.node_area_flat)


# --------------------------------------------------------------------------- priors / truth


@dataclass
class Priors:
    """Priors over the hidden parameters (all positive, handled in log space).

    k_pad  ~ LogUniform[k_low, k_high]       [N/mm^3]
    K0     ~ LogNormal(log K0_median, K0_sigma_log)   [mm^2/N]
    lambda ~ LogNormal(log lam_median, lam_sigma_log) [1/mm^3]
    """

    k_low: float
    k_high: float
    K0_median: float
    K0_sigma_log: float
    lam_median: float
    lam_sigma_log: float

    def sample(self, rng: np.random.Generator, n: int | None = None) -> dict:
        size = n
        return {
            "k_pad": np.exp(rng.uniform(np.log(self.k_low), np.log(self.k_high), size)),
            "K0": np.exp(rng.normal(np.log(self.K0_median), self.K0_sigma_log, size)),
            "lam": np.exp(rng.normal(np.log(self.lam_median), self.lam_sigma_log, size)),
        }

    def log_mean(self) -> dict:
        """Prior mean of each log-parameter (the 'nominal' parameter set)."""
        return {
            "k_pad": float(np.exp(0.5 * (np.log(self.k_low) + np.log(self.k_high)))),
            "K0": float(self.K0_median),
            "lam": float(self.lam_median),
        }


@dataclass
class HiddenTruth:
    k_pad: float
    K0: float
    lam: float


@dataclass
class TruthTrajectory:
    """What actually happened (hidden from the estimators)."""

    truth: HiddenTruth
    K: np.ndarray                  # effectiveness during each scanned pass
    removal: np.ndarray            # (n_passes, n_pix) true removal maps [mm]
    volume: np.ndarray             # removed volume per scanned pass [mm^3]
    crossing_pass: int | None      # first pass (1-based) with K < tau*K0
    K_extended: np.ndarray = field(repr=False, default=None)  # K beyond the scanned passes


def action_at(schedule: list[PassAction], i: int) -> PassAction:
    """Action for pass index ``i`` (0-based); past the end, repeat the schedule's cycle."""
    if i < len(schedule):
        return schedule[i]
    p = cycle_length(schedule)
    return schedule[i % p]


def cycle_length(schedule: list[PassAction]) -> int:
    """Length of the shortest repeating force/rpm cycle of a schedule."""
    n = len(schedule)
    keys = [(a.force, a.rpm) for a in schedule]
    for p in range(1, n + 1):
        if all(keys[i] == keys[i % p] for i in range(n)):
            return p
    return n


def simulate_truth(model: ProcessModel, truth: HiddenTruth, schedule: list[PassAction],
                   wear_sigma: float, rng: np.random.Generator, threshold: float = 0.5,
                   max_extend: int = 400) -> TruthTrajectory:
    """Run the hidden process for the scheduled passes.

    The wear trajectory is continued past the last scanned pass (repeating the
    schedule's force cycle) until the threshold crossing, so the true change
    point is defined even when it falls after the experiment ends.
    """
    cache: dict[tuple[float, float], np.ndarray] = {}

    def exposure(a: PassAction) -> np.ndarray:
        key = (a.force, a.rpm)
        if key not in cache:
            cache[key] = model.exposure(a.force, truth.k_pad, a.rpm)
        return cache[key]

    n = len(schedule)
    K = truth.K0
    Ks, maps, vols, K_ext = [], [], [], []
    crossing = None
    for i in range(n + max_extend):
        if crossing is None and K < threshold * truth.K0:
            crossing = i + 1
        K_ext.append(K)
        if i >= n and crossing is not None:
            break
        a = action_at(schedule, i)
        E = exposure(a)
        dV = K * float(E @ model.node_area_flat)
        if i < n:
            Ks.append(K)
            maps.append(K * E)
            vols.append(dV)
        K = float(wear_step(K, truth.lam, dV, wear_sigma * rng.standard_normal()))
    return TruthTrajectory(truth, np.array(Ks), np.array(maps), np.array(vols), crossing,
                           np.array(K_ext))
