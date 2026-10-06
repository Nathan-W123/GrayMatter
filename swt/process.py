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

Wear acts continuously during a pass. Within a pass K follows the closed form
K(U) = K_start / (1 + lambda * K_start * U), where U is the volume the pass would
have removed at unit effectiveness so far; it is evaluated at the resolution of
one raster line, using the exact line-average of K (so the volume of the removal
map equals the volume that drives the wear). Between passes a small
multiplicative fluctuation is applied (grain fracture, loading and batch
variation are not deterministic):

    K_{n+1} = K_n * exp(-lambda * dV_n + sigma_w * xi_n),   xi_n ~ N(0, 1)

The "realistic" world can replace this law with a two-stage law (a fast
break-in of the sharpest grit tips plus slow dulling), use a foam pad that
stiffens as it densifies, and apply a force-calibration error; the tracker
always assumes the single-stage law, a linear pad and the commanded force.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Panel, RasterPath
from .pad import LINEAR, ContactGeometry, PadLaw


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


P_REF = 0.01   # MPa: reference pressure of the Preston exponent law (10 kPa)


class ProcessModel:
    """Exact forward model for one pass: exposure, removal map, removed volume."""

    def __init__(self, panel: Panel, path: RasterPath, pad_radius: float, sander: Sander):
        self.panel = panel
        self.path = path
        self.sander = sander
        self.contact = ContactGeometry(panel, path.stations, pad_radius)
        self.pad_radius = float(pad_radius)
        # v * dt for every footprint point at the reference spindle speed.
        self.vdt = sander.speed(self.contact.r) * path.dt[self.contact.station_of_point]
        self.node_area_flat = panel.node_area.ravel()
        self.line_of_point = path.line[self.contact.station_of_point]
        self.n_lines = int(path.line.max()) + 1

    def speed_scale(self, rpm: float) -> float:
        """Both sliding-speed components scale linearly with spindle speed."""
        return rpm / self.sander.spindle_rpm

    def line_exposures(self, force: float, k_pad: float, rpm: float | None = None,
                       law: PadLaw = LINEAR, alpha: float = 1.0, rings: int = 0,
                       tilt_stiffness: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Exposure map of every raster line, (n_lines, n_pix) [N/mm], and each line's
        removed volume per unit effectiveness [mm^3 per (mm^2/N)].

        ``alpha`` != 1: Preston's law with a pressure exponent, dh = K p_ref (p / p_ref)^alpha v dt
        (p_ref = ``P_REF``). ``rings`` > 0: the contributions of ``rings`` equal-width
        rings of the pad face are kept apart, shapes (rings, n_lines, n_pix) and
        (rings, n_lines), for abrasive wear that varies across the pad.
        ``tilt_stiffness`` > 0: the pad holder tilts (linear pad only; see
        :meth:`ContactGeometry.tilt_solution`).
        """
        n = self.panel.n_pix
        R = max(rings, 1)
        shape = (R, self.n_lines) if rings else (self.n_lines,)
        if force <= 0:
            return np.zeros(shape + (n,)), np.zeros(shape)
        if tilt_stiffness > 0:
            if law.kind != "linear":
                raise ValueError("the tilting holder is implemented for the linear pad law only")
            pts, p = self.contact.tilted_contact_points(force, k_pad, tilt_stiffness)
        else:
            pts, p = self.contact.contact_points(force, k_pad, law)
        scale = 1.0 if rpm is None else self.speed_scale(rpm)
        if alpha != 1.0:
            p = P_REF * (p / P_REF) ** alpha
        flat = self.line_of_point[pts] * n + self.contact.idx[pts]
        if rings:
            ring = np.minimum((self.contact.r[pts] / self.pad_radius * R).astype(np.int64), R - 1)
            flat = flat + ring * (self.n_lines * n)
        E = scale * np.bincount(flat, weights=p * self.vdt[pts], minlength=R * self.n_lines * n)
        E = E.reshape(shape + (n,))
        return E, E @ self.node_area_flat

    def exposure(self, force: float, k_pad: float, rpm: float | None = None,
                 law: PadLaw = LINEAR) -> np.ndarray:
        """Exposure map E = sum p v dt [N/mm] (flat, n_pix): removal at constant K is K * E."""
        if force <= 0:
            return np.zeros(self.panel.n_pix)
        pts, p = self.contact.contact_points(force, k_pad, law)
        scale = 1.0 if rpm is None else self.speed_scale(rpm)
        return scale * np.bincount(self.contact.idx[pts], weights=p * self.vdt[pts],
                                   minlength=self.panel.n_pix)

    def removal(self, force: float, k_pad: float, K: float, rpm: float | None = None) -> np.ndarray:
        """Removal depth map at constant effectiveness K [mm] (flat, n_pix)."""
        return K * self.exposure(force, k_pad, rpm)

    def volume(self, removal_flat: np.ndarray) -> float:
        """Removed volume [mm^3]."""
        return float(removal_flat @ self.node_area_flat)


def ring_effectiveness(K0: float, lam: float, u: np.ndarray, x_start: np.ndarray, fluct: float,
                       wear: "WearLaw") -> np.ndarray:
    """Line-averaged effectiveness of every ring of the pad face during a pass.

    Ring r has used up x_r of abrasive (removed volume scaled to the whole pad,
    see :func:`simulate_truth`) and its effectiveness is K0 * fluct * phi(x_r);
    ``u`` (rings, n_lines) is each ring's removed volume per unit effectiveness in
    each line, already scaled the same way. Single-stage law: exact closed form;
    otherwise RK4 with 8 sub-steps per line. Returns (rings, n_lines).
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    x = np.asarray(x_start, dtype=float).copy()
    K_start = K0 * fluct * wear.phi(lam, x)
    if lam == 0:
        return np.repeat(K_start[:, None], u.shape[1], axis=1)
    if wear.kind == "single":
        U_end = np.cumsum(u, axis=1)
        U_start = U_end - u
        dW = (np.log1p(lam * K_start[:, None] * U_end) - np.log1p(lam * K_start[:, None] * U_start)) / lam
        return np.divide(dW, u, out=np.repeat(K_start[:, None], u.shape[1], axis=1), where=u > 0)
    out = np.empty_like(u)

    def rate(w):
        return K0 * fluct * wear.phi(lam, w)

    for i in range(u.shape[1]):
        x0, h = x.copy(), u[:, i] / 8.0
        for _ in range(8):
            k1 = rate(x)
            k2 = rate(x + 0.5 * h * k1)
            k3 = rate(x + 0.5 * h * k2)
            k4 = rate(x + h * k3)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        out[:, i] = np.where(u[:, i] > 0, (x - x0) / np.where(u[:, i] > 0, u[:, i], 1.0), rate(x))
    return out


def line_effectiveness(K_start: float, lam: float, u_lines: np.ndarray, wear: "WearLaw" = None,
                       K0: float | None = None, W_start: float = 0.0, fluct: float = 1.0) -> np.ndarray:
    """Line-averaged effectiveness during a pass in which K wears continuously.

    Single-stage law: exact closed form, K(U) = K_start / (1 + lam K_start U).
    Two-stage law: dW/dU = K(W) integrated with RK4 (8 sub-steps per line).
    Each line's value is (volume removed by the line) / u_line, so the removal
    map's volume equals the volume that drives the wear.
    """
    u = np.asarray(u_lines, dtype=float)
    if lam == 0:
        return np.full(u.size, K_start)
    if wear is None or wear.kind == "single":
        U_end = np.cumsum(u)
        U_start = U_end - u
        dW = (np.log1p(lam * K_start * U_end) - np.log1p(lam * K_start * U_start)) / lam
        return np.divide(dW, u, out=np.full(u.size, K_start), where=u > 0)
    W = W_start
    out = np.empty(u.size)

    def rate(w):
        return K0 * fluct * wear.phi(lam, w)

    for i, ul in enumerate(u):
        W0, h = W, ul / 8.0
        for _ in range(8):
            k1 = rate(W)
            k2 = rate(W + 0.5 * h * k1)
            k3 = rate(W + 0.5 * h * k2)
            k4 = rate(W + h * k3)
            W += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        out[i] = (W - W0) / ul if ul > 0 else rate(W)
    return out


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


@dataclass(frozen=True)
class WearLaw:
    """K / K0 as a function of cumulative removed volume W (without the fluctuation).

    single:    exp(-lam W)
    two_stage: (1 - f) exp(-lam W) + f exp(-r lam W)   (fast break-in of a fraction f)
    """

    kind: str = "single"
    fast_fraction: float = 0.25
    fast_ratio: float = 6.0

    def phi(self, lam, W):
        if self.kind == "single":
            return np.exp(-lam * W)
        f, r = self.fast_fraction, self.fast_ratio
        return (1 - f) * np.exp(-lam * W) + f * np.exp(-r * lam * W)


@dataclass(frozen=True)
class World:
    """How the hidden process differs from the tracker's model ("matched" = not at all)."""

    name: str = "matched"
    pad_law: PadLaw = LINEAR
    force_gain_sigma: float = 0.0       # log-sd of the run's force-calibration error
    force_ripple: float = 0.0           # relative s.d. of the force of each raster line (force-control ripple)
    ripple_corr: float = 0.5            # correlation of the ripple between consecutive lines
    wear: WearLaw = WearLaw()
    wear_rings: int = 1                 # > 1: abrasive wears ring by ring with its own local work
    preston_exponent: float = 1.0       # removal ~ p^alpha
    loading_max: float = 0.0            # abrasive loading (clogging): largest fractional loss of cutting
    loading_volume: float = 50.0        # removed volume [mm^3] over which loading builds up (e-folding)
    loading_clean: float = 0.5          # fraction of the loading shed between passes
    tilt_stiffness: float = 0.0         # > 0: pad holder tilts with this rotational stiffness [N mm/rad]


MATCHED = World()


@dataclass
class HiddenTruth:
    k_pad: float
    K0: float
    lam: float
    force_gain: float = 1.0             # actual force = gain * commanded force


@dataclass
class TruthTrajectory:
    """What actually happened (hidden from the estimators)."""

    truth: HiddenTruth
    K: np.ndarray                  # effectiveness at the start of each scanned pass
    removal: np.ndarray            # (n_passes, n_pix) true removal maps [mm]
    volume: np.ndarray             # removed volume per scanned pass [mm^3]
    crossing_pass: int | None      # first pass (1-based) whose starting K < tau*K0
    K_extended: np.ndarray = field(repr=False, default=None)  # K beyond the scanned passes
    oracle: np.ndarray = field(repr=False, default=None)      # (n_passes, n_pix) oracle predictions
    world: World = MATCHED


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
                   max_extend: int = 400, world: World = MATCHED,
                   ripple_rng: np.random.Generator | None = None) -> TruthTrajectory:
    """Run the hidden process for the scheduled passes.

    The pad face is split into ``world.wear_rings`` equal-width rings (1 = one
    uniform abrasive). Ring r has removed a volume V_r so far; its abrasive has
    been used as much as the whole pad would have after removing
    x_r = (A_pad / A_r) V_r, so its effectiveness is K0 * phi(x_r) * exp(eps),
    with phi the wear law and eps the accumulated log fluctuation. With one ring
    this is the uniform law K = K0 * phi(W) * exp(eps). Rings that do more work
    per unit area (the pad centre, for stiff pads) dull faster. The reported K
    of a pass is the work-weighted mean of the rings' starting effectiveness
    for that pass's contact pattern; the threshold crossing uses it. The
    trajectory continues past the last scanned pass (repeating the schedule's
    force cycle) until the threshold crossing.

    The oracle prediction for pass n uses the true physics and the true state
    at the start of pass n-1, but not the fluctuation drawn after it, nor the
    force ripple of pass n.

    Abrasive loading (``world.loading_max`` > 0): swarf clogs the abrasive while it
    cuts. A loading state L multiplies the effectiveness of each raster line by
    (1 - L); along a pass L grows towards ``loading_max`` with the removed volume
    (e-folding volume ``loading_volume``), and between passes a fraction
    ``loading_clean`` of it is shed. Loading is temporary, so the reported K and
    the abrasive-change threshold refer to the wear alone.

    Tilting holder (``world.tilt_stiffness`` > 0): the pad tilts against a rotational
    spring until the pressure moment balances it, which moves pressure towards an
    overhanging edge (linear pad law only).

    Force ripple (``world.force_ripple`` > 0): raster line l of a pass runs at
    force F (1 + e_l), with e_l an AR(1) sequence along the pass. Its effect on
    the line exposures is applied to first order, using the derivative of the
    exact exposures with respect to the force (relative error ~ e_l^2).
    """
    cache: dict[tuple[float, float], tuple] = {}
    h = 0.05
    R = max(int(world.wear_rings), 1)
    edges = np.linspace(0.0, 1.0, R + 1)
    scale = 1.0 / (edges[1:] ** 2 - edges[:-1] ** 2)     # A_pad / A_ring

    def lines(a: PassAction):
        key = (a.force, a.rpm)
        if key not in cache:
            F = truth.force_gain * a.force
            E, u = model.line_exposures(F, truth.k_pad, a.rpm, world.pad_law, world.preston_exponent, R,
                                        world.tilt_stiffness)
            if world.force_ripple > 0:
                E2, u2 = model.line_exposures(F * (1 + h), truth.k_pad, a.rpm, world.pad_law,
                                              world.preston_exponent, R, world.tilt_stiffness)
                cache[key] = (E, u, (E2 - E) / h, (u2 - u) / h)
            else:
                cache[key] = (E, u, None, None)
        return cache[key]

    def ripple() -> np.ndarray | None:
        if world.force_ripple <= 0:
            return None
        r, e = world.ripple_corr, np.empty(model.n_lines)
        xi = ripple_rng.standard_normal(model.n_lines)
        e[0] = xi[0]
        for i in range(1, model.n_lines):
            e[i] = r * e[i - 1] + np.sqrt(1 - r * r) * xi[i]
        return world.force_ripple * e

    def k_start(a: PassAction, V: np.ndarray, fluct: float) -> float:
        """Work-weighted mean starting effectiveness of the rings for action a."""
        _, u, _, _ = lines(a)
        Kr = truth.K0 * world.wear.phi(truth.lam, scale * V) * fluct
        wr = u.sum(axis=1)
        return float(Kr @ wr / wr.sum()) if wr.sum() > 0 else float(Kr.mean())

    def run_pass(a: PassAction, V: np.ndarray, fluct: float, e: np.ndarray | None = None, L: float = 0.0):
        E, u, dE, du = lines(a)
        if e is not None:
            E = E + e[None, :, None] * dE
            u = u + e[None, :] * du
        K_rl = ring_effectiveness(truth.K0, truth.lam, scale[:, None] * u, scale * V, fluct, world.wear)
        if world.loading_max > 0:     # loading builds up line by line with the removed volume
            Lm, VL = world.loading_max, world.loading_volume
            for li in range(K_rl.shape[1]):
                K_rl[:, li] *= 1.0 - L
                L = Lm - (Lm - L) * np.exp(-float(K_rl[:, li] @ u[:, li]) / VL)
        dV = (K_rl * u).sum(axis=1)
        return np.einsum("rl,rln->n", K_rl, E), dV, L

    n = len(schedule)
    V, eps, L = np.zeros(R), 0.0, 0.0
    Ks, maps, vols, K_ext, oracle = [], [], [], [], []
    crossing = None
    eps_prev = None
    for i in range(n + max_extend):
        a = action_at(schedule, i)
        K = k_start(a, V, np.exp(eps))
        if crossing is None and K < threshold * truth.K0:
            crossing = i + 1
        K_ext.append(K)
        if i >= n and crossing is not None:
            break
        removal, dV, L_end = run_pass(a, V, np.exp(eps), ripple(), L)
        if i < n:
            Ks.append(K)
            maps.append(removal)
            vols.append(float(dV.sum()))
            # oracle: the true state at the start of the previous pass, without the fluctuation
            # drawn after it, and without this pass's force ripple (pass 1: the true initial state);
            # it knows the abrasive's loading at the start of the pass
            oracle.append(run_pass(a, V, np.exp(eps if i == 0 else eps_prev), None, L)[0])
        eps_prev = eps
        V = V + dV
        L = (1.0 - world.loading_clean) * L_end     # loading partly shed between passes
        eps += wear_sigma * rng.standard_normal()
    return TruthTrajectory(truth, np.array(Ks), np.array(maps), np.array(vols), crossing,
                           np.array(K_ext), np.array(oracle), world)
