"""Estimators that predict the next pass's removal map from commanded actions and scans.

A. :class:`Nominal`        – prior-mean parameters, open loop, never updated.
B. :class:`CalibrateOnce`  – least-squares fit of (k_pad, K) to the first scan, then frozen.
C. :class:`ParticleFilter` – joint Bayesian tracker of log k_pad and the log K and log lambda paths.
D. :class:`RefitEachPass`  – joint least squares over all scans: one pooled k_pad, a K per
                             scan, wear rate and force exponent from a log-linear fit of those
                             K with C's priors (the point-estimate counterpart of C, without
                             uncertainty).

All three see the same information: the commanded action of every pass and
the noisy post-pass scans. None of them sees the hidden parameters. All
predictions are removal maps at the scan points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .process import PassAction, Priors, ProcessModel, action_at, line_effectiveness
from .surrogate import ExposureTable, ScanStats


class _ExactPredictor:
    """Shared helper: exact forward model (linear pad, commanded force) with a cache."""

    def __init__(self, model: ProcessModel, obs_index: np.ndarray, cache: dict | None = None):
        self.model = model
        self.obs_index = obs_index
        self._cache: dict = {} if cache is None else cache

    def lines(self, action: PassAction, k_pad: float) -> tuple[np.ndarray, np.ndarray]:
        key = (action.force, action.rpm, k_pad)
        if key not in self._cache:
            E, u = self.model.line_exposures(action.force, k_pad, action.rpm)
            self._cache[key] = (E[:, self.obs_index], u)
        return self._cache[key]


# --------------------------------------------------------------------------- A


class Nominal(_ExactPredictor):
    """Open-loop simulator with prior-mean parameters (k, K0, lambda).

    Wear (during and between passes) is propagated with the nominal lambda,
    so it is a complete but uncalibrated world model.
    """

    name = "A: nominal"

    def __init__(self, model: ProcessModel, priors: Priors, obs_index: np.ndarray, cache: dict | None = None):
        super().__init__(model, obs_index, cache)
        nom = priors.log_mean()
        self.k_pad = nom["k_pad"]
        self.K0 = nom["K0"]
        self.lam = nom["lam"]
        self.K = self.K0

    def predict(self, action: PassAction) -> np.ndarray:
        E, u = self.lines(action, self.k_pad)
        return line_effectiveness(self.K, self.lam, u) @ E

    def update(self, scan: np.ndarray, action: PassAction) -> None:
        pass  # never updated

    def advance(self, action: PassAction) -> None:
        _, u = self.lines(action, self.k_pad)
        self.K = self.K / (1.0 + self.lam * self.K * float(u.sum()))

    def crossing_pass(self, schedule: list[PassAction], threshold: float, horizon: int = 400) -> int:
        """Open-loop threshold crossing pass under the nominal parameters."""
        K = self.K0
        for i in range(horizon):
            if K < threshold * self.K0:
                return i + 1
            _, u = self.lines(action_at(schedule, i), self.k_pad)
            K = K / (1.0 + self.lam * K * float(u.sum()))
        return horizon + 1


# --------------------------------------------------------------------------- B


def least_squares_fit(table: ExposureTable, scan: np.ndarray, action: PassAction,
                      k_bounds: tuple[float, float], lam: float = 0.0, valid: np.ndarray | None = None,
                      iterations: int = 2) -> tuple[float, float]:
    """Least-squares (k_pad, K) for one scan: K profiled out, 1-D search in log k.

    ``lam`` = 0 ignores wear during the pass (B); otherwise the within-pass wear
    at that rate is included, with its exponent z = lam * K F s updated from the
    fitted K for a few fixed-point ``iterations``. Returns K at the start of the pass.
    """
    st = table.scan_stats(scan, valid)
    s = action.rpm / table.model.sander.spindle_rpm
    F = action.force
    lk = np.linspace(np.log(k_bounds[0]), np.log(k_bounds[1]), 4001)
    z = 0.0
    for _ in range(1 + (iterations if lam > 0 else 0)):
        _, sse = table.profile_sse(F / np.exp(lk), st, z)
        i = int(np.argmin(sse))
        lo, hi = lk[max(i - 1, 0)], lk[min(i + 1, lk.size - 1)]

        def obj(x):
            return float(table.profile_sse(np.array([F / np.exp(x)]), st, z)[1][0])

        if hi > lo:
            res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8})
            best = res.x if res.fun <= sse[i] else lk[i]
        else:
            best = lk[i]
        a, _ = table.profile_sse(np.array([F / np.exp(best)]), st, z)
        z = lam * float(a[0])
    return float(np.exp(best)), float(a[0] / (F * s))


class CalibrateOnce(_ExactPredictor):
    """Fit (k_pad, K) to the first scan by least squares, then freeze both (no wear model)."""

    name = "B: calibrate-once"

    def __init__(self, model: ProcessModel, priors: Priors, table: ExposureTable, obs_index: np.ndarray,
                 cache: dict | None = None):
        super().__init__(model, obs_index)
        self._nominal_cache = cache   # shared cache for the pre-calibration (nominal) prediction
        self.table = table
        self.k_bounds = (priors.k_low, priors.k_high)
        nom = priors.log_mean()
        self.k_pad = nom["k_pad"]
        self.K = nom["K0"]
        self.calibrated = False

    def predict(self, action: PassAction) -> np.ndarray:
        if not self.calibrated and self._nominal_cache is not None:
            key = (action.force, action.rpm, self.k_pad)
            if key not in self._nominal_cache:
                E, u = self.model.line_exposures(action.force, self.k_pad, action.rpm)
                self._nominal_cache[key] = (E[:, self.obs_index], u)
            return self.K * self._nominal_cache[key][0].sum(axis=0)
        return self.K * self.lines(action, self.k_pad)[0].sum(axis=0)

    def update(self, scan: np.ndarray, action: PassAction) -> None:
        if not self.calibrated:
            self.k_pad, self.K = least_squares_fit(self.table, scan, action, self.k_bounds)
            self.calibrated = True

    def advance(self, action: PassAction) -> None:
        pass  # frozen: no wear model


# --------------------------------------------------------------------------- D


class RefitEachPass:
    """Joint least-squares tracker: the point-estimate counterpart of C.

    Same model structure as C (linear pad, exponential wear acting during the
    pass, force exponent, outlier rejection), without the probabilistic
    treatment. After every scan:

    1. points more than ``outlier_z`` robust s.d. from a fit of that scan alone
       are rejected;
    2. one stiffness is fitted to *all* scans so far (sum of the per-scan
       least-squares errors, each with its own K profiled out, minimised over
       log k_pad), and each scan's K_i (at the start of its pass) is refitted at
       that stiffness;
    3. the wear rate and the force exponent come from a fit of log K_i against
       the cumulative removed volume at the start of each pass and log F_i,
       penalised by C's priors (a MAP estimate: log-normal prior on lambda,
       linearised at its median; N(1, sd^2) on beta; log K_i noise ``obs_sd``),
       so that one or two scans do not give a wild wear rate;
    4. steps 2-3 are repeated twice so the within-pass wear in the fits uses the
       fitted rate.

    Predictions use the pooled k and the latest K, extrapolated through the
    pass just scanned and rescaled to the next force. K0 for the abrasive
    change is the first scan's K. Point predictions only.
    """

    name = "D: joint least squares"

    def __init__(self, priors: Priors, table: ExposureTable, outlier_z: float = 5.0, force_ref: float = 30.0,
                 force_exponent_sd: float = 0.15, obs_sd: float = 0.02):
        self.table = table
        self.k_bounds = (priors.k_low, priors.k_high)
        nom = priors.log_mean()
        self.lam_prior = nom["lam"]
        self.lam_sd = priors.lam_sigma_log * nom["lam"]     # log-normal prior, linearised at its median
        self.gamma_sd = force_exponent_sd
        self.obs_sd = obs_sd
        self.k_pad, self.K, self.lam = nom["k_pad"], nom["K0"], nom["lam"]
        self.outlier_z = outlier_z
        self.force_ref = force_ref
        self.gamma = 0.0                   # beta - 1
        self.rpm_ref = table.model.sander.spindle_rpm
        self.stats: list = []              # per scan: (ScanStats, action)
        self.Ks: list[float] = []          # fitted K at the start of each scanned pass (effective, at its force)
        self.lk_grid = np.linspace(np.log(self.k_bounds[0]), np.log(self.k_bounds[1]), 2001)

    @property
    def last_action(self) -> PassAction | None:
        return self.stats[-1][1] if self.stats else None

    def _s(self, action: PassAction) -> float:
        return action.rpm / self.rpm_ref

    def _f(self, action: PassAction) -> float:
        """Force factor (F / F_ref)^(beta - 1) of the effective K."""
        return float((action.force / self.force_ref) ** self.gamma)

    def _volume(self, K: float, action: PassAction, k_pad: float | None = None) -> float:
        """Volume removed by a pass that starts at effective K (wear during the pass)."""
        k = self.k_pad if k_pad is None else k_pad
        U = action.force * self._s(action) * float(self.table.volume(np.array([action.force / k]))[0])
        return float(np.log1p(self.lam * K * U) / self.lam) if self.lam > 0 else K * U

    def _K_next(self, action: PassAction | None = None) -> float:
        """Effective K at the start of the next pass (at that pass's force if given)."""
        if not self.stats:
            return self.K
        last = self.last_action
        K = self.K * np.exp(-self.lam * self._volume(self.K, last))
        return float(K if action is None else K / self._f(last) * self._f(action))

    def predict(self, action: PassAction) -> np.ndarray:
        a = self._K_next(action) * action.force * self._s(action)
        return self.table.map_obs(action.force / self.k_pad, a, self.lam * a)

    def _fit_k(self) -> None:
        """Pooled stiffness and per-scan K for the current wear rate."""
        tab, lk = self.table, self.lk_grid
        z = [self.lam * K * act.force * self._s(act) for K, (_, act) in zip(self.Ks, self.stats)]

        def total(lks):
            return sum(tab.profile_sse(act.force / np.exp(lks), st, zi)[1] for (st, act), zi in zip(self.stats, z))

        sse = total(lk)
        i = int(np.argmin(sse))
        lo, hi = lk[max(i - 1, 0)], lk[min(i + 1, lk.size - 1)]
        best = lk[i]
        if hi > lo:
            res = minimize_scalar(lambda x: float(total(np.array([x]))[0]), bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-8})
            if res.fun <= sse[i]:
                best = res.x
        self.k_pad = float(np.exp(best))
        self.Ks = [float(tab.profile_sse(np.array([act.force / self.k_pad]), st, zi)[0][0] / (act.force * self._s(act)))
                   for (st, act), zi in zip(self.stats, z)]

    def _fit_wear(self) -> None:
        """Wear rate and force exponent: MAP fit of log K_i = c - lam W_i + (beta - 1) log(F_i / F_ref)
        with flat c and C's priors on lam and beta (one scan gives the prior medians)."""
        W, Ws = 0.0, []
        for K, (_, act) in zip(self.Ks, self.stats):
            Ws.append(W)
            W += self._volume(K, act)
        logF = np.array([np.log(act.force / self.force_ref) for _, act in self.stats])
        X = np.column_stack([np.ones(len(Ws)), -np.array(Ws), logF]) / self.obs_sd
        y = np.log(self.Ks) / self.obs_sd
        P = np.array([[0.0, 1.0 / self.lam_sd, 0.0], [0.0, 0.0, 1.0 / self.gamma_sd]])
        yp = np.array([self.lam_prior / self.lam_sd, 0.0])
        coef = np.linalg.lstsq(np.vstack([X, P]), np.concatenate([y, yp]), rcond=None)[0]
        self.lam = float(max(coef[1], 0.0))
        self.gamma = float(np.clip(coef[2], -0.5, 0.5))

    def update(self, scan: np.ndarray, action: PassAction) -> None:
        finite = np.isfinite(scan)
        k1, K1 = least_squares_fit(self.table, scan, action, self.k_bounds, self.lam, finite)
        a1 = K1 * action.force * self._s(action)
        r = scan - self.table.map_obs(action.force / k1, a1, self.lam * a1)
        med = np.nanmedian(r)
        mad = 1.4826 * np.nanmedian(np.abs(r - med))
        valid = finite & (np.abs(r - med) <= self.outlier_z * mad)
        self.stats.append((self.table.scan_stats(scan, valid), action))
        self.Ks.append(K1)
        for _ in range(2):
            self._fit_k()
            self._fit_wear()
        self.K = self.Ks[-1]

    def advance(self, action: PassAction) -> None:
        pass   # the extrapolation happens in predict()

    def K0(self) -> float:
        """Fresh effectiveness at F_ref: the first scan's K."""
        return float(self.Ks[0] / self._f(self.stats[0][1])) if self.Ks else self.K

    def next_ratio(self) -> float:
        """Forecast K / K0 at the start of the next pass (both at F_ref)."""
        if not self.stats:
            return 1.0
        return float(self._K_next() / self._f(self.last_action) / self.K0())

    def crossing_pass(self, passes_done: int, schedule: list[PassAction], threshold: float,
                      horizon: int = 400) -> int:
        """Point prediction of the first pass whose starting K < threshold * K0."""
        thr = threshold * self.K0()
        if not self.stats:
            return passes_done + 1
        K = self._K_next() / self._f(self.last_action)          # at F_ref
        for i in range(horizon):
            if K < thr:
                return passes_done + i + 1
            a = action_at(schedule, passes_done + i)
            K = K * np.exp(-self.lam * self._volume(K * self._f(a), a))
        return passes_done + horizon + 1


# --------------------------------------------------------------------------- C


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: one uniform offset, N evenly spaced pointers."""
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions, side="left")


def ess(logw: np.ndarray) -> float:
    w = np.exp(logw - logw.max())
    return float(w.sum() ** 2 / (w * w).sum())


def weighted_quantiles(x: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    """Quantiles of a weighted sample (inverse of the weighted empirical CDF)."""
    order = np.argsort(x, kind="stable")
    xs, cw = x[order], np.cumsum(w[order])
    cw /= cw[-1]
    idx = np.searchsorted(cw, np.asarray(q, dtype=float), side="left")
    return xs[np.clip(idx, 0, xs.size - 1)]


def block_inflation(resid: np.ndarray, shape: tuple[int, int], block: int) -> float:
    """Variance inflation of block means relative to independent residuals (>= 1).

    1 for independent residuals; larger when the residual field is spatially
    correlated (structured model error), because then neighbouring scan points
    do not carry independent information.
    """
    ny, nx = shape
    r = resid.reshape(ny, nx)
    by, bx = ny // block, nx // block
    if by < 2 or bx < 2:
        return 1.0
    sub = r[:by * block, :bx * block].reshape(by, block, bx, block)
    cnt = np.isfinite(sub).sum(axis=(1, 3))
    means = np.nansum(sub, axis=(1, 3)) / np.maximum(cnt, 1)
    ok = cnt >= 0.75 * block * block
    v = np.nanvar(r)
    if ok.sum() < 4 or v <= 0:
        return 1.0
    return float(max(1.0, np.var(means[ok]) * np.mean(cnt[ok]) / v))


@dataclass
class PFConfig:
    n_particles: int = 2000
    ess_fraction: float = 0.75       # tempering target ESS / N
    wear_noise_range: tuple[float, float] = (0.005, 0.05)   # log-uniform prior of the wear fluctuation s.d.
    transient_range: tuple[float, float] = (0.002, 0.05)    # log-uniform prior of the per-pass gain s.d. (0: off)
    rate_drift: float = 0.05         # random-walk s.d. of log lambda per pass (0 = constant wear rate)
    force_exponent_sd: float = 0.15  # prior s.d. of beta in removal ~ F^beta at fixed contact shape (0 = beta = 1)
    force_ref: float = 30.0          # N: force at which K is defined when beta != 1
    sigma_floor_um: float = 0.05     # likelihood noise floor (surrogate error allowance)
    threshold: float = 0.5           # abrasive-change threshold, fraction of K0
    max_stages: int = 200
    mcmc_sweeps: int = 1             # full-path Metropolis-Hastings sweeps after each update
    robust: bool = True              # outlier rejection + discrepancy-inflated noise
    outlier_z: float = 5.0
    inflation_block: int = 4         # scan points per block side for the correlation check
    inflation_threshold: float = 1.2 # inflate only when the residual variance exceeds this x noise^2
    profile_offsets: bool = True     # random offset per scan profile, variance estimated online
    redo_ratio: float = 1.5          # redo an update if the noise estimate moves by more than this factor


class ParticleFilter:
    """Sequential Monte Carlo tracker with tempering and resample-move rejuvenation.

    State of each particle: log k_pad, a force exponent beta, log sigma_w and
    log sigma_g (static), and the paths of log K, log lambda and a transient
    log gain g at every pass so far (K0 = K at pass 1). Model: the removal of
    pass j follows the surrogate with within-pass wear at rate lambda_j and
    scale a = K exp(g_j) F s (F / F_ref)^(beta - 1). beta = 1 is Preston's law;
    beta lets the tracker learn a different force dependence (prior
    N(1, force_exponent_sd^2) on [0.5, 1.5]). g_j ~ N(0, sigma_g^2) is a gain of
    pass j alone (force ripple, local loading) that does not carry over to
    the next pass, unlike K; sigma_g has a log-uniform prior on
    ``transient_range``. Between passes

        log K_{j+1}      = log K_j - log(1 + lambda_j a_j U_j) + N(0, sigma_w^2)
        log lambda_{j+1} = log lambda_j + N(0, rate_drift^2)

    The wear-fluctuation level sigma_w is unknown, with a log-uniform prior on
    ``wear_noise_range``: it is learned from how much K changes from pass to
    pass beyond the wear law, so the predictive spread is wide while little is
    known and grows if K varies more than expected. The drift lets the wear
    rate follow the recent decay of K when the true wear law is not the
    assumed exponential one (``rate_drift`` = 0: a constant rate). Scans are
    Gaussian with the instrument noise (see robust mode below).

    Update after each scan: the scan likelihood is applied in tempered stages
    whose exponent increments keep the effective sample size at
    ``ess_fraction * N``; after each stage the particles are systematically
    resampled and rejuvenated with Metropolis-Hastings moves that leave the
    exact tempered path posterior invariant: a joint random walk on log k_pad
    and a common shift of the whole log lambda path, a random walk on
    log sigma_w, and local random walks on log lambda at the last two passes
    and on log K at pass 1 (= K0) and the current pass. After the last stage,
    ``mcmc_sweeps`` full-path sweeps also move log K and log lambda at every
    earlier pass, so the stored paths do not collapse onto a few ancestors.
    Every past scan is kept as O(1) sufficient statistics, so every move uses
    the full path posterior.

    Robust mode (``cfg.robust``), for scans the model does not fully explain:
    points more than ``outlier_z`` robust standard deviations from the
    tracker's own prediction for the pass are rejected before the update
    (innovation gating; on the first pass, where the prior prediction is
    poor, the test uses the fitted map instead and the update is redone once
    without them). The noise level of the likelihood is re-estimated from the
    residuals at the posterior mean, inflated when the residuals are spatially
    correlated (block means vary more than independent noise allows), and a
    random offset per scan profile is added to the noise model when the
    profile means vary significantly more than the noise explains (variance
    estimated from the residuals; folded exactly into the sufficient
    statistics). If these estimates differ from the levels used by more than
    ``redo_ratio``, the update is redone once with the new levels, which then
    carry over to the next scan.
    """

    name = "C: joint tracker"

    def __init__(self, table: ExposureTable, priors: Priors, cfg: PFConfig, rng: np.random.Generator,
                 rollout_rng: np.random.Generator | None = None, n_passes_max: int = 64,
                 obs_shape: tuple[int, int] | None = None):
        self.table = table
        self.priors = priors
        self.cfg = cfg
        self.rng = rng
        self.rollout_rng = rng.spawn(1)[0] if rollout_rng is None else rollout_rng
        n = cfg.n_particles
        p = priors.sample(rng, n)
        self.lk = np.log(p["k_pad"])
        self.beta = np.clip(1.0 + cfg.force_exponent_sd * rng.standard_normal(n), 0.5, 1.5)
        lo, hi = (np.log(float(v)) for v in cfg.wear_noise_range)
        self.lsw_bounds = (lo, hi)
        self.lsw = rng.uniform(lo, hi, n) if hi > lo else np.full(n, lo)
        tlo, thi = (float(v) for v in cfg.transient_range)
        self.transient = thi > 0
        self.lsg_bounds = (np.log(tlo), np.log(thi)) if self.transient else (0.0, 0.0)
        lo_g, hi_g = self.lsg_bounds
        self.lsg = rng.uniform(lo_g, hi_g, n) if hi_g > lo_g else np.full(n, lo_g)
        self.gpath = np.zeros((n, n_passes_max))        # transient log gain of each pass
        self.GP = np.zeros((n, n_passes_max))           # its log prior density
        if self.transient:
            xi = rng.standard_normal(n)
            self.gpath[:, 0] = np.exp(self.lsg) * xi
            self.GP[:, 0] = -0.5 * xi**2 - self.lsg
        self.path = np.zeros((n, n_passes_max))         # log K at the start of each pass
        self.path[:, 0] = np.log(p["K0"])
        self.lpath = np.zeros((n, n_passes_max))        # log lambda during each pass
        self.lpath[:, 0] = np.log(p["lam"])
        self.n_k = 1                                    # number of passes with K defined
        self.w = np.full(n, 1.0 / n)
        self.LL = np.zeros((n, n_passes_max))           # untempered log-likelihood of each scan
        self.TR = np.zeros((n, n_passes_max))           # log transition density of K into pass j+1
        self.stats: list[ScanStats | None] = [None] * n_passes_max
        self.F = np.zeros(n_passes_max)
        self.s = np.zeros(n_passes_max)
        self.sigma2 = np.ones(n_passes_max)
        self.noise_est: tuple[float, float] | None = None    # (sigma^2, rho) from the last scan
        self.lk_bounds = (np.log(priors.k_low), np.log(priors.k_high))
        self.rpm_ref = table.model.sander.spindle_rpm
        self.obs_shape = obs_shape
        self.last_stages = 0
        self.last_sigma_um = None
        self.last_profile_sd_um = 0.0
        self.last_outliers = 0
        self.last_redone = False
        self.rw_scale = {"theta": 1.0, "sw": 1.0, "K": 1.0, "lam": 1.0, "sg": 1.0, "ridge": 1.0}

    # ------------------------------------------------------------------ model pieces
    def _scale(self, action: PassAction) -> float:
        return action.rpm / self.rpm_ref

    def _coeffs(self, lk, llam, lK, F, s, beta=None, g=0.0):
        """eta = F / k, scale a = K exp(g) F s (F/F_ref)^(beta-1) and wear exponent z = lambda a of a pass."""
        beta = self.beta if beta is None else beta
        eta = F / np.exp(lk)
        a = np.exp(lK + g + (beta - 1.0) * np.log(F / self.cfg.force_ref)) * (F * s)
        return eta, a, np.exp(llam) * a

    def _loglik(self, p: int, lk, llam, lK, beta=None, g=None) -> np.ndarray:
        """Untempered log-likelihood of scan ``p`` (transient gain of that pass unless given)."""
        g = self.gpath[:, p] if g is None else g
        eta, a, z = self._coeffs(lk, llam, lK, self.F[p], self.s[p], beta, g)
        return -self.table.sse(eta, a, z, self.stats[p]) / (2.0 * self.sigma2[p])

    def _mu(self, p, lk, llam, lK_p, beta=None, g=None):
        """Expected log K at the start of pass p+1 given the state of pass p (the wear of
        pass p uses its actual removal, transient gain included). Pass ``g`` when p is an array."""
        g = self.gpath[:, p] if g is None else g
        eta, _, z = self._coeffs(lk, llam, lK_p, self.F[p], self.s[p], beta, g)
        return lK_p - np.log1p(z * self.table.volume(eta))

    def _transition(self, p, lk, llam, lK_p, lK_next, lsw, beta=None, g=None) -> np.ndarray:
        """log N(log K_{p+1}; mu_p, sigma_w^2), up to a constant."""
        return -0.5 * ((lK_next - self._mu(p, lk, llam, lK_p, beta, g)) / np.exp(lsw)) ** 2 - lsw

    def _logprior_beta(self, beta) -> np.ndarray:
        sd = self.cfg.force_exponent_sd
        if sd <= 0:
            return np.where(beta == 1.0, 0.0, -np.inf)
        return np.where((beta >= 0.5) & (beta <= 1.5), -0.5 * ((beta - 1.0) / sd) ** 2, -np.inf)

    def _logprior_k(self, lk) -> np.ndarray:
        lo, hi = self.lk_bounds
        return np.where((lk >= lo) & (lk <= hi), 0.0, -np.inf)

    def _logprior_lam0(self, llam0) -> np.ndarray:
        pr = self.priors
        return -0.5 * ((llam0 - np.log(pr.lam_median)) / pr.lam_sigma_log) ** 2

    def _logprior_K0(self, lK0) -> np.ndarray:
        pr = self.priors
        return -0.5 * ((lK0 - np.log(pr.K0_median)) / pr.K0_sigma_log) ** 2

    def _drift(self, l_prev, l_next) -> np.ndarray:
        return -0.5 * ((l_next - l_prev) / self.cfg.rate_drift) ** 2

    def _lam_prior(self, i: int, l_i) -> np.ndarray:
        """Prior term of log lambda_i given lambda_{i-1} (or the prior of lambda_0)."""
        return self._logprior_lam0(l_i) if i == 0 else self._drift(self.lpath[:, i - 1], l_i)

    # ------------------------------------------------------------------ MCMC rejuvenation
    def _wcov(self, X: np.ndarray) -> np.ndarray:
        w = self.w
        m = w @ X
        D = X - m
        return (D * w[:, None]).T @ D + 1e-12 * np.eye(X.shape[1])

    def _rw_sd(self, x: np.ndarray, key: str) -> float:
        return float(np.sqrt(self._wcov(x[:, None])[0, 0]) * 2.38 * self.rw_scale[key])

    def _mh(self, key: str, log_ratio: np.ndarray) -> np.ndarray:
        acc = np.log(self.rng.random(log_ratio.size)) < log_ratio
        self._adapt(key, acc.mean())
        return acc

    def _move_theta(self, j: int, tempers: np.ndarray) -> None:
        """Joint random walk on log k, beta and a common shift of the whole log lambda path."""
        n = self.lk.size
        free_beta = self.cfg.force_exponent_sd > 0
        X = np.column_stack([self.lk, self.lpath[:, j]] + ([self.beta] if free_beta else []))
        d = X.shape[1]
        L = np.linalg.cholesky(self._wcov(X) * (2.38**2 / d) * self.rw_scale["theta"] ** 2)
        step = self.rng.standard_normal((n, d)) @ L.T
        betap = self.beta + step[:, 2] if free_beta else self.beta
        okp = np.isfinite(self._logprior_k(self.lk + step[:, 0]) + self._logprior_beta(betap))
        lkp = np.where(okp, self.lk + step[:, 0], self.lk)
        betap = np.where(okp, betap, self.beta)
        shift = step[:, 1]
        LLp = np.column_stack([self._loglik(i, lkp, self.lpath[:, i] + shift, self.path[:, i], betap)
                               for i in range(j + 1)])
        cur = self._logprior_lam0(self.lpath[:, 0]) + self._logprior_beta(self.beta) + self.LL[:, :j + 1] @ tempers
        new = self._logprior_lam0(self.lpath[:, 0] + shift) + self._logprior_beta(betap) + LLp @ tempers
        if j:
            TRp = self._transition(np.arange(j)[None, :], lkp[:, None], self.lpath[:, :j] + shift[:, None],
                                   self.path[:, :j], self.path[:, 1:j + 1], self.lsw[:, None], betap[:, None],
                                   self.gpath[:, :j])
            cur = cur + self.TR[:, :j].sum(axis=1)
            new = new + TRp.sum(axis=1)
        acc = okp & self._mh("theta", new - cur)
        self.lk = np.where(acc, lkp, self.lk)
        self.beta = np.where(acc, betap, self.beta)
        self.lpath[acc, :j + 1] += shift[acc, None]
        self.LL[acc, :j + 1] = LLp[acc]
        if j:
            self.TR[acc, :j] = TRp[acc]

    def _move_sw(self, j: int) -> None:
        """Random walk on log sigma_w; only the K transitions depend on it."""
        lo, hi = self.lsw_bounds
        if j == 0 or hi <= lo:
            return
        prop = self.lsw + self._rw_sd(self.lsw, "sw") * self.rng.standard_normal(self.lsw.size)
        ok = (prop >= lo) & (prop <= hi)
        r2 = -2.0 * np.exp(2 * self.lsw)[:, None] * (self.TR[:, :j] + self.lsw[:, None])   # squared innovations
        tr_new = -0.5 * r2 / np.exp(2 * prop)[:, None] - prop[:, None]
        acc = ok & self._mh("sw", np.where(ok, tr_new.sum(axis=1) - self.TR[:, :j].sum(axis=1), -np.inf))
        self.lsw = np.where(acc, prop, self.lsw)
        self.TR[acc, :j] = tr_new[acc]

    def _move_sg(self, j: int) -> None:
        """Random walk on log sigma_g; only the priors of the transient gains depend on it."""
        lo, hi = self.lsg_bounds
        if not self.transient or hi <= lo:
            return
        prop = self.lsg + self._rw_sd(self.lsg, "sg") * self.rng.standard_normal(self.lsg.size)
        ok = (prop >= lo) & (prop <= hi)
        gp_new = -0.5 * (self.gpath[:, :j + 1] / np.exp(prop)[:, None]) ** 2 - prop[:, None]
        acc = ok & self._mh("sg", np.where(ok, gp_new.sum(axis=1) - self.GP[:, :j + 1].sum(axis=1), -np.inf))
        self.lsg = np.where(acc, prop, self.lsg)
        self.GP[acc, :j + 1] = gp_new[acc]

    def _move_ridge(self, i: int, j: int) -> None:
        """Move log K_i up and the transient gain g_i down by the same amount: the removal of
        pass i (and so its scan likelihood) is unchanged, only the split between the
        persistent effectiveness and the gain of that pass changes."""
        if not self.transient:
            return
        d = self._rw_sd(self.gpath[:, i], "ridge") * self.rng.standard_normal(self.lk.size)
        lK_new = self.path[:, i] + d
        g_new = self.gpath[:, i] - d
        gp_new = -0.5 * (g_new / np.exp(self.lsg)) ** 2 - self.lsg
        cur, new = self.GP[:, i].copy(), gp_new
        if i == 0:
            cur = cur + self._logprior_K0(self.path[:, 0])
            new = new + self._logprior_K0(lK_new)
        else:
            tr_in = self._transition(i - 1, self.lk, self.lpath[:, i - 1], self.path[:, i - 1], lK_new, self.lsw)
            cur, new = cur + self.TR[:, i - 1], new + tr_in
        if i < j:
            tr_out = self._transition(i, self.lk, self.lpath[:, i], lK_new, self.path[:, i + 1], self.lsw, g=g_new)
            cur, new = cur + self.TR[:, i], new + tr_out
        acc = self._mh("ridge", new - cur)
        self.path[acc, i] = lK_new[acc]
        self.gpath[acc, i] = g_new[acc]
        self.GP[acc, i] = gp_new[acc]
        if i > 0:
            self.TR[acc, i - 1] = tr_in[acc]
        if i < j:
            self.TR[acc, i] = tr_out[acc]

    def _move_K(self, i: int, j: int, temper: float) -> None:
        """Random walk on log K at pass i (pass 1: K0), given its neighbours on the path."""
        prop = self.path[:, i] + self._rw_sd(self.path[:, i], "K") * self.rng.standard_normal(self.lk.size)
        ll_new = self._loglik(i, self.lk, self.lpath[:, i], prop)
        cur = temper * self.LL[:, i]
        new = temper * ll_new
        if i == 0:
            cur = cur + self._logprior_K0(self.path[:, 0])
            new = new + self._logprior_K0(prop)
        else:
            tr_in = self._transition(i - 1, self.lk, self.lpath[:, i - 1], self.path[:, i - 1], prop, self.lsw)
            cur, new = cur + self.TR[:, i - 1], new + tr_in
        if i < j:
            tr_out = self._transition(i, self.lk, self.lpath[:, i], prop, self.path[:, i + 1], self.lsw)
            cur, new = cur + self.TR[:, i], new + tr_out
        acc = self._mh("K", new - cur)
        self.path[acc, i] = prop[acc]
        self.LL[acc, i] = ll_new[acc]
        if i > 0:
            self.TR[acc, i - 1] = tr_in[acc]
        if i < j:
            self.TR[acc, i] = tr_out[acc]

    def _move_lam(self, i: int, j: int, temper: float) -> None:
        """Random walk on log lambda at pass i, given its neighbours on the path."""
        if self.cfg.rate_drift <= 0:
            return
        prop = self.lpath[:, i] + self._rw_sd(self.lpath[:, i], "lam") * self.rng.standard_normal(self.lk.size)
        ll_new = self._loglik(i, self.lk, prop, self.path[:, i])
        cur = temper * self.LL[:, i] + self._lam_prior(i, self.lpath[:, i])
        new = temper * ll_new + self._lam_prior(i, prop)
        if i < j:
            tr_new = self._transition(i, self.lk, prop, self.path[:, i], self.path[:, i + 1], self.lsw)
            cur = cur + self.TR[:, i] + self._drift(self.lpath[:, i], self.lpath[:, i + 1])
            new = new + tr_new + self._drift(prop, self.lpath[:, i + 1])
        acc = self._mh("lam", new - cur)
        self.lpath[acc, i] = prop[acc]
        self.LL[acc, i] = ll_new[acc]
        if i < j:
            self.TR[acc, i] = tr_new[acc]

    def _mh_sweep(self, j: int, phi: float, full: bool = False) -> None:
        """One sweep of MH moves leaving the tempered path posterior invariant:
        prior x drift x transitions x scans 0..j-1 x scan j^phi. ``full`` also moves
        log K and log lambda at every earlier pass."""
        tempers = np.ones(j + 1)
        tempers[j] = phi
        self._move_theta(j, tempers)
        self._move_sw(j)
        self._move_sg(j)
        lam_idx = range(j + 1) if full else range(max(j - 1, 0), j + 1)
        k_idx = range(j + 1) if full else sorted({0, j})
        for i in lam_idx:
            self._move_lam(i, j, tempers[i])
        for i in k_idx:
            self._move_K(i, j, tempers[i])
            self._move_ridge(i, j)

    def _adapt(self, key: str, rate: float) -> None:
        """Keep acceptance near 0.3 (scale of the random-walk proposals)."""
        self.rw_scale[key] = float(np.clip(self.rw_scale[key] * np.exp(rate - 0.3), 0.05, 3.0))

    def _resample(self) -> None:
        idx = systematic_resample(self.w, self.rng)
        self.lk, self.lsw, self.beta, self.lsg = self.lk[idx], self.lsw[idx], self.beta[idx], self.lsg[idx]
        self.gpath, self.GP = self.gpath[idx], self.GP[idx]
        self.path, self.lpath, self.LL, self.TR = self.path[idx], self.lpath[idx], self.LL[idx], self.TR[idx]
        self.w = np.full(self.lk.size, 1.0 / self.lk.size)

    # ------------------------------------------------------------------ filter steps
    def _current(self) -> int:
        return self.n_k - 1

    def _next_pass(self, action: PassAction):
        j = self._current()
        return self._coeffs(self.lk, self.lpath[:, j], self.path[:, j], action.force, self._scale(action),
                            g=self.gpath[:, j])

    def predict(self, action: PassAction) -> np.ndarray:
        """Posterior predictive mean removal map for the next pass (scan points)."""
        eta, a, z = self._next_pass(action)
        return self.table.mixture_obs(eta, self.table.coefficients(a, z), self.w)

    def predict_mean_removal(self, action: PassAction, q=(0.05, 0.5, 0.95)) -> np.ndarray:
        """Predictive quantiles of the next pass's mean removal over the scan points."""
        eta, a, z = self._next_pass(action)
        return weighted_quantiles(self.table.mean_removal(eta, self.table.coefficients(a, z)), self.w, q)

    def predict_region_removal(self, action: PassAction, region_tables: np.ndarray, q=(0.05, 0.5, 0.95)) -> np.ndarray:
        """Predictive quantiles of the next pass's mean removal over regions of the scan.
        ``region_tables[r]`` (N_TAB, n_eta) is the mean of the surrogate tables over the scan
        points of region r. Returns (n_regions, len(q))."""
        eta, a, z = self._next_pass(action)
        b = self.table.coefficients(a, z)
        e, t = self.table.locate(eta)
        out = []
        for M in region_tables:
            m = (1 - t)[:, None] * M[:, e].T + t[:, None] * M[:, e + 1].T
            out.append(weighted_quantiles(np.einsum("nc,nc->n", b, m), self.w, q))
        return np.array(out)

    def _tempered_update(self, j: int) -> int:
        n = self.lk.size
        target = self.cfg.ess_fraction * n
        self.LL[:, j] = self._loglik(j, self.lk, self.lpath[:, j], self.path[:, j])
        logw0 = np.log(self.w)
        phi, stages = 0.0, 0
        while phi < 1.0 and stages < self.cfg.max_stages:
            ll = self.LL[:, j]
            rem = 1.0 - phi
            if ess(logw0 + rem * ll) >= target:
                delta = rem
            else:
                lo, hi = 0.0, rem
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if ess(logw0 + mid * ll) >= target:
                        lo = mid
                    else:
                        hi = mid
                delta = max(lo, 1e-12)
            logw = logw0 + delta * ll
            w = np.exp(logw - logw.max())
            self.w = w / w.sum()
            phi = min(1.0, phi + delta)
            stages += 1
            if phi < 1.0 or 1.0 / (self.w @ self.w) < target:
                self._resample()
                self._mh_sweep(j, phi)
            logw0 = np.log(self.w)
        for _ in range(self.cfg.mcmc_sweeps):
            self._mh_sweep(j, 1.0, full=True)
        return stages

    def _snapshot(self):
        return (self.lk.copy(), self.lsw.copy(), self.beta.copy(), self.lsg.copy(), self.path.copy(),
                self.lpath.copy(), self.gpath.copy(), self.GP.copy(), self.w.copy(), self.LL.copy(),
                self.TR.copy(), dict(self.rw_scale))

    def _restore(self, snap) -> None:
        (self.lk, self.lsw, self.beta, self.lsg, self.path, self.lpath, self.gpath, self.GP, self.w, self.LL,
         self.TR, rw) = snap
        self.rw_scale = dict(rw)

    def _outliers(self, resid: np.ndarray, base2: float) -> np.ndarray:
        """Points whose residual is more than ``outlier_z`` robust s.d. from the median."""
        f = np.isfinite(resid)
        med = np.median(resid[f])
        mad = 1.4826 * np.median(np.abs(resid[f] - med))
        return f & (np.abs(resid - med) > self.cfg.outlier_z * max(mad, np.sqrt(base2)))

    def _noise_estimate(self, resid: np.ndarray, valid: np.ndarray, base2: float) -> tuple[float, float]:
        """Noise model (sigma^2, rho = tau^2 / sigma^2) from the residuals at the posterior mean.

        tau^2 is the variance of a random offset per scan profile (column),
        estimated from the excess variance of the column means and kept only
        if it is significant; sigma^2 is the within-profile variance, inflated
        if the remaining residuals are spatially correlated.
        """
        r = np.where(valid, resid, np.nan)
        rho = 0.0
        if self.obs_shape is not None and self.cfg.profile_offsets:
            g = r.reshape(self.obs_shape)
            n_c = np.isfinite(g).sum(axis=0)
            ok = n_c >= 4
            m_c = np.nanmean(g[:, ok], axis=0)
            within = g[:, ok] - m_c
            s2w = float(np.nansum(within**2) / max(n_c[ok].sum() - ok.sum(), 1))
            ratio = float(np.mean(n_c[ok] * m_c**2) / s2w)        # ~ 1 without offsets
            if ratio > 1.0 + 3.0 * np.sqrt(2.0 / ok.sum()):
                tau2 = max(0.0, float(np.mean(m_c**2 - s2w / n_c[ok])))
                full = np.full_like(g, np.nan)
                full[:, ok] = within
                r = full.ravel()
            else:
                tau2 = 0.0
        else:
            tau2 = 0.0
        ms = float(np.nanmean(r**2))
        infl = block_inflation(r, self.obs_shape, self.cfg.inflation_block) if self.obs_shape else 1.0
        cand = infl * ms
        sigma2 = cand if cand > self.cfg.inflation_threshold * base2 else base2
        if tau2 > 0:
            rho = tau2 / sigma2
        return sigma2, rho

    def update(self, scan: np.ndarray, action: PassAction, noise_um: float) -> None:
        j = self._current()
        base2 = (max(noise_um, self.cfg.sigma_floor_um) * 1e-3) ** 2
        robust = self.cfg.robust
        finite = np.isfinite(scan)
        valid = finite.copy()
        if robust and j >= 1:
            # innovation gating: test the scan against the tracker's own prediction for this pass
            valid &= ~self._outliers(scan - self.predict(action), base2)
        if robust and self.noise_est is not None:
            sigma2, rho = max(base2, self.noise_est[0]), self.noise_est[1]
        else:
            sigma2, rho = base2, 0.0
        self.F[j], self.s[j] = action.force, self._scale(action)
        snap = self._snapshot() if robust else None
        self.last_redone = False
        n_cols = self.obs_shape[1] if self.obs_shape is not None else 1
        n_bar = finite.sum() / n_cols

        def col_var(s2, rh):          # variance of a profile mean under the noise model
            return s2 * (1.0 / n_bar + rh)

        for attempt in range(2):
            self.stats[j] = self.table.scan_stats(scan, valid, self.obs_shape, rho)
            self.sigma2[j] = sigma2
            stages = self._tempered_update(j)
            if not robust:
                break
            resid = scan - self.predict(action)
            # first pass: the prior prediction is too poor for gating, so test the fitted map
            new_valid = (finite & ~self._outliers(resid, base2)) if j == 0 else valid
            est = self._noise_estimate(resid, new_valid, base2)
            self.noise_est = est
            lim = np.log(self.cfg.redo_ratio)
            changed = ((new_valid != valid).any() or abs(np.log(est[0] / sigma2)) > lim
                       or abs(np.log(col_var(*est) / col_var(sigma2, rho))) > lim)
            if attempt == 0 and changed:
                valid, (sigma2, rho) = new_valid, est
                self._restore(snap)
                self.last_redone = True
                continue
            break
        self.last_stages = stages
        self.last_sigma_um = float(np.sqrt(self.sigma2[j]) * 1e3)
        self.last_profile_sd_um = float(np.sqrt(rho * self.sigma2[j]) * 1e3)
        self.last_outliers = int(finite.sum() - valid.sum())

    def advance(self, action: PassAction) -> None:
        """Propagate K through the pass just completed (within-pass wear + fluctuation)
        and let the wear rate drift."""
        j = self._current()
        n = self.lk.size
        self.F[j], self.s[j] = action.force, self._scale(action)
        mu = self._mu(j, self.lk, self.lpath[:, j], self.path[:, j])
        xi = self.rng.standard_normal(n)
        self.path[:, j + 1] = mu + np.exp(self.lsw) * xi
        self.TR[:, j] = -0.5 * xi**2 - self.lsw
        self.lpath[:, j + 1] = self.lpath[:, j] + self.cfg.rate_drift * self.rng.standard_normal(n)
        if self.transient:     # the gain of the coming pass, drawn from its prior
            xi = self.rng.standard_normal(n)
            self.gpath[:, j + 1] = np.exp(self.lsg) * xi
            self.GP[:, j + 1] = -0.5 * xi**2 - self.lsg
        self.n_k += 1

    # ------------------------------------------------------------------ reporting
    def wear_noise(self) -> float:
        """Posterior median of the wear-fluctuation s.d. sigma_w (log K per pass)."""
        return float(np.exp(weighted_quantiles(self.lsw, self.w, [0.5])[0]))

    def ancestral_diversity(self, i: int) -> int:
        """Number of distinct log K values left at pass index i (path degeneracy check)."""
        return int(np.unique(self.path[:, i]).size)

    def summary(self, q: tuple[float, float] = (0.05, 0.95)) -> dict:
        """Posterior of k_pad, K0, and of lambda and K at the current pass."""
        j = self._current()
        comps = {"k_pad": self.lk, "K0": self.path[:, 0], "lam": self.lpath[:, j], "K": self.path[:, j],
                 "beta": np.log(self.beta)}
        out = {}
        w = self.w
        for name, x in comps.items():
            v = np.exp(x)
            lo, med, hi = weighted_quantiles(v, w, [q[0], 0.5, q[1]])
            out[name] = {"mean": float(w @ v), "median": float(med), "lo": float(lo), "hi": float(hi)}
        cov = np.cov(np.column_stack([self.lk, self.path[:, j]]), rowvar=False, aweights=w)
        out["corr_logk_logK"] = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
        out["stages"] = int(self.last_stages)
        out["ess"] = float(1.0 / (w @ w))
        out["sigma_um"] = self.last_sigma_um
        out["profile_sd_um"] = self.last_profile_sd_um
        out["wear_noise"] = self.wear_noise()
        out["transient_noise"] = float(np.exp(weighted_quantiles(self.lsg, self.w, [0.5])[0])) if self.transient else 0.0
        out["min_path_diversity"] = min(self.ancestral_diversity(i) for i in range(j + 1))
        out["outliers"] = self.last_outliers
        out["redone"] = self.last_redone
        return out

    def predict_crossing(self, passes_done: int, schedule: list[PassAction], threshold: float,
                         horizon: int = 400, q: tuple[float, float] = (0.05, 0.95)) -> dict:
        """Distribution of the first pass with K < threshold * K0, given scans so far.

        A particle whose K path already went below the threshold reports the pass
        at which it did; the others are rolled forward through the planned
        schedule with their own state, wear-fluctuation level and random future
        wear fluctuations and wear-rate drift.
        """
        i0 = passes_done - 1
        lthr = np.log(threshold) + self.path[:, 0]
        below = self.path[:, :passes_done] < lthr[:, None]
        done = below.any(axis=1)
        n = self.lk.size
        pred = np.full(n, passes_done + horizon + 1, dtype=float)
        pred[done] = np.argmax(below[done], axis=1) + 1
        lK = self.path[:, i0].copy()
        llam = self.lpath[:, i0].copy()
        sw = np.exp(self.lsw)
        for j in range(horizon):
            if done.all():
                break
            a = action_at(schedule, i0 + j)   # pass whose wear we apply
            g = self.gpath[:, i0] if j == 0 else np.exp(self.lsg) * self.rollout_rng.standard_normal(n)
            eta, _, z = self._coeffs(self.lk, llam, lK, a.force, a.rpm / self.rpm_ref, g=g)
            lK = lK - np.log1p(z * self.table.volume(eta)) + sw * self.rollout_rng.standard_normal(n)
            llam = llam + self.cfg.rate_drift * self.rollout_rng.standard_normal(n)
            new = (~done) & (lK < lthr)
            pred[new] = passes_done + j + 1
            done |= new
        lo, med, hi = weighted_quantiles(pred, self.w, [q[0], 0.5, q[1]])
        mass = float(self.w @ ((pred >= lo) & (pred <= hi)))   # >= q[1]-q[0] for integer passes
        return {"median": float(med), "lo": float(lo), "hi": float(hi), "mean": float(self.w @ pred),
                "band_mass": mass}
