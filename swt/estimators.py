"""Estimators that predict the next pass's removal map from commanded actions and scans.

A. :class:`Nominal`        – prior-mean parameters, open loop, never updated.
B. :class:`CalibrateOnce`  – least-squares fit of (k_pad, K) to the first scan, then frozen.
C. :class:`ParticleFilter` – joint Bayesian tracker of log k_pad and the log K and log lambda paths.
D. :class:`RefitEachPass`  – least-squares (k_pad, K) from every scan, wear rate from a
                             log-linear fit of those K against removed volume (an engineering
                             baseline without a probabilistic model).

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
                      k_bounds: tuple[float, float]) -> tuple[float, float]:
    """Least-squares (k_pad, K) for one scan, ignoring wear: K profiled out, 1-D search in log k."""
    st = table.scan_stats(scan)
    s = action.rpm / table.model.sander.spindle_rpm
    lk = np.linspace(np.log(k_bounds[0]), np.log(k_bounds[1]), 4001)
    _, sse = table.profile_sse(action.force / np.exp(lk), st)
    i = int(np.argmin(sse))
    lo, hi = lk[max(i - 1, 0)], lk[min(i + 1, lk.size - 1)]

    def obj(x):
        return float(table.profile_sse(np.array([action.force / np.exp(x)]), st)[1][0])

    if hi > lo:
        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8})
        best = res.x if res.fun <= sse[i] else lk[i]
    else:
        best = lk[i]
    a, _ = table.profile_sse(np.array([action.force / np.exp(best)]), st)
    return float(np.exp(best)), float(a[0] / (action.force * s))


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
    """Refit (k_pad, K) to every scan by least squares; extrapolate K with a fitted wear rate.

    After each scan: (k_i, K_i) = least-squares fit (as in B) and the volume the
    pass removed, V_i. The wear rate is the slope of a least-squares line
    through log K_i against the cumulative volume at mid-pass (the prior median
    until two scans exist), so K_next = K_last * exp(-lambda (V_last + V_next) / 2).
    Predictions use the latest k and the surrogate table (no within-pass wear,
    like the fit). It gives point predictions only.
    """

    name = "D: refit each pass"

    def __init__(self, priors: Priors, table: ExposureTable):
        self.table = table
        self.k_bounds = (priors.k_low, priors.k_high)
        nom = priors.log_mean()
        self.lam_prior = nom["lam"]
        self.k_pad, self.K, self.lam = nom["k_pad"], nom["K0"], nom["lam"]
        self.rpm_ref = table.model.sander.spindle_rpm
        self.W_mid: list[float] = []      # cumulative removed volume at mid-pass
        self.logK: list[float] = []
        self.W = 0.0
        self.V_last = 0.0

    def _volume(self, K: float, action: PassAction) -> float:
        a = K * action.force * action.rpm / self.rpm_ref
        return float(a * self.table.volume(np.array([action.force / self.k_pad]))[0])

    def _K_next(self, action: PassAction) -> float:
        if not self.logK:
            return self.K
        K = self.K
        for _ in range(3):    # V_next depends on K_next: a few fixed-point steps
            K = self.K * np.exp(-self.lam * 0.5 * (self.V_last + self._volume(K, action)))
        return float(K)

    def predict(self, action: PassAction) -> np.ndarray:
        a = self._K_next(action) * action.force * action.rpm / self.rpm_ref
        return self.table.map_obs(action.force / self.k_pad, a, 0.0)

    def update(self, scan: np.ndarray, action: PassAction) -> None:
        self.k_pad, self.K = least_squares_fit(self.table, scan, action, self.k_bounds)
        self.V_last = self._volume(self.K, action)
        self.W_mid.append(self.W + 0.5 * self.V_last)
        self.W += self.V_last
        self.logK.append(float(np.log(self.K)))
        if len(self.logK) >= 2:
            slope = np.polyfit(self.W_mid, self.logK, 1)[0]
            self.lam = float(max(-slope, 0.0))
        else:
            self.lam = self.lam_prior

    def advance(self, action: PassAction) -> None:
        pass   # the extrapolation happens in predict()

    def K0(self) -> float:
        """Intercept of the wear fit (K at zero removed volume)."""
        if len(self.logK) >= 2:
            return float(np.exp(np.polyval(np.polyfit(self.W_mid, self.logK, 1), 0.0)))
        return float(np.exp(self.logK[0] + self.lam * self.W_mid[0])) if self.logK else self.K

    def crossing_pass(self, passes_done: int, schedule: list[PassAction], threshold: float,
                      horizon: int = 400) -> int:
        """Point prediction of the first pass whose starting K < threshold * K0."""
        thr = threshold * self.K0()
        # K at the start of the next pass, then pass by pass with the fitted rate
        K = self.K * np.exp(-self.lam * 0.5 * self.V_last)
        for i in range(horizon):
            if K < thr:
                return passes_done + i + 1
            K = K * np.exp(-self.lam * self._volume(K, action_at(schedule, passes_done + i)))
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
    wear_sigma: float = 0.01         # assumed process noise on log K per pass
    adaptive_wear_noise: bool = True # inflate that noise if the filter's own K forecasts miss by more
    wear_noise_prior_passes: float = 4.0   # weight of wear_sigma in that estimate, in passes
    rate_drift: float = 0.05         # random-walk s.d. of log lambda per pass (0 = constant wear rate)
    sigma_floor_um: float = 0.05     # likelihood noise floor (surrogate error allowance)
    threshold: float = 0.5           # abrasive-change threshold, fraction of K0
    max_stages: int = 200
    mcmc_sweeps: int = 2             # Metropolis-Hastings sweeps after each update
    robust: bool = True              # outlier rejection + discrepancy-inflated noise
    outlier_z: float = 5.0
    inflation_block: int = 4         # scan points per block side for the correlation check
    inflation_threshold: float = 1.2 # inflate only when the residual variance exceeds this x noise^2
    profile_offsets: bool = True     # random offset per scan profile, variance estimated online
    redo_ratio: float = 1.5          # redo an update if the noise estimate moves by more than this factor


class ParticleFilter:
    """Sequential Monte Carlo tracker with tempering and resample-move rejuvenation.

    State of each particle: log k_pad (static), and the paths of log K and
    log lambda at the start of every pass so far (K0 = K at pass 1). Model:
    the removal of pass j follows the surrogate with within-pass wear at rate
    lambda_j; between passes

        log K_{j+1}      = log K_j - log(1 + lambda_j a_j U_j) + N(0, sw_j^2)
        log lambda_{j+1} = log lambda_j + N(0, rate_drift^2)

    (``rate_drift`` = 0: a constant wear rate). The drift lets the wear rate
    follow the recent decay of K when the true wear law is not the assumed
    exponential one. sw_j is ``wear_sigma``, or with ``adaptive_wear_noise``
    inflated by the filter's own one-step forecast errors (see
    :meth:`wear_noise`), so the predictive spread grows when K decays
    differently from the model. Scans are Gaussian with the instrument noise.

    Update after each scan: the scan likelihood is applied in tempered stages
    whose exponent increments keep the effective sample size at
    ``ess_fraction * N``; after each stage the particles are systematically
    resampled and rejuvenated with Metropolis-Hastings moves that leave the
    exact tempered path posterior invariant: a joint random walk on log k_pad
    and a common shift of the whole log lambda path, local random walks on
    log lambda at the last two passes, and on log K at pass 1 (= K0) and at the
    current pass. Every past scan is kept as O(1) sufficient statistics, so
    the full path posterior can be evaluated and the static stiffness does not
    degenerate over time.

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
        self.sw = np.full(n_passes_max, cfg.wear_sigma)   # process noise of each K transition
        self.forecast_checks: list[float] = []          # squared one-step K forecast errors minus their
        self._forecast: tuple[float, float] | None = None   # predicted variance other than process noise
        self.noise_est: tuple[float, float] | None = None    # (sigma^2, rho) from the last scan
        self.lk_bounds = (np.log(priors.k_low), np.log(priors.k_high))
        self.rpm_ref = table.model.sander.spindle_rpm
        self.obs_shape = obs_shape
        self.crossed_at = np.zeros(n, dtype=np.int64)
        self.last_stages = 0
        self.last_sigma_um = None
        self.last_profile_sd_um = 0.0
        self.last_outliers = 0
        self.last_redone = False
        self.rw_scale = {"theta": 1.0, "K0": 1.0, "Kj": 1.0, "lam_j": 1.0, "lam_prev": 1.0}

    # ------------------------------------------------------------------ model pieces
    def _scale(self, action: PassAction) -> float:
        return action.rpm / self.rpm_ref

    def _coeffs(self, lk, llam, lK, F, s):
        """eta = F / k, scale a = K F s and wear exponent z = lambda a of a pass."""
        eta = F / np.exp(lk)
        a = np.exp(lK) * (F * s)
        return eta, a, np.exp(llam) * a

    def _loglik(self, p: int, lk, llam, lK) -> np.ndarray:
        """Untempered log-likelihood of scan ``p``."""
        eta, a, z = self._coeffs(lk, llam, lK, self.F[p], self.s[p])
        return -self.table.sse(eta, a, z, self.stats[p]) / (2.0 * self.sigma2[p])

    def _transition(self, p, lk, llam, lK_j, lK_next) -> np.ndarray:
        """log N(log K_{p+1}; log K_p - log(1 + z_p U_p), sw_p^2) (constant dropped)."""
        eta, _, z = self._coeffs(lk, llam, lK_j, self.F[p], self.s[p])
        mu = lK_j - np.log1p(z * self.table.volume(eta))
        return -0.5 * ((lK_next - mu) / self.sw[p]) ** 2

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

    def _mh_sweep(self, j: int, phi: float) -> None:
        """One sweep of MH moves leaving the tempered path posterior invariant:
        prior x drift x transitions x scans 0..j-1 x scan j^phi."""
        n = self.lk.size
        rng = self.rng
        tempers = np.ones(j + 1)
        tempers[j] = phi
        drift = self.cfg.rate_drift > 0

        # (1) joint random walk on log k and a common shift of the log lambda path
        X = np.column_stack([self.lk, self.lpath[:, j]])
        L = np.linalg.cholesky(self._wcov(X) * (2.38**2 / 2) * self.rw_scale["theta"] ** 2)
        step = rng.standard_normal((n, 2)) @ L.T
        lpk = self._logprior_k(self.lk + step[:, 0])
        okp = np.isfinite(lpk)
        lkp = np.where(okp, self.lk + step[:, 0], self.lk)
        shift = step[:, 1]
        LLp = np.column_stack([self._loglik(i, lkp, self.lpath[:, i] + shift, self.path[:, i]) for i in range(j + 1)])
        cur = self._logprior_lam0(self.lpath[:, 0]) + self.LL[:, :j + 1] @ tempers
        new = self._logprior_lam0(self.lpath[:, 0] + shift) + LLp @ tempers
        if j:
            TRp = self._transition(np.arange(j)[None, :], lkp[:, None], self.lpath[:, :j] + shift[:, None],
                                   self.path[:, :j], self.path[:, 1:j + 1])
            cur = cur + self.TR[:, :j].sum(axis=1)
            new = new + TRp.sum(axis=1)
        acc = okp & self._mh("theta", new - cur)
        self.lk = np.where(acc, lkp, self.lk)
        self.lpath[acc, :j + 1] += shift[acc, None]
        self.LL[acc, :j + 1] = LLp[acc]
        if j:
            self.TR[acc, :j] = TRp[acc]

        # (2) local moves on log lambda at the previous and the current pass
        if drift and j >= 1:
            i = j - 1
            prop = self.lpath[:, i] + self._rw_sd(self.lpath[:, i], "lam_prev") * rng.standard_normal(n)
            ll_new = self._loglik(i, self.lk, prop, self.path[:, i])
            tr_new = self._transition(i, self.lk, prop, self.path[:, i], self.path[:, i + 1])
            cur = self.LL[:, i] + self.TR[:, i] + self._lam_prior(i, self.lpath[:, i]) \
                + self._drift(self.lpath[:, i], self.lpath[:, j])
            new = ll_new + tr_new + self._lam_prior(i, prop) + self._drift(prop, self.lpath[:, j])
            acc = self._mh("lam_prev", new - cur)
            self.lpath[acc, i] = prop[acc]
            self.LL[acc, i] = ll_new[acc]
            self.TR[acc, i] = tr_new[acc]

            prop = self.lpath[:, j] + self._rw_sd(self.lpath[:, j], "lam_j") * rng.standard_normal(n)
            ll_new = self._loglik(j, self.lk, prop, self.path[:, j])
            cur = phi * self.LL[:, j] + self._lam_prior(j, self.lpath[:, j])
            new = phi * ll_new + self._lam_prior(j, prop)
            acc = self._mh("lam_j", new - cur)
            self.lpath[acc, j] = prop[acc]
            self.LL[acc, j] = ll_new[acc]

        # (3) random walk on log K at pass 1 (= log K0)
        prop = self.path[:, 0] + self._rw_sd(self.path[:, 0], "K0") * rng.standard_normal(n)
        ll_new = self._loglik(0, self.lk, self.lpath[:, 0], prop)
        cur = self._logprior_K0(self.path[:, 0]) + tempers[0] * self.LL[:, 0]
        new = self._logprior_K0(prop) + tempers[0] * ll_new
        if j >= 1:
            tr_new = self._transition(0, self.lk, self.lpath[:, 0], prop, self.path[:, 1])
            cur = cur + self.TR[:, 0]
            new = new + tr_new
        acc = self._mh("K0", new - cur)
        self.path[acc, 0] = prop[acc]
        self.LL[acc, 0] = ll_new[acc]
        if j >= 1:
            self.TR[acc, 0] = tr_new[acc]

        # (4) random walk on log K at the current pass
        if j >= 1:
            prop = self.path[:, j] + self._rw_sd(self.path[:, j], "Kj") * rng.standard_normal(n)
            ll_new = self._loglik(j, self.lk, self.lpath[:, j], prop)
            tr_new = self._transition(j - 1, self.lk, self.lpath[:, j - 1], self.path[:, j - 1], prop)
            cur = phi * self.LL[:, j] + self.TR[:, j - 1]
            new = phi * ll_new + tr_new
            acc = self._mh("Kj", new - cur)
            self.path[acc, j] = prop[acc]
            self.LL[acc, j] = ll_new[acc]
            self.TR[acc, j - 1] = tr_new[acc]

    def _adapt(self, key: str, rate: float) -> None:
        """Keep acceptance near 0.3 (scale of the random-walk proposals)."""
        self.rw_scale[key] = float(np.clip(self.rw_scale[key] * np.exp(rate - 0.3), 0.05, 3.0))

    def _resample(self) -> None:
        idx = systematic_resample(self.w, self.rng)
        self.lk = self.lk[idx]
        self.path, self.lpath, self.LL, self.TR = self.path[idx], self.lpath[idx], self.LL[idx], self.TR[idx]
        self.crossed_at = self.crossed_at[idx]
        self.w = np.full(self.lk.size, 1.0 / self.lk.size)

    # ------------------------------------------------------------------ filter steps
    def _current(self) -> int:
        return self.n_k - 1

    def _next_pass(self, action: PassAction):
        j = self._current()
        return self._coeffs(self.lk, self.lpath[:, j], self.path[:, j], action.force, self._scale(action))

    def predict(self, action: PassAction) -> np.ndarray:
        """Posterior predictive mean removal map for the next pass (scan points)."""
        eta, a, z = self._next_pass(action)
        return self.table.mixture_obs(eta, self.table.coefficients(a, z), self.w)

    def predict_mean_removal(self, action: PassAction, q=(0.05, 0.5, 0.95)) -> np.ndarray:
        """Predictive quantiles of the next pass's mean removal over the scan points."""
        eta, a, z = self._next_pass(action)
        return weighted_quantiles(self.table.mean_removal(eta, self.table.coefficients(a, z)), self.w, q)

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
            self._mh_sweep(j, 1.0)
        return stages

    def _snapshot(self):
        return (self.lk.copy(), self.path.copy(), self.lpath.copy(), self.w.copy(), self.LL.copy(),
                self.TR.copy(), self.crossed_at.copy(), dict(self.rw_scale))

    def _restore(self, snap) -> None:
        (self.lk, self.path, self.lpath, self.w, self.LL, self.TR, self.crossed_at, rw) = snap
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
        if self._forecast is not None:   # one-step forecast of log K_j vs. its estimate from scan j
            m, v_other = self._forecast
            self.forecast_checks.append(float((self.w @ self.path[:, j] - m) ** 2 - v_other))
            self._forecast = None
        self.last_stages = stages
        self.last_sigma_um = float(np.sqrt(self.sigma2[j]) * 1e3)
        self.last_profile_sd_um = float(np.sqrt(rho * self.sigma2[j]) * 1e3)
        self.last_outliers = int(finite.sum() - valid.sum())
        below = self.path[:, j] < np.log(self.cfg.threshold) + self.path[:, 0]
        self.crossed_at[(self.crossed_at == 0) & below] = j + 1

    def advance(self, action: PassAction) -> None:
        """Propagate K through the pass just completed (within-pass wear + fluctuation)
        and let the wear rate drift."""
        j = self._current()
        n = self.lk.size
        self.sw[j] = self.wear_noise()
        eta, _, z = self._coeffs(self.lk, self.lpath[:, j], self.path[:, j], action.force, self._scale(action))
        mu = self.path[:, j] - np.log1p(z * self.table.volume(eta))
        self.path[:, j + 1] = mu + self.sw[j] * self.rng.standard_normal(n)
        self.TR[:, j] = -0.5 * ((self.path[:, j + 1] - mu) / self.sw[j]) ** 2
        m = float(self.w @ self.path[:, j + 1])
        v = float(self.w @ (self.path[:, j + 1] - m) ** 2)
        self._forecast = (m, max(v - self.sw[j] ** 2, 0.0))   # forecast variance not due to process noise
        self.lpath[:, j + 1] = self.lpath[:, j] + self.cfg.rate_drift * self.rng.standard_normal(n)
        self.n_k += 1

    def wear_noise(self) -> float:
        """Process noise for the next K transition.

        ``wear_sigma``, or (adaptive) the process-noise variance implied by the
        filter's own one-step forecasts of log K: the mean squared forecast error
        (forecast made before each scan, checked against the estimate from that
        scan) minus the part of the forecast variance that is not process noise,
        shrunk towards ``wear_sigma``^2 with a weight of
        ``wear_noise_prior_passes`` passes. Never below ``wear_sigma``; larger when
        K changes from pass to pass more than the model predicts.
        """
        sw2 = self.cfg.wear_sigma**2
        if not self.cfg.adaptive_wear_noise:
            return self.cfg.wear_sigma
        n0 = self.cfg.wear_noise_prior_passes
        v = (n0 * sw2 + float(np.sum(self.forecast_checks))) / (n0 + len(self.forecast_checks))
        return float(np.sqrt(max(sw2, v)))

    # ------------------------------------------------------------------ reporting
    def summary(self, q: tuple[float, float] = (0.05, 0.95)) -> dict:
        """Posterior of k_pad, K0, and of lambda and K at the current pass."""
        j = self._current()
        comps = {"k_pad": self.lk, "K0": self.path[:, 0], "lam": self.lpath[:, j], "K": self.path[:, j]}
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
        out["outliers"] = self.last_outliers
        out["redone"] = self.last_redone
        return out

    def predict_crossing(self, passes_done: int, schedule: list[PassAction], threshold: float,
                         horizon: int = 400, q: tuple[float, float] = (0.05, 0.95)) -> dict:
        """Distribution of the first pass with K < threshold * K0, given scans so far.

        Each particle is rolled forward through the planned schedule with its
        own state and random future wear fluctuations and wear-rate drift. A
        particle whose history already went below the threshold reports the
        pass at which that happened (tracked through resampling).
        """
        k = np.exp(self.lk)
        i0 = passes_done - 1
        lK0 = self.path[:, 0]
        lK = self.path[:, i0].copy()
        llam = self.lpath[:, i0].copy()
        n = lK.size
        pred = np.full(n, passes_done + horizon + 1, dtype=float)
        done = self.crossed_at > 0
        pred[done] = self.crossed_at[done]
        late = (~done) & (lK < np.log(threshold) + lK0)
        pred[late] = passes_done
        done |= late
        lthr = np.log(threshold) + lK0
        sw = self.wear_noise()
        for j in range(horizon):
            if done.all():
                break
            a = action_at(schedule, i0 + j)   # pass whose wear we apply
            s = a.rpm / self.rpm_ref
            z = np.exp(llam + lK) * a.force * s
            lK = lK - np.log1p(z * self.table.volume(a.force / k)) + sw * self.rollout_rng.standard_normal(n)
            llam = llam + self.cfg.rate_drift * self.rollout_rng.standard_normal(n)
            new = (~done) & (lK < lthr)
            pred[new] = passes_done + j + 1
            done |= new
        lo, med, hi = weighted_quantiles(pred, self.w, [q[0], 0.5, q[1]])
        mass = float(self.w @ ((pred >= lo) & (pred <= hi)))   # >= q[1]-q[0] for integer passes
        return {"median": float(med), "lo": float(lo), "hi": float(hi), "mean": float(self.w @ pred),
                "band_mass": mass}
