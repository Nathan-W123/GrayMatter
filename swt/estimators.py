"""Estimators that predict the next pass's removal map from commanded actions and scans.

A. :class:`Nominal`        – prior-mean parameters, open loop, never updated.
B. :class:`CalibrateOnce`  – least-squares fit of (k_pad, K) to the first scan, then frozen.
C. :class:`ParticleFilter` – joint Bayesian tracker of [log k_pad, log K0, log lambda, log K].

All three see the same information: the commanded action of every pass and
the noisy post-pass scans. None of them sees the hidden parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .process import PassAction, Priors, ProcessModel, action_at, wear_step
from .surrogate import ExposureTable


class _ExactPredictor:
    """Shared helper: exact forward model with a per-(force, rpm, k) exposure cache."""

    def __init__(self, model: ProcessModel, cache: dict | None = None):
        self.model = model
        self._cache: dict[tuple[float, float, float], np.ndarray] = {} if cache is None else cache

    def exposure(self, action: PassAction, k_pad: float) -> np.ndarray:
        key = (action.force, action.rpm, k_pad)
        if key not in self._cache:
            self._cache[key] = self.model.exposure(action.force, k_pad, action.rpm)
        return self._cache[key]


# --------------------------------------------------------------------------- A


class Nominal(_ExactPredictor):
    """Open-loop simulator with prior-mean parameters (k, K0, lambda).

    Abrasive wear is propagated with the nominal lambda and the model's own
    predicted removed volume, so it is a complete but uncalibrated world model.
    """

    name = "A: nominal"

    def __init__(self, model: ProcessModel, priors: Priors, cache: dict | None = None):
        super().__init__(model, cache)
        nom = priors.log_mean()
        self.k_pad = nom["k_pad"]
        self.K0 = nom["K0"]
        self.lam = nom["lam"]
        self.K = self.K0

    def predict(self, action: PassAction) -> np.ndarray:
        return self.K * self.exposure(action, self.k_pad)

    def update(self, scan: np.ndarray, action: PassAction) -> None:
        pass  # never updated

    def advance(self, action: PassAction) -> None:
        dV = self.model.volume(self.predict(action))
        self.K = float(wear_step(self.K, self.lam, dV))

    def crossing_pass(self, schedule: list[PassAction], threshold: float, horizon: int = 400) -> int:
        """Open-loop threshold crossing pass under the nominal parameters."""
        K = self.K0
        for i in range(horizon):
            if K < threshold * self.K0:
                return i + 1
            a = action_at(schedule, i)
            K = float(wear_step(K, self.lam, K * self.model.volume(self.exposure(a, self.k_pad))))
        return horizon + 1


# --------------------------------------------------------------------------- B


def least_squares_fit(table: ExposureTable, scan: np.ndarray, action: PassAction,
                      k_bounds: tuple[float, float]) -> tuple[float, float]:
    """Least-squares (k_pad, K) for one scan: K profiled out, 1-D search in log k."""
    ym, yy = table.scan_terms(scan)
    s = action.rpm / table.model.sander.spindle_rpm
    lk = np.linspace(np.log(k_bounds[0]), np.log(k_bounds[1]), 4001)
    _, sse = table.profile_sse(action.force / np.exp(lk), ym, yy)
    i = int(np.argmin(sse))
    lo, hi = lk[max(i - 1, 0)], lk[min(i + 1, lk.size - 1)]

    def obj(x):
        return float(table.profile_sse(np.array([action.force / np.exp(x)]), ym, yy)[1][0])

    if hi > lo:
        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8})
        best = res.x if res.fun <= sse[i] else lk[i]
    else:
        best = lk[i]
    a, _ = table.profile_sse(np.array([action.force / np.exp(best)]), ym, yy)
    return float(np.exp(best)), float(a[0] / (action.force * s))


class CalibrateOnce(_ExactPredictor):
    """Fit (k_pad, K) to the first scan by least squares, then freeze both."""

    name = "B: calibrate-once"

    def __init__(self, model: ProcessModel, priors: Priors, table: ExposureTable, cache: dict | None = None):
        super().__init__(model)
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
                self._nominal_cache[key] = self.model.exposure(action.force, self.k_pad, action.rpm)
            return self.K * self._nominal_cache[key]
        return self.K * self.exposure(action, self.k_pad)

    def update(self, scan: np.ndarray, action: PassAction) -> None:
        if not self.calibrated:
            self.k_pad, self.K = least_squares_fit(self.table, scan, action, self.k_bounds)
            self.calibrated = True

    def advance(self, action: PassAction) -> None:
        pass  # frozen: no wear model


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


@dataclass
class PFConfig:
    n_particles: int = 10000
    ess_fraction: float = 0.75       # tempering target ESS / N
    shrinkage: float = 0.98          # Liu-West kernel shrinkage a; jitter scale h = sqrt(1 - a^2)
    wear_sigma: float = 0.01         # assumed process noise on log K per pass
    sigma_floor_um: float = 0.05     # likelihood noise floor (surrogate error allowance)
    threshold: float = 0.5           # abrasive-change threshold, fraction of K0
    max_stages: int = 200


class ParticleFilter:
    """Particle filter over x = [log k_pad, log K0, log lambda, log K_current].

    * Prediction between passes: K_current follows the wear model using the
      particle's own predicted removed volume, plus N(0, wear_sigma^2) noise on
      log K. k_pad, K0 and lambda are static.
    * Update after each scan: weights from the Gaussian likelihood of the whole
      removal scan. A single scan has thousands of pixels, so the likelihood is
      far sharper than the prior; it is applied in tempered stages (progressive
      correction): at each stage the exponent increment is chosen so the
      effective sample size stays at ``ess_fraction * N``, then particles are
      systematically resampled and roughened with a Liu-West shrinkage kernel
      (jitter that keeps the cloud's mean and covariance, so it fights
      collapse without inflating the posterior).
    * K0 (fresh-abrasive effectiveness) is carried as a static component
      because the abrasive-change threshold is defined relative to it.
    """

    name = "C: joint tracker"
    labels = ("k_pad", "K0", "lam", "K")

    def __init__(self, table: ExposureTable, priors: Priors, cfg: PFConfig, rng: np.random.Generator,
                 rollout_rng: np.random.Generator | None = None):
        self.table = table
        self.priors = priors
        self.cfg = cfg
        self.rng = rng
        # separate stream for crossing-pass rollouts, so asking for a prediction
        # never changes the filter's own random sequence
        self.rollout_rng = rng.spawn(1)[0] if rollout_rng is None else rollout_rng
        n = cfg.n_particles
        p = priors.sample(rng, n)
        self.X = np.column_stack([np.log(p["k_pad"]), np.log(p["K0"]), np.log(p["lam"]), np.log(p["K0"])])
        self.w = np.full(n, 1.0 / n)          # normalised particle weights
        self.n_resample = 0
        self.n_updates = 0
        # pass at which each particle's own history first fell below the
        # threshold (0 = not yet); copied with its particle on resampling
        self.crossed_at = np.zeros(n, dtype=np.int64)
        self.lk_bounds = (np.log(priors.k_low), np.log(priors.k_high))
        self.rpm_ref = table.model.sander.spindle_rpm
        self.last_stages = 0

    # ------------------------------------------------------------------ helpers
    def _scale(self, action: PassAction, X: np.ndarray | None = None) -> np.ndarray:
        X = self.X if X is None else X
        return np.exp(X[:, 3]) * action.force * (action.rpm / self.rpm_ref)

    def _eta(self, action: PassAction, X: np.ndarray | None = None) -> np.ndarray:
        X = self.X if X is None else X
        return action.force / np.exp(X[:, 0])

    def _reflect(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.lk_bounds
        x = X[:, 0]
        x = np.where(x < lo, 2 * lo - x, x)
        x = np.where(x > hi, 2 * hi - x, x)
        X[:, 0] = np.clip(x, lo, hi)
        return X

    def _roughen(self) -> None:
        a = self.cfg.shrinkage
        h = np.sqrt(1.0 - a * a)
        m = self.X.mean(axis=0)
        V = np.cov(self.X, rowvar=False)
        w, U = np.linalg.eigh(V)
        L = U * np.sqrt(np.clip(w, 0.0, None))
        eps = self.rng.standard_normal(self.X.shape)
        self.X = a * self.X + (1 - a) * m + h * eps @ L.T
        self.X = self._reflect(self.X)

    # ------------------------------------------------------------------ filter steps
    def predict(self, action: PassAction) -> np.ndarray:
        """Posterior predictive mean removal map for the next pass (full grid)."""
        return self.table.mixture_full(self._eta(action), self._scale(action), self.w)

    def update(self, scan: np.ndarray, action: PassAction, noise_um: float) -> None:
        sigma = max(noise_um, self.cfg.sigma_floor_um) * 1e-3
        ym, yy = self.table.scan_terms(scan)
        n = self.X.shape[0]
        target = self.cfg.ess_fraction * n

        def loglik(X):
            return -self.table.sse(self._eta(action, X), self._scale(action, X), ym, yy) / (2 * sigma**2)

        ll = loglik(self.X)
        logw0 = np.log(self.w)
        phi, stages = 0.0, 0
        while phi < 1.0 and stages < self.cfg.max_stages:
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
            w /= w.sum()
            phi = min(1.0, phi + delta)
            stages += 1
            if phi < 1.0 or ess(np.log(w)) < target:
                # intermediate stage (or degenerate final stage): resample and roughen
                idx = systematic_resample(w, self.rng)
                self.X = self.X[idx]
                self.crossed_at = self.crossed_at[idx]
                self.w = np.full(n, 1.0 / n)
                self._roughen()
                self.n_resample += 1
                if phi < 1.0:
                    ll = loglik(self.X)
                logw0 = np.log(self.w)
            else:
                self.w = w
                logw0 = np.log(w)
        self.last_stages = stages
        self.n_updates += 1
        below = self.X[:, 3] < np.log(self.cfg.threshold) + self.X[:, 1]
        self.crossed_at[(self.crossed_at == 0) & below] = self.n_updates

    def advance(self, action: PassAction) -> None:
        """Propagate K_current through the pass just completed (wear + noise)."""
        dV = self._scale(action) * self.table.volume(self._eta(action))
        lam = np.exp(self.X[:, 2])
        noise = self.cfg.wear_sigma * self.rng.standard_normal(self.X.shape[0])
        self.X[:, 3] = self.X[:, 3] - lam * dV + noise

    # ------------------------------------------------------------------ reporting
    def summary(self, q: tuple[float, float] = (0.05, 0.95)) -> dict:
        out = {}
        w = self.w
        for j, name in enumerate(self.labels):
            v = np.exp(self.X[:, j])
            lo, med, hi = weighted_quantiles(v, w, [q[0], 0.5, q[1]])
            out[name] = {"mean": float(w @ v), "median": float(med), "lo": float(lo), "hi": float(hi)}
        cov = np.cov(self.X[:, [0, 3]], rowvar=False, aweights=w)
        out["corr_logk_logK"] = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
        out["stages"] = int(self.last_stages)
        out["ess"] = float(1.0 / (w @ w))
        return out

    def predict_crossing(self, passes_done: int, schedule: list[PassAction], threshold: float,
                         horizon: int = 400, q: tuple[float, float] = (0.05, 0.95)) -> dict:
        """Distribution of the first pass with K < threshold * K0, given scans so far.

        Each particle is rolled forward through the planned schedule with its
        own (k, K0, lambda) and random future wear fluctuations. A particle
        whose history already went below the threshold reports the pass at
        which that happened (tracked through resampling).
        """
        k = np.exp(self.X[:, 0])
        K0 = np.exp(self.X[:, 1])
        lam = np.exp(self.X[:, 2])
        K = np.exp(self.X[:, 3])
        n = K.size
        pred = np.full(n, passes_done + horizon + 1, dtype=float)
        done = self.crossed_at > 0
        pred[done] = self.crossed_at[done]
        late = (~done) & (K < threshold * K0)   # only if threshold differs from cfg.threshold
        pred[late] = passes_done
        done |= late
        for j in range(horizon):
            if done.all():
                break
            a = action_at(schedule, passes_done - 1 + j)   # pass whose wear we apply
            dV = K * a.force * (a.rpm / self.rpm_ref) * self.table.volume(a.force / k)
            K = K * np.exp(-lam * dV + self.cfg.wear_sigma * self.rollout_rng.standard_normal(n))
            new = (~done) & (K < threshold * K0)
            pred[new] = passes_done + j + 1
            done |= new
        lo, med, hi = weighted_quantiles(pred, self.w, [q[0], 0.5, q[1]])
        mass = float(self.w @ ((pred >= lo) & (pred <= hi)))   # >= q[1]-q[0] for integer passes
        return {"median": float(med), "lo": float(lo), "hi": float(hi), "mean": float(self.w @ pred),
                "band_mass": mass}
