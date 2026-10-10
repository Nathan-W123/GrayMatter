"""Tracker C2: the frozen particle filter C plus an online model-discrepancy layer.

C (``swt.estimators.ParticleFilter``) is used unchanged. C2 watches C's
one-pass-ahead errors and learns two discrepancies that C's model family
cannot represent, each with a Kalman filter on log-ratios of observed to
predicted removal:

* **Level** ``m``: a persistent multiplicative error of the next pass's mean
  removal: an exponentially weighted (``level_memory``), precision-weighted
  mean of C's log errors, shrunk towards zero by a positive-part James-Stein
  factor, so that it stays at zero unless the errors are consistently on one
  side. It absorbs slowly varying effects such as abrasive loading, which C
  reads as wear, and a break-in that C's exponential law extrapolates too far.
* **Shape** ``d_b``: a log factor for each of the B blocks of the scan
  (default 4 x 4), constrained to a removal-weighted mean of zero, so that it
  changes the distribution of removal over the part but not its level. Each
  block keeps an exponentially weighted (``shape_memory``), precision-weighted
  mean of its centred log-ratios; these are shrunk towards zero by an
  empirical-Bayes factor whose signal variance is estimated across blocks
  (method of moments), so the layer stays at zero when C's map shape is
  right and grows only when the errors are consistent. It absorbs map-shape
  errors such as uneven abrasive wear, a tilting holder or a foam pad. At the
  scan points the block factors are interpolated bilinearly between block
  centres.

The variance of each observation is C's own predictive variance (per block
or for the mean, from its particles), plus the scan noise averaged over the
points, plus an *unexplained* variance estimated online from the innovations
(exponentially weighted method of moments). The predictive distributions of
C2 are log-normal: the weighted mean and variance of the log of C's particle
predictions, shifted by the discrepancy and widened by its variance.
Pass 1 (a prior prediction for every estimator) is not used for learning.

C2 changes predictions only. A level error of the removal cannot be pinned on
the abrasive from the scans alone (a tilting holder or a foam pad also change
it), so by default (``level_to_K`` False) C2 does not move C's estimate of K
and its abrasive-change decisions are C's; on held-out draws, attributing the
level error to K helped under abrasive loading but hurt under a tilting holder
and in the tracker's own world, about equally.

Nothing in C2 feeds back into C, so C's results are identical with or
without C2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimators import ParticleFilter, weighted_quantiles
from .process import PassAction

UM = 1000.0


@dataclass(frozen=True)
class C2Config:
    level_memory: float = 0.8           # weight of the past in the mean log error of the level
    level_to_K: bool = False            # apply the level discrepancy to K in the abrasive-change probability
    shape_memory: float = 0.8           # weight of the past in each block's mean log-ratio
    memory: float = 0.8                 # weight of the past in the unexplained-variance estimates
    draws: int = 4                      # Monte Carlo draws per particle for the crossing probability
    learn_from_pass: int = 2            # first scanned pass whose error is learned from


class DiscrepancyTracker:
    """C2 = C + level and block-shape discrepancies learned from C's own errors."""

    name = "C2: C + discrepancy layer"

    def __init__(self, pf: ParticleFilter, region_index: list[np.ndarray], region_tables: np.ndarray,
                 obs_shape: tuple[int, int], cfg: C2Config, rng: np.random.Generator,
                 blocks: tuple[int, int] = (4, 4)):
        self.pf = pf
        self.cfg = cfg
        self.rng = rng
        self.region_index = region_index
        self.region_tables = region_tables                  # (B, N_TAB, n_eta) block means of the tables
        self.n_b = np.array([ix.size for ix in region_index], dtype=float)
        B = len(region_index)
        self.m, self.V = 0.0, 0.0
        self.lvl_wy, self.lvl_w = 0.0, 0.0
        self.d, self.P = np.zeros(B), np.zeros(B)
        self.sum_wy, self.sum_w = np.zeros(B), np.zeros(B)     # discounted precision-weighted sums
        self.signal2 = 0.0                                       # estimated variance of the true shape factors
        self.tau2_level, self.tau2_shape = 0.0, 0.0
        self.W = _bilinear_weights(obs_shape, blocks)     # (n_obs, B)
        self.pending: dict | None = None
        self.passes = 0

    # ------------------------------------------------------------------ prediction
    def _particle_predictions(self, action: PassAction):
        """C's per-particle predictions of the next pass: mean removal (n,) and block means (n, B)."""
        pf = self.pf
        eta, a, z = pf._next_pass(action)
        b = pf.table.coefficients(a, z)
        e, t = pf.table.locate(eta)
        lo = self.region_tables[:, :, e]                   # (B, N_TAB, n)
        hi = self.region_tables[:, :, e + 1]
        m = (1 - t)[None, None, :] * lo + t[None, None, :] * hi
        blocks = np.einsum("nc,bcn->nb", b, m)
        return pf.table.mean_removal(eta, b), blocks

    def prepare(self, action: PassAction) -> None:
        """Store C's predictions for the coming pass (before its scan) for learning and intervals."""
        mean_pp, block_pp = self._particle_predictions(action)
        w = self.pf.w
        self.pending = {"mean_pp": mean_pp, "block_pp": block_pp, "w": w.copy(),
                        "mean": float(w @ mean_pp), "block_mean": w @ block_pp}

    def predict_map(self, c_map: np.ndarray) -> np.ndarray:
        """C's predicted map at the scan points corrected by the level and shape discrepancies."""
        return c_map * np.exp(self.m + self.W @ self.d)

    @staticmethod
    def _lognormal_quantiles(values: np.ndarray, w: np.ndarray, shift: float, extra_var: float, q) -> np.ndarray:
        """Quantiles of C's particle predictions times exp(shift + N(0, extra_var)), with C's predictive
        spread summarised by the weighted mean and variance of the log predictions (log-normal)."""
        x = np.log(np.maximum(values, 1e-300))
        mu = float(w @ x)
        var = float(w @ (x - mu) ** 2) + extra_var
        z = np.array([_norm_ppf(p) for p in q])
        return np.exp(mu + shift + z * np.sqrt(var))

    def predict_mean_removal(self, q=(0.05, 0.5, 0.95)) -> np.ndarray:
        """Predictive quantiles of the coming pass's mean removal (call after ``prepare``)."""
        p = self.pending
        return self._lognormal_quantiles(p["mean_pp"], p["w"], self.m, self.V + self.tau2_level, q)

    def predict_region_removal(self, q=(0.05, 0.5, 0.95)) -> np.ndarray:
        """Predictive quantiles of the coming pass's mean removal over each block, (B, len(q))."""
        p = self.pending
        lev = self.V + self.tau2_level
        return np.array([self._lognormal_quantiles(p["block_pp"][:, bi], p["w"], self.m + self.d[bi],
                                                   lev + self.P[bi] + self.tau2_shape, q)
                         for bi in range(self.d.size)])

    # ------------------------------------------------------------------ learning
    def update(self, scan: np.ndarray, noise_um: float) -> None:
        """Learn from the error of C's prediction for the pass just scanned (call after C.update)."""
        self.passes += 1
        p = self.pending
        self.pending = None
        if p is None or self.passes < self.cfg.learn_from_pass:
            return
        cfg = self.cfg
        noise = noise_um / UM
        valid = np.isfinite(scan)
        obs = float(np.mean(scan[valid]))
        if not obs > 0 or not p["mean"] > 0:
            return
        # level: log(observed / predicted) mean removal
        e = np.log(obs / p["mean"])
        var_c = _log_var(p["mean_pp"], p["w"])
        r_lvl = var_c + (noise / np.sqrt(valid.sum()) / obs) ** 2
        innov = e - self.m
        self.tau2_level = cfg.memory * self.tau2_level + (1 - cfg.memory) * max(0.0, innov ** 2 - self.V - r_lvl)
        prec = 1.0 / (r_lvl + self.tau2_level)
        self.lvl_wy = cfg.level_memory * self.lvl_wy + prec * e
        self.lvl_w = cfg.level_memory * self.lvl_w + prec
        ebar, v = self.lvl_wy / self.lvl_w, 1.0 / self.lvl_w
        shrink = max(0.0, 1.0 - v / ebar ** 2) if ebar != 0 else 0.0
        self.m, self.V = shrink * ebar, shrink * v
        # shape: block log-ratios relative to the level error, centred (removal-weighted)
        obs_b = np.array([np.nanmean(scan[ix]) if np.isfinite(scan[ix]).any() else np.nan for ix in self.region_index])
        pred_b = p["block_mean"]
        ok = np.isfinite(obs_b) & (obs_b > 0) & (pred_b > 0)
        if ok.sum() < 2:
            return
        o = np.full(self.d.size, np.nan)
        o[ok] = np.log(obs_b[ok] / pred_b[ok]) - e
        wts = np.where(ok, pred_b * self.n_b, 0.0)
        o[ok] -= np.sum(wts[ok] * o[ok]) / wts[ok].sum()
        shape_pp = np.log(np.maximum(p["block_pp"], 1e-300)) - np.log(np.maximum(p["mean_pp"], 1e-300))[:, None]
        var_cb = p["w"] @ (shape_pp - p["w"] @ shape_pp) ** 2           # C's own uncertainty of the shape
        n_valid_b = np.array([np.isfinite(scan[ix]).sum() for ix in self.region_index], dtype=float)
        r_b = var_cb + (noise / np.sqrt(np.maximum(n_valid_b, 1.0)) / np.maximum(pred_b, 1e-12)) ** 2
        # unexplained per-pass block noise, from the innovations against the current shape estimate
        innov_b = o - self.d
        excess = np.nanmean(np.where(ok, innov_b ** 2 - self.P - r_b, np.nan))
        self.tau2_shape = cfg.memory * self.tau2_shape + (1 - cfg.memory) * max(0.0, float(excess))
        # discounted precision-weighted mean log-ratio of each block
        prec = np.where(ok, 1.0 / (r_b + self.tau2_shape), 0.0)
        self.sum_wy = cfg.shape_memory * self.sum_wy + prec * np.nan_to_num(o)
        self.sum_w = cfg.shape_memory * self.sum_w + prec
        has = self.sum_w > 0
        ybar = np.where(has, self.sum_wy / np.where(has, self.sum_w, 1.0), 0.0)
        v = np.where(has, 1.0 / np.where(has, self.sum_w, 1.0), np.inf)
        # empirical Bayes: signal variance across blocks, then shrink each block towards zero
        fin = has & np.isfinite(v)
        self.signal2 = max(0.0, float(np.mean(ybar[fin] ** 2 - v[fin]))) if fin.any() else 0.0
        shrink = np.where(fin, self.signal2 / (self.signal2 + np.where(fin, v, 1.0)), 0.0)
        self.d = shrink * ybar
        self.P = np.where(fin, shrink * np.where(fin, v, 0.0), self.signal2)
        wts = pred_b * self.n_b
        self.d -= np.sum(wts * self.d) / wts.sum()                 # keep the shape free of level

    # ------------------------------------------------------------------ decisions
    def prob_crossed_by_next(self, action: PassAction, threshold: float, use_level: bool | None = None) -> float:
        """Probability that K is below ``threshold`` * K0 at the start of the next pass (or already was),
        from C's particles with the level discrepancy applied to the next pass's K.
        Call after C.update for the pass just scanned (``action``)."""
        pf = self.pf
        j = pf._current()
        lthr = np.log(threshold) + pf.path[:, 0]
        done = (pf.path[:, : j + 1] < lthr[:, None]).any(axis=1)
        eta, _, z = pf._coeffs(pf.lk, pf.lpath[:, j], pf.path[:, j], action.force, pf._scale(action),
                               g=pf.gpath[:, j])
        lK1 = pf.path[:, j] - np.log1p(z * pf.table.volume(eta))
        use = self.cfg.level_to_K if use_level is None else use_level
        shift, extra = (self.m, self.V) if use else (0.0, 0.0)
        sd = np.sqrt(np.exp(2 * pf.lsw) + extra)
        k = self.cfg.draws
        eps = self.rng.standard_normal((k, lK1.size))
        below = (lK1[None, :] + shift + sd[None, :] * eps) < lthr[None, :]
        return float(np.mean([pf.w @ (done | below[i]) for i in range(k)]))

    def summary(self) -> dict:
        return {"level": float(self.m), "level_sd": float(np.sqrt(self.V)),
                "shape_rms": float(np.sqrt(np.mean(self.d ** 2))), "shape_signal_sd": float(np.sqrt(self.signal2)),
                "tau_level": float(np.sqrt(self.tau2_level)), "tau_shape": float(np.sqrt(self.tau2_shape))}


def _norm_ppf(p: float) -> float:
    """Standard normal quantile."""
    from scipy.stats import norm
    return float(norm.ppf(p))


def _log_var(values: np.ndarray, w: np.ndarray) -> float:
    """Weighted variance of log(values) (values > 0)."""
    x = np.log(np.maximum(values, 1e-300))
    mu = float(w @ x)
    return float(w @ (x - mu) ** 2)


def _bilinear_weights(shape: tuple[int, int], blocks: tuple[int, int]) -> np.ndarray:
    """(n_points, n_blocks) weights that interpolate block values bilinearly between block
    centres (constant beyond the outer centres), for a row-major grid of scan points."""
    ny, nx = shape
    by, bx = blocks

    def axis(n, nb):
        c = (np.arange(n) + 0.5) / n * nb - 0.5           # position in block-centre units
        c = np.clip(c, 0.0, nb - 1.0)
        i0 = np.minimum(np.floor(c).astype(int), nb - 2) if nb > 1 else np.zeros(n, int)
        f = c - i0 if nb > 1 else np.zeros(n)
        return i0, f

    iy, fy = axis(ny, by)
    ix, fx = axis(nx, bx)
    W = np.zeros((ny, nx, by * bx))
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            yy = np.minimum(iy + dy, by - 1)
            xx = np.minimum(ix + dx, bx - 1)
            idx = yy[:, None] * bx + xx[None, :]
            np.add.at(W.reshape(ny * nx, -1), (np.arange(ny * nx), idx.ravel()), (wy[:, None] * wx[None, :]).ravel())
    return W.reshape(ny * nx, by * bx)
