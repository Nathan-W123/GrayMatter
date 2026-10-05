"""Fast surrogate of the forward model for the estimators.

For a fixed toolpath the exposure map depends on stiffness and force only
through eta = F / k_pad, and on force and spindle speed through a linear
factor:

    E(F, k, rpm) = F * (rpm / rpm_ref) * m(eta),   eta = F / k_pad.

``m`` (exposure per newton) is tabulated once on a log-spaced eta grid with the
exact contact model and linearly interpolated in log(eta). A predicted removal
map is then ``a * m(eta)`` with scale ``a = K * F * rpm / rpm_ref``.

Because interpolation is linear in the table rows, the Gaussian scan
log-likelihood of any (eta, a) reduces to a handful of precomputed inner
products (``<y, m_e>`` and the tri-diagonal Gram matrix of the table), so
evaluating thousands of particles costs O(1) each instead of O(n_pixels).
"""
from __future__ import annotations

import numpy as np

from .process import ProcessModel


class ExposureTable:
    def __init__(self, model: ProcessModel, eta_min: float, eta_max: float, n_eta: int,
                 obs_index: np.ndarray, chunk: int = 64):
        if not (0 < eta_min < eta_max):
            raise ValueError("need 0 < eta_min < eta_max")
        self.model = model
        self.log_eta = np.linspace(np.log(eta_min), np.log(eta_max), n_eta)
        self.eta = np.exp(self.log_eta)
        self.dlog = float(self.log_eta[1] - self.log_eta[0])
        self.n_eta = n_eta
        cg = model.contact
        table = np.zeros((model.panel.n_pix, n_eta))
        for s in range(cg.n_stations):
            seg = cg.segment(s)
            g, C, S, B = cg.gap[seg], cg.C[seg], cg.S[seg], cg.B[seg]
            idx, w = cg.idx[seg], model.vdt[seg]
            j = np.searchsorted(B, self.eta, side="right") - 1
            d = (self.eta + S[j]) / C[j]
            for e0 in range(0, n_eta, chunk):
                e1 = min(n_eta, e0 + chunk)
                n_c = int(j[e1 - 1]) + 1           # contact set grows with eta
                M = np.maximum(d[None, e0:e1] - g[:n_c, None], 0.0)
                M *= w[:n_c, None]
                M /= self.eta[None, e0:e1]
                table[idx[:n_c], e0:e1] += M
        self.obs_index = np.asarray(obs_index)
        self.obs = np.ascontiguousarray(table[self.obs_index])   # (n_obs, n_eta)
        self.gram_diag = np.einsum("ij,ij->j", self.obs, self.obs)
        self.gram_off = np.einsum("ij,ij->j", self.obs[:, :-1], self.obs[:, 1:])
        self.vol = model.node_area_flat @ table             # removed volume per unit scale a
        self.full = np.ascontiguousarray(table.T)           # (n_eta, n_pix): one row per eta node
        del table

    # ------------------------------------------------------------------ interpolation
    def locate(self, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower node index and fractional position of each eta (clipped to the table)."""
        u = (np.log(np.asarray(eta, dtype=float)) - self.log_eta[0]) / self.dlog
        u = np.clip(u, 0.0, self.n_eta - 1 - 1e-12)
        e = np.floor(u).astype(np.int64)
        return e, u - e

    def in_range(self, eta: np.ndarray) -> np.ndarray:
        le = np.log(np.asarray(eta, dtype=float))
        return (le >= self.log_eta[0] - 1e-12) & (le <= self.log_eta[-1] + 1e-12)

    def map_full(self, eta: float) -> np.ndarray:
        """Interpolated m(eta) on the full grid (flat)."""
        e, t = self.locate(np.array([eta]))
        e, t = int(e[0]), float(t[0])
        return (1 - t) * self.full[e] + t * self.full[e + 1]

    def volume(self, eta: np.ndarray) -> np.ndarray:
        """Removed volume per unit scale a, interpolated."""
        e, t = self.locate(eta)
        return (1 - t) * self.vol[e] + t * self.vol[e + 1]

    def mixture_full(self, eta: np.ndarray, a: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """sum_i w_i * a_i * m(eta_i) on the full grid (posterior predictive mean)."""
        e, t = self.locate(eta)
        node_w = np.zeros(self.n_eta)
        np.add.at(node_w, e, weights * a * (1 - t))
        np.add.at(node_w, e + 1, weights * a * t)
        nz = np.flatnonzero(node_w)
        return node_w[nz] @ self.full[nz]

    # ------------------------------------------------------------------ likelihood terms
    def scan_terms(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Precompute <y, m_e> for all nodes and ||y||^2 for one scan."""
        return self.obs.T @ y, float(y @ y)

    def sse(self, eta: np.ndarray, a: np.ndarray, ym: np.ndarray, yy: float) -> np.ndarray:
        """||y - a m(eta)||^2 at the scan points for many (eta, a) pairs, exactly."""
        e, t = self.locate(eta)
        s = 1 - t
        y_m = s * ym[e] + t * ym[e + 1]
        m_m = s * s * self.gram_diag[e] + 2 * s * t * self.gram_off[e] + t * t * self.gram_diag[e + 1]
        return np.maximum(yy - 2 * a * y_m + a * a * m_m, 0.0)

    def profile_sse(self, eta: np.ndarray, ym: np.ndarray, yy: float) -> tuple[np.ndarray, np.ndarray]:
        """Best scale a*(eta) and the resulting SSE (least squares in a)."""
        e, t = self.locate(eta)
        s = 1 - t
        y_m = s * ym[e] + t * ym[e + 1]
        m_m = s * s * self.gram_diag[e] + 2 * s * t * self.gram_off[e] + t * t * self.gram_diag[e + 1]
        a = y_m / m_m
        return a, np.maximum(yy - y_m * y_m / m_m, 0.0)
