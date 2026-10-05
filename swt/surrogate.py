"""Fast surrogate of the tracker's forward model, at the scan points.

For a fixed toolpath and the tracker's linear pad, the exposure of raster line l
depends on stiffness and force only through eta = F / k_pad:

    E_l(F, k, rpm) = F * (rpm / rpm_ref) * m_l(eta).

With wear acting during the pass (process.line_effectiveness, single-stage law)
the removal map is

    h = a * sum_l c_l(z) m_l,   a = K F rpm/rpm_ref,   z = lambda * a,
    c_l(z) = [ln(1 + z U_end,l) - ln(1 + z U_start,l)] / (z u_l)
           = 1 - z (U_start,l + U_end,l)/2 + z^2 (U_s^2 + U_s U_e + U_e^2)/3 - ...

where u_l(eta) is line l's volume per unit a and U_start/end its cumulative
bounds. Since 1/(1 + zU) = sum_k (-z)^k U^k, c_l(z) = sum_k (-z)^k avg_l(U^k), with
avg_l(U^k) = (U_end^(k+1) - U_start^(k+1)) / ((k+1) u_l). Truncating after z^3
gives h = a sum_k (-z)^k X_k with four tables X_k = sum_l avg_l(U^k) m_l,
tabulated once on a log-spaced eta grid with the exact contact model and
linearly interpolated in log(eta). (Relative truncation error ~ (lambda dV_pass)^4:
<1e-3 for typical wear, ~5e-3 for the fastest-wearing draws.)

Because interpolation is linear in the table rows, the Gaussian scan
log-likelihood of any particle reduces to precomputed inner products
(<y, X_c,e> and 4x4 Gram blocks between neighbouring nodes). The weights of
the tables are powers of (-z), so each Gram block collapses to the 7
coefficients of a polynomial in (-z) and thousands of particles cost O(1)
each. Missing or rejected scan points are handled by subtracting their
contribution from the Gram blocks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .process import ProcessModel

N_TAB = 4   # orders 0..3 of the within-pass wear expansion


@dataclass
class ScanStats:
    """Sufficient statistics of one (masked) scan for the surrogate likelihood."""

    Y: np.ndarray        # (N_TAB, n_eta)  <y, X_c,e> over valid points
    yy: float            # ||y||^2 over valid points
    Gd: np.ndarray       # (n_eta, N_TAB, N_TAB) <X_c,e, X_c',e>
    Go: np.ndarray       # (n_eta-1, N_TAB, N_TAB) <X_c,e, X_c',e+1>
    n_valid: int

    def __post_init__(self):
        # the same Gram terms collapsed to polynomials in w = -z (powers 0..2*N_TAB-2)
        n_pow = 2 * N_TAB - 1
        self.Yt = np.ascontiguousarray(self.Y.T)                       # (n_eta, N_TAB)
        self.gd = np.zeros((self.Gd.shape[0], n_pow))
        self.go = np.zeros((self.Go.shape[0], n_pow))
        for c in range(N_TAB):
            for d in range(N_TAB):
                self.gd[:, c + d] += self.Gd[:, c, d]
                self.go[:, c + d] += self.Go[:, c, d] + self.Go[:, d, c]


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
        self.obs_index = np.asarray(obs_index)
        n_obs = self.obs_index.size
        cg = model.contact
        area = model.node_area_flat[cg.idx]
        line = model.path.line
        n_lines = model.n_lines

        # 1) volume of every station per unit a, for every eta (prefix sums, no maps needed)
        u_st = np.zeros((cg.n_stations, n_eta))
        wa = model.vdt * area
        for s in range(cg.n_stations):
            seg = cg.segment(s)
            j = np.searchsorted(cg.B[seg], self.eta, side="right") - 1
            d = (self.eta + cg.S[seg][j]) / cg.C[seg][j]
            P1 = np.cumsum(wa[seg])
            P2 = np.cumsum(wa[seg] * cg.gap[seg])
            u_st[s] = (d * P1[j] - P2[j]) / self.eta
        u_line = np.zeros((n_lines, n_eta))
        np.add.at(u_line, line, u_st)
        U_end = np.cumsum(u_line, axis=0)
        U_start = U_end - u_line
        # line-average of U^k: (U_end^(k+1) - U_start^(k+1)) / ((k+1) u_l), written stably
        wk = np.empty((N_TAB, n_lines, n_eta))
        for k in range(N_TAB):
            wk[k] = sum(U_end ** (k - i) * U_start ** i for i in range(k + 1)) / (k + 1)
        self.vol = U_end[-1]                                  # pass volume per unit a
        self.u_line = u_line

        # 2) exposure tables at the scan points
        obs_col = np.full(model.panel.n_pix, -1, dtype=np.int64)
        obs_col[self.obs_index] = np.arange(n_obs)
        tables = np.zeros((n_obs, N_TAB, n_eta))
        for s in range(cg.n_stations):
            seg = cg.segment(s)
            g, B, C, S = cg.gap[seg], cg.B[seg], cg.C[seg], cg.S[seg]
            col = obs_col[cg.idx[seg]]
            pos = np.flatnonzero(col >= 0)                 # sorted positions of scan points
            if pos.size == 0:
                continue
            j = np.searchsorted(B, self.eta, side="right") - 1
            d = (self.eta + S[j]) / C[j]
            l = line[s]
            for e0 in range(0, n_eta, chunk):
                e1 = min(n_eta, e0 + chunk)
                n_c = int(np.searchsorted(pos, j[e1 - 1], side="right"))   # scan points in contact
                if n_c == 0:
                    continue
                pp = pos[:n_c]
                M = np.maximum(d[None, e0:e1] - g[pp, None], 0.0)
                M *= model.vdt[seg][pp, None]
                M /= self.eta[None, e0:e1]
                rows = col[pp]
                for k in range(N_TAB):
                    tables[rows, k, e0:e1] += M * wk[k, l, None, e0:e1]
        self.X = tables                                       # (n_obs, N_TAB, n_eta), point-major
        self.mean_X = tables.mean(axis=0)                     # (N_TAB, n_eta): mean over scan points
        self.Gd_full = np.einsum("iae,ibe->eab", tables, tables)
        self.Go_full = np.einsum("iae,ibe->eab", tables[:, :, :-1], tables[:, :, 1:])

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

    def volume(self, eta: np.ndarray) -> np.ndarray:
        """Pass volume per unit scale a (no wear), interpolated."""
        e, t = self.locate(eta)
        return (1 - t) * self.vol[e] + t * self.vol[e + 1]

    @staticmethod
    def powers(w: np.ndarray, n: int) -> np.ndarray:
        """w^0 .. w^(n-1) by repeated multiplication -> (len(w), n)."""
        w = np.asarray(w, dtype=float)
        out = np.empty(w.shape + (n,))
        out[..., 0] = 1.0
        for k in range(1, n):
            out[..., k] = out[..., k - 1] * w
        return out

    @classmethod
    def coefficients(cls, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Weights of the tables: a * (-z)^k, k = 0..N_TAB-1 -> (N, N_TAB)."""
        return np.asarray(a, dtype=float)[..., None] * cls.powers(-np.asarray(z, dtype=float), N_TAB)

    def map_obs(self, eta: float, a: float = 1.0, z: float = 0.0) -> np.ndarray:
        """Predicted removal at the scan points for one parameter set."""
        e, t = self.locate(np.array([eta]))
        e, t = int(e[0]), float(t[0])
        b = self.coefficients(np.array([a]), np.array([z]))[0]
        return ((1 - t) * self.X[:, :, e] + t * self.X[:, :, e + 1]) @ b

    def mixture_obs(self, eta: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """sum_i w_i * predicted map_i at the scan points (posterior predictive mean)."""
        e, t = self.locate(eta)
        node_w = np.empty((N_TAB, self.n_eta))
        for c in range(N_TAB):
            wc = weights * b[:, c]
            node_w[c] = np.bincount(e, wc * (1 - t), self.n_eta) + np.bincount(e + 1, wc * t, self.n_eta)
        lo, hi = int(e.min()), int(e.max()) + 2
        if hi - lo > 64:
            return self.X.reshape(self.X.shape[0], -1) @ node_w.ravel()
        return np.einsum("ice,ce->i", self.X[:, :, lo:hi], node_w[:, lo:hi])

    def mean_removal(self, eta: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Mean predicted removal over the scan points, per particle."""
        e, t = self.locate(eta)
        m = (1 - t)[:, None] * self.mean_X[:, e].T + t[:, None] * self.mean_X[:, e + 1].T
        return np.einsum("nc,nc->n", b, m)

    # ------------------------------------------------------------------ likelihood terms
    def scan_stats(self, y: np.ndarray, valid: np.ndarray | None = None, shape: tuple[int, int] | None = None,
                   rho: float = 0.0) -> ScanStats:
        """Sufficient statistics of a scan; ``valid`` masks missing / rejected points.

        ``rho`` > 0 adds a random offset per scan profile (one column of the
        ``shape`` grid) with variance rho * sigma^2 to the noise model. By the
        Woodbury identity the Gaussian log-likelihood is then
        -[SSE - sum_c rho / (1 + n_c rho) * S_c^2] / (2 sigma^2), with S_c the sum of
        the residuals in column c, which has the same form as SSE with adjusted
        statistics (built from the column sums of y and of the tables).
        """
        if valid is None:
            valid = np.isfinite(y)
        yv = np.where(valid, y, 0.0)
        Y = (yv @ self.X.reshape(yv.size, -1)).reshape(N_TAB, self.n_eta)
        Gd, Go = self.Gd_full, self.Go_full
        bad = np.flatnonzero(~valid)
        if bad.size:
            Xb = self.X[bad]
            Gd = Gd - np.einsum("iae,ibe->eab", Xb, Xb)
            Go = Go - np.einsum("iae,ibe->eab", Xb[:, :, :-1], Xb[:, :, 1:])
        yy = float(yv @ yv)
        if rho > 0 and shape is not None:
            ny, nx = shape
            vg = valid.reshape(ny, nx).astype(float)
            Xg = self.X.reshape(ny, nx, -1).transpose(1, 0, 2)              # (nx, ny, N_TAB * n_eta)
            Xs = np.matmul(vg.T[:, None, :], Xg)[:, 0].reshape(nx, N_TAB, self.n_eta)
            ys = yv.reshape(ny, nx).sum(axis=0)
            kap = rho / (1.0 + vg.sum(axis=0) * rho)
            yy -= float(kap @ ys**2)
            Y = Y - np.einsum("x,x,xce->ce", kap, ys, Xs)
            Gd = Gd - np.einsum("x,xae,xbe->eab", kap, Xs, Xs)
            Go = Go - np.einsum("x,xae,xbe->eab", kap, Xs[:, :, :-1], Xs[:, :, 1:])
        return ScanStats(Y, yy, Gd, Go, int(valid.sum()))

    def sse(self, eta: np.ndarray, a: np.ndarray, z: np.ndarray, st: ScanStats) -> np.ndarray:
        """||y - predicted||^2 over the valid scan points for many particles, exactly
        (up to the table interpolation), for removal = a * sum_k (-z)^k X_k(eta)."""
        e, t = self.locate(eta)
        s = 1 - t
        W = self.powers(-np.asarray(z, dtype=float), 2 * N_TAB - 1)
        Wl = W[:, :N_TAB]
        lin = s * np.einsum("nc,nc->n", Wl, st.Yt[e]) + t * np.einsum("nc,nc->n", Wl, st.Yt[e + 1])
        quad = (s * s) * np.einsum("nm,nm->n", W, st.gd[e]) \
            + (s * t) * np.einsum("nm,nm->n", W, st.go[np.minimum(e, self.n_eta - 2)]) \
            + (t * t) * np.einsum("nm,nm->n", W, st.gd[e + 1])
        return np.maximum(st.yy - 2 * a * lin + a * a * quad, 0.0)

    def profile_sse(self, eta: np.ndarray, st: ScanStats, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Best scale a*(eta) and the resulting SSE for a fixed within-pass wear exponent z
        (z = 0: no wear during the pass)."""
        e, t = self.locate(eta)
        s = 1 - t
        W = self.powers(np.array(-float(z)), 2 * N_TAB - 1)
        y_m = s * (st.Yt[e] @ W[:N_TAB]) + t * (st.Yt[e + 1] @ W[:N_TAB])
        m_m = s * s * (st.gd[e] @ W) + s * t * (st.go[np.minimum(e, self.n_eta - 2)] @ W) + t * t * (st.gd[e + 1] @ W)
        return y_m / m_m, np.maximum(st.yy - y_m * y_m / m_m, 0.0)
