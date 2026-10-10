"""Tracker C3: tracker C with an abrasive-loading (clogging) state.

C reads abrasive loading as wear. Swarf clogs the abrasive while it cuts, so removal
falls along a pass; part of the clogging is shed between passes, so the next pass
starts higher again. C explains the fall within each pass with a high wear rate and
forecasts the abrasive change far too early. C3 keeps C's model, priors, robust scan
model, tempering and moves, and adds a loading state to every particle:

* along pass j the effectiveness is K_j (1 - L_j(u)), with u the fraction of the pass's
  volume removed so far and
      L_j(u) = L_inf - (L_inf - S_j) exp(-kappa u):
  loading builds from its start level S_j towards L_inf, kappa e-folds per pass;
* fresh abrasive is clean (S_1 = 0), and between passes a fraction c of the loading is
  shed: S_{j+1} = (1 - c) L_j(1);
* wear, within and between passes, is driven by the removed (loaded) volume.

K_j is the effectiveness of the clean abrasive, so K_j / K_1 measures wear alone and the
abrasive-change rule uses it. Whether the abrasive loads at all is itself unknown (prior
probability p_loading); if it does, the static unknowns L_inf (uniform on [0, load_max]),
c (uniform on [clean_min, 1]) and kappa (uniform over a few values) are learned from the
scans: loading makes removal fall faster within a pass than from one pass to the next,
which wear alone cannot do. Loading that is hardly shed would be indistinguishable from
wear (a break-in, say), and should count as wear, hence c >= clean_min; and a loading
that saturates within the first raster line would be a fixed map shape, hence the
smallest kappa values only. These settings were chosen on held-out draws (200-215) of
the matched, realistic, break-in-only, loading, tilting-holder and six-ring worlds. Moves: a joint random walk on (L_inf, c) that shifts the whole
log K path so that the loaded level a (1 - L_inf) is unchanged, and a jump of kappa
to another value, and switching loading off or on; all leave the tempered path
posterior invariant.

Surrogate: with 1 - L(u) = (1 - L_inf) + (L_inf - S) psi(u), psi(u) = exp(-kappa u), the
removal map is A sum_k (-z)^k X_k(eta) + B Psi_kappa(eta), with A = a (1 - L_inf),
B = a (L_inf - S), C's tables X_k and one more table per kappa,
Psi_kappa = sum_l avg_l(psi) m_l (line averages of psi over each raster line's share
of the pass volume; the small product of within-pass wear and loading is neglected).
The scan log-likelihood is again a quadratic form in the coefficients, computed from
precomputed inner products of the tables.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimators import ParticleFilter, PFConfig, systematic_resample, weighted_quantiles
from .process import PassAction, Priors, action_at
from .surrogate import N_TAB, ExposureTable


@dataclass(frozen=True)
class C3Config:
    kappas: tuple[float, ...] = (1.5, 3.0, 6.0)    # loading e-folds per pass (uniform prior over these)
    load_max: float = 0.5                          # L_inf ~ uniform on [0, load_max]
    clean_min: float = 0.4                         # c ~ uniform on [clean_min, 1]: loading is temporary
    p_loading: float = 0.5                         # prior probability that the abrasive loads at all
    rate_drift: float | None = None                # C3's own wear-rate drift (None: C's tuned value)


def _gram(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """sum_i A[i, a, e] * B[i, b, e] -> (e, a, b)."""
    return np.einsum("iae,ibe->eab", A, B)


@dataclass
class LoadingStats:
    """Sufficient statistics of one (masked) scan over the augmented tables [X_0..X_3, Psi_k...],
    collapsed as in C's surrogate: the X-X block to polynomials in w = -z (powers 0..6), the
    X-Psi blocks to cubics in w per kappa, and the Psi-Psi diagonal to one number per kappa."""

    Yx: np.ndarray       # (n_eta, N_TAB) <y, X_c,e>
    Yp: np.ndarray       # (n_eta, n_k) <y, Psi_k,e>
    yy: float
    gd: np.ndarray       # (n_eta, 2 N_TAB - 1) X-X, same node
    go: np.ndarray       # (n_eta-1, 2 N_TAB - 1) X-X, neighbouring nodes (both orders)
    xd: np.ndarray       # (n_eta, n_k, N_TAB) X-Psi, same node
    xo: np.ndarray       # (n_eta-1, n_k, N_TAB) X-Psi, neighbouring nodes (both orders)
    pd: np.ndarray       # (n_eta, n_k) Psi-Psi, same node
    po: np.ndarray       # (n_eta-1, n_k) Psi-Psi, neighbouring nodes (both orders)
    n_valid: int

    @classmethod
    def collapse(cls, Y, yy, Gd, Go, n_valid, n_k):
        GoS = Go + Go.transpose(0, 2, 1)
        n_pow = 2 * N_TAB - 1
        gd = np.zeros((Gd.shape[0], n_pow))
        go = np.zeros((GoS.shape[0], n_pow))
        for c in range(N_TAB):
            for d in range(N_TAB):
                gd[:, c + d] += Gd[:, c, d]
                go[:, c + d] += GoS[:, c, d]
        k = np.arange(n_k)
        return cls(np.ascontiguousarray(Y[:N_TAB].T), np.ascontiguousarray(Y[N_TAB:].T), yy, gd, go,
                   np.ascontiguousarray(Gd[:, N_TAB:, :N_TAB]), np.ascontiguousarray(GoS[:, N_TAB:, :N_TAB]),
                   np.ascontiguousarray(Gd[:, N_TAB + k, N_TAB + k]), np.ascontiguousarray(GoS[:, N_TAB + k, N_TAB + k]),
                   n_valid)


class LoadingTable:
    """C's surrogate tables plus one loading table Psi_kappa per kappa, at the scan points."""

    def __init__(self, base: ExposureTable, kappas=(1.5, 5.0, 15.0), chunk: int = 64):
        self.base = base
        self.model = base.model
        self.kappas = np.asarray(kappas, dtype=float)
        self.n_k = len(self.kappas)
        self.n_eta = base.n_eta
        self.p = N_TAB + self.n_k
        model, cg = base.model, base.model.contact
        line = model.path.line
        n_lines = model.n_lines
        eta = base.eta
        # line averages of psi = exp(-kappa u) over each line's share of the pass volume
        U_end = np.cumsum(base.u_line, axis=0)
        U_start = U_end - base.u_line
        frac_s, frac_e = U_start / base.vol, U_end / base.vol
        wl = np.empty((self.n_k, n_lines, self.n_eta))
        for i, kap in enumerate(self.kappas):
            d = kap * (frac_e - frac_s)
            safe = np.maximum(d, 1e-12)
            wl[i] = np.where(d > 1e-9, (np.exp(-kap * frac_s) - np.exp(-kap * frac_e)) / safe, np.exp(-kap * frac_s))
        # the same station loop as ExposureTable, with the loading line weights
        n_obs = base.obs_index.size
        obs_col = np.full(model.panel.n_pix, -1, dtype=np.int64)
        obs_col[base.obs_index] = np.arange(n_obs)
        P = np.zeros((n_obs, self.n_k, self.n_eta))
        for s in range(cg.n_stations):
            seg = cg.segment(s)
            g, B, C, S = cg.gap[seg], cg.B[seg], cg.C[seg], cg.S[seg]
            col = obs_col[cg.idx[seg]]
            pos = np.flatnonzero(col >= 0)
            if pos.size == 0:
                continue
            j = np.searchsorted(B, eta, side="right") - 1
            dd = (eta + S[j]) / C[j]
            ln = line[s]
            for e0 in range(0, self.n_eta, chunk):
                e1 = min(self.n_eta, e0 + chunk)
                n_c = int(np.searchsorted(pos, j[e1 - 1], side="right"))
                if n_c == 0:
                    continue
                pp = pos[:n_c]
                M = np.maximum(dd[None, e0:e1] - g[pp, None], 0.0)
                M *= model.vdt[seg][pp, None]
                M /= eta[None, e0:e1]
                rows = col[pp]
                for k in range(self.n_k):
                    P[rows, k, e0:e1] += M * wl[k, ln, None, e0:e1]
        self.P = P                                            # (n_obs, n_k, n_eta)
        self.mean_P = P.mean(axis=0)                          # (n_k, n_eta)
        X = base.X
        p = self.p
        self.Gd_full = np.empty((self.n_eta, p, p))
        self.Gd_full[:, :N_TAB, :N_TAB] = base.Gd_full
        xp = _gram(X, P)
        self.Gd_full[:, :N_TAB, N_TAB:] = xp
        self.Gd_full[:, N_TAB:, :N_TAB] = xp.transpose(0, 2, 1)
        self.Gd_full[:, N_TAB:, N_TAB:] = _gram(P, P)
        Go = np.empty((self.n_eta - 1, p, p))
        Go[:, :N_TAB, :N_TAB] = base.Go_full
        Go[:, :N_TAB, N_TAB:] = _gram(X[:, :, :-1], P[:, :, 1:])
        Go[:, N_TAB:, :N_TAB] = _gram(P[:, :, :-1], X[:, :, 1:])
        Go[:, N_TAB:, N_TAB:] = _gram(P[:, :, :-1], P[:, :, 1:])
        self.Go_full = Go
        self._colsum: dict = {}
        self._regions: dict = {}

    # ------------------------------------------------------------------ helpers
    def locate(self, eta):
        return self.base.locate(eta)

    def volume(self, eta):
        return self.base.volume(eta)

    def _rows(self, idx: np.ndarray) -> np.ndarray:
        return np.concatenate([self.base.X[idx], self.P[idx]], axis=1)

    def _column_sums(self, shape: tuple[int, int]) -> np.ndarray:
        if shape not in self._colsum:
            ny, nx = shape
            self._colsum[shape] = np.concatenate(
                [self.base._column_sums(shape), self.P.reshape(ny, nx, self.n_k, self.n_eta).sum(axis=0)], axis=1)
        return self._colsum[shape]

    def coefficients(self, A, B, z, kidx) -> np.ndarray:
        """Weights of the augmented tables per particle -> (N, p)."""
        b = np.zeros((np.size(A), self.p))
        b[:, :N_TAB] = ExposureTable.coefficients(A, z)
        b[np.arange(np.size(A)), N_TAB + np.asarray(kidx)] = B
        return b

    # ------------------------------------------------------------------ likelihood terms
    def scan_stats(self, y: np.ndarray, valid: np.ndarray | None = None, shape: tuple[int, int] | None = None,
                   rho: float = 0.0) -> LoadingStats:
        """As ExposureTable.scan_stats (masked points, random offset per scan profile), for the
        augmented tables."""
        if valid is None:
            valid = np.isfinite(y)
        yv = np.where(valid, y, 0.0)
        n = yv.size
        Y = np.concatenate([(yv @ self.base.X.reshape(n, -1)).reshape(N_TAB, self.n_eta),
                            (yv @ self.P.reshape(n, -1)).reshape(self.n_k, self.n_eta)])
        Gd, Go = self.Gd_full, self.Go_full
        bad = np.flatnonzero(~valid)
        if bad.size:
            Zb = self._rows(bad)
            Gd = Gd - _gram(Zb, Zb)
            Go = Go - _gram(Zb[:, :, :-1], Zb[:, :, 1:])
        yy = float(yv @ yv)
        if rho > 0 and shape is not None:
            ny, nx = shape
            n_col = valid.reshape(ny, nx).sum(axis=0).astype(float)
            Zs = self._column_sums(shape)
            if bad.size:
                onehot = np.zeros((nx, bad.size))
                onehot[bad % nx, np.arange(bad.size)] = 1.0
                Zs = Zs - (onehot @ Zb.reshape(bad.size, -1)).reshape(Zs.shape)
            ys = yv.reshape(ny, nx).sum(axis=0)
            kap = rho / (1.0 + n_col * rho)
            yy -= float(kap @ ys**2)
            Y = Y - ((kap * ys) @ Zs.reshape(nx, -1)).reshape(self.p, self.n_eta)
            Zw = Zs * np.sqrt(kap)[:, None, None]
            Gd = Gd - _gram(Zw, Zw)
            Go = Go - _gram(Zw[:, :, :-1], Zw[:, :, 1:])
        return LoadingStats.collapse(Y, yy, Gd, Go, int(valid.sum()), self.n_k)

    def sse(self, eta, A, B, z, kidx, st: LoadingStats) -> np.ndarray:
        """||y - predicted||^2 over the valid scan points, predicted = A sum_k (-z)^k X_k + B Psi_kappa."""
        e, t = self.locate(eta)
        s = 1 - t
        W = ExposureTable.powers(-np.asarray(z, dtype=float), 2 * N_TAB - 1)
        Wl = W[:, :N_TAB]
        lin = A * (s * np.einsum("nc,nc->n", Wl, st.Yx[e]) + t * np.einsum("nc,nc->n", Wl, st.Yx[e + 1])) \
            + B * (s * st.Yp[e, kidx] + t * st.Yp[e + 1, kidx])
        xx = (s * s) * np.einsum("nm,nm->n", W, st.gd[e]) + (s * t) * np.einsum("nm,nm->n", W, st.go[e]) \
            + (t * t) * np.einsum("nm,nm->n", W, st.gd[e + 1])
        xp = (s * s) * np.einsum("nc,nc->n", Wl, st.xd[e, kidx]) + (s * t) * np.einsum("nc,nc->n", Wl, st.xo[e, kidx]) \
            + (t * t) * np.einsum("nc,nc->n", Wl, st.xd[e + 1, kidx])
        pp = (s * s) * st.pd[e, kidx] + (s * t) * st.po[e, kidx] + (t * t) * st.pd[e + 1, kidx]
        return np.maximum(st.yy - 2 * lin + A * A * xx + 2 * A * B * xp + B * B * pp, 0.0)

    # ------------------------------------------------------------------ predictions
    def mixture_obs(self, eta, A, B, z, kidx, weights) -> np.ndarray:
        """sum_i w_i * predicted map_i at the scan points."""
        out = self.base.mixture_obs(eta, ExposureTable.coefficients(A, z), weights)
        e, t = self.locate(eta)
        node_w = np.zeros((self.n_k, self.n_eta))
        for k in range(self.n_k):
            wk = weights * B * (kidx == k)
            node_w[k] = np.bincount(e, wk * (1 - t), self.n_eta) + np.bincount(e + 1, wk * t, self.n_eta)
        return out + self.P.reshape(self.P.shape[0], -1) @ node_w.ravel()

    def mean_removal(self, eta, A, B, z, kidx) -> np.ndarray:
        e, t = self.locate(eta)
        m_psi = (1 - t) * self.mean_P[kidx, e] + t * self.mean_P[kidx, e + 1]
        return self.base.mean_removal(eta, ExposureTable.coefficients(A, z)) + B * m_psi

    def region_means(self, region_index) -> tuple[np.ndarray, np.ndarray]:
        """Mean of X and of Psi over each region's scan points: (n_r, N_TAB, n_eta), (n_r, n_k, n_eta)."""
        key = id(region_index)
        if key not in self._regions:
            self._regions[key] = (np.array([self.base.X[ix].mean(axis=0) for ix in region_index]),
                                  np.array([self.P[ix].mean(axis=0) for ix in region_index]))
        return self._regions[key]


class LoadingTracker(ParticleFilter):
    """Tracker C with an abrasive-loading state (module docstring)."""

    name = "C3: C + abrasive loading"

    def __init__(self, table: LoadingTable, priors: Priors, cfg: PFConfig, rng: np.random.Generator,
                 rollout_rng: np.random.Generator | None = None, n_passes_max: int = 64,
                 obs_shape: tuple[int, int] | None = None, c3: C3Config = C3Config(), region_index=None):
        super().__init__(table, priors, cfg, rng, rollout_rng, n_passes_max, obs_shape)
        n = cfg.n_particles
        self.c3 = c3
        self.kap = table.kappas
        self.Linf = rng.uniform(0.0, c3.load_max, n)        # loading limit if the abrasive loads
        self.clean = rng.uniform(c3.clean_min, 1.0, n)
        self.kidx = rng.integers(0, table.n_k, n)
        self.region_index = region_index
        self.on = (rng.random(n) < c3.p_loading) if c3.p_loading < 1.0 else np.ones(n, dtype=bool)
        self.rw_scale.update({"load": 1.0, "kappa": 1.0, "toggle": 1.0})
        # passes after which r^i < 1e-12 (r = (1 - c) exp(-kappa) <= its largest value): the start
        # loading has reached its steady state to that precision
        r_max = (1.0 - c3.clean_min) * float(np.exp(-self.kap.min()))
        self._n_transient = int(np.ceil(np.log(1e-12) / np.log(max(r_max, 1e-300)))) + 1 if r_max > 0 else 1
        self.S, self.Lbar = self._paths(self.Leff, self.clean, self.kidx)

    # ------------------------------------------------------------------ loading model
    @property
    def Leff(self) -> np.ndarray:
        """The loading limit in effect: L_inf if the particle's abrasive loads, else 0."""
        return np.where(self.on, self.Linf, 0.0)

    def _paths(self, Linf, clean, kidx) -> tuple[np.ndarray, np.ndarray]:
        """Loading at the start of every pass (S) and its volume-weighted mean over the pass (Lbar)."""
        kap = self.kap[kidx]
        E = np.exp(-kap)
        gbar = (1.0 - E) / kap
        r = (1.0 - clean) * E                    # S_{i+1} = r S_i + q, S_1 = 0: S_i = S_inf (1 - r^i)
        S_inf = (1.0 - clean) * Linf * (1.0 - E) / np.maximum(1.0 - r, 1e-12)
        S = np.repeat(S_inf[:, None], self.path.shape[1], axis=1)
        i = np.arange(min(self._n_transient, S.shape[1]))
        S[:, i] *= 1.0 - r[:, None] ** i[None, :]
        return S, Linf[:, None] - (Linf[:, None] - S) * gbar[:, None]

    @staticmethod
    def _at(M: np.ndarray, p):
        """Column(s) p of a per-particle path; p may be an index array shaped (1, j)."""
        p = np.asarray(p)
        if p.ndim == 0:
            return M[:, int(p)]
        return M[:, p.reshape(-1)].reshape((M.shape[0],) + p.shape[1:])

    def _load(self, load):
        return (self.Leff, self.S, self.Lbar, self.kidx) if load is None else load

    def _loglik(self, p: int, lk, llam, lK, beta=None, g=None, load=None) -> np.ndarray:
        g = self.gpath[:, p] if g is None else g
        Linf, S, Lb, kidx = self._load(load)
        eta, a, z = self._coeffs(lk, llam, lK, self.F[p], self.s[p], beta, g)
        A, B = a * (1.0 - Linf), a * (Linf - S[:, p])
        return -self.table.sse(eta, A, B, z * (1.0 - Lb[:, p]), kidx, self.stats[p]) / (2.0 * self.sigma2[p])

    def _mu(self, p, lk, llam, lK_p, beta=None, g=None, load=None):
        g = self.gpath[:, p] if g is None else g
        Lb = self._load(load)[2]
        eta, _, z = self._coeffs(lk, llam, lK_p, self.F[p], self.s[p], beta, g)
        return lK_p - np.log1p(z * (1.0 - self._at(Lb, p)) * self.table.volume(eta))

    def _transition(self, p, lk, llam, lK_p, lK_next, lsw, beta=None, g=None, load=None) -> np.ndarray:
        return -0.5 * ((lK_next - self._mu(p, lk, llam, lK_p, beta, g, load)) / np.exp(lsw)) ** 2 - lsw

    # ------------------------------------------------------------------ moves
    def _rescore(self, j: int, tempers, path, load):
        """Tempered log posterior terms that depend on the loading state and the K path (passes 0..j)."""
        LLp = np.column_stack([self._loglik(i, self.lk, self.lpath[:, i], path[:, i], load=load)
                               for i in range(j + 1)])
        val = self._logprior_K0(path[:, 0]) + LLp @ tempers
        TRp = None
        if j:
            TRp = self._transition(np.arange(j)[None, :], self.lk[:, None], self.lpath[:, :j], path[:, :j],
                                   path[:, 1:j + 1], self.lsw[:, None], self.beta[:, None], self.gpath[:, :j],
                                   load=load)
            val = val + TRp.sum(axis=1)
        return val, LLp, TRp

    def _current_score(self, j: int, tempers):
        val = self._logprior_K0(self.path[:, 0]) + self.LL[:, :j + 1] @ tempers
        return val + self.TR[:, :j].sum(axis=1) if j else val

    def _accept(self, acc, j, path, load, LLp, TRp) -> None:
        _, S, Lb, kidx = load
        self.kidx = np.where(acc, kidx, self.kidx)
        self.S[acc], self.Lbar[acc] = S[acc], Lb[acc]
        self.path[acc, :j + 1] = path[acc]
        self.LL[acc, :j + 1] = LLp[acc]
        if j:
            self.TR[acc, :j] = TRp[acc]

    @staticmethod
    def _reflect(x, lo, hi):
        x = np.where(x < lo, 2 * lo - x, x)
        x = np.where(x > hi, 2 * hi - x, x)
        return np.clip(x, lo, hi)

    def _move_load(self, j: int, tempers) -> None:
        """Joint random walk on (L_inf, c); the whole log K path moves by log(1 - L_inf) - log(1 - L_inf')
        so that the loaded level a (1 - L_inf) is unchanged (a shift with unit Jacobian)."""
        n = self.lk.size
        X = np.column_stack([self.Linf, self.clean])
        L = np.linalg.cholesky(self._wcov(X) * (2.38**2 / 2) * self.rw_scale["load"] ** 2)
        step = self.rng.standard_normal((n, 2)) @ L.T
        Linf_p = self._reflect(self.Linf + step[:, 0], 0.0, self.c3.load_max)
        clean_p = self._reflect(self.clean + step[:, 1], self.c3.clean_min, 1.0)
        Leff_p = np.where(self.on, Linf_p, 0.0)
        path_p = self.path[:, :j + 1] + (np.log1p(-self.Leff) - np.log1p(-Leff_p))[:, None]
        S_p, Lb_p = self._paths(Leff_p, clean_p, self.kidx)
        load = (Leff_p, S_p, Lb_p, self.kidx)
        new, LLp, TRp = self._rescore(j, tempers, path_p, load)
        acc = self._mh("load", new - self._current_score(j, tempers))
        self.Linf = np.where(acc, Linf_p, self.Linf)
        self.clean = np.where(acc, clean_p, self.clean)
        self._accept(acc, j, path_p, load, LLp, TRp)

    def _move_toggle(self, j: int, tempers) -> None:
        """Switch loading off or on (keeping L_inf, c and kappa, which have their priors either way), with
        the same shift of the log K path as in _move_load; the prior odds enter the acceptance ratio."""
        pl = self.c3.p_loading
        if not 0.0 < pl < 1.0:
            return
        on_p = ~self.on
        Leff_p = np.where(on_p, self.Linf, 0.0)
        path_p = self.path[:, :j + 1] + (np.log1p(-self.Leff) - np.log1p(-Leff_p))[:, None]
        S_p, Lb_p = self._paths(Leff_p, self.clean, self.kidx)
        load = (Leff_p, S_p, Lb_p, self.kidx)
        new, LLp, TRp = self._rescore(j, tempers, path_p, load)
        odds = np.log(pl / (1.0 - pl))
        acc = self._mh("toggle", new - self._current_score(j, tempers) + np.where(on_p, odds, -odds))
        self.on = np.where(acc, on_p, self.on)
        self._accept(acc, j, path_p, load, LLp, TRp)

    def _move_kappa(self, j: int, tempers) -> None:
        """kappa jumps to one of the other values, uniformly (a symmetric proposal)."""
        if self.table.n_k < 2:
            return
        n = self.lk.size
        prop = (self.kidx + self.rng.integers(1, self.table.n_k, n)) % self.table.n_k
        S_p, Lb_p = self._paths(self.Leff, self.clean, prop)
        load = (self.Leff, S_p, Lb_p, prop)
        path = self.path[:, :j + 1]
        new, LLp, TRp = self._rescore(j, tempers, path, load)
        acc = self._mh("kappa", new - self._current_score(j, tempers))
        self._accept(acc, j, path.copy(), load, LLp, TRp)

    def _mh_sweep(self, j: int, phi: float, full: bool = False) -> None:
        """C's moves, then (L_inf, c); kappa only in the full sweeps after each update."""
        super()._mh_sweep(j, phi, full)
        tempers = np.ones(j + 1)
        tempers[j] = phi
        self._move_load(j, tempers)
        self._move_toggle(j, tempers)
        if full:
            self._move_kappa(j, tempers)

    def _adapt(self, key: str, rate: float) -> None:
        if key not in ("kappa", "toggle"):     # discrete jumps have no scale to adapt
            super()._adapt(key, rate)

    def _resample(self) -> None:
        idx = systematic_resample(self.w, self.rng)
        self.lk, self.lsw, self.beta, self.lsg = self.lk[idx], self.lsw[idx], self.beta[idx], self.lsg[idx]
        self.gpath, self.GP = self.gpath[idx], self.GP[idx]
        self.path, self.lpath, self.LL, self.TR = self.path[idx], self.lpath[idx], self.LL[idx], self.TR[idx]
        self.Linf, self.clean, self.kidx, self.on = self.Linf[idx], self.clean[idx], self.kidx[idx], self.on[idx]
        self.S, self.Lbar = self.S[idx], self.Lbar[idx]
        self.w = np.full(self.lk.size, 1.0 / self.lk.size)

    def _snapshot(self):
        return (super()._snapshot(), self.Linf.copy(), self.clean.copy(), self.kidx.copy(), self.on.copy(),
                self.S.copy(), self.Lbar.copy())

    def _restore(self, snap) -> None:
        base, self.Linf, self.clean, self.kidx, self.on, self.S, self.Lbar = snap
        super()._restore(base)

    # ------------------------------------------------------------------ predictions
    def _next(self, action: PassAction):
        j = self._current()
        eta, a, z = self._coeffs(self.lk, self.lpath[:, j], self.path[:, j], action.force, self._scale(action),
                                 g=self.gpath[:, j])
        L = self.Leff
        return eta, a * (1.0 - L), a * (L - self.S[:, j]), z * (1.0 - self.Lbar[:, j])

    def predict(self, action: PassAction) -> np.ndarray:
        eta, A, B, z = self._next(action)
        return self.table.mixture_obs(eta, A, B, z, self.kidx, self.w)

    def predict_mean_removal(self, action: PassAction, q=(0.05, 0.5, 0.95)) -> np.ndarray:
        eta, A, B, z = self._next(action)
        return weighted_quantiles(self.table.mean_removal(eta, A, B, z, self.kidx), self.w, q)

    def predict_region_removal(self, action: PassAction, region_tables=None, q=(0.05, 0.5, 0.95)) -> np.ndarray:
        """Predictive quantiles of the next pass's mean removal over each region (``region_index``)."""
        eta, A, B, z = self._next(action)
        b = ExposureTable.coefficients(A, z)
        e, t = self.table.locate(eta)
        Xr, Pr = self.table.region_means(self.region_index)
        out = []
        for M, Q in zip(Xr, Pr):
            m = (1 - t)[:, None] * M[:, e].T + t[:, None] * M[:, e + 1].T
            psi = (1 - t) * Q[self.kidx, e] + t * Q[self.kidx, e + 1]
            out.append(weighted_quantiles(np.einsum("nc,nc->n", b, m) + B * psi, self.w, q))
        return np.array(out)

    def predict_crossing(self, passes_done: int, schedule: list[PassAction], threshold: float,
                         horizon: int = 400, q: tuple[float, float] = (0.05, 0.95)) -> dict:
        """As C's, with the wear of each future pass driven by its loaded volume."""
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
        nmax = self.Lbar.shape[1]
        for j in range(horizon):
            if done.all():
                break
            a = action_at(schedule, i0 + j)
            g = self.gpath[:, i0] if j == 0 else np.exp(self.lsg) * self.rollout_rng.standard_normal(n)
            eta, _, z = self._coeffs(self.lk, llam, lK, a.force, a.rpm / self.rpm_ref, g=g)
            Lb = self.Lbar[:, min(i0 + j, nmax - 1)]
            lK = lK - np.log1p(z * (1.0 - Lb) * self.table.volume(eta)) + sw * self.rollout_rng.standard_normal(n)
            llam = llam + self.cfg.rate_drift * self.rollout_rng.standard_normal(n)
            new = (~done) & (lK < lthr)
            pred[new] = passes_done + j + 1
            done |= new
        lo, med, hi = weighted_quantiles(pred, self.w, [q[0], 0.5, q[1]])
        mass = float(self.w @ ((pred >= lo) & (pred <= hi)))
        return {"median": float(med), "lo": float(lo), "hi": float(hi), "mean": float(self.w @ pred),
                "band_mass": mass}

    def summary(self, q: tuple[float, float] = (0.05, 0.95)) -> dict:
        out = super().summary(q)
        j = self._current()
        w = self.w
        L = self.Leff
        lo, med, hi = weighted_quantiles(L, w, [q[0], 0.5, q[1]])
        out["L_inf"] = {"median": float(med), "lo": float(lo), "hi": float(hi), "mean": float(w @ L)}
        out["p_loading"] = float(w @ self.on)
        out["clean"] = float(weighted_quantiles(self.clean, w, [0.5])[0])
        out["kappa"] = float(weighted_quantiles(self.kap[self.kidx], w, [0.5])[0])
        out["load_start"] = float(w @ self.S[:, j])
        return out
