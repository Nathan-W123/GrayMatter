"""Simulated post-pass surface scan.

After each pass the scanner reports the removal map (depth removed in that
pass) at a subset of grid nodes, corrupted by independent Gaussian noise. A
``stride`` > 1 keeps every ``stride``-th node in x and y, mimicking a line
scanner whose line spacing and point spacing are coarser than the simulation
grid.

Optional artefacts (used by the "realistic" world; the tracker is not told
about them):

* registration error: the whole map is shifted by a random sub-millimetre
  offset before sampling (the scan is not perfectly aligned with the part);
* profile bias: every scan profile (one column of sample points along y)
  gets its own constant offset;
* outliers: a small fraction of points get a large error (e.g. reflections);
* dropouts: a fraction of points is missing (returned as NaN), half of them
  as isolated points and half as short gaps along a profile;
* surface texture: the removal is measured as the difference of the height
  scans before and after the pass, and each pass leaves a new random surface
  texture (spatially correlated over ``texture_corr_mm``). The texture left by
  pass n therefore enters the scans of pass n (with a minus sign) and pass n+1
  (with a plus sign).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as nd_shift

from .geometry import Panel


@dataclass(frozen=True)
class ScanArtefacts:
    registration_sigma_mm: float = 0.0
    profile_bias_um: float = 0.0
    outlier_fraction: float = 0.0
    outlier_um: float = 15.0
    dropout_fraction: float = 0.0
    dropout_gap_points: int = 8
    texture_um: float = 0.0          # RMS of the surface texture each pass leaves
    texture_corr_mm: float = 1.0     # its correlation length (Gaussian kernel s.d.)

    @property
    def any(self) -> bool:
        return (self.registration_sigma_mm > 0 or self.profile_bias_um > 0
                or self.outlier_fraction > 0 or self.dropout_fraction > 0)


class Scanner:
    def __init__(self, panel: Panel, noise_um: float = 2.0, stride: int = 2,
                 artefacts: ScanArtefacts | None = None):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.panel = panel
        self.noise_mm = float(noise_um) * 1e-3
        self.stride = int(stride)
        self.artefacts = artefacts or ScanArtefacts()
        ny, nx = panel.shape
        iy = np.arange(0, ny, stride)
        ix = np.arange(0, nx, stride)
        self.obs_shape = (iy.size, ix.size)
        self.obs_index = (iy[:, None] * nx + ix[None, :]).ravel()
        self._texture: np.ndarray | None = None   # texture of the surface before the next pass

    def _new_texture(self, rng: np.random.Generator) -> np.ndarray:
        art = self.artefacts
        g = gaussian_filter(rng.standard_normal(self.panel.shape), art.texture_corr_mm / self.panel.grid,
                            mode="wrap")
        return (art.texture_um * 1e-3 / g.std() * g).ravel()

    @property
    def n_obs(self) -> int:
        return self.obs_index.size

    def observe(self, removal_flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Noisy scan of a removal map [mm] at the scanner's sample points (NaN = missing)."""
        art = self.artefacts
        field = np.asarray(removal_flat)
        if art.texture_um > 0:           # (height before + old texture) - (height after + new texture)
            if self._texture is None:
                self._texture = self._new_texture(rng)
            new = self._new_texture(rng)
            field = field + self._texture - new
            self._texture = new
        if art.registration_sigma_mm > 0:
            off = rng.normal(0.0, art.registration_sigma_mm, 2) / self.panel.grid
            field = nd_shift(field.reshape(self.panel.shape), off, order=1, mode="nearest").ravel()
        clean = field[self.obs_index]
        y = clean.copy()
        if self.noise_mm > 0:
            y = y + self.noise_mm * rng.standard_normal(clean.size)
        if not art.any:
            return y
        ny, nx = self.obs_shape
        grid = y.reshape(ny, nx)
        if art.profile_bias_um > 0:
            grid += art.profile_bias_um * 1e-3 * rng.standard_normal(nx)[None, :]
        if art.outlier_fraction > 0:
            hit = rng.random(grid.shape) < art.outlier_fraction
            grid[hit] += art.outlier_um * 1e-3 * rng.standard_normal(int(hit.sum()))
        if art.dropout_fraction > 0:
            drop = rng.random(grid.shape) < 0.5 * art.dropout_fraction
            n_gaps = int(round(0.5 * art.dropout_fraction * grid.size / art.dropout_gap_points))
            for _ in range(n_gaps):
                c, r0 = rng.integers(nx), rng.integers(ny)
                drop[r0:r0 + art.dropout_gap_points, c] = True
            grid[drop] = np.nan
        return grid.ravel()
