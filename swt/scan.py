"""Simulated post-pass surface scan.

After each pass the scanner reports the removal map (depth removed in that
pass) at a subset of grid nodes, corrupted by independent Gaussian noise. A
``stride`` > 1 keeps every ``stride``-th node in x and y, mimicking a line
scanner whose line spacing and point spacing are coarser than the simulation
grid.
"""
from __future__ import annotations

import numpy as np

from .geometry import Panel


class Scanner:
    def __init__(self, panel: Panel, noise_um: float = 2.0, stride: int = 2):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.panel = panel
        self.noise_mm = float(noise_um) * 1e-3
        self.stride = int(stride)
        ny, nx = panel.shape
        iy = np.arange(0, ny, stride)
        ix = np.arange(0, nx, stride)
        self.obs_shape = (iy.size, ix.size)
        self.obs_index = (iy[:, None] * nx + ix[None, :]).ravel()

    @property
    def n_obs(self) -> int:
        return self.obs_index.size

    def observe(self, removal_flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Noisy scan of a removal map [mm] at the scanner's sample points."""
        clean = np.asarray(removal_flat)[self.obs_index]
        if self.noise_mm == 0:
            return clean.copy()
        return clean + self.noise_mm * rng.standard_normal(clean.size)
