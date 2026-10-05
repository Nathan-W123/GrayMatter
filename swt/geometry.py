"""Workpiece geometry and toolpath.

The workpiece is a curved panel stored as a height map z(x, y) on a regular
plan-view grid. The default is a convex cylindrical section: the cylinder axis
runs along x (straight direction) and the panel curves across y, with the crown
at the middle of the panel width.

The toolpath for one pass is a serpentine raster: lines run along x and step
over in y. Each line is discretised into "stations" (pad-centre positions). A
station stands for a short segment of the path, so it is assigned the time the
pad spends on that segment (segment length / feed), plus an optional dwell at
the two ends of every line where the robot turns around.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Panel:
    """Height map of a cylindrical panel on a regular grid.

    Parameters
    ----------
    length_x, width_y : panel size in plan view [mm].
    radius : cylinder radius [mm]; ``None`` or ``inf`` gives a flat panel.
    grid : grid spacing [mm] (same in x and y).
    """

    def __init__(self, length_x: float = 200.0, width_y: float = 150.0,
                 radius: float | None = 300.0, grid: float = 1.0):
        nx = int(round(length_x / grid)) + 1
        ny = int(round(width_y / grid)) + 1
        if abs((nx - 1) * grid - length_x) > 1e-9 or abs((ny - 1) * grid - width_y) > 1e-9:
            raise ValueError("panel size must be a multiple of the grid spacing")
        self.length_x = float(length_x)
        self.width_y = float(width_y)
        self.radius = None if radius is None or not np.isfinite(radius) else float(radius)
        self.grid = float(grid)
        self.x = np.arange(nx) * grid
        self.y = np.arange(ny) * grid
        self.X, self.Y = np.meshgrid(self.x, self.y)  # shape (ny, nx)
        if self.radius is None:
            self.Z = np.zeros_like(self.X)
        else:
            if width_y / 2 >= self.radius:
                raise ValueError("panel half-width must be smaller than the radius")
            yc = width_y / 2.0
            self.Z = np.sqrt(self.radius**2 - (self.Y - yc) ** 2) - self.radius
        gy, gx = np.gradient(self.Z, grid, grid, edge_order=2)
        n = np.stack([-gx, -gy, np.ones_like(self.Z)], axis=-1)
        self.normals = n / np.linalg.norm(n, axis=-1, keepdims=True)
        self.max_slope = float(np.max(np.hypot(gx, gy)))
        # Surface area represented by each grid node: trapezoidal weights in
        # plan view (half at edges, quarter at corners) times the slope factor.
        wx = np.ones(nx); wx[[0, -1]] = 0.5
        wy = np.ones(ny); wy[[0, -1]] = 0.5
        self.node_area = (wy[:, None] * wx[None, :]) * grid * grid * np.sqrt(1.0 + gx**2 + gy**2)

    @property
    def shape(self) -> tuple[int, int]:
        return self.Z.shape

    @property
    def n_pix(self) -> int:
        return self.Z.size

    @property
    def area(self) -> float:
        """Total surface area of the panel [mm^2]."""
        return float(self.node_area.sum())

    def node_index(self, x0: float, y0: float) -> tuple[int, int]:
        """Grid indices (iy, ix) of the node at plan position (x0, y0)."""
        ix = int(round(x0 / self.grid))
        iy = int(round(y0 / self.grid))
        if not (0 <= ix < self.x.size and 0 <= iy < self.y.size):
            raise ValueError(f"point ({x0}, {y0}) is outside the panel")
        return iy, ix


@dataclass
class RasterPath:
    """Discretised raster pass: pad-centre stations and time spent at each."""

    stations: np.ndarray  # (n_st, 2) plan positions (x0, y0) [mm]
    dt: np.ndarray        # (n_st,) time attributed to each station [s]
    line: np.ndarray      # (n_st,) raster line index of each station
    feed: float
    dwell: float
    stepover: float

    @property
    def n_stations(self) -> int:
        return self.stations.shape[0]

    @property
    def duration(self) -> float:
        return float(self.dt.sum())


def raster_path(panel: Panel, stepover: float = 15.0, station_spacing: float = 5.0,
                feed: float = 25.0, dwell: float = 0.25, margin: float = 0.0) -> RasterPath:
    """Serpentine raster with lines along x, stepping over in y.

    Stations are spread evenly over each line (the requested spacing is
    rounded so both line ends are reached) and snapped to grid nodes. Each
    station is credited with the time to travel half-way to each neighbour;
    the two end stations also get the turnaround ``dwell``.
    """
    g = panel.grid
    if station_spacing < g:
        raise ValueError("station spacing must be at least the grid spacing")
    span_y = panel.width_y - 2 * margin
    n_lines = int(round(span_y / stepover)) + 1
    ys = np.round(np.linspace(margin, panel.width_y - margin, n_lines) / g) * g
    span_x = panel.length_x - 2 * margin
    n_st_line = max(2, int(round(span_x / station_spacing)) + 1)
    xs = np.round(np.linspace(margin, panel.length_x - margin, n_st_line) / g) * g
    seg = np.diff(xs)
    dt_line = np.zeros(xs.size)
    dt_line[:-1] += 0.5 * seg / feed
    dt_line[1:] += 0.5 * seg / feed
    dt_line[[0, -1]] += dwell
    stations, dts, lines = [], [], []
    for li, y0 in enumerate(ys):
        order = slice(None) if li % 2 == 0 else slice(None, None, -1)
        stations.append(np.column_stack([xs[order], np.full(xs.size, y0)]))
        dts.append(dt_line[order])
        lines.append(np.full(xs.size, li))
    actual_stepover = float(ys[1] - ys[0]) if n_lines > 1 else 0.0
    return RasterPath(np.vstack(stations), np.concatenate(dts), np.concatenate(lines),
                      feed=feed, dwell=dwell, stepover=actual_stepover)
