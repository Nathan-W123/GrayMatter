"""Configuration loading and construction of the shared simulation context."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .estimators import PFConfig
from .geometry import Panel, RasterPath, raster_path
from .process import PassAction, Priors, ProcessModel, Sander, make_schedule
from .scan import Scanner
from .surrogate import ExposureTable

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def rng_for(seed: int, *keys: int) -> np.random.Generator:
    """Independent, reproducible random stream for a (seed, keys...) tuple."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), *[int(k) for k in keys]]))


@dataclass
class Context:
    """Everything that is fixed across runs: geometry, path, models, surrogate."""

    cfg: dict
    panel: Panel
    path: RasterPath
    sander: Sander
    model: ProcessModel
    priors: Priors
    table: ExposureTable
    obs_index: np.ndarray
    pf: PFConfig

    @property
    def pad_radius(self) -> float:
        return 0.5 * self.cfg["pad"]["diameter_mm"]

    @property
    def wear_sigma(self) -> float:
        return float(self.cfg["priors"]["wear_sigma_log"])

    @property
    def threshold(self) -> float:
        return float(self.cfg["abrasive"]["threshold_fraction"])

    def schedule(self, kind: str = "alternating") -> list[PassAction]:
        s = self.cfg["schedule"]
        forces = s["forces_N"] if kind == "alternating" else [s["constant_force_N"]]
        return make_schedule(forces, int(s["n_passes"]), float(s["rpm"]))


def build_context(cfg: dict) -> Context:
    t0 = time.time()
    g, p, sd, pa, sc = cfg["geometry"], cfg["pad"], cfg["sander"], cfg["path"], cfg["scan"]
    panel = Panel(g["length_x_mm"], g["width_y_mm"], g["radius_mm"], g["grid_mm"])
    path = raster_path(panel, pa["stepover_mm"], pa["station_spacing_mm"], pa["feed_mm_s"],
                       pa["dwell_s"], pa["edge_margin_mm"])
    sander = Sander(sd["spindle_rpm"], sd["orbit_diameter_mm"], sd["pad_spin_ratio"])
    model = ProcessModel(panel, path, 0.5 * p["diameter_mm"], sander)
    pr = cfg["priors"]
    priors = Priors(pr["k_low"], pr["k_high"], pr["K0_median"], pr["K0_sigma_log"],
                    pr["lam_median"], pr["lam_sigma_log"])
    obs_index = Scanner(panel, sc["noise_um"], sc["stride"]).obs_index
    forces = list(cfg["schedule"]["forces_N"]) + [cfg["schedule"]["constant_force_N"]]
    margin = 1.02
    eta_min = min(forces) / priors.k_high / margin
    eta_max = max(forces) / priors.k_low * margin
    f = cfg["filter"]
    table = ExposureTable(model, eta_min, eta_max, int(f["surrogate_n_eta"]), obs_index)
    pf = PFConfig(n_particles=int(f["n_particles"]), ess_fraction=float(f["ess_fraction"]),
                  shrinkage=float(f["shrinkage"]), wear_sigma=float(pr["wear_sigma_log"]),
                  sigma_floor_um=float(f["sigma_floor_um"]),
                  threshold=float(cfg["abrasive"]["threshold_fraction"]))
    ctx = Context(cfg, panel, path, sander, model, priors, table, obs_index, pf)
    ctx.build_seconds = time.time() - t0
    return ctx
