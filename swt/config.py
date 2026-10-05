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
from .pad import LINEAR, PadLaw
from .process import MATCHED, PassAction, Priors, ProcessModel, Sander, WearLaw, World, make_schedule
from .scan import ScanArtefacts, Scanner
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
    obs_shape: tuple[int, int]
    pf: PFConfig
    worlds: dict[str, tuple[World, ScanArtefacts]]

    @property
    def pad_radius(self) -> float:
        return 0.5 * self.cfg["pad"]["diameter_mm"]

    @property
    def wear_sigma(self) -> float:
        return float(self.cfg["priors"]["wear_sigma_log"])

    @property
    def threshold(self) -> float:
        return float(self.cfg["abrasive"]["threshold_fraction"])

    def world(self, name: str) -> tuple[World, ScanArtefacts]:
        return self.worlds[name]

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
    scanner = Scanner(panel, sc["noise_um"], sc["stride"])
    obs_index, obs_shape = scanner.obs_index, scanner.obs_shape
    forces = list(cfg["schedule"]["forces_N"]) + [cfg["schedule"]["constant_force_N"]]
    margin = 1.02
    eta_min = min(forces) / priors.k_high / margin
    eta_max = max(forces) / priors.k_low * margin
    f = cfg["filter"]
    table = ExposureTable(model, eta_min, eta_max, int(f["surrogate_n_eta"]), obs_index)
    pf = PFConfig(n_particles=int(f["n_particles"]), ess_fraction=float(f["ess_fraction"]),
                  wear_noise_range=tuple(float(v) for v in f.get("wear_noise_range", (0.005, 0.05))),
                  rate_drift=_drift(f), sigma_floor_um=float(f["sigma_floor_um"]),
                  threshold=float(cfg["abrasive"]["threshold_fraction"]),
                  mcmc_sweeps=int(f.get("mcmc_sweeps", 1)), robust=bool(f.get("robust", True)),
                  outlier_z=float(f.get("outlier_z", 5.0)), inflation_block=int(f.get("inflation_block", 4)),
                  inflation_threshold=float(f.get("inflation_threshold", 1.2)),
                  profile_offsets=bool(f.get("profile_offsets", True)))
    worlds = build_worlds(cfg.get("worlds", {}))
    ctx = Context(cfg, panel, path, sander, model, priors, table, obs_index, obs_shape, pf, worlds)
    ctx.build_seconds = time.time() - t0
    return ctx


def _drift(f: dict) -> float:
    """``filter.rate_drift``: a number, or 'auto' (tuned by the tuning experiment;
    ``rate_drift_fallback`` until then)."""
    v = f.get("rate_drift", "auto")
    return float(f.get("rate_drift_fallback", 0.1)) if str(v) == "auto" else float(v)


def make_world(name: str, spec: dict) -> tuple[World, ScanArtefacts]:
    """A hidden-process world from its config entry (missing keys = the tracker's model)."""
    pad = spec.get("pad", {})
    law = PadLaw("foam", float(pad["thickness_mm"])) if pad.get("kind", "linear") == "foam" else LINEAR
    w = spec.get("wear", {})
    wear = WearLaw(w.get("kind", "single"), float(w.get("fast_fraction", 0.25)), float(w.get("fast_ratio", 6.0)))
    f = spec.get("force", {})
    world = World(name, law, float(f.get("gain_sigma_log", 0.0)), float(f.get("ripple", 0.0)),
                  float(f.get("ripple_corr", 0.5)), wear, int(w.get("rings", 1)),
                  float(spec.get("removal", {}).get("preston_exponent", 1.0)))
    return world, ScanArtefacts(**spec.get("scan", {}))


GROUPS = ("pad", "removal", "force", "wear", "scan")


def build_worlds(spec: dict) -> dict[str, tuple[World, ScanArtefacts]]:
    """``matched`` (the tracker's own model), the configured worlds, and for the
    ``realistic`` world one world per group of mismatches it contains."""
    worlds = {"matched": (MATCHED, ScanArtefacts())}
    for name, s in spec.items():
        worlds[name] = make_world(name, s or {})
    real = spec.get("realistic")
    if real:
        for key in GROUPS:
            if key in real:
                worlds["only_" + key] = make_world("only_" + key, {key: real[key]})
    return worlds
