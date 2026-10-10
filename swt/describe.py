"""Model-description experiments: contact versus stiffness, and the derived constants and
mismatch sizes quoted in the README."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Context, rng_for
from .geometry import Panel
from .pad import ContactGeometry, contact_fraction_sweep
from .process import HiddenTruth, WearLaw, line_effectiveness, simulate_truth
from .stats import UM, write_csv, write_json


def experiment_contact_sweep(ctx: Context, out: Path, verbose: bool = True) -> dict:
    cfg = ctx.cfg
    kmin, kmax, nk = cfg["experiments"]["contact_sweep_k"]
    ks = np.logspace(np.log10(kmin), np.log10(kmax), int(nk))
    F_ref = cfg["pad"]["reference_force_N"]
    forces = sorted(set(cfg["schedule"]["forces_N"]) | {F_ref})
    rows = []
    pr = ctx.priors
    for F in forces:
        cf = contact_fraction_sweep(ctx.panel, ctx.pad_radius, F, ks)
        rows += [{"force_N": F, "k_pad": float(k), "contact_fraction": float(c)} for k, c in zip(ks, cf)]
    # grid-convergence reference: the same sweep on a 4x finer grid at the reference force
    fine_grid = ctx.panel.grid / 4
    fine = Panel(ctx.panel.length_x, ctx.panel.width_y, ctx.panel.radius, fine_grid)
    cf_fine = contact_fraction_sweep(fine, ctx.pad_radius, F_ref, ks)
    rows += [{"force_N": F_ref, "k_pad": float(k), "contact_fraction": float(c), "grid_mm": fine_grid}
             for k, c in zip(ks, cf_fine)]
    # the realistic world's foam pad (same small-strain stiffness, stiffens as it densifies)
    foam = next((w.pad_law for w, _ in ctx.worlds.values() if w.pad_law.kind == "foam"), None)
    foam_bounds = None
    if foam is not None:
        cf_foam = contact_fraction_sweep(ctx.panel, ctx.pad_radius, F_ref, ks, foam)
        rows += [{"force_N": F_ref, "k_pad": float(k), "contact_fraction": float(c), "pad_law": "foam"}
                 for k, c in zip(ks, cf_foam)]
        lo, hi = contact_fraction_sweep(ctx.panel, ctx.pad_radius, F_ref, np.array([pr.k_low, pr.k_high]), foam)
        foam_bounds = {"thickness_mm": foam.thickness, "soft_k_low": float(lo), "stiff_k_high": float(hi),
                       "force_N": F_ref}
    for r in rows:
        r.setdefault("grid_mm", ctx.panel.grid)
        r.setdefault("pad_law", "linear")
    fine_bounds = contact_fraction_sweep(fine, ctx.pad_radius, F_ref, np.array([pr.k_low, pr.k_high]))
    at_bounds = {}
    for F in forces:
        lo, hi = contact_fraction_sweep(ctx.panel, ctx.pad_radius, F, np.array([pr.k_low, pr.k_high]))
        at_bounds[f"{F:g}N"] = {"soft_k_low": float(lo), "stiff_k_high": float(hi)}
    summary = {"k_low": pr.k_low, "k_high": pr.k_high, "reference_force_N": F_ref,
               "grid_mm": ctx.panel.grid, "radius_mm": ctx.panel.radius,
               "contact_fraction_at_prior_bounds": at_bounds,
               "fine_grid_reference": {"grid_mm": fine_grid, "soft_k_low": float(fine_bounds[0]),
                                       "stiff_k_high": float(fine_bounds[1]), "force_N": F_ref},
               "foam_pad_at_prior_bounds": foam_bounds,
               "pad_face_area_mm2": float(np.pi * ctx.pad_radius**2)}
    # pressure profile across the cylinder (through the pad centre) for stiff / mid / soft pads
    centre = np.array([[round(ctx.panel.length_x / 2 / ctx.panel.grid) * ctx.panel.grid,
                        round(ctx.panel.width_y / 2 / ctx.panel.grid) * ctx.panel.grid]])
    cg = ContactGeometry(ctx.panel, centre, ctx.pad_radius)
    xs = ctx.panel.X.ravel()[cg.idx]
    ys = ctx.panel.Y.ravel()[cg.idx]
    on_line = np.isclose(xs, centre[0, 0])
    prof = []
    k_mid = float(np.sqrt(pr.k_low * pr.k_high))
    for label, k in (("stiff", pr.k_high), ("mid", k_mid), ("soft", pr.k_low)):
        p = cg.pressure(F_ref, k)
        cf = float(cg.contact_fraction(F_ref, k)[0])
        order = np.argsort(ys[on_line])
        for yy, pp in zip(ys[on_line][order], p[on_line][order]):
            prof.append({"pad": label, "k_pad": float(k), "contact_fraction": cf,
                         "offset_mm": float(yy - centre[0, 1]), "pressure_kPa": float(pp * 1e3)})
    write_csv(out / "contact_profiles.csv", prof)
    write_csv(out / "contact_sweep.csv", rows)
    write_json(out / "contact_sweep.json", summary)
    if verbose:
        print("\nContact fraction vs pad stiffness (pad centred on the panel)")
        print("  k_pad [N/mm^3] " + "".join(f"  F={F:>4g} N" for F in forces))
        for k in np.logspace(np.log10(kmin), np.log10(kmax), 15):
            vals = [contact_fraction_sweep(ctx.panel, ctx.pad_radius, F, np.array([k]))[0] for F in forces]
            print(f"  {k:12.4g}   " + "".join(f"  {v:9.1%}" for v in vals))
        b = at_bounds[f"{F_ref:g}N"]
        print(f"  prior range k in [{pr.k_low:g}, {pr.k_high:g}] -> contact {b['stiff_k_high']:.1%} "
              f"(stiff) to {b['soft_k_low']:.1%} (soft) at {F_ref:g} N on the {ctx.panel.grid:g} mm grid; "
              f"{fine_bounds[1]:.1%} to {fine_bounds[0]:.1%} on a {fine_grid:g} mm grid")
    return summary


def experiment_model_info(ctx: Context, out: Path, verbose: bool = True) -> dict:
    """Derived constants of the default model quoted in the README (geometry, kinematics,
    the near-independence of removed volume from stiffness, surrogate accuracy, and the
    size of each mismatch of the realistic world)."""
    cfg, pr, m = ctx.cfg, ctx.priors, ctx.model
    a = ctx.pad_radius
    R = ctx.panel.radius
    F_ref = float(cfg["pad"]["reference_force_N"])
    forces_all = sorted(set(cfg["schedule"]["forces_N"]) | {cfg["schedule"]["constant_force_N"]})
    ks = np.geomspace(pr.k_low, pr.k_high, 9)
    vol = np.array([m.volume(m.exposure(F_ref, k)) for k in ks])          # per unit K
    # surrogate vs the exact forward model (linear pad, within-pass wear), at the scan points
    rng = rng_for(cfg["seed"], 9)
    oi = ctx.obs_index
    errs, zmax = [], 0.0
    for _ in range(30):
        k = float(np.exp(rng.uniform(np.log(pr.k_low), np.log(pr.k_high))))
        K = float(pr.K0_median * np.exp(pr.K0_sigma_log * rng.standard_normal()) * rng.uniform(0.4, 1.0))
        lam = float(pr.lam_median * np.exp(pr.lam_sigma_log * rng.standard_normal()))
        for F in forces_all:
            E, u = m.line_exposures(F, k)
            exact = line_effectiveness(K, lam, u) @ E[:, oi]
            approx = ctx.table.map_obs(F / k, K * F, lam * K * F)
            errs.append(float(np.sqrt(np.mean((exact - approx) ** 2)) / np.sqrt(np.mean(exact**2))))
            zmax = max(zmax, lam * K * float(u.sum()))
    nom = pr.log_mean()
    centre = np.array([[round(ctx.panel.length_x / 2 / ctx.panel.grid) * ctx.panel.grid,
                        round(ctx.panel.width_y / 2 / ctx.panel.grid) * ctx.panel.grid]])
    cg = ContactGeometry(ctx.panel, centre, a)
    penetration = {f"{F:g}N": {"soft_k_low_mm": float(cg.penetration(F / pr.k_low)[0]),
                               "stiff_k_high_mm": float(cg.penetration(F / pr.k_high)[0])}
                   for F in forces_all}
    fresh = {f"{F:g}N": float(nom["K0"] * m.exposure(F, nom["k_pad"])[oi].mean() * UM) for F in forces_all}
    info = {
        "grid_nodes": int(ctx.panel.n_pix), "scan_points": int(len(oi)),
        "scan_grid_mm": float(ctx.panel.grid * cfg["scan"]["stride"]),
        "panel_sag_mm": float(ctx.panel.Z.max() - ctx.panel.Z.min()),
        "pad_rim_gap_mm": float(R - np.sqrt(R**2 - a**2)) if R else 0.0,
        "raster_lines": int(ctx.path.line.max() + 1), "stations_per_pass": int(ctx.path.n_stations),
        "actual_stepover_mm": ctx.path.stepover, "pass_duration_s": ctx.path.duration,
        "sliding_speed_centre_mm_s": float(ctx.sander.speed(0.0)),
        "sliding_speed_rim_mm_s": float(ctx.sander.speed(a)),
        "nominal_parameters": nom,
        "nominal_fresh_mean_removal_um": fresh,
        "pad_penetration_at_prior_bounds": penetration,
        "volume_per_unit_K_vs_k": {"k_pad": ks.tolist(), "volume_mm3_per_mm2N": vol.tolist(),
                                   "relative_spread": float((vol.max() - vol.min()) / vol.mean()),
                                   "force_N": F_ref},
        "surrogate": {"n_eta": int(ctx.table.n_eta), "eta_min_mm3": float(ctx.table.eta[0]),
                      "eta_max_mm3": float(ctx.table.eta[-1]), "wear_terms": 4,
                      "max_relative_rms_error": float(max(errs)),
                      "median_relative_rms_error": float(np.median(errs)), "n_checks": len(errs),
                      "max_lambda_times_pass_volume": zmax},
        "worlds": _world_info(ctx, cg, forces_all, nom),
    }
    write_json(out / "model_info.json", info)
    if verbose:
        print(f"\nModel: {info['grid_nodes']} grid nodes, {info['scan_points']} scan points, "
              f"{info['stations_per_pass']} stations/pass ({info['pass_duration_s']:.1f} s), sliding speed "
              f"{info['sliding_speed_centre_mm_s']:.0f}-{info['sliding_speed_rim_mm_s']:.0f} mm/s; removed volume "
              f"per unit K varies by {info['volume_per_unit_K_vs_k']['relative_spread']:.2%} across the k_pad "
              f"prior; surrogate max relative RMS error {info['surrogate']['max_relative_rms_error']:.2e}")
    return info


def _world_info(ctx: Context, cg: ContactGeometry, forces: list[float], nom: dict) -> dict:
    """How large each mismatch of every configured world is, in numbers."""
    from scipy.optimize import brentq
    pr = ctx.priors
    out = {}
    for name, (w, art) in ctx.worlds.items():
        d = {"pad_law": w.pad_law.kind, "force_gain_sigma_log": w.force_gain_sigma,
             "force_ripple": w.force_ripple, "ripple_corr": w.ripple_corr, "wear_law": w.wear.kind,
             "wear_rings": w.wear_rings, "preston_exponent": w.preston_exponent,
             "scan_artefacts": {k: getattr(art, k) for k in art.__dataclass_fields__}}
        if w.pad_law.kind == "foam":
            h = w.pad_law.thickness
            pk = {}
            for label, k in (("soft_k_low", pr.k_low), ("mid", nom["k_pad"]), ("stiff_k_high", pr.k_high)):
                F = max(forces)
                _, p_lin = cg.contact_points(F, k)
                _, p_foam = cg.contact_points(F, k, w.pad_law)
                strain = float(p_foam.max() / (k + p_foam.max() / h) / h)     # invert p = k d / (1 - d/h)
                pk[label] = {"k_pad": float(k), "force_N": F, "max_strain_foam": strain,
                             "peak_pressure_ratio_foam_over_linear": float(p_foam.max() / p_lin.max()),
                             "contact_fraction_linear": float(cg.contact_fraction(F, k)[0]),
                             "contact_fraction_foam": float(cg.contact_fraction(F, k, w.pad_law)[0])}
            d["foam"] = {"thickness_mm": h, "at_max_force": pk}
        if w.wear.kind != "single":
            lam = nom["lam"]
            single = WearLaw()
            v50 = {law: float(brentq(lambda W: wl.phi(lam, W) - 0.5, 0.0, 50.0 / lam))
                   for law, wl in (("single", single), ("two_stage", w.wear))}
            # K/K0 after the volume removed by the first nominal pass at the low force
            V1 = float(nom["K0"] * ctx.model.volume(ctx.model.exposure(min(forces), nom["k_pad"])))
            d["two_stage"] = {"fast_fraction": w.wear.fast_fraction, "fast_ratio": w.wear.fast_ratio,
                              "volume_to_half_K0_mm3": v50, "nominal_first_pass_volume_mm3": V1,
                              "K_over_K0_after_first_pass": {"single": single.phi(lam, V1),
                                                             "two_stage": w.wear.phi(lam, V1)}}
        if w.preston_exponent != 1.0:
            F = float(ctx.cfg["pad"]["reference_force_N"])
            vol = {lab: float(ctx.model.line_exposures(F, k, alpha=w.preston_exponent)[1].sum()
                              / ctx.model.line_exposures(F, k)[1].sum())
                   for lab, k in (("soft_k_low", pr.k_low), ("mid", nom["k_pad"]), ("stiff_k_high", pr.k_high))}
            d["preston"] = {"exponent": w.preston_exponent, "p_ref_kPa": 10.0, "force_N": F,
                            "volume_ratio_vs_linear": vol,
                            "soft_over_stiff_volume_ratio": vol["soft_k_low"] / vol["stiff_k_high"]}
        if w.wear_rings > 1:
            from .process import World
            sched = ctx.schedule()
            k20 = {}
            for lab, k in (("soft_k_low", pr.k_low), ("mid", nom["k_pad"]), ("stiff_k_high", pr.k_high)):
                truth = HiddenTruth(float(k), nom["K0"], nom["lam"])
                rings = simulate_truth(ctx.model, truth, sched, 0.0, rng_for(0, 0), max_extend=0,
                                       world=World("r", wear=w.wear, wear_rings=w.wear_rings))
                uni = simulate_truth(ctx.model, truth, sched, 0.0, rng_for(0, 0), max_extend=0,
                                     world=World("u", wear=w.wear))
                k20[lab] = {"rings": float(rings.K[-1] / nom["K0"]), "uniform": float(uni.K[-1] / nom["K0"])}
            d["radial_wear"] = {"rings": w.wear_rings, "K_over_K0_at_last_pass_nominal": k20}
        if w.loading_max > 0:
            from .process import World
            sched = ctx.schedule()
            truth = HiddenTruth(nom["k_pad"], nom["K0"], nom["lam"])
            base = World("b", wear=w.wear, wear_rings=w.wear_rings)
            load = World("l", wear=w.wear, wear_rings=w.wear_rings, loading_max=w.loading_max,
                         loading_volume=w.loading_volume, loading_clean=w.loading_clean)
            v0 = simulate_truth(ctx.model, truth, sched, 0.0, rng_for(0, 0), max_extend=0, world=base).volume
            v1 = simulate_truth(ctx.model, truth, sched, 0.0, rng_for(0, 0), max_extend=0, world=load).volume
            d["loading"] = {"max": w.loading_max, "volume_mm3": w.loading_volume, "clean": w.loading_clean,
                            "volume_loss_nominal": {"pass_1": float(1 - v1[0] / v0[0]),
                                                    "last_pass": float(1 - v1[-1] / v0[-1])}}
        if w.tilt_stiffness > 0:
            F = max(forces)
            tilt = {}
            for lab, k in (("soft_k_low", pr.k_low), ("mid", nom["k_pad"]), ("stiff_k_high", pr.k_high)):
                _, a, b = ctx.model.contact.tilt_solution(F, k, w.tilt_stiffness)   # every station of a pass
                E_t = ctx.model.line_exposures(F, k, tilt_stiffness=w.tilt_stiffness)[0].sum(axis=0)
                E_r = ctx.model.line_exposures(F, k)[0].sum(axis=0)
                tilt[lab] = {"max_tilt_deg": float(np.degrees(np.max(np.hypot(a, b)))),
                             "map_change_rel_rms": float(np.sqrt(np.mean((E_t - E_r) ** 2)) / np.mean(E_r))}
            d["tilt"] = {"stiffness_Nmm_per_rad": w.tilt_stiffness, "force_N": F, "at_max_force": tilt}
        out[name] = d
    return out
