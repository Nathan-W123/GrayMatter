"""run-all on a tiny configuration: every experiment, every summary key the figures
need, and every figure, so the results schema and the plotting code cannot drift apart."""
import json

from swt import experiments, plots
from swt.config import build_context, load_config

TINY = {
    "geometry": {"grid_mm": 2.0},
    "path": {"station_spacing_mm": 10.0},
    "schedule": {"n_passes": 6},
    "filter": {"surrogate_n_eta": 96, "n_particles": 600},
    "experiments": {"robustness_draws": 3, "noise_draws": 2, "ablation_draws": 2, "breakdown_draws": 2,
                    "validation_draws": 2,
                    "noise_levels_um": [2.0, 5.0], "contact_sweep_k": [1.0e-4, 30.0, 9]},
    "tuning": {"draws": [200, 202], "rate_drift": [0.0, 0.1]},
    "extended": {"draws": 2, "stress_draws": 1, "stress_worlds": ["loading", "edge_tilt"]},
}


def test_run_all_and_figures(tmp_path, capsys):
    ctx = build_context(load_config(overrides=TINY))
    res, figs = tmp_path / "results", tmp_path / "figures"
    summary = experiments.run_all(ctx, res, verbose=True, workers=1)   # verbose: exercise the printing path
    plots.make_all(res, figs)
    ext = experiments.run_extended(ctx, res / "extended", res, verbose=True, workers=1)
    plots.make_extended(res, figs)
    for name in ("fig1_contact_vs_stiffness", "fig2_removal_maps", "fig3_rmse_per_pass",
                 "fig4_parameter_tracking", "fig5_abrasive_change", "fig6_ablation_and_noise",
                 "fig7_mismatch_and_calibration", "fig8_tracker_c2", "fig9_tracker_c3"):
        assert (figs / f"{name}.png").stat().st_size > 10_000
    on_disk = json.loads((res / "summary.json").read_text())
    assert on_disk["robustness"]["worlds"]["realistic"]["n_draws"] == 3
    assert set(summary) == {"contact_sweep", "model_info", "tuning", "main_run", "robustness", "ablation",
                            "noise_sensitivity", "abrasive_change", "world_breakdown"}
    assert on_disk["tuning"]["chosen_rate_drift"] in (0.0, 0.1)
    assert ctx.pf.rate_drift == on_disk["tuning"]["chosen_rate_drift"]
    main = on_disk["main_run"]["worlds"][on_disk["main_run"]["world"]]
    assert {"mean", "median", "lo", "hi"} <= set(main["C_final"]["k_pad"])
    assert set(on_disk["world_breakdown"]["worlds"]) == {"matched", "only_pad", "only_removal", "only_force",
                                                         "only_wear", "only_scan", "realistic", "radial_wear",
                                                         "edge_tilt", "loading"}
    assert set(on_disk["robustness"]["worlds"]) == {"matched", "realistic"}
    assert "C2" in on_disk["robustness"]["worlds"]["realistic"]
    assert {w: v["n_draws"] for w, v in ext["robustness"]["worlds"].items()} == \
        {"matched": 2, "realistic": 2, "realistic_b": 2, "realistic_c": 2, "realistic_d": 2, "realistic_e": 2}
    assert set(ext["stress_tests"]["worlds"]) == {"loading", "edge_tilt"}
    assert all("C2" in v and "C3" in v for v in ext["stress_tests"]["worlds"].values())
    assert all("C3" in v for v in ext["robustness"]["worlds"].values())
    by_r = ext["abrasive_change"]["worlds"]["realistic_d"]["decision_rules"]["by_cost_ratio"]["r3"]
    assert {"p_perm", "p_holm", "p_holm_all"} <= set(by_r["C_minus_D_r3"])
    assert {"p_perm", "p_holm_C3"} <= set(by_r["C3_minus_C_r3"]) and "p_holm_C3" in by_r["C3_minus_D_r3"]
    # the repeated core worlds count once in the family over core and extended worlds
    ch = ext["abrasive_change"]["worlds"]
    for w in ("matched", "realistic"):
        assert all("p_holm_all" not in e[f"C_minus_D_{tag}"] for tag, e in ch[w]["decision_rules"]["by_cost_ratio"].items())
    n_all = sum(1 for w in ("realistic_b", "realistic_c", "realistic_d", "realistic_e")
                for e in ch[w]["decision_rules"]["by_cost_ratio"].values() for k in e if k.startswith("C_minus_") and
                "p_holm_all" in e[k])
    assert ext["abrasive_change"]["holm_all_family_size"] >= n_all
    n_c3 = sum(1 for v in ch.values() for e in v["decision_rules"]["by_cost_ratio"].values()
               for k in e if k.startswith("C3_minus_"))
    assert ext["abrasive_change"]["holm_C3_family_size"] == n_c3
    rules = on_disk["abrasive_change"]["worlds"]["realistic"]["decision_rules"]
    assert {"A", "D", "R", "C_r1", "C_r3", "C_r19", "D_r3", "R_r19"} <= set(rules["rules"])
    assert set(rules["by_cost_ratio"]) == {"r1", "r3", "r19"} == set(on_disk["tuning"]["decision_margins"])
    assert {"startup", "steady", "last_two"} <= set(on_disk["robustness"]["worlds"]["realistic"]["C_vs_D"])
    assert "tilt" in on_disk["model_info"]["worlds"]["edge_tilt"]
    assert "loading" in on_disk["model_info"]["worlds"]["loading"]
    assert (res / "abrasive_change_by_lead.csv").exists()
    assert "Robustness, realistic world, 3 hidden truths" in capsys.readouterr().out
