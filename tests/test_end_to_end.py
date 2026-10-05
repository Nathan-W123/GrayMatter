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
}


def test_run_all_and_figures(tmp_path, capsys):
    ctx = build_context(load_config(overrides=TINY))
    res, figs = tmp_path / "results", tmp_path / "figures"
    summary = experiments.run_all(ctx, res, verbose=True, workers=1)   # verbose: exercise the printing path
    plots.make_all(res, figs)
    for name in ("fig1_contact_vs_stiffness", "fig2_removal_maps", "fig3_rmse_per_pass",
                 "fig4_parameter_tracking", "fig5_abrasive_change", "fig6_ablation_and_noise",
                 "fig7_mismatch_and_calibration"):
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
                                                         "only_wear", "only_scan", "realistic", "radial_wear"}
    assert on_disk["robustness"]["worlds"]["realistic_b"]["n_draws"] == 2
    assert {"C", "D", "A", "R"} <= set(on_disk["abrasive_change"]["worlds"]["realistic"]["decision_rules"])
    assert (res / "abrasive_change_by_lead.csv").exists()
    assert "Robustness, realistic world, 3 hidden truths" in capsys.readouterr().out
