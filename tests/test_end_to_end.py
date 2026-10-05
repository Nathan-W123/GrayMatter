"""run-all on a tiny configuration: every experiment, every summary key the figures
need, and every figure, so the results schema and the plotting code cannot drift apart."""
import json

from swt import experiments, plots
from swt.config import build_context, load_config

TINY = {
    "geometry": {"grid_mm": 2.0},
    "path": {"station_spacing_mm": 10.0},
    "schedule": {"n_passes": 6},
    "filter": {"surrogate_n_eta": 96, "n_particles": 800},
    "experiments": {"robustness_draws": 3, "noise_levels_um": [2.0, 5.0], "contact_sweep_k": [1.0e-4, 30.0, 9]},
}


def test_run_all_and_figures(tmp_path, capsys):
    ctx = build_context(load_config(overrides=TINY))
    res, figs = tmp_path / "results", tmp_path / "figures"
    summary = experiments.run_all(ctx, res, verbose=True, workers=1)   # verbose: exercise the printing path
    plots.make_all(res, figs)
    for name in ("fig1_contact_vs_stiffness", "fig2_removal_maps", "fig3_rmse_per_pass",
                 "fig4_parameter_tracking", "fig5_abrasive_change", "fig6_ablation_and_noise"):
        assert (figs / f"{name}.png").stat().st_size > 10_000
    on_disk = json.loads((res / "summary.json").read_text())
    assert on_disk["robustness"]["n_draws"] == 3
    assert set(summary) == {"contact_sweep", "model_info", "main_run", "robustness", "ablation",
                            "noise_sensitivity", "abrasive_change"}
    assert {"mean", "median", "lo", "hi"} <= set(on_disk["main_run"]["C_final"]["k_pad"])
    assert (res / "abrasive_change_by_lead.csv").exists()
    assert "Robustness over 3 hidden-truth draws" in capsys.readouterr().out
