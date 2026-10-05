# sanding-wear-tracker

A simulation study of tracking sanding-pad stiffness and abrasive wear together, so that a sanding process model could stay accurate as the paper dulls and say when to change it. The tracker is tested both inside its own model and in a "realistic" simulated world with five effects it does not model. All results are simulated; nothing has been tested on physical parts.

![Prediction error per pass](figures/fig3_rmse_per_pass.png)

## 1. Summary

A model that predicts how much material a robotic sanding pass removes needs two things it cannot measure directly: how the compliant pad presses on the part (set by a stiffness that is fixed but unknown) and how sharp the abrasive is (which drops every pass). This project simulates a 125 mm pad sanding a curved panel with hidden stiffness and wear parameters and compares four predictors of the next pass's removal map: a nominal model with fixed prior-mean parameters (A), a model calibrated once on the first scan (B), an engineering baseline that refits stiffness and effectiveness to every scan and extrapolates a fitted wear rate (D), and a particle filter that tracks stiffness, current effectiveness and a drifting wear rate with a robust scan model (C). In a *realistic* world that adds a foam pad that stiffens as it compresses, force-calibration error and line-to-line force ripple, a two-stage wear law and scanner artefacts, none of which any predictor models, C's median pass-20 error over {{robustness.worlds.realistic.n_draws:d}} hidden truths was {{robustness.worlds.realistic.final_rmse_um.C.median:.3f}} µm, against {{robustness.worlds.realistic.final_rmse_um.D.median:.3f}} µm for D (C better in {{robustness.worlds.realistic.draws_C_better_than_D_at_final_pass:d}} of {{robustness.worlds.realistic.n_draws:d}} draws), {{robustness.worlds.realistic.final_rmse_um.A.median:.2f}} µm for A, {{robustness.worlds.realistic.final_rmse_um.B.median:.2f}} µm for B and {{robustness.worlds.realistic.final_rmse_um.oracle.median:.3f}} µm for an oracle that knows the true physics and state (median true removal {{robustness.worlds.realistic.final_true_mean_removal_um.median:.2f}} µm); from the last pass before the abrasive fell below 50% of fresh effectiveness, C named that pass to within one pass in {{abrasive_change.worlds.realistic.by_decision.last_pass_before_crossing.within_1_pass_C:d}} of {{abrasive_change.worlds.realistic.by_decision.last_pass_before_crossing.n:d}} draws (D: {{abrasive_change.worlds.realistic.by_decision.last_pass_before_crossing.within_1_pass_D:d}}). In the tracker's own (matched) world C stays close to the oracle ({{robustness.worlds.matched.final_rmse_um.C.median:.3f}} vs {{robustness.worlds.matched.final_rmse_um.oracle.median:.3f}} µm) and its 90% intervals are calibrated ({{robustness.worlds.matched.predictive_90.passes_2_on.fraction:pct0}}% predictive coverage; {{robustness.worlds.matched.coverage_90.k_pad.inside:d}}–{{robustness.worlds.matched.coverage_90.lam.inside:d}} of 100 for the parameters), but the realistic world exposes three weaknesses: its 90% predictive intervals for the next pass's mean removal cover only {{robustness.worlds.realistic.predictive_90.passes_2_on.fraction:pct0}}% ({{robustness.worlds.realistic.predictive_90.passes_2_to_5.fraction:pct0}}% in passes 2–5, {{robustness.worlds.realistic.predictive_90.passes_11_on.fraction:pct0}}% from pass 11), its parameter estimates become effective values of its simplified model rather than the true ones, and abrasive-change forecasts made many passes ahead are no better than the nominal schedule because the fast break-in of the abrasive is extrapolated.

## 2. Credit and scope

This project builds on the problem described in GrayMatter Robotics' post
**[World Models for Manufacturing Processes](https://factory.graymatter-robotics.com/world-models-for-manufacturing-processes/)**
(Hantao Ye, Omey Manyar and Satyandra K. Gupta, 2 September 2026). Three points from the post shape this work:

- **The model.** A sanding world model takes the scanned surface and a candidate action (path, force, grit, spindle speed, dwell) and predicts the result.
- **The stiffness range.** On a curved panel, GrayMatter's simulator predicts contact over roughly 5% to 97% of the pad face across physically plausible stiffnesses. The post does not give the panel radius, pad size or force.
- **The calibration problem.** Pad compliance, abrasive behaviour and wear are hard to identify exactly, so the model is conditioned on real observations.

**This is an independent, simplified simulation of that problem. It uses no GrayMatter data, code or models.** Every result below comes from the simulator in this repository (`results/`, produced by `python -m swt run-all`); the 5–97% contact range is quoted from GrayMatter's post. The extension studied here is having two unknowns at once, one fixed (pad stiffness) and one that changes every pass (abrasive wear), and testing the tracker against effects it does not model.

## 3. Model

Units are mm, N, s and MPa (= N/mm²). Pad stiffness `k_pad` is in N/mm³, abrasive effectiveness `K` in mm²/N (= 1/MPa), and wear rate `λ` in 1/mm³. All parameter values are in `configs/default.yaml`; the copy used for the reported run is in `results/run_info.json`.

The simulator has two layers. The **tracker's model** (below) is what estimators A–D assume. The **hidden process** runs either in that same model (the *matched* world) or in a *realistic* world that adds five effects the estimators do not model (section 3.6).

### 3.1 Workpiece and toolpath (`swt/geometry.py`)
* **The part** is a height map on a 1 mm grid ({{model_info.grid_nodes:d}} nodes): a convex cylindrical section, 200 mm along the straight x direction and 150 mm across the curved y direction, radius {{contact_sweep.radius_mm:g}} mm (edges {{model_info.panel_sag_mm:.1f}} mm below the crown).
* **One pass** is a serpentine raster: {{model_info.raster_lines:d}} lines along x, {{model_info.actual_stepover_mm:.0f}} mm apart; pad-centre stations every 5 mm ({{model_info.stations_per_pass:d}} per pass); feed 25 mm/s; 0.25 s dwell at both ends of each line; {{model_info.pass_duration_s:.1f}} s per pass. The pad centre runs to the panel edge, so edge stations overhang the part.
* **Between passes** only the commanded normal force changes (alternating 20 N / 40 N). Spindle speed is part of the action but constant here.

### 3.2 Pad contact (`swt/pad.py`)
At each station the pad face is held tangent to the surface at the pad centre; a point under the pad lies a gap `g` below the pad plane (0 at the centre, {{model_info.pad_rim_gap_mm:.2f}} mm at the rim of the 125 mm pad). Pushing the pad in by `d` gives a Winkler pressure and a force balance:

```
p = k_pad · max(0, d − g)          [MPa]
Σ_i p_i · A_i = F                  (A_i = pad-face area over grid node i)
```

The balance is piecewise linear in `d` and is solved exactly (sorted gaps, one linear piece); a Brent solver gives the same root in the tests. A stiff pad touches a thin strip under its centre; a soft pad wraps around the curvature.

**Stiffness prior.** At {{contact_sweep.reference_force_N:.0f}} N, contact runs from **{{contact_sweep.contact_fraction_at_prior_bounds.30N.stiff_k_high:pct1}}% of the pad face for the stiffest pad to {{contact_sweep.contact_fraction_at_prior_bounds.30N.soft_k_low:pct1}}% for the softest**, reproducing the 5–97% range quoted in GrayMatter's post (their geometry is not published, so this calibrates the prior to the quoted range rather than reproducing their setup). At 20 N and 40 N the range is {{contact_sweep.contact_fraction_at_prior_bounds.20N.stiff_k_high:pct1}}–{{contact_sweep.contact_fraction_at_prior_bounds.20N.soft_k_low:pct1}}% and {{contact_sweep.contact_fraction_at_prior_bounds.40N.stiff_k_high:pct1}}–{{contact_sweep.contact_fraction_at_prior_bounds.40N.soft_k_low:pct1}}%; on a {{contact_sweep.fine_grid_reference.grid_mm:g}} mm grid the 30 N range is {{contact_sweep.fine_grid_reference.stiff_k_high:pct1}}–{{contact_sweep.fine_grid_reference.soft_k_low:pct1}}% (the stiff end is quantised by the 1 mm grid). The softest pads indent {{model_info.pad_penetration_at_prior_bounds.20N.soft_k_low_mm:.1f}}–{{model_info.pad_penetration_at_prior_bounds.40N.soft_k_low_mm:.1f}} mm at 20–40 N, beyond where a linear law holds for real foam; the realistic world's foam pad addresses this (section 3.6).

![Contact fraction vs stiffness](figures/fig1_contact_vs_stiffness.png)

### 3.3 Removal (`swt/process.py`): Preston's law with wear during the pass
```
dh = K · p · v · dt
```
`v` is the RMS sliding speed of a random-orbital sander (5 mm orbit at 8000 rpm plus free pad rotation at 2% of spindle speed), `v(r) = sqrt((π·D_orb·f)² + (2π·f·0.02·r)²)`, from {{model_info.sliding_speed_centre_mm_s:.0f}} mm/s at the pad centre to {{model_info.sliding_speed_rim_mm_s:.0f}} mm/s at the rim. With nominal parameters a fresh abrasive removes on average {{model_info.nominal_fresh_mean_removal_um.20N:.1f}} µm at 20 N and {{model_info.nominal_fresh_mean_removal_um.40N:.1f}} µm at 40 N.

**Wear.** `K = K0 · exp(−λ·W)`, with `W` the cumulative removed volume (by analogy with the grinding G-ratio; an assumption, untested here). Wear acts *during* the pass: within a pass `K(U) = K_start / (1 + λ·K_start·U)`, where `U` is the volume the pass would have removed at unit effectiveness so far. It is applied at the resolution of one raster line, using each line's exact average effectiveness, so the volume of the removal map is exactly the volume that drives the wear (a test checks this to 1e-9). Between passes a small multiplicative fluctuation stands in for abrasive variability:

```
K_{n+1} = K_n · exp(−λ·ΔV_n + σ_w·ξ_n),   ξ_n ~ N(0, 1),   σ_w = 0.01
```

### 3.4 Scan (`swt/scan.py`)
The scanner reports each pass's removal on a 2 mm sub-grid ({{model_info.scan_points:d}} points, read as profiles along y, one every 2 mm in x) with independent Gaussian noise, σ = 2 µm. In the realistic world it also has artefacts (section 3.6).

### 3.5 Hidden truth and priors
| Parameter | Prior | Notes |
|---|---|---|
| `k_pad` | log-uniform on [{{contact_sweep.k_low:g}}, {{contact_sweep.k_high:g}}] N/mm³ | {{contact_sweep.contact_fraction_at_prior_bounds.30N.soft_k_low:pct0}}% to {{contact_sweep.contact_fraction_at_prior_bounds.30N.stiff_k_high:pct0}}% contact at 30 N |
| `K0` (fresh effectiveness) | log-normal, median 6.0e-5 mm²/N, σ_log 0.25 | |
| `λ` (wear rate) | log-normal, median 2.5e-4 1/mm³, σ_log 0.35 | |
| wear fluctuation | σ_w = 0.01 per pass | the tracker's starting assumption |

Each draw id fixes `k_pad`, `K0`, `λ` (and the realistic world's force error) through its own random streams, so every world and schedule sees the same hidden parameters for the same draw. Draws 0–99 are used for the reported experiments and draws 200–219 only for tuning (section 3.8).

### 3.6 The realistic world: five effects no estimator models
| Effect | What the hidden process does | Size |
|---|---|---|
| **Foam pad** | `p = k·δ / (1 − δ/h)`, h = {{model_info.worlds.realistic.foam.thickness_mm:g}} mm: the pad stiffens as it densifies (same small-strain stiffness `k`) | softest pad at 40 N: peak strain {{model_info.worlds.realistic.foam.at_max_force.soft_k_low.max_strain_foam:pct0}}%, peak pressure ×{{model_info.worlds.realistic.foam.at_max_force.soft_k_low.peak_pressure_ratio_foam_over_linear:.2f}}, contact {{model_info.worlds.realistic.foam.at_max_force.soft_k_low.contact_fraction_foam:pct0}}% instead of {{model_info.worlds.realistic.foam.at_max_force.soft_k_low.contact_fraction_linear:pct0}}%; negligible for mid and stiff pads (×{{model_info.worlds.realistic.foam.at_max_force.mid.peak_pressure_ratio_foam_over_linear:.2f}} at the nominal pad) |
| **Force-calibration error** | actual force = gain × commanded, one gain per run | log-normal, σ_log = {{model_info.worlds.realistic.force_gain_sigma_log:g}} |
| **Force-control ripple** | each raster line runs at its own force, AR(1) along the pass (first-order exposure model, checked against the exact model to < 0.1%) | σ = {{model_info.worlds.realistic.force_ripple:pct0}}% per line, correlation {{model_info.worlds.realistic.ripple_corr:g}} between lines |
| **Two-stage wear** | `K/K0 = (1−f)·e^{−λW} + f·e^{−rλW}`: a fast break-in of the sharpest grit tips plus slow dulling (f = {{model_info.worlds.realistic.two_stage.fast_fraction:g}}, r = {{model_info.worlds.realistic.two_stage.fast_ratio:g}}), integrated within the pass | `K` halves after {{model_info.worlds.realistic.two_stage.volume_to_half_K0_mm3.two_stage:.0f}} mm³ instead of {{model_info.worlds.realistic.two_stage.volume_to_half_K0_mm3.single:.0f}} mm³ (nominal λ) |
| **Scanner artefacts** | per-scan misregistration of the whole map; a constant offset per scan profile; outliers; missing points (isolated and 8-point gaps along profiles) | registration σ = {{model_info.worlds.realistic.scan_artefacts.registration_sigma_mm:g}} mm per axis; profile offset σ = {{model_info.worlds.realistic.scan_artefacts.profile_bias_um:g}} µm; {{model_info.worlds.realistic.scan_artefacts.outlier_fraction:pct1}}% outliers of σ = {{model_info.worlds.realistic.scan_artefacts.outlier_um:g}} µm; {{model_info.worlds.realistic.scan_artefacts.dropout_fraction:pct0}}% missing |

The force error and ripple appear in the breakdown as one group ("force errors"). The **oracle** reference always uses the hidden process's own physics.

### 3.7 Estimators (`swt/estimators.py`)
All four see the commanded action of every pass and the scans, and nothing else; before each pass each predicts that pass's removal at the scan points.

**A: nominal.** Prior-mean parameters (geometric-mean stiffness {{model_info.nominal_parameters.k_pad:.3g}} N/mm³, prior medians of `K0` and `λ`), wear propagated with its own predicted volume, never updated.

**B: calibrate-once.** Least-squares `k_pad` and `K` from the first scan (`K` profiled out, 1-D search over log `k_pad`), then frozen, no wear model. The "learn the pad once" baseline.

**D: refit each pass** (the strong engineering baseline). The same least-squares fit on *every* scan; the wear rate is the slope of a straight line through log `K_i` against cumulative removed volume, and the next pass uses the latest `k_pad` and `K` extrapolated with that slope. Its abrasive-change forecast extrapolates the same line. It gives point predictions only.

**C: joint tracker.** A sequential Monte Carlo filter over log `k_pad` (static) and the paths of log `K` and log `λ` (one value per pass), {{info.config.filter.n_particles:d}} particles:
* *Process model:* within-pass wear as in 3.3; between passes `log K` follows the wear law plus noise of s.d. `σ_w`, and `log λ` takes a random-walk step of s.d. `δ` (the *wear-rate drift*), so the wear rate can follow the recent decay when the true law is not exponential. `δ` is tuned on held-out draws (3.8); `δ = 0` is the constant-rate model.
* *Update:* the scan likelihood is applied in tempered stages that keep the effective sample size at 75% of N. After each stage the particles are resampled systematically and moved by Metropolis–Hastings steps that leave the exact path posterior invariant: a joint random walk on log `k_pad` and a common shift of the whole log `λ` path, local moves on log `λ` at the last two passes, and moves on log `K` at pass 1 (= `K0`) and at the current pass. Every past scan is kept as O(1) sufficient statistics, so the full path posterior is evaluated in every move and the static stiffness does not degenerate.
* *Robust measurement model:* (i) points more than 5 robust s.d. from C's own prediction for the pass are rejected (on pass 1, from the fitted map, with one redo); (ii) a random offset per scan profile is added to the noise model when the profile means vary significantly more than the noise explains (variance estimated from the residuals, folded exactly into the sufficient statistics via the Woodbury identity); (iii) the noise level is re-estimated from the residuals and inflated when they are spatially correlated. In the matched world they almost never act (median {{robustness.worlds.matched.rejected_points_C.median:.0f}} rejected points per scan, {{robustness.worlds.matched.passes_redone_C:d}} of the {{robustness.n_draws:d}} × 20 updates redone); in the realistic world the median is {{robustness.worlds.realistic.rejected_points_C.median:.0f}} rejected points per scan and {{robustness.worlds.realistic.passes_redone_C:d}} updates were redone.
* *Adaptive process noise:* `σ_w` is replaced by the variance implied by C's own one-step forecast errors of log `K` (shrunk towards 0.01 with a weight of 4 passes, never below it), so the predictive spread widens when `K` changes from pass to pass more than the model expects.
* *Speed:* removal depends on stiffness only through `η = F/k_pad`; maps at the scan points are tabulated once on {{model_info.surrogate.n_eta:d}} log-spaced η values, with the within-pass wear expanded to third order. Against the exact model the relative RMS error is at most {{model_info.surrogate.max_relative_rms_error:.1e}} (median {{model_info.surrogate.median_relative_rms_error:.1e}}, {{model_info.surrogate.n_checks:d}} checks). The hidden process and A, B use the exact model.
* *Outputs:* posterior median and 90% interval of `k_pad`, `K0`, current `K` and current `λ`; a 90% predictive interval of the next pass's mean removal; and the predicted abrasive-change pass with a ≥90% band (particles rolled forward with random future wear and wear-rate drift).

**Oracle (reference).** Knows the hidden process's physics and its true state after the previous pass, but not the wear fluctuation drawn since then, nor the coming pass's force ripple. Its error is a floor in expectation; on single passes C can beat it by luck.

### 3.8 Tuning on held-out draws
The wear-rate drift `δ` is the only tuned setting. `run-all` runs C with each `δ` of {{info.config.tuning.rate_drift.0:g}}, {{info.config.tuning.rate_drift.1:g}}, {{info.config.tuning.rate_drift.2:g}}, {{info.config.tuning.rate_drift.3:g}} and {{info.config.tuning.rate_drift.4:g}} on held-out draws {{tuning.first_draw:d}}–{{tuning.last_draw:d}} in the matched and realistic worlds, and keeps the value with the lowest mean *interval score* of the 90% predictive interval of next-pass mean removal (a proper scoring rule: interval width plus 20× any miss; Gneiting & Raftery 2007). It chose **δ = {{tuning.chosen_rate_drift:g}}**. Other settings (75% ESS target, 2 MH sweeps, outlier threshold 5, 4 passes of prior weight for the process noise) were fixed during development, which used held-out draws 200–239 and short runs on draws 0–3; no setting was changed after the full results on draws 0–99 had been seen.
## 4. Results

All numbers come from `results/` and were produced by `python -m swt run-all` with seed {{info.config.seed:d}}. Unless a section says otherwise, every run has 20 passes, forces alternating 20/40 N, 2 µm scan noise and the 50%-of-K0 abrasive-change threshold. Errors are RMSE over the {{model_info.scan_points:d}} scan points against the simulator's noise-free removal (not against the scan).

**At a glance** ({{robustness.n_draws:d}} hidden truths per world, pass 20; RMSE median [IQR] in µm):

| | A: nominal | B: calibrate-once | D: refit each pass | C: joint tracker | oracle |
|---|---|---|---|---|---|
| matched world | {{robustness.worlds.matched.final_rmse_um.A.median:.2f}} [{{robustness.worlds.matched.final_rmse_um.A.q25:.2f}}–{{robustness.worlds.matched.final_rmse_um.A.q75:.2f}}] | {{robustness.worlds.matched.final_rmse_um.B.median:.2f}} [{{robustness.worlds.matched.final_rmse_um.B.q25:.2f}}–{{robustness.worlds.matched.final_rmse_um.B.q75:.2f}}] | {{robustness.worlds.matched.final_rmse_um.D.median:.3f}} [{{robustness.worlds.matched.final_rmse_um.D.q25:.3f}}–{{robustness.worlds.matched.final_rmse_um.D.q75:.3f}}] | **{{robustness.worlds.matched.final_rmse_um.C.median:.3f}}** [{{robustness.worlds.matched.final_rmse_um.C.q25:.3f}}–{{robustness.worlds.matched.final_rmse_um.C.q75:.3f}}] | {{robustness.worlds.matched.final_rmse_um.oracle.median:.3f}} |
| realistic world | {{robustness.worlds.realistic.final_rmse_um.A.median:.2f}} [{{robustness.worlds.realistic.final_rmse_um.A.q25:.2f}}–{{robustness.worlds.realistic.final_rmse_um.A.q75:.2f}}] | {{robustness.worlds.realistic.final_rmse_um.B.median:.2f}} [{{robustness.worlds.realistic.final_rmse_um.B.q25:.2f}}–{{robustness.worlds.realistic.final_rmse_um.B.q75:.2f}}] | {{robustness.worlds.realistic.final_rmse_um.D.median:.3f}} [{{robustness.worlds.realistic.final_rmse_um.D.q25:.3f}}–{{robustness.worlds.realistic.final_rmse_um.D.q75:.3f}}] | **{{robustness.worlds.realistic.final_rmse_um.C.median:.3f}}** [{{robustness.worlds.realistic.final_rmse_um.C.q25:.3f}}–{{robustness.worlds.realistic.final_rmse_um.C.q75:.3f}}] | {{robustness.worlds.realistic.final_rmse_um.oracle.median:.3f}} |

| C's calibration (target 90%) | matched | realistic |
|---|---|---|
| next-pass mean removal inside the 90% predictive interval, passes 2–20 | {{robustness.worlds.matched.predictive_90.passes_2_on.inside:d}} / {{robustness.worlds.matched.predictive_90.passes_2_on.n:d}} ({{robustness.worlds.matched.predictive_90.passes_2_on.fraction:pct1}}%) | {{robustness.worlds.realistic.predictive_90.passes_2_on.inside:d}} / {{robustness.worlds.realistic.predictive_90.passes_2_on.n:d}} ({{robustness.worlds.realistic.predictive_90.passes_2_on.fraction:pct1}}%) |
| … truth below / above the interval | {{robustness.worlds.matched.predictive_90.passes_2_on.below:d}} / {{robustness.worlds.matched.predictive_90.passes_2_on.above:d}} | {{robustness.worlds.realistic.predictive_90.passes_2_on.below:d}} / {{robustness.worlds.realistic.predictive_90.passes_2_on.above:d}} |
| … passes 2–5 / passes 11–20 | {{robustness.worlds.matched.predictive_90.passes_2_to_5.fraction:pct0}}% / {{robustness.worlds.matched.predictive_90.passes_11_on.fraction:pct0}}% | {{robustness.worlds.realistic.predictive_90.passes_2_to_5.fraction:pct0}}% / {{robustness.worlds.realistic.predictive_90.passes_11_on.fraction:pct0}}% |
| true parameter inside the 90% interval at pass 20: k_pad / λ / K / K0 (of {{robustness.n_draws:d}}) | {{robustness.worlds.matched.coverage_90.k_pad.inside:d}} / {{robustness.worlds.matched.coverage_90.lam.inside:d}} / {{robustness.worlds.matched.coverage_90.K.inside:d}} / {{robustness.worlds.matched.coverage_90.K0.inside:d}} | {{robustness.worlds.realistic.coverage_90.k_pad.inside:d}} / {{robustness.worlds.realistic.coverage_90.lam.inside:d}} / {{robustness.worlds.realistic.coverage_90.K.inside:d}} / {{robustness.worlds.realistic.coverage_90.K0.inside:d}} (effective parameters; see 4.2) |

With 100 draws, the binomial standard error of a calibrated 90% parameter coverage is {{robustness.worlds.matched.coverage_90.binomial_se_if_calibrated:pct0}} points. The λ interval is wider than needed in the matched world because C allows the wear rate to drift while the matched truth's rate is constant.

### 4.1 Main run: one hidden truth in both worlds

The draw is chosen by a rule that looks only at the sampled truths: among the {{robustness.n_draws:d}} robustness draws, the one at the median standardised distance from the prior centre in (log k_pad, log K0, log λ). It is draw {{main_run.draw:d}}: `k_pad` = {{main_run.worlds.matched.truth.k_pad:.4f}} N/mm³, `K0` = {{main_run.worlds.matched.truth.K0:.3g}} mm²/N, `λ` = {{main_run.worlds.matched.truth.lam:.3g}} 1/mm³, and in the realistic world a force gain of {{main_run.worlds.realistic.truth.force_gain:.3f}}. Its abrasive crosses 50% at pass {{main_run.worlds.matched.true_crossing_pass:d}} in the matched world and at pass {{main_run.worlds.realistic.true_crossing_pass:d}} in the realistic world (the break-in brings it forward); A's nominal model says pass {{main_run.worlds.realistic.A_crossing_pass:d}}.

| Removal-map RMSE [µm] | A | B | D | C | oracle |
|---|---|---|---|---|---|
| realistic, pass {{main_run.worlds.realistic.early_pass:d}} | {{main_run.worlds.realistic.early_rmse_um.A:.2f}} | {{main_run.worlds.realistic.early_rmse_um.B:.2f}} | {{main_run.worlds.realistic.early_rmse_um.D:.2f}} | {{main_run.worlds.realistic.early_rmse_um.C:.2f}} | {{main_run.worlds.realistic.early_rmse_um.oracle:.2f}} |
| realistic, pass {{main_run.worlds.realistic.final_pass:d}} | {{main_run.worlds.realistic.final_rmse_um.A:.2f}} | {{main_run.worlds.realistic.final_rmse_um.B:.2f}} | {{main_run.worlds.realistic.final_rmse_um.D:.3f}} | {{main_run.worlds.realistic.final_rmse_um.C:.3f}} | {{main_run.worlds.realistic.final_rmse_um.oracle:.3f}} |
| matched, pass {{main_run.worlds.matched.final_pass:d}} | {{main_run.worlds.matched.final_rmse_um.A:.2f}} | {{main_run.worlds.matched.final_rmse_um.B:.2f}} | {{main_run.worlds.matched.final_rmse_um.D:.3f}} | {{main_run.worlds.matched.final_rmse_um.C:.3f}} | {{main_run.worlds.matched.final_rmse_um.oracle:.3f}} |

![Removal maps](figures/fig2_removal_maps.png)

* **B** is good right after its calibration but never learns that the paper is dulling, so its error grows with every pass.
* **A** has the wear law but the wrong parameters.
* **D and C** both follow the wear. On this one draw they are close, and D is slightly better at pass 20 in the matched world; single passes are noisy, and section 4.2 has the statistics over 100 draws.
* **Parameters at pass 20 (figure 4):**

  | | C: median [90% interval], matched world | C: median [90% interval], realistic world | truth |
  |---|---|---|---|
  | `k_pad` [N/mm³] | {{main_run.worlds.matched.C_final.k_pad.median:.4f}} [{{main_run.worlds.matched.C_final.k_pad.lo:.4f}}, {{main_run.worlds.matched.C_final.k_pad.hi:.4f}}] | {{main_run.worlds.realistic.C_final.k_pad.median:.4f}} [{{main_run.worlds.realistic.C_final.k_pad.lo:.4f}}, {{main_run.worlds.realistic.C_final.k_pad.hi:.4f}}] | {{main_run.worlds.matched.truth.k_pad:.4f}} |
  | `λ` [1/mm³] | {{main_run.worlds.matched.C_final.lam.median:.3g}} [{{main_run.worlds.matched.C_final.lam.lo:.3g}}, {{main_run.worlds.matched.C_final.lam.hi:.3g}}] | {{main_run.worlds.realistic.C_final.lam.median:.3g}} [{{main_run.worlds.realistic.C_final.lam.lo:.3g}}, {{main_run.worlds.realistic.C_final.lam.hi:.3g}}] | {{main_run.worlds.matched.truth.lam:.3g}} |
  | `K0` [mm²/N] | {{main_run.worlds.matched.C_final.K0.median:.3g}} [{{main_run.worlds.matched.C_final.K0.lo:.3g}}, {{main_run.worlds.matched.C_final.K0.hi:.3g}}] | {{main_run.worlds.realistic.C_final.K0.median:.3g}} [{{main_run.worlds.realistic.C_final.K0.lo:.3g}}, {{main_run.worlds.realistic.C_final.K0.hi:.3g}}] | {{main_run.worlds.matched.truth.K0:.3g}} |

  In the realistic world the same tracker converges on an *effective* stiffness and an effective wear rate that is high during the break-in and falls later: the best linear-pad, exponential-wear description of a process that is neither, with the force gain absorbed into the effectiveness. Its map predictions remain good because they only need the effective values to reproduce the removal, not to equal the true parameters.

![Parameter tracking](figures/fig4_parameter_tracking.png)

### 4.2 Robustness: {{robustness.n_draws:d}} hidden truths per world

* **Map accuracy.** C has the lowest median error of the four predictors at every pass after the first in the matched world, and at almost every pass in the realistic world, where D comes close (figure 3). At pass 20, D's error is {{robustness.worlds.matched.final_rmse_D_over_C.median:.2f}}× C's (median ratio) in the matched world and {{robustness.worlds.realistic.final_rmse_D_over_C.median:.2f}}× in the realistic world; C is better than D in {{robustness.worlds.matched.draws_C_better_than_D_at_final_pass:d}} and {{robustness.worlds.realistic.draws_C_better_than_D_at_final_pass:d}} of {{robustness.n_draws:d}} draws at pass 20, and in {{robustness.worlds.matched.draws_C_better_than_D_mean_over_passes:d}} and {{robustness.worlds.realistic.draws_C_better_than_D_mean_over_passes:d}} averaged over passes. In the realistic world C's pass-20 error is {{robustness.worlds.realistic.final_rmse_C_over_oracle.median:.2f}}× the oracle's (median) and {{robustness.worlds.realistic.final_rmse_C_relative_to_true_removal.median:pct1}}% of the true removal.
* **The advantage over D shrinks under mismatch.** D takes `k_pad` and `K` from the latest scan alone and fits a straight line for the wear; C pools every scan in one posterior and models wear within the pass. In the matched world that structure is exactly right; in the realistic world part of the advantage is lost to the effects neither models.
* **Calibration.** In the matched world the predictive intervals and parameter intervals are close to nominal. In the realistic world the predictive intervals under-cover, most of all in the first passes, before C's adaptive noise estimate has seen enough forecast errors, and the misses are mostly on one side (the truth above the interval: the abrasive keeps cutting better than an exponential law fitted through the break-in predicts). Parameter intervals in the realistic world describe effective parameters, so their "coverage" of the true foam-pad stiffness, gain-scaled `K` or two-stage `λ` is not expected to be 90% and is reported for completeness.

### 4.3 What each unmodelled effect costs

Each effect of the realistic world was also run alone on the first {{world_breakdown.n_draws:d}} draws (same draws in every world; figure 7, left).

| World ({{world_breakdown.n_draws:d}} draws, pass 20) | C RMSE [µm] | oracle [µm] | C / oracle | D RMSE [µm] | predictive coverage | crossing error, last pass before (mean, passes) |
|---|---|---|---|---|---|---|
| matched | {{world_breakdown.worlds.matched.final_rmse_um.C.median:.3f}} | {{world_breakdown.worlds.matched.final_rmse_um.oracle.median:.3f}} | {{world_breakdown.worlds.matched.final_rmse_C_over_oracle.median:.2f}} | {{world_breakdown.worlds.matched.final_rmse_um.D.median:.3f}} | {{world_breakdown.worlds.matched.predictive_90.passes_2_on.fraction:pct0}}% | {{world_breakdown.worlds.matched.crossing.last_pass_before_crossing.mean_abs_err_C_passes:.2f}} |
| foam pad only | {{world_breakdown.worlds.only_pad.final_rmse_um.C.median:.3f}} | {{world_breakdown.worlds.only_pad.final_rmse_um.oracle.median:.3f}} | {{world_breakdown.worlds.only_pad.final_rmse_C_over_oracle.median:.2f}} | {{world_breakdown.worlds.only_pad.final_rmse_um.D.median:.3f}} | {{world_breakdown.worlds.only_pad.predictive_90.passes_2_on.fraction:pct0}}% | {{world_breakdown.worlds.only_pad.crossing.last_pass_before_crossing.mean_abs_err_C_passes:.2f}} |
| force errors only | {{world_breakdown.worlds.only_force.final_rmse_um.C.median:.3f}} | {{world_breakdown.worlds.only_force.final_rmse_um.oracle.median:.3f}} | {{world_breakdown.worlds.only_force.final_rmse_C_over_oracle.median:.2f}} | {{world_breakdown.worlds.only_force.final_rmse_um.D.median:.3f}} | {{world_breakdown.worlds.only_force.predictive_90.passes_2_on.fraction:pct0}}% | {{world_breakdown.worlds.only_force.crossing.last_pass_before_crossing.mean_abs_err_C_passes:.2f}} |
| two-stage wear only | {{world_breakdown.worlds.only_wear.final_rmse_um.C.median:.3f}} | {{world_breakdown.worlds.only_wear.final_rmse_um.oracle.median:.3f}} | {{world_breakdown.worlds.only_wear.final_rmse_C_over_oracle.median:.2f}} | {{world_breakdown.worlds.only_wear.final_rmse_um.D.median:.3f}} | {{world_breakdown.worlds.only_wear.predictive_90.passes_2_on.fraction:pct0}}% | {{world_breakdown.worlds.only_wear.crossing.last_pass_before_crossing.mean_abs_err_C_passes:.2f}} |
| scanner artefacts only | {{world_breakdown.worlds.only_scan.final_rmse_um.C.median:.3f}} | {{world_breakdown.worlds.only_scan.final_rmse_um.oracle.median:.3f}} | {{world_breakdown.worlds.only_scan.final_rmse_C_over_oracle.median:.2f}} | {{world_breakdown.worlds.only_scan.final_rmse_um.D.median:.3f}} | {{world_breakdown.worlds.only_scan.predictive_90.passes_2_on.fraction:pct0}}% | {{world_breakdown.worlds.only_scan.crossing.last_pass_before_crossing.mean_abs_err_C_passes:.2f}} |
| realistic (all five) | {{world_breakdown.worlds.realistic.final_rmse_um.C.median:.3f}} | {{world_breakdown.worlds.realistic.final_rmse_um.oracle.median:.3f}} | {{world_breakdown.worlds.realistic.final_rmse_C_over_oracle.median:.2f}} | {{world_breakdown.worlds.realistic.final_rmse_um.D.median:.3f}} | {{world_breakdown.worlds.realistic.predictive_90.passes_2_on.fraction:pct0}}% | {{world_breakdown.worlds.realistic.crossing.last_pass_before_crossing.mean_abs_err_C_passes:.2f}} |

* **Force errors** raise everyone's error, the oracle's included: the line-to-line ripple is unpredictable, and it is the largest single cause of C's under-coverage (the misses are on both sides: intervals too narrow until the adaptive process noise catches up).
* **Two-stage wear** biases the predictions (misses mostly on one side) and the long-range crossing forecasts (section 4.4).
* **Scanner artefacts:** C does not correct misregistration; per-profile offsets are absorbed by its offset model and spikes by its gating (the tests check both). Its misses in this world also lean towards the truth being above the interval ({{world_breakdown.worlds.only_scan.predictive_90.passes_2_on.above:d}} above, {{world_breakdown.worlds.only_scan.predictive_90.passes_2_on.below:d}} below), a bias I have not traced.
* **The foam pad** changes the map noticeably only for soft pads and costs the least of the five; stiffness becomes an effective value.

![Mismatch and calibration](figures/fig7_mismatch_and_calibration.png)

### 4.4 Abrasive-change prediction (threshold: K < 50% of K0)

| Prediction made | world | n | C mean abs. error [passes] | D | A (nominal) | C within ±1 pass | C's ≥90% band contains the truth |
|---|---|---|---|---|---|---|---|
| after pass 2 | matched | {{abrasive_change.worlds.matched.by_decision.after_pass_2.n:d}} | {{abrasive_change.worlds.matched.by_decision.after_pass_2.mean_abs_err_C_passes:.2f}} | {{abrasive_change.worlds.matched.by_decision.after_pass_2.mean_abs_err_D_passes:.2f}} | {{abrasive_change.worlds.matched.by_decision.after_pass_2.mean_abs_err_A_passes:.2f}} | {{abrasive_change.worlds.matched.by_decision.after_pass_2.within_1_pass_C:d}} | {{abrasive_change.worlds.matched.by_decision.after_pass_2.band_coverage_C:pct0}}% |
| last pass before the crossing | matched | {{abrasive_change.worlds.matched.by_decision.last_pass_before_crossing.n:d}} | {{abrasive_change.worlds.matched.by_decision.last_pass_before_crossing.mean_abs_err_C_passes:.2f}} | {{abrasive_change.worlds.matched.by_decision.last_pass_before_crossing.mean_abs_err_D_passes:.2f}} | {{abrasive_change.worlds.matched.by_decision.last_pass_before_crossing.mean_abs_err_A_passes:.2f}} | {{abrasive_change.worlds.matched.by_decision.last_pass_before_crossing.within_1_pass_C:d}} | {{abrasive_change.worlds.matched.by_decision.last_pass_before_crossing.band_coverage_C:pct0}}% |
| after pass 2 | realistic | {{abrasive_change.worlds.realistic.by_decision.after_pass_2.n:d}} | {{abrasive_change.worlds.realistic.by_decision.after_pass_2.mean_abs_err_C_passes:.2f}} | {{abrasive_change.worlds.realistic.by_decision.after_pass_2.mean_abs_err_D_passes:.2f}} | {{abrasive_change.worlds.realistic.by_decision.after_pass_2.mean_abs_err_A_passes:.2f}} | {{abrasive_change.worlds.realistic.by_decision.after_pass_2.within_1_pass_C:d}} | {{abrasive_change.worlds.realistic.by_decision.after_pass_2.band_coverage_C:pct0}}% |
| after pass 5 | realistic | {{abrasive_change.worlds.realistic.by_decision.after_pass_5.n:d}} | {{abrasive_change.worlds.realistic.by_decision.after_pass_5.mean_abs_err_C_passes:.2f}} | {{abrasive_change.worlds.realistic.by_decision.after_pass_5.mean_abs_err_D_passes:.2f}} | {{abrasive_change.worlds.realistic.by_decision.after_pass_5.mean_abs_err_A_passes:.2f}} | {{abrasive_change.worlds.realistic.by_decision.after_pass_5.within_1_pass_C:d}} | {{abrasive_change.worlds.realistic.by_decision.after_pass_5.band_coverage_C:pct0}}% |
| last pass before the crossing | realistic | {{abrasive_change.worlds.realistic.by_decision.last_pass_before_crossing.n:d}} | {{abrasive_change.worlds.realistic.by_decision.last_pass_before_crossing.mean_abs_err_C_passes:.2f}} | {{abrasive_change.worlds.realistic.by_decision.last_pass_before_crossing.mean_abs_err_D_passes:.2f}} | {{abrasive_change.worlds.realistic.by_decision.last_pass_before_crossing.mean_abs_err_A_passes:.2f}} | {{abrasive_change.worlds.realistic.by_decision.last_pass_before_crossing.within_1_pass_C:d}} | {{abrasive_change.worlds.realistic.by_decision.last_pass_before_crossing.band_coverage_C:pct0}}% |

Only predictions made before the true crossing count; the true crossing is simulated beyond pass 20 when needed. In the matched world C is more accurate than A at every lead time and than D at every lead but one, where they tie (figure 5, right). In the realistic world, early forecasts are biased early (mean signed error after pass 2: {{abrasive_change.worlds.realistic.by_decision.after_pass_2.mean_err_C_passes:+.2f}} passes): the break-in makes the abrasive look as if it wears faster than it will, and neither C (whose wear-rate drift lets λ fall only gradually) nor D's straight-line fit anticipates the slow-down. At long leads this is no better than the fixed nominal schedule (figure 5, right), and at intermediate leads D's straight-line forecast is slightly better than C's: on the {{abrasive_change.worlds.realistic.by_lead.population:d}} realistic truths whose crossing every lead can see, the mean error 5 passes ahead is {{abrasive_change.worlds.realistic.by_lead.rows.4.mean_abs_err_C:.2f}} passes for C and {{abrasive_change.worlds.realistic.by_lead.rows.4.mean_abs_err_D:.2f}} for D, against {{abrasive_change.worlds.realistic.by_lead.rows.0.mean_abs_err_C:.2f}} and {{abrasive_change.worlds.realistic.by_lead.rows.0.mean_abs_err_D:.2f}} one pass ahead. Close to the crossing, C is the most accurate and its band is calibrated.

![Abrasive change](figures/fig5_abrasive_change.png)

### 4.5 Tuning: what the drifting wear rate buys

| wear-rate drift δ (held-out draws {{tuning.first_draw:d}}–{{tuning.last_draw:d}}) | 0 (constant rate) | chosen: {{tuning.chosen_rate_drift:g}} |
|---|---|---|
| predictive coverage, matched | {{tuning.constant_rate.worlds.matched.coverage:pct1}}% | {{tuning.chosen.worlds.matched.coverage:pct1}}% |
| predictive coverage, realistic | {{tuning.constant_rate.worlds.realistic.coverage:pct1}}% | {{tuning.chosen.worlds.realistic.coverage:pct1}}% |
| median relative width of the 90% interval, realistic | {{tuning.constant_rate.worlds.realistic.relative_width:pct2}}% | {{tuning.chosen.worlds.realistic.relative_width:pct2}}% |
| mean interval score (lower is better; both worlds) | {{tuning.constant_rate.mean_interval_score:.4f}} | {{tuning.chosen.mean_interval_score:.4f}} |

Larger drifts raise the realistic world's coverage further but widen the intervals more than they gain (figure 7, right), so the proper scoring rule stops at {{tuning.chosen_rate_drift:g}}. The choice is made inside `run-all`, on draws not used anywhere else.

### 4.6 Identifiability ablation: constant 30 N instead of alternating 20/40 N

On the first {{ablation.n_draws:d}} draws of the {{ablation.world}} world, a constant 30 N schedule (same mean force) was compared with the alternating one, and a re-run of the alternating schedule with a different filter seed gives the spread expected from Monte Carlo noise alone.

| Paired over draws, pass 20 | constant ÷ alternating | other seed ÷ alternating |
|---|---|---|
| k_pad 90% width, median ratio (draws wider) | {{ablation.paired_width_ratio_constant_over_alternating.k_pad.median:.3f}} ({{ablation.paired_width_ratio_constant_over_alternating.draws_constant_wider_k_pad:d}} of {{ablation.n_draws:d}}) | {{ablation.paired_width_ratio_control_over_alternating.k_pad.median:.3f}} ({{ablation.paired_width_ratio_control_over_alternating.draws_control_wider_k_pad:d}}) |
| λ 90% width, median ratio (draws wider) | {{ablation.paired_width_ratio_constant_over_alternating.lam.median:.3f}} ({{ablation.paired_width_ratio_constant_over_alternating.draws_constant_wider_lam:d}}) | {{ablation.paired_width_ratio_control_over_alternating.lam.median:.3f}} ({{ablation.paired_width_ratio_control_over_alternating.draws_control_wider_lam:d}}) |

**This did not show what was expected.** Alternating the force did not narrow the stiffness interval at all. Because the force is commanded and known, a single force level already pins `F / k_pad` from the shape of one removal map; a second force adds a second view of the same thing. Alternating did narrow the wear-rate interval slightly and consistently, because two forces remove different volumes and so trace the wear law at two rates. Varying the force would matter if something else also changed the contact shape (an unknown force offset, uncertain curvature, a drifting pad); with the realistic world's force-calibration error it could help separate gain from effectiveness, which was not tested.

![Ablation and noise](figures/fig6_ablation_and_noise.png)

### 4.7 Scan-noise sensitivity ({{noise_sensitivity.world}} world, first {{noise_sensitivity.n_draws:d}} draws)

| scan noise σ [µm] | 1 | 2 | 5 | 10 |
|---|---|---|---|---|
| C pass-20 RMSE, median [µm] | {{noise_sensitivity.levels.1.final_rmse_C_um.median:.3f}} | {{noise_sensitivity.levels.2.final_rmse_C_um.median:.3f}} | {{noise_sensitivity.levels.5.final_rmse_C_um.median:.3f}} | {{noise_sensitivity.levels.10.final_rmse_C_um.median:.3f}} |
| oracle, median [µm] | {{noise_sensitivity.levels.1.final_rmse_oracle_um.median:.3f}} | {{noise_sensitivity.levels.2.final_rmse_oracle_um.median:.3f}} | {{noise_sensitivity.levels.5.final_rmse_oracle_um.median:.3f}} | {{noise_sensitivity.levels.10.final_rmse_oracle_um.median:.3f}} |
| predictive coverage | {{noise_sensitivity.levels.1.predictive_90.passes_2_on.fraction:pct0}}% | {{noise_sensitivity.levels.2.predictive_90.passes_2_on.fraction:pct0}}% | {{noise_sensitivity.levels.5.predictive_90.passes_2_on.fraction:pct0}}% | {{noise_sensitivity.levels.10.predictive_90.passes_2_on.fraction:pct0}}% |

In the realistic world C's error barely depends on scan noise up to 10 µm: with thousands of points per scan the noise averages out, and the error is dominated by what neither scans nor the oracle can predict (force ripple, fluctuations) and by model mismatch.

## 5. Limitations

* **Simulated only.** Nothing here has touched a physical part. The realistic world is a richer simulator, not reality: its five effects and their sizes were chosen by me, not measured, and real processes will contain effects that are in neither world (loading and clogging of the abrasive, heat, grit changes, pad wear, robot path errors).
* **Same author for the world and the tracker.** The scanner-offset model and outlier gating address artefacts that I also put into the realistic world; per-profile offsets and spikes are well-known line-scanner artefacts, but a tracker designed without knowing the test world would likely do worse.
* **Under-coverage under mismatch.** C's 90% predictive intervals cover {{robustness.worlds.realistic.predictive_90.passes_2_on.fraction:pct0}}% in the realistic world ({{robustness.worlds.realistic.predictive_90.passes_2_to_5.fraction:pct0}}% in passes 2–5). Decisions taken on those intervals early in an abrasive's life would be over-confident.
* **Parameters are only meaningful in the matched world.** Under mismatch C's stiffness and wear rate are effective values; they should not be read as physical properties of the pad or the abrasive.
* **Long-range abrasive-change forecasts** are poor when the wear law has a break-in phase (section 4.4); only forecasts within a few passes of the crossing are reliable there.
* **Registration error is not corrected.** C treats misregistration as structured noise; it biases the effective stiffness.
* **Winkler-type pad.** Both pad laws are independent springs: no shear coupling, no bending of the backing plate, no tilt over an overhanging edge. The contact uses the nominal (CAD) surface; removal is not fed back into the contact.
* **One abrasive, removal depth only.** One grit and one wear law per world; no roughness or other finish metric.
* **Grid effects.** The 1 mm grid quantises the stiffest contact strips (section 3.2).
* **Surrogate.** C's table has a worst-case relative error of {{model_info.surrogate.max_relative_rms_error:.1e}} against the exact model, at the fastest-wearing draws (third-order expansion of the within-pass wear).
* **Not validated on physical coupons.**

## 6. Next steps that would make it real

1. **Fit the wear law to coupon data.** Scan coupons after every pass at several forces and grits, test whether removed volume or force × distance drives the decay, measure break-in and the real pass-to-pass fluctuation. This decides whether the tracker needs a break-in state.
2. **Add a transient per-pass removal term** (force ripple, local loading) to C's model, separate from the persistent wear, so that early-pass intervals are calibrated instead of waiting for the adaptive noise estimate.
3. **Register each scan to the prediction** (two shifts per scan as nuisance parameters) before updating.
4. **Replace the spring pad** with a finite-element or learned pad model, including tilt over edges.
5. **Run on GrayMatter-style scan data** with real registration error, missing data and noise.
6. **Use the tracker for decisions**: choose the next pass's force or dwell to hit a target removal, and the abrasive change from the predicted crossing distribution and the cost of a bad pass.

## 7. How to reproduce

Requires Python ≥ 3.11. The recorded run used Python {{info.python}} and NumPy {{info.numpy}}; see `results/run_info.json`.

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib, pyyaml, pytest
python -m swt run-all                    # every experiment, every figure, every number in results/
python -m swt report                     # re-render README.md and SUMMARY.md from results/
python -m swt run --config configs/default.yaml    # main 20-pass experiment only (uses results/tuning.json if present)
python -m pytest -q                      # tests
```

* **Runtime.** `run-all` runs {{info.episodes:d}} episodes (tuning included) using `min(4, CPUs)` worker processes. In the recorded run, building the model and surrogate took {{info.context_build_seconds:.0f}} s and the experiments {{info.experiment_seconds:.0f}} s with {{info.workers:d}} workers on a {{info.cpu_count:d}}-CPU machine. `--workers 1` runs serially; results are identical for any worker count because every task has its own seeds.
* **Outputs.** `results/` holds per-pass CSVs, per-draw CSVs, JSON summaries and `run_info.json` (config, versions, source hash). `summary.json` gathers everything. `figures/` holds seven PNGs drawn only from `results/` (`python -m swt figures` redraws them).
* **How the documents are made.** `README.md` and `SUMMARY.md` are rendered from `docs/*.template.md`, in which every number is a lookup into `results/summary.json` or `results/run_info.json`. A test fails if the shipped documents differ from what the templates render from the shipped results.
* **Changing the model.** Edit `configs/default.yaml` (geometry, pad, sander, path, schedule, scan, priors, filter, worlds, experiments, tuning) and rerun. The prose describes the default configuration.

Repository layout:

```
swt/geometry.py     panel height map and raster toolpath
swt/pad.py          contact and force balance (linear and foam pad laws)
swt/process.py      Preston removal, sliding speed, within-pass wear, worlds, hidden-truth simulator
swt/scan.py         noisy, downsampled scans with optional artefacts
swt/surrogate.py    scan-point table and O(1) likelihood terms (missing points, profile offsets)
swt/estimators.py   A nominal, B calibrate-once, D refit each pass, C particle filter
swt/experiments.py  all experiments, including tuning; writes results/
swt/plots.py        all figures, drawn from results/ only
swt/report.py       renders README.md / SUMMARY.md from docs/*.template.md and results/
swt/cli.py          command-line entry point (python -m swt ...)
configs/default.yaml
docs/               README and SUMMARY templates
tests/              pytest suite (incl. an end-to-end run on a tiny configuration)
```
