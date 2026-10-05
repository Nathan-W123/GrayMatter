# sanding-wear-tracker

A simulation study of tracking sanding-pad stiffness and abrasive wear together, so that a sanding process model could stay accurate as the paper dulls and say when to change it. All results are simulated; nothing has been tested on physical parts.

![Prediction error per pass](figures/fig3_rmse_per_pass.png)

## 1. Summary

A model that predicts how much material a robotic sanding pass removes needs two things it cannot measure directly: how the compliant pad presses on the part (set by its stiffness, which is fixed but uncertain) and how sharp the abrasive is (which drops every pass), so a model calibrated once on the first scan predicts the next pass well (median error {{robustness.per_pass.median_rmse_um.B.1:.2f}} µm at pass 2) and then gets steadily worse as the paper wears ({{robustness.final_rmse_um.B.median:.2f}} µm at pass 20). This project simulates a 125 mm compliant pad sanding a curved panel with hidden stiffness and wear parameters, and compares that calibrate-once model, a nominal model with fixed prior-mean parameters, and a particle filter that re-estimates stiffness, current abrasive effectiveness and wear rate from every post-pass scan. Over {{robustness.n_draws:d}} random hidden truths, the filter's median pass-20 error was {{robustness.final_rmse_um.C.median:.3f}} µm, against {{robustness.final_rmse_um.A.median:.2f}} µm for the nominal model and {{robustness.final_rmse_um.B.median:.2f}} µm for the calibrate-once model, with a median true removal of {{robustness.final_true_mean_removal_um.median:.2f}} µm in that pass; that is close to the {{robustness.final_rmse_um.oracle.median:.3f}} µm of an oracle that knows the true parameters but not the random pass-to-pass wear fluctuation, and after two scans the filter also predicts the pass at which effectiveness falls below 50% of fresh with a median error of {{abrasive_change.by_decision.after_pass_2.abs_err_C_passes.median:.0f}} pass (fixed nominal schedule: {{abrasive_change.by_decision.after_pass_2.abs_err_A_passes.median:.0f}}). These are best-case numbers, because the filter shares the simulator's physics, priors and noise levels and errors are measured against the noise-free simulated removal map, and two results were weaker than hoped: alternating the force between passes barely helped (no measurable effect on stiffness, slightly narrower wear-rate intervals with no lower error), and the 90% intervals for wear rate and fresh effectiveness contained the truth in {{robustness.coverage_90.lam.inside:d}} and {{robustness.coverage_90.K0.inside:d}} of {{robustness.n_draws:d}} runs, slightly below nominal (stiffness {{robustness.coverage_90.k_pad.inside:d}}, current effectiveness {{robustness.coverage_90.K.inside:d}}).

## 2. Credit and scope

This project builds on the problem described in GrayMatter Robotics' post
**[World Models for Manufacturing Processes](https://factory.graymatter-robotics.com/world-models-for-manufacturing-processes/)**
(Hantao Ye, Omey Manyar and Satyandra K. Gupta, 2 September 2026). Three points from the post shape this work:

- **The model.** A sanding world model takes the scanned surface and a candidate action (path, force, grit, spindle speed, dwell) and predicts the result.
- **The stiffness range.** On a curved panel, GrayMatter's simulator predicts contact over roughly 5% to 97% of the pad face across physically plausible stiffnesses. The post does not give the panel radius, pad size or force.
- **The calibration problem.** Pad compliance, abrasive behaviour and wear are hard to identify exactly, so the model is conditioned on real observations.

**This is an independent, simplified simulation of that problem. It uses no GrayMatter data, code or models.** Every result below comes from the simulator in this repository (`results/`, produced by `python -m swt run-all`); the 5–97% contact range is quoted from GrayMatter's post.

The extension studied here is having **two** unknowns at once: one fixed (pad stiffness) and one that changes every pass (abrasive wear).

## 3. Model

Units are mm, N, s and MPa (= N/mm²). Pad stiffness `k_pad` is in N/mm³, abrasive effectiveness `K` in mm²/N (= 1/MPa), and wear rate `λ` in 1/mm³. All parameter values are in `configs/default.yaml`, and the copy used for the reported run is in `results/run_info.json`.

### Workpiece and toolpath (`swt/geometry.py`)
* **The part** is a height map `z(x, y)` on a 1 mm grid ({{model_info.grid_nodes:d}} nodes). It is a convex cylindrical section, 200 mm along the straight x direction and 150 mm across the curved y direction, with radius {{contact_sweep.radius_mm:g}} mm. The crown runs down the middle and the edges are {{model_info.panel_sag_mm:.1f}} mm lower.
* **One pass** is a serpentine raster:
  * lines run along x, with a {{model_info.actual_stepover_mm:.0f}} mm stepover in y ({{model_info.raster_lines:d}} lines);
  * pad-centre stations are 5 mm apart ({{model_info.stations_per_pass:d}} per pass) and the feed is 25 mm/s;
  * *dwell* means 0.25 s of extra time at the two ends of every line (the turnaround);
  * one pass takes {{model_info.pass_duration_s:.1f}} s.

  The pad centre runs all the way to the panel edge, so edge stations overhang the part and remove more material there. Edge over-sanding is common in practice, but its size here is untested.
* **What varies between passes.** Path, feed and dwell are the same for every pass; only the commanded normal force changes, alternating 20 N / 40 N. Spindle speed is part of the action too, but is constant here.

### Pad contact (`swt/pad.py`): Winkler foundation
At each station the pad face is held tangent to the surface at the pad centre. A surface point under the pad lies a distance `g` below the pad plane: `g = 0` at the centre, rising to {{model_info.pad_rim_gap_mm:.2f}} mm at the rim of a 125 mm pad on this cylinder. Pushing the pad in by `d` gives

```
p = k_pad · max(0, d − g)          [MPa]
Σ_i p_i · A_i = F                  (force balance; A_i = pad-face area over grid node i)
```

**Solving the force balance.** It is monotone and piecewise linear in `d`, so the 1-D root is found exactly: bracket it between two sorted gaps, then solve the linear piece. A generic Brent root finder finds the same root in the tests. The integrated pressure equals the applied force at every station, including stations that overhang the edge.

**What stiffness does to the contact.** A stiff pad touches only a thin strip along the line under its centre. A soft pad wraps around the curvature.

**The stiffness prior** is chosen so that, at a reference force of {{contact_sweep.reference_force_N:.0f}} N, contact runs from **{{contact_sweep.contact_fraction_at_prior_bounds.30N.stiff_k_high:pct1}}% of the pad face for the stiffest pad to {{contact_sweep.contact_fraction_at_prior_bounds.30N.soft_k_low:pct1}}% for the softest**, reproducing the range quoted in GrayMatter's post. Because the post does not give its geometry, this is a calibration of the prior to the quoted range, not a reproduction of their setup. Three caveats:

- **Other forces give other ranges.** At the 20 N and 40 N forces actually used in the alternating schedule, the range is {{contact_sweep.contact_fraction_at_prior_bounds.20N.stiff_k_high:pct1}}–{{contact_sweep.contact_fraction_at_prior_bounds.20N.soft_k_low:pct1}}% and {{contact_sweep.contact_fraction_at_prior_bounds.40N.stiff_k_high:pct1}}–{{contact_sweep.contact_fraction_at_prior_bounds.40N.soft_k_low:pct1}}%.
- **The stiff end is quantised by the grid.** On a {{contact_sweep.fine_grid_reference.grid_mm:g}} mm grid the same 30 N range is {{contact_sweep.fine_grid_reference.stiff_k_high:pct1}}–{{contact_sweep.fine_grid_reference.soft_k_low:pct1}}% (figure 1, dashed line).
- **The softest pads need large indentations:** {{model_info.pad_penetration_at_prior_bounds.30N.soft_k_low_mm:.1f}} mm at 30 N, and {{model_info.pad_penetration_at_prior_bounds.20N.soft_k_low_mm:.1f}}–{{model_info.pad_penetration_at_prior_bounds.40N.soft_k_low_mm:.1f}} mm over 20–40 N. That is beyond where a linear Winkler law holds for real foam, but it is what a 97% contact fraction on a 300 mm radius requires.

![Contact fraction vs stiffness](figures/fig1_contact_vs_stiffness.png)

### Removal (`swt/process.py`): Preston's law
```
dh = K · p · v · dt
```

**Sliding speed.** `v` is the RMS sliding speed of a random-orbital sander. It combines the orbital speed π·D_orb·f (5 mm orbit, 8000 rpm) with slow free rotation of the pad at 2% of spindle speed:

```
v(r) = sqrt((π·D_orb·f)² + (2π·f·0.02·r)²)
```

so it rises from {{model_info.sliding_speed_centre_mm_s:.0f}} mm/s at the pad centre to {{model_info.sliding_speed_rim_mm_s:.0f}} mm/s at the rim.

**Removal per pass** is `h(x) = K · Σ_stations p·v·dt`. Under the nominal parameters, a fresh abrasive removes {{model_info.nominal_fresh_mean_removal_um.20N:.1f}}, {{model_info.nominal_fresh_mean_removal_um.30N:.1f}} and {{model_info.nominal_fresh_mean_removal_um.40N:.1f}} µm on average at 20, 30 and 40 N.

### Wear
`K = K0 · exp(−λ·W)`, where `W` is the cumulative **removed volume** in mm³.

**This is an assumption, untested here.** Wear is taken to be driven by removed volume rather than by force × sliding distance, by analogy with the grinding G-ratio (material removed per unit of abrasive wear). Section 6 lists testing it on coupons.

**Per-pass update.** `K` is held constant within a pass and updated between passes, with a small random fluctuation standing in for abrasive variability (its level is assumed):

```
K_{n+1} = K_n · exp(−λ·ΔV_n + σ_w·ξ_n),   ξ_n ~ N(0, 1),   σ_w = 0.01
```

Because ΔV_n ∝ K_n, blunt paper both cuts and wears more slowly; without noise, the decay is hyperbolic in pass count.

**This is a coarse time step.** Over the {{robustness.n_draws:d}} truths, the first 40 N pass alone reduced `K` by a median of {{robustness.per_pass_K_drop_fraction.pass2_40N.median:pct0}}% and at most {{robustness.per_pass_K_drop_fraction.pass2_40N.max:pct0}}%. The truth and all three estimators share this discretisation, so it does not bias the comparison, but a real abrasive would also leave a gradient within the pass.

### Scan (`swt/scan.py`)
The observed per-pass removal map is the true map plus independent Gaussian noise with σ = 2 µm. It is sampled on a regular 2 mm sub-grid ({{model_info.scan_points:d}} points), a simplified stand-in for a line scanner.

### Hidden truth and priors
| Parameter | Prior | Notes |
|---|---|---|
| `k_pad` | log-uniform on [{{contact_sweep.k_low:g}}, {{contact_sweep.k_high:g}}] N/mm³ | {{contact_sweep.contact_fraction_at_prior_bounds.30N.soft_k_low:pct0}}% to {{contact_sweep.contact_fraction_at_prior_bounds.30N.stiff_k_high:pct0}}% contact at 30 N |
| `K0` (fresh effectiveness) | log-normal, median 6.0e-5 mm²/N, σ_log 0.25 | |
| `λ` (wear rate) | log-normal, median 2.5e-4 1/mm³, σ_log 0.35 | nominal crossing of 50% of K0 at pass {{main_run.A_crossing_pass:d}} |
| wear fluctuation | σ_w = 0.01 per pass | known to the tracker |

Each run draws `k_pad`, `K0` and `λ` from these priors with a fixed seed. The estimators never see them.

Over the {{robustness.n_draws:d}} draws, the true crossing pass has a median of {{abrasive_change.true_crossing_pass.median:.0f}} (IQR {{abrasive_change.true_crossing_pass.q25:.0f}}–{{abrasive_change.true_crossing_pass.q75:.0f}}). {{abrasive_change.true_crossings_within_experiment:d}} of {{robustness.n_draws:d}} cross within the 20 scanned passes; for the others, the truth is simulated past pass 20 to find the crossing.

### Why stiffness and wear can be told apart
* **Wear only changes the scale.** Preston's law is linear in `K`, so a sharper or blunter abrasive multiplies the whole removal map by a constant.
* **Stiffness changes the shape, and barely the scale.** The force balance fixes the total load at `F` for every stiffness.
  * *Scale:* the removed volume per unit `K` depends on `k_pad` only through where the contact sits relative to the faster-moving rim, plus a surface-slope factor. Across the whole prior that moves it by {{model_info.volume_per_unit_K_vs_k.relative_spread:pct1}}%.
  * *Shape:* stiffness sets the contact patch, whose width and pressure profile depend only on `F / k_pad` (figure 1, right). In the removal map, stiffer pads give stronger raster ripple between lines and different over-sanding at the panel edges. Figure 2 compares the main run's pad ({{main_run.truth.k_pad:.2f}} N/mm³) with A's nominal pad ({{model_info.nominal_parameters.k_pad:.2g}} N/mm³); it does not include a soft pad.
* **Varying the force** gives two views of the same pad. At 20 N and 40 N the contact patch differs in a known way, so in principle a stiffness error cannot hide as a scale error.
  * In this model, though, a *single* force level already pins `F / k_pad`, and hence `k_pad`, because the force is commanded and known. The ablation in section 4.4 confirms that the alternating schedule adds little.
  * It would matter if something else also changed the patch shape: an unknown force offset, uncertain local curvature, a pad whose stiffness drifts, or a scanner too coarse to resolve the patch.

### Estimators (`swt/estimators.py`)
All three see the commanded action of every pass and the noisy scans, and nothing else. Before each pass, each predicts that pass's removal map.

**A: nominal.** Prior-mean parameters, taken as the mean in log space: `k_pad` = {{model_info.nominal_parameters.k_pad:.3g}} N/mm³ (the geometric mean of the prior range), plus the prior medians of `K0` and `λ`. Wear is propagated with the nominal `λ` and A's own predicted volume, and no parameter is ever updated.

**B: calibrate-once.** After pass 1, a least-squares fit of `k_pad` and `K` to the first scan. `K` is profiled out in closed form, followed by a 1-D search over log `k_pad`, using the same surrogate table as C. Both values are then frozen and there is no wear model. This is the "learn the pad once" baseline.

**C: joint tracker.** A particle filter with 10,000 particles over `[log k_pad, log K0, log λ, log K_current]`. `K0` is carried as a static component because the abrasive-change threshold is defined relative to it.
* *Predict:* between passes, `K_current` follows the wear law using each particle's own predicted removed volume, plus N(0, 0.01²) noise on log K. The other components are static.
* *Update:* after each scan, particles are weighted by the Gaussian likelihood of the whole scan. With thousands of pixels that likelihood is far sharper than the prior, so it is applied in tempered stages (progressive correction):
  1. choose each stage's exponent so the effective sample size stays at 75% of N;
  2. resample systematically;
  3. roughen with a Liu–West shrinkage kernel (a = 0.98), which keeps the cloud's mean and covariance.

  A final stage that leaves the effective sample size above 75% is not resampled; its weights are kept.
* *Speed:* removal depends on stiffness only through `η = F/k_pad`, so the exposure map `m(η)` is tabulated once with the exact contact model, on {{model_info.surrogate.n_eta:d}} log-spaced η values. The largest relative RMS interpolation error over {{model_info.surrogate.n_checks:d}} spot checks is {{model_info.surrogate.max_relative_rms_error:.1e}}. With linear interpolation, the scan likelihood reduces to precomputed inner products, so each particle costs O(1). The hidden truth and the predictions of A and B use the exact model.
* *Outputs after every pass:* posterior mean, median and 90% credible interval of every component, plus the predicted crossing pass. For the crossing, particles are rolled forward through the planned schedule with random future wear, and each particle remembers the pass at which its own history crossed.

**Oracle (reference only).** Knows the true `k_pad` and `λ` and the true `K` of the previous pass, but not the current pass's wear fluctuation. Its error is a floor *in expectation*; on individual passes C can beat it by luck.

Filter settings (10,000 particles, ESS target 75% of N, shrinkage a = 0.98) are fixed in `configs/default.yaml`. They were set during development on draws other than the reported ones; that tuning is not part of `run-all` and is not recorded in `results/`.

### Modelling choices
* **Panel size.** It is 200 × **150** mm, not the 200 × 100 mm panel suggested in the original project brief, because a 125 mm pad cannot sit fully on a 100 mm wide panel. On a 100 mm panel the 97% end of the contact range would be geometrically unreachable (this was not simulated).
* **State vector.** The tracker's state adds `log K0` to `[log k_pad, log K_current, log λ]`, because the change threshold is a fraction of `K0`.
* **Update step.** The update is tempered (several weight → resample → roughen rounds per scan) rather than a single round.
* **Root finder.** The 1-D root is computed exactly from the piecewise-linear structure; Brent's method is kept as a reference in the tests.
* **Wear law.** The true law is `K0·exp(−λW)` plus the small per-pass fluctuation above, applied once per pass.
* **A well-specified world.** The tracker uses the same physics, priors, scan noise level and wear-fluctuation level as the simulated truth (apart from the surrogate's interpolation error). Its results are therefore a best case; see Limitations.

## 4. Results

All numbers come from `results/` and were produced by `python -m swt run-all` with seed {{info.config.seed:d}}. Unless a section says otherwise (constant 30 N in 4.4, other noise levels in 4.5), every run has:
- 20 passes;
- forces alternating 20/40 N;
- 2 µm scan noise;
- the 50%-of-K0 threshold.

All errors are measured against the simulator's noise-free removal map.

**At a glance**, over {{robustness.n_draws:d}} hidden truths at pass 20:

| | A: nominal | B: calibrate-once | C: joint tracker | oracle |
|---|---|---|---|---|
| removal-map RMSE, median [IQR] µm | {{robustness.final_rmse_um.A.median:.2f}} [{{robustness.final_rmse_um.A.q25:.2f}}–{{robustness.final_rmse_um.A.q75:.2f}}] | {{robustness.final_rmse_um.B.median:.2f}} [{{robustness.final_rmse_um.B.q25:.2f}}–{{robustness.final_rmse_um.B.q75:.2f}}] | **{{robustness.final_rmse_um.C.median:.3f}}** [{{robustness.final_rmse_um.C.q25:.3f}}–{{robustness.final_rmse_um.C.q75:.3f}}] | {{robustness.final_rmse_um.oracle.median:.3f}} |
| 90% coverage (k_pad / λ / K / K0) | | | {{robustness.coverage_90.k_pad.inside:d}} / {{robustness.coverage_90.lam.inside:d}} / {{robustness.coverage_90.K.inside:d}} / {{robustness.coverage_90.K0.inside:d}} of {{robustness.n_draws:d}} | |
| abrasive-change error after pass 2, median (mean) passes | {{abrasive_change.by_decision.after_pass_2.abs_err_A_passes.median:.0f}} ({{abrasive_change.by_decision.after_pass_2.mean_abs_err_A_passes:.2f}}) | no wear model | {{abrasive_change.by_decision.after_pass_2.abs_err_C_passes.median:.0f}} ({{abrasive_change.by_decision.after_pass_2.mean_abs_err_C_passes:.2f}}) | |

### 4.1 Main run: one hidden truth

**How the draw was chosen.** A fixed rule that looks only at the sampled truths and never at estimator results: among the {{robustness.n_draws:d}} robustness draws, take the one at the *median* standardised distance from the prior centre in (log k_pad, log K0, log λ). That is a typical draw rather than a tail draw or one that happens to match the nominal model.
- It is draw {{main_run.draw:d}}: `k_pad` = {{main_run.truth.k_pad:.4f}} N/mm³, `K0` = {{main_run.truth.K0:.3g}} mm²/N, `λ` = {{main_run.truth.lam:.3g}} 1/mm³.
- Its abrasive crosses 50% at pass {{main_run.true_crossing_pass:d}}; the nominal schedule says pass {{main_run.A_crossing_pass:d}}.
- An earlier version used an arbitrary draw id, which turned out to be an unusually fast-wearing truth. An intermediate rule (closest to the prior medians) picked a truth where the nominal model is almost exact. This rule avoids both. The robustness results in section 4.2 cover the full spread.

| Removal-map RMSE [µm] | A: nominal | B: calibrate-once | C: joint tracker | oracle |
|---|---|---|---|---|
| pass 1 (before any scan) | {{main_run.first_pass_rmse_um.A:.2f}} | {{main_run.first_pass_rmse_um.B:.2f}} | {{main_run.first_pass_rmse_um.C:.2f}} | {{main_run.first_pass_rmse_um.oracle:.2f}} |
| pass {{main_run.early_pass:d}} | {{main_run.early_rmse_um.A:.2f}} | {{main_run.early_rmse_um.B:.2f}} | {{main_run.early_rmse_um.C:.2f}} | {{main_run.early_rmse_um.oracle:.2f}} |
| pass {{main_run.final_pass:d}} | {{main_run.final_rmse_um.A:.2f}} | {{main_run.final_rmse_um.B:.2f}} | {{main_run.final_rmse_um.C:.3f}} | {{main_run.final_rmse_um.oracle:.3f}} |

* **B** is good right after its calibration, but never learns that the paper is dulling. At pass 20 it predicts a mean removal of {{main_run.predicted_mean_removal_um.pass20.B:.1f}} µm against a true {{main_run.predicted_mean_removal_um.pass20.true:.1f}} µm.
* **A** has the wear law but wrong parameters. The level error comes mostly from fresh effectiveness, the ripple difference from the pad:

  | Parameter | A uses | Truth |
  |---|---|---|
  | fresh effectiveness `K0` [mm²/N] | {{model_info.nominal_parameters.K0:.3g}} | {{main_run.truth.K0:.3g}} |
  | pad stiffness `k_pad` [N/mm³] | {{model_info.nominal_parameters.k_pad:.3g}} | {{main_run.truth.k_pad:.3g}} |
  | wear rate `λ` [1/mm³] | {{model_info.nominal_parameters.lam:.3g}} | {{main_run.truth.lam:.3g}} |

  | Mean removal [µm] | A predicts | True |
  |---|---|---|
  | pass 2 | {{main_run.predicted_mean_removal_um.pass2.A:.1f}} | {{main_run.predicted_mean_removal_um.pass2.true:.1f}} |
  | pass 20 | {{main_run.predicted_mean_removal_um.pass20.A:.1f}} | {{main_run.predicted_mean_removal_um.pass20.true:.1f}} |
* **C at pass 1** has seen no scan yet; its prediction is the prior-predictive average over all plausible pads.
* **C at pass 20.** Its posterior means and 90% intervals are:

  | Parameter | C: mean [90% interval] | Truth |
  |---|---|---|
  | `k_pad` [N/mm³] | {{main_run.C_final.k_pad.mean:.4f}} [{{main_run.C_final.k_pad.lo:.4f}}, {{main_run.C_final.k_pad.hi:.4f}}] | {{main_run.truth.k_pad:.4f}} |
  | `λ` [1/mm³] | {{main_run.C_final.lam.mean:.3g}} [{{main_run.C_final.lam.lo:.3g}}, {{main_run.C_final.lam.hi:.3g}}] | {{main_run.truth.lam:.3g}} |
  | `K` [mm²/N] | {{main_run.C_final.K.mean:.4g}} [{{main_run.C_final.K.lo:.4g}}, {{main_run.C_final.K.hi:.4g}}] | {{main_run.true_final_K:.4g}} |

  All four of C's 90% intervals (these three and `K0`) contain the truth at pass 20 (`results/main_run.json`, `C_final_inside_90`).
* **Crossing prediction.**

  | Prediction made after | Predicted crossing pass | Band |
  |---|---|---|
  | pass 1 | {{main_run.C_crossing_prediction.0.median:.0f}} | {{main_run.C_crossing_prediction.0.lo:.0f}}–{{main_run.C_crossing_prediction.0.hi:.0f}} |
  | pass 2 | {{main_run.C_crossing_prediction.1.median:.0f}} | {{main_run.C_crossing_prediction.1.lo:.0f}}–{{main_run.C_crossing_prediction.1.hi:.0f}} |
  | pass 3 | {{main_run.C_crossing_prediction.2.median:.0f}} | {{main_run.C_crossing_prediction.2.lo:.0f}}–{{main_run.C_crossing_prediction.2.hi:.0f}} |

  The truth is pass {{main_run.true_crossing_pass:d}}.

![Removal maps](figures/fig2_removal_maps.png)

![Parameter tracking](figures/fig4_parameter_tracking.png)

### 4.2 Robustness: {{robustness.n_draws:d}} hidden truths

| | A: nominal | B: calibrate-once | C: joint tracker | oracle |
|---|---|---|---|---|
| final-pass RMSE, median [IQR] µm | {{robustness.final_rmse_um.A.median:.2f}} [{{robustness.final_rmse_um.A.q25:.2f}}–{{robustness.final_rmse_um.A.q75:.2f}}] | {{robustness.final_rmse_um.B.median:.2f}} [{{robustness.final_rmse_um.B.q25:.2f}}–{{robustness.final_rmse_um.B.q75:.2f}}] | **{{robustness.final_rmse_um.C.median:.3f}}** [{{robustness.final_rmse_um.C.q25:.3f}}–{{robustness.final_rmse_um.C.q75:.3f}}] | {{robustness.final_rmse_um.oracle.median:.3f}} [{{robustness.final_rmse_um.oracle.q25:.3f}}–{{robustness.final_rmse_um.oracle.q75:.3f}}] |
| RMSE averaged over the 20 passes, median µm | {{robustness.mean_over_passes_rmse_um.A.median:.2f}} | {{robustness.mean_over_passes_rmse_um.B.median:.2f}} | {{robustness.mean_over_passes_rmse_um.C.median:.3f}} | |
| draws where C is better at pass 20 | {{robustness.draws_C_better_than_A_at_final_pass:d}} / {{robustness.n_draws:d}} | {{robustness.draws_C_better_than_B_at_final_pass:d}} / {{robustness.n_draws:d}} | | |

For scale, the true mean removal is {{robustness.first_true_mean_removal_um.median:.1f}} µm in pass 1 and {{robustness.final_true_mean_removal_um.median:.2f}} µm in pass 20 (medians over draws). B's pass-20 error exceeds the true removal in {{robustness.draws_B_error_exceeds_true_removal_at_final_pass:d}} of {{robustness.n_draws:d}} draws.

* **C against B, pass by pass** (draws where C's error is below B's):

  | Pass | Draws | Note |
  |---|---|---|
  | 1 | {{robustness.per_pass.draws_C_better_than_B.0:d}} | before any scan; B equals the nominal model |
  | 2 | {{robustness.per_pass.draws_C_better_than_B.1:d}} | the first pass after B's calibration; B's fresh one-scan fit is still ahead in the rest |
  | 3 to 20 | {{robustness.per_pass.draws_C_better_than_B.2:d}} at every pass | |

  C's 20-pass average is dominated by passes 1–2, before the wear rate is observable.
* **B against A.** By pass 20, B is worse than the uncalibrated nominal model A in {{robustness.draws_B_worse_than_A_at_final_pass:d}} of {{robustness.n_draws:d}} draws. A has a wear law, even a wrong one; B has none, and ignoring wear costs more than starting from wrong parameters. This does not mean calibration is useless: C is calibration that keeps going.
* **Parameter accuracy at pass 20.** C's posterior mean has a median absolute error of {{robustness.final_relative_error_C.k_pad.median:pct2}}% for `k_pad`, {{robustness.final_relative_error_C.K.median:pct2}}% for the current `K`, and {{robustness.final_relative_error_C.lam.median:pct1}}% for `λ`. B's one-scan stiffness estimate is off by a median {{robustness.B_abs_relative_error_k_pad.median:pct2}}%.

**Coverage of C's 90% credible intervals at pass 20:**

| `k_pad` | `λ` | `K_current` | `K0` | all of k_pad, λ, K inside |
|---|---|---|---|---|
| {{robustness.coverage_90.k_pad.inside:d}}/{{robustness.n_draws:d}} | {{robustness.coverage_90.lam.inside:d}}/{{robustness.n_draws:d}} | {{robustness.coverage_90.K.inside:d}}/{{robustness.n_draws:d}} | {{robustness.coverage_90.K0.inside:d}}/{{robustness.n_draws:d}} | {{robustness.coverage_90.all_three_marginals_inside.inside:d}}/{{robustness.n_draws:d}} (≈{{robustness.coverage_90.all_three_marginals_inside.expected_if_calibrated_independent:pct0}} expected if the three were calibrated and independent) |

With {{robustness.n_draws:d}} draws, a calibrated 90% interval gives a coverage estimate with a binomial standard error of about {{robustness.coverage_90.binomial_se_if_calibrated:pct0}} percentage points.

- **`k_pad` and `K_current`** are on target.
- **`λ` and `K0`** are a little low: {{robustness.coverage_90.lam.inside:d}} and {{robustness.coverage_90.K0.inside:d}} against 90, with a standard error of about {{robustness.coverage_90.binomial_se_if_calibrated:pct0}}.

The per-pass records show where the shortfall comes from:

| Coverage of | pass 1 | pass 2 | pass 10 | pass 20 |
|---|---|---|---|---|
| `K0` | {{robustness.per_pass.coverage_90.K0.0:d}} | | {{robustness.per_pass.coverage_90.K0.9:d}} | {{robustness.per_pass.coverage_90.K0.19:d}} |
| `λ` | | {{robustness.per_pass.coverage_90.lam.1:d}} | {{robustness.per_pass.coverage_90.lam.9:d}} | {{robustness.per_pass.coverage_90.lam.19:d}} |

(out of {{robustness.n_draws:d}})

- **`K0`'s** shortfall is already there after the first update and stays flat, so it comes from the first-scan posterior. One untested possibility is that the tempered update with roughening makes that posterior slightly too narrow.
- **`λ`'s** coverage declines over the passes. That fits the known weakness of particle filters with static parameters: repeated resampling and roughening narrow the interval a little more than the data justify.

Even with the filter matching the simulator (apart from the surrogate's {{model_info.surrogate.max_relative_rms_error:.1e}} interpolation error), coverage is not above nominal, so with real data the intervals would need to be wider.

### 4.3 Abrasive-change prediction (threshold: K < 50% of K0)

| Prediction made | draws | C: median \|error\| (mean) [passes] | C: within ±1 pass | C: band contains truth | A (fixed nominal schedule): median \|error\| (mean) |
|---|---|---|---|---|---|
| after pass 2 | {{abrasive_change.by_decision.after_pass_2.n:d}} | {{abrasive_change.by_decision.after_pass_2.abs_err_C_passes.median:.0f}} ({{abrasive_change.by_decision.after_pass_2.mean_abs_err_C_passes:.2f}}) | {{abrasive_change.by_decision.after_pass_2.within_1_pass_C:d}} | {{abrasive_change.by_decision.after_pass_2.band_hits_C:d}} | {{abrasive_change.by_decision.after_pass_2.abs_err_A_passes.median:.0f}} ({{abrasive_change.by_decision.after_pass_2.mean_abs_err_A_passes:.2f}}) |
| after pass 5 | {{abrasive_change.by_decision.after_pass_5.n:d}} | {{abrasive_change.by_decision.after_pass_5.abs_err_C_passes.median:.0f}} ({{abrasive_change.by_decision.after_pass_5.mean_abs_err_C_passes:.2f}}) | {{abrasive_change.by_decision.after_pass_5.within_1_pass_C:d}} | {{abrasive_change.by_decision.after_pass_5.band_hits_C:d}} | {{abrasive_change.by_decision.after_pass_5.abs_err_A_passes.median:.0f}} ({{abrasive_change.by_decision.after_pass_5.mean_abs_err_A_passes:.2f}}) |
| one pass before the crossing | {{abrasive_change.by_decision.last_pass_before_crossing.n:d}} | {{abrasive_change.by_decision.last_pass_before_crossing.abs_err_C_passes.median:.0f}} ({{abrasive_change.by_decision.last_pass_before_crossing.mean_abs_err_C_passes:.2f}}) | {{abrasive_change.by_decision.last_pass_before_crossing.within_1_pass_C:d}} | {{abrasive_change.by_decision.last_pass_before_crossing.band_hits_C:d}} | {{abrasive_change.by_decision.last_pass_before_crossing.abs_err_A_passes.median:.0f}} ({{abrasive_change.by_decision.last_pass_before_crossing.mean_abs_err_A_passes:.2f}}) |

Only predictions made before the true crossing count, so rows can have fewer than {{robustness.n_draws:d}} draws.

* **How early.** After two scans the tracker has seen one wear step, and already places the change point within ±1 pass in {{abrasive_change.by_decision.after_pass_2.within_1_pass_C:d}} of {{abrasive_change.by_decision.after_pass_2.n:d}} runs.
* **Error against lead time** (figure 5, right), on a fixed population: the {{abrasive_change.by_lead.population:d}} draws that cross between pass {{abrasive_change.by_lead.crossing_range.0:d}} and {{abrasive_change.by_lead.crossing_range.1:d}}, for which every lead from 1 to {{abrasive_change.by_lead.L_max:d}} passes is observable. C's mean error grows from {{abrasive_change.by_lead.rows.0.mean_abs_err_C:.2f}} passes one pass ahead to {{abrasive_change.by_lead.rows.9.mean_abs_err_C:.2f}} passes ten passes ahead. A's fixed schedule is off by {{abrasive_change.by_lead.rows.0.mean_abs_err_A:.2f}} passes on average.
* **The band is conservative.** The crossing pass is an integer, so the 5–95% band (median width {{abrasive_change.by_decision.after_pass_2.band_width_passes.median:.0f}} passes after pass 2) holds more than 90% of the filter's own predictive probability. The median share inside the band is {{abrasive_change.by_decision.after_pass_2.band_predictive_mass.median:pct0}}% after pass 2, {{abrasive_change.by_decision.after_pass_5.band_predictive_mass.median:pct0}}% after pass 5 and {{abrasive_change.by_decision.last_pass_before_crossing.band_predictive_mass.median:pct0}}% one pass before the crossing. That largely explains the empirical coverage: {{abrasive_change.by_decision.after_pass_2.band_coverage_C:pct0}}–{{abrasive_change.by_decision.last_pass_before_crossing.band_coverage_C:pct0}}% at the decision points in the table, and {{abrasive_change.by_lead.band_coverage_min:pct0}}–{{abrasive_change.by_lead.band_coverage_max:pct0}}% across the leads in figure 5.

![Abrasive change](figures/fig5_abrasive_change.png)

### 4.4 Identifiability ablation: constant force instead of alternating 20/40 N
The same {{ablation.n_draws:d}} truths were rerun with 30 N on every pass, the same mean force. A *null control* reruns the alternating schedule with only the particle filter's random seed changed; it shows how much the comparison moves from Monte Carlo noise alone. Medians over draws at pass 20:

| | 90% width of `k_pad` / estimate | 90% width of `λ` | \|corr(log k_pad, log K)\| | C's RMSE / true removal | passes until `λ` width < 20% (never reached) |
|---|---|---|---|---|---|
| alternating 20/40 N | {{ablation.by_schedule.alternating.final_relative_width.k_pad.median:pct2}}% | {{ablation.by_schedule.alternating.final_relative_width.lam.median:pct1}}% | {{ablation.by_schedule.alternating.final_abs_corr_logk_logK.median:.3f}} | {{ablation.by_schedule.alternating.final_rmse_C_relative_to_true_removal.median:pct2}}% | {{ablation.by_schedule.alternating.pass_lam_width_below_20pct.median:.0f}} ({{ablation.by_schedule.alternating.pass_lam_width_below_20pct.n_never_reached:d}}) |
| constant 30 N | {{ablation.by_schedule.constant.final_relative_width.k_pad.median:pct2}}% | {{ablation.by_schedule.constant.final_relative_width.lam.median:pct1}}% | {{ablation.by_schedule.constant.final_abs_corr_logk_logK.median:.3f}} | {{ablation.by_schedule.constant.final_rmse_C_relative_to_true_removal.median:pct2}}% | {{ablation.by_schedule.constant.pass_lam_width_below_20pct.median:.0f}} ({{ablation.by_schedule.constant.pass_lam_width_below_20pct.n_never_reached:d}}) |
| null control (alternating, other filter seed) | {{ablation.by_schedule.alternating_reseeded.final_relative_width.k_pad.median:pct2}}% | {{ablation.by_schedule.alternating_reseeded.final_relative_width.lam.median:pct1}}% | {{ablation.by_schedule.alternating_reseeded.final_abs_corr_logk_logK.median:.3f}} | {{ablation.by_schedule.alternating_reseeded.final_rmse_C_relative_to_true_removal.median:pct2}}% | {{ablation.by_schedule.alternating_reseeded.pass_lam_width_below_20pct.median:.0f}} ({{ablation.by_schedule.alternating_reseeded.pass_lam_width_below_20pct.n_never_reached:d}}) |

Paired per-draw comparisons with the alternating run:

| compared with alternating | `k_pad` interval wider in | median width ratio | `λ` interval wider in | median width ratio |
|---|---|---|---|---|
| constant 30 N | {{ablation.paired_width_ratio_constant_over_alternating.draws_constant_wider_k_pad:d}} / {{ablation.n_draws:d}} | {{ablation.paired_width_ratio_constant_over_alternating.k_pad.median:.3f}} | {{ablation.paired_width_ratio_constant_over_alternating.draws_constant_wider_lam:d}} / {{ablation.n_draws:d}} | {{ablation.paired_width_ratio_constant_over_alternating.lam.median:.3f}} |
| null control | {{ablation.paired_width_ratio_control_over_alternating.draws_control_wider_k_pad:d}} / {{ablation.n_draws:d}} | {{ablation.paired_width_ratio_control_over_alternating.k_pad.median:.3f}} | {{ablation.paired_width_ratio_control_over_alternating.draws_control_wider_lam:d}} / {{ablation.n_draws:d}} | {{ablation.paired_width_ratio_control_over_alternating.lam.median:.3f}} |

Constant force against the null control directly: the `k_pad` interval is wider in {{ablation.paired_width_ratio_constant_over_control.draws_constant_wider_k_pad:d}} of {{ablation.n_draws:d}} draws and the `λ` interval in {{ablation.paired_width_ratio_constant_over_control.draws_constant_wider_lam:d}} of {{ablation.n_draws:d}}.

**This mostly did not show what was expected.**

- **Stiffness.** Constant force did not widen the interval any more often than reseeding the filter does.
- **Wear rate.** Constant force gave a small but consistent loss of precision: the `λ` interval was slightly wider (median ratio {{ablation.paired_width_ratio_constant_over_alternating.lam.median:.3f}}). The narrower interval under alternating force did *not* come with better accuracy or coverage:

  | Schedule | median `λ` error at pass 20 | `λ` coverage |
  |---|---|---|
  | alternating | {{ablation.by_schedule.alternating.final_abs_relative_error.lam.median:pct2}}% | {{ablation.by_schedule.alternating.coverage_90.lam.inside:d}} |
  | constant | {{ablation.by_schedule.constant.final_abs_relative_error.lam.median:pct2}}% | {{ablation.by_schedule.constant.coverage_90.lam.inside:d}} |
  | null control | {{ablation.by_schedule.alternating_reseeded.final_abs_relative_error.lam.median:pct2}}% | {{ablation.by_schedule.alternating_reseeded.coverage_90.lam.inside:d}} |

- **Correlation between stiffness and current effectiveness.** At pass 20 they are essentially uncorrelated in the posterior under both schedules (median \|corr\| {{ablation.by_schedule.alternating.per_pass_abs_corr.19.median:.3f}} alternating, {{ablation.by_schedule.constant.per_pass_abs_corr.19.median:.3f}} constant). After the first scan alone the correlation is larger: {{ablation.by_schedule.alternating.per_pass_abs_corr.0.median:.2f}} after a 20 N first pass and {{ablation.by_schedule.constant.per_pass_abs_corr.0.median:.2f}} after a 30 N one.

The reason is in section 3: the commanded force is known, the contact shape depends only on `F/k_pad`, and the scale barely depends on `k_pad`, so one force level already identifies both parameters.

Two raw differences are not identifiability effects:

- **`K` interval width.** At pass 20 it is {{ablation.by_schedule.alternating.final_relative_width.K.median:pct2}}% (alternating) against {{ablation.by_schedule.constant.final_relative_width.K.median:pct2}}% (constant), because pass 20 is a 40 N pass, which removes more material. At pass 19, a 20 N pass, the order reverses: {{ablation.by_schedule.alternating.per_pass_relw_K.18.median:pct2}}% vs {{ablation.by_schedule.constant.per_pass_relw_K.18.median:pct2}}%.
- **Prediction error.** Normalised by the true removal, C's pass-20 error is {{ablation.by_schedule.alternating.final_rmse_C_relative_to_true_removal.median:pct2}}% (alternating), {{ablation.by_schedule.constant.final_rmse_C_relative_to_true_removal.median:pct2}}% (constant) and {{ablation.by_schedule.alternating_reseeded.final_rmse_C_relative_to_true_removal.median:pct2}}% (control). Relative to the oracle it is {{ablation.by_schedule.alternating.final_rmse_C_over_oracle.median:.2f}}×, {{ablation.by_schedule.constant.final_rmse_C_over_oracle.median:.2f}}× and {{ablation.by_schedule.alternating_reseeded.final_rmse_C_over_oracle.median:.2f}}×. Neither schedule gives a prediction advantage.

### 4.5 Scan-noise sensitivity

| scan noise σ | 1 µm | 2 µm | 5 µm | 10 µm |
|---|---|---|---|---|
| C final-pass RMSE, median [IQR] µm | {{noise_sensitivity.levels.1.final_rmse_C_um.median:.3f}} [{{noise_sensitivity.levels.1.final_rmse_C_um.q25:.3f}}–{{noise_sensitivity.levels.1.final_rmse_C_um.q75:.3f}}] | {{noise_sensitivity.levels.2.final_rmse_C_um.median:.3f}} [{{noise_sensitivity.levels.2.final_rmse_C_um.q25:.3f}}–{{noise_sensitivity.levels.2.final_rmse_C_um.q75:.3f}}] | {{noise_sensitivity.levels.5.final_rmse_C_um.median:.3f}} [{{noise_sensitivity.levels.5.final_rmse_C_um.q25:.3f}}–{{noise_sensitivity.levels.5.final_rmse_C_um.q75:.3f}}] | {{noise_sensitivity.levels.10.final_rmse_C_um.median:.3f}} [{{noise_sensitivity.levels.10.final_rmse_C_um.q25:.3f}}–{{noise_sensitivity.levels.10.final_rmse_C_um.q75:.3f}}] |
| 90% width of `k_pad` / estimate | {{noise_sensitivity.levels.1.final_relative_width.k_pad.median:pct2}}% | {{noise_sensitivity.levels.2.final_relative_width.k_pad.median:pct2}}% | {{noise_sensitivity.levels.5.final_relative_width.k_pad.median:pct2}}% | {{noise_sensitivity.levels.10.final_relative_width.k_pad.median:pct2}}% |
| 90% width of `λ` / estimate | {{noise_sensitivity.levels.1.final_relative_width.lam.median:pct1}}% | {{noise_sensitivity.levels.2.final_relative_width.lam.median:pct1}}% | {{noise_sensitivity.levels.5.final_relative_width.lam.median:pct1}}% | {{noise_sensitivity.levels.10.final_relative_width.lam.median:pct1}}% |
| coverage k_pad / λ / K (of {{robustness.n_draws:d}}) | {{noise_sensitivity.levels.1.coverage_90.k_pad.inside:d}} / {{noise_sensitivity.levels.1.coverage_90.lam.inside:d}} / {{noise_sensitivity.levels.1.coverage_90.K.inside:d}} | {{noise_sensitivity.levels.2.coverage_90.k_pad.inside:d}} / {{noise_sensitivity.levels.2.coverage_90.lam.inside:d}} / {{noise_sensitivity.levels.2.coverage_90.K.inside:d}} | {{noise_sensitivity.levels.5.coverage_90.k_pad.inside:d}} / {{noise_sensitivity.levels.5.coverage_90.lam.inside:d}} / {{noise_sensitivity.levels.5.coverage_90.K.inside:d}} | {{noise_sensitivity.levels.10.coverage_90.k_pad.inside:d}} / {{noise_sensitivity.levels.10.coverage_90.lam.inside:d}} / {{noise_sensitivity.levels.10.coverage_90.K.inside:d}} |

* **Stiffness interval:** grows roughly in proportion to σ.
* **Prediction error:** grows much more slowly, from {{noise_sensitivity.levels.1.final_rmse_C_um.median:.3f}} to {{noise_sensitivity.levels.10.final_rmse_C_um.median:.3f}} µm, against an oracle level of {{noise_sensitivity.levels.1.final_rmse_oracle_um.median:.3f}} µm. A scan has thousands of points, so even 10 µm noise pins the removal scale well. At 1–2 µm noise the remaining error is mostly the unpredictable wear fluctuation; at 10 µm, parameter uncertainty adds about as much again.
* **`λ` interval:** set mainly by that fluctuation, so it hardly changes.

![Ablation and noise](figures/fig6_ablation_and_noise.png)

## 5. Limitations

* **One abrasive.** There is a single grit and a single wear law. Grit changes, loading versus dulling, and recovery after cleaning are not modelled.
* **Winkler pad, not a full FE pad.** The independent-spring pad has no shear coupling, no bending of the backing plate, and a linear law even at the {{model_info.pad_penetration_at_prior_bounds.20N.soft_k_low_mm:.1f}}–{{model_info.pad_penetration_at_prior_bounds.40N.soft_k_low_mm:.1f}} mm indentations of the softest pads. The pad is held tangent at its centre even when it overhangs an edge, so it never tilts.
* **Contact uses the nominal (CAD) surface.** Micrometre-scale removal is not fed back into the contact. That is a good approximation when removal per pass is much smaller than the pad indentation, and weakest for the stiffest pads, whose indentation (~{{model_info.pad_penetration_at_prior_bounds.30N.stiff_k_high_mm:.3f}} mm at 30 N) is comparable to the removal.
* **No heat, no surface roughness, no material response.** The model predicts removal depth only.
* **Wear is updated once per pass.** That is a coarse step for fast-wearing paper (section 3).
* **Simulated scans.** The noise is independent and Gaussian, with perfect registration and no outliers or missing data.
* **The wear law's form is assumed.** It is exponential in removed volume, and the tracker is told the form and its fluctuation level.
* **Best case for the tracker.** The truth and the tracker share physics, priors and noise levels (apart from the surrogate's {{model_info.surrogate.max_relative_rms_error:.1e}} interpolation error), so there is no model mismatch. Real performance would be worse, and real intervals would need to be wider.
* **The grid quantises stiff contact.** At the stiff end the 1 mm grid resolves the contact strip with only a few nodes across; see the 1 mm vs fine-grid contact fractions in section 3.
* **Weak results.** Coverage of `λ` and `K0` is slightly below nominal (section 4.2), and the 20/40 N schedule did not measurably help identifiability (section 4.4).
* **Not validated on physical coupons.**

## 6. Next steps that would make it real

1. **Fit the wear law to physical coupon data.** Scan coupons after every pass, at several forces and grits. Test whether removed volume or force × distance drives the decay, and measure the real pass-to-pass fluctuation.
2. **Replace the Winkler pad** with a finite-element or learned pad model, including pad tilt over edges and non-linear foam. Then check whether stiffness is still pinned by a single removal map. If it is not, the force schedule becomes important.
3. **Add roughness prediction** alongside removal depth, since finish quality is usually the acceptance criterion.
4. **Test on GrayMatter-style scan data**, with registration error, missing data and real noise, and with model mismatch between the simulator and the tracker.
5. **Use the tracker for decisions.** Pick the next pass's force or dwell to hit a target removal, and choose the abrasive change from the predicted crossing distribution and the cost of a bad pass.

## 7. How to reproduce

Requires Python ≥ 3.11. The recorded run used Python {{info.python}} and NumPy {{info.numpy}}; see `results/run_info.json`.

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib, pyyaml, pytest
python -m swt run-all                    # every experiment, every figure, every number in results/
python -m swt report                     # re-render README.md and SUMMARY.md from results/
python -m swt run --config configs/default.yaml    # main 20-pass experiment only
python -m pytest -q                      # tests
```

* **Runtime.** `run-all` uses `min(4, CPUs)` worker processes for the multi-draw experiments. In the recorded run, building the model and surrogate took {{info.context_build_seconds:.0f}} s and the experiments took {{info.experiment_seconds:.0f}} s, with {{info.workers:d}} workers on a {{info.cpu_count:d}}-CPU machine. `--workers 1` runs serially; results are identical for any worker count because every draw has its own seeds.
* **Outputs.** `results/` holds per-pass CSVs for every run, per-draw CSVs, JSON summaries and `run_info.json`. `summary.json` gathers everything; `run_info.json` has the config, versions and a hash of the source code. `figures/` holds the six PNGs, drawn only from `results/` (`python -m swt figures` redraws them).
* **How the documents are made.** `README.md` and `SUMMARY.md` are rendered from `docs/*.template.md`, in which every number is a lookup into `results/summary.json` or `results/run_info.json`. A test fails if the shipped documents differ from what the templates render from the shipped results.
* **Changing the model.** Edit `configs/default.yaml` (geometry, pad, sander, path, schedule, scan, priors, filter, experiments) and rerun. The prose in the templates describes the default configuration; with other settings, re-read it before relying on it.

Repository layout:

```
swt/geometry.py     panel height map and raster toolpath
swt/pad.py          Winkler contact and force balance
swt/process.py      Preston removal, sliding speed, wear, hidden-truth simulator
swt/scan.py         noisy, downsampled scans
swt/surrogate.py    exposure table and O(1) likelihood terms for the filter
swt/estimators.py   A nominal, B calibrate-once, C particle filter
swt/experiments.py  all experiments; writes results/
swt/plots.py        all figures, drawn from results/ only
swt/report.py       renders README.md / SUMMARY.md from docs/*.template.md and results/
swt/cli.py          command-line entry point (python -m swt ...)
configs/default.yaml
docs/               README and SUMMARY templates
tests/              pytest suite (incl. an end-to-end run on a tiny configuration)
```
