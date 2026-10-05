# sanding-wear-tracker

A simulation study of tracking sanding-pad stiffness and abrasive wear together, so that a sanding process model could stay accurate as the paper dulls and say when to change it. All results are simulated; nothing has been tested on physical parts.

![Prediction error per pass](figures/fig3_rmse_per_pass.png)

## 1. Summary

A model that predicts how much material a robotic sanding pass removes needs two things it cannot measure directly: how the compliant pad presses on the part (set by its stiffness, which is fixed but uncertain) and how sharp the abrasive is (which drops every pass), so a model calibrated once on the first scan predicts the next pass well (median error 1.03 µm at pass 2) and then gets steadily worse as the paper wears (11.05 µm at pass 20). This project simulates a 125 mm compliant pad sanding a curved panel with hidden stiffness and wear parameters, and compares that calibrate-once model, a nominal model with fixed prior-mean parameters, and a particle filter that re-estimates stiffness, current abrasive effectiveness and wear rate from every post-pass scan. Over 100 random hidden truths, the filter's median pass-20 error was 0.053 µm, against 2.24 µm for the nominal model and 11.05 µm for the calibrate-once model, with a median true removal of 5.92 µm in that pass; that is close to the 0.043 µm of an oracle that knows the true parameters but not the random pass-to-pass wear fluctuation, and after two scans the filter also predicts the pass at which effectiveness falls below 50% of fresh with a median error of 1 pass (fixed nominal schedule: 3). These are best-case numbers, because the filter shares the simulator's physics, priors and noise levels and errors are measured against the noise-free simulated removal map, and two results were weaker than hoped: alternating the force between passes barely helped (no measurable effect on stiffness, slightly narrower wear-rate intervals with no lower error), and the 90% intervals for wear rate and fresh effectiveness contained the truth in 86 and 86 of 100 runs, slightly below nominal (stiffness 89, current effectiveness 90).

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
* **The part** is a height map `z(x, y)` on a 1 mm grid (30351 nodes). It is a convex cylindrical section, 200 mm along the straight x direction and 150 mm across the curved y direction, with radius 300 mm. The crown runs down the middle and the edges are 9.5 mm lower.
* **One pass** is a serpentine raster:
  * lines run along x, with a 15 mm stepover in y (11 lines);
  * pad-centre stations are 5 mm apart (451 per pass) and the feed is 25 mm/s;
  * *dwell* means 0.25 s of extra time at the two ends of every line (the turnaround);
  * one pass takes 93.5 s.

  The pad centre runs all the way to the panel edge, so edge stations overhang the part and remove more material there. Edge over-sanding is common in practice, but its size here is untested.
* **What varies between passes.** Path, feed and dwell are the same for every pass; only the commanded normal force changes, alternating 20 N / 40 N. Spindle speed is part of the action too, but is constant here.

### Pad contact (`swt/pad.py`): Winkler foundation
At each station the pad face is held tangent to the surface at the pad centre. A surface point under the pad lies a distance `g` below the pad plane: `g = 0` at the centre, rising to 6.58 mm at the rim of a 125 mm pad on this cylinder. Pushing the pad in by `d` gives

```
p = k_pad · max(0, d − g)          [MPa]
Σ_i p_i · A_i = F                  (force balance; A_i = pad-face area over grid node i)
```

**Solving the force balance.** It is monotone and piecewise linear in `d`, so the 1-D root is found exactly: bracket it between two sorted gaps, then solve the linear piece. A generic Brent root finder finds the same root in the tests. The integrated pressure equals the applied force at every station, including stations that overhang the edge.

**What stiffness does to the contact.** A stiff pad touches only a thin strip along the line under its centre. A soft pad wraps around the curvature.

**The stiffness prior** is chosen so that, at a reference force of 30 N, contact runs from **5.1% of the pad face for the stiffest pad to 97.3% for the softest**, reproducing the range quoted in GrayMatter's post. Because the post does not give its geometry, this is a calibration of the prior to the quoted range, not a reproduction of their setup. Three caveats:

- **Other forces give other ranges.** At the 20 N and 40 N forces actually used in the alternating schedule, the range is 5.1–89.0% and 7.1–100.0%.
- **The stiff end is quantised by the grid.** On a 0.25 mm grid the same 30 N range is 5.8–97.0% (figure 1, dashed line).
- **The softest pads need large indentations:** 5.5 mm at 30 N, and 4.1–6.8 mm over 20–40 N. That is beyond where a linear Winkler law holds for real foam, but it is what a 97% contact fraction on a 300 mm radius requires.

![Contact fraction vs stiffness](figures/fig1_contact_vs_stiffness.png)

### Removal (`swt/process.py`): Preston's law
```
dh = K · p · v · dt
```

**Sliding speed.** `v` is the RMS sliding speed of a random-orbital sander. It combines the orbital speed π·D_orb·f (5 mm orbit, 8000 rpm) with slow free rotation of the pad at 2% of spindle speed:

```
v(r) = sqrt((π·D_orb·f)² + (2π·f·0.02·r)²)
```

so it rises from 2094 mm/s at the pad centre to 2342 mm/s at the rim.

**Removal per pass** is `h(x) = K · Σ_stations p·v·dt`. Under the nominal parameters, a fresh abrasive removes 8.0, 12.0 and 16.1 µm on average at 20, 30 and 40 N.

### Wear
`K = K0 · exp(−λ·W)`, where `W` is the cumulative **removed volume** in mm³.

**This is an assumption, untested here.** Wear is taken to be driven by removed volume rather than by force × sliding distance, by analogy with the grinding G-ratio (material removed per unit of abrasive wear). Section 6 lists testing it on coupons.

**Per-pass update.** `K` is held constant within a pass and updated between passes, with a small random fluctuation standing in for abrasive variability (its level is assumed):

```
K_{n+1} = K_n · exp(−λ·ΔV_n + σ_w·ξ_n),   ξ_n ~ N(0, 1),   σ_w = 0.01
```

Because ΔV_n ∝ K_n, blunt paper both cuts and wears more slowly; without noise, the decay is hyperbolic in pass count.

**This is a coarse time step.** Over the 100 truths, the first 40 N pass alone reduced `K` by a median of 10% and at most 25%. The truth and all three estimators share this discretisation, so it does not bias the comparison, but a real abrasive would also leave a gradient within the pass.

### Scan (`swt/scan.py`)
The observed per-pass removal map is the true map plus independent Gaussian noise with σ = 2 µm. It is sampled on a regular 2 mm sub-grid (7676 points), a simplified stand-in for a line scanner.

### Hidden truth and priors
| Parameter | Prior | Notes |
|---|---|---|
| `k_pad` | log-uniform on [0.00063, 5] N/mm³ | 97% to 5% contact at 30 N |
| `K0` (fresh effectiveness) | log-normal, median 6.0e-5 mm²/N, σ_log 0.25 | |
| `λ` (wear rate) | log-normal, median 2.5e-4 1/mm³, σ_log 0.35 | nominal crossing of 50% of K0 at pass 12 |
| wear fluctuation | σ_w = 0.01 per pass | known to the tracker |

Each run draws `k_pad`, `K0` and `λ` from these priors with a fixed seed. The estimators never see them.

Over the 100 draws, the true crossing pass has a median of 13 (IQR 10–16). 90 of 100 cross within the 20 scanned passes; for the others, the truth is simulated past pass 20 to find the crossing.

### Why stiffness and wear can be told apart
* **Wear only changes the scale.** Preston's law is linear in `K`, so a sharper or blunter abrasive multiplies the whole removal map by a constant.
* **Stiffness changes the shape, and barely the scale.** The force balance fixes the total load at `F` for every stiffness.
  * *Scale:* the removed volume per unit `K` depends on `k_pad` only through where the contact sits relative to the faster-moving rim, plus a surface-slope factor. Across the whole prior that moves it by 1.7%.
  * *Shape:* stiffness sets the contact patch, whose width and pressure profile depend only on `F / k_pad` (figure 1, right). In the removal map, stiffer pads give stronger raster ripple between lines and different over-sanding at the panel edges. Figure 2 compares the main run's pad (0.20 N/mm³) with A's nominal pad (0.056 N/mm³); it does not include a soft pad.
* **Varying the force** gives two views of the same pad. At 20 N and 40 N the contact patch differs in a known way, so in principle a stiffness error cannot hide as a scale error.
  * In this model, though, a *single* force level already pins `F / k_pad`, and hence `k_pad`, because the force is commanded and known. The ablation in section 4.4 confirms that the alternating schedule adds little.
  * It would matter if something else also changed the patch shape: an unknown force offset, uncertain local curvature, a pad whose stiffness drifts, or a scanner too coarse to resolve the patch.

### Estimators (`swt/estimators.py`)
All three see the commanded action of every pass and the noisy scans, and nothing else. Before each pass, each predicts that pass's removal map.

**A: nominal.** Prior-mean parameters, taken as the mean in log space: `k_pad` = 0.0561 N/mm³ (the geometric mean of the prior range), plus the prior medians of `K0` and `λ`. Wear is propagated with the nominal `λ` and A's own predicted volume, and no parameter is ever updated.

**B: calibrate-once.** After pass 1, a least-squares fit of `k_pad` and `K` to the first scan. `K` is profiled out in closed form, followed by a 1-D search over log `k_pad`, using the same surrogate table as C. Both values are then frozen and there is no wear model. This is the "learn the pad once" baseline.

**C: joint tracker.** A particle filter with 10,000 particles over `[log k_pad, log K0, log λ, log K_current]`. `K0` is carried as a static component because the abrasive-change threshold is defined relative to it.
* *Predict:* between passes, `K_current` follows the wear law using each particle's own predicted removed volume, plus N(0, 0.01²) noise on log K. The other components are static.
* *Update:* after each scan, particles are weighted by the Gaussian likelihood of the whole scan. With thousands of pixels that likelihood is far sharper than the prior, so it is applied in tempered stages (progressive correction):
  1. choose each stage's exponent so the effective sample size stays at 75% of N;
  2. resample systematically;
  3. roughen with a Liu–West shrinkage kernel (a = 0.98), which keeps the cloud's mean and covariance.

  A final stage that leaves the effective sample size above 75% is not resampled; its weights are kept.
* *Speed:* removal depends on stiffness only through `η = F/k_pad`, so the exposure map `m(η)` is tabulated once with the exact contact model, on 512 log-spaced η values. The largest relative RMS interpolation error over 90 spot checks is 2.3e-04. With linear interpolation, the scan likelihood reduces to precomputed inner products, so each particle costs O(1). The hidden truth and the predictions of A and B use the exact model.
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

All numbers come from `results/` and were produced by `python -m swt run-all` with seed 2026. Unless a section says otherwise (constant 30 N in 4.4, other noise levels in 4.5), every run has:
- 20 passes;
- forces alternating 20/40 N;
- 2 µm scan noise;
- the 50%-of-K0 threshold.

All errors are measured against the simulator's noise-free removal map.

**At a glance**, over 100 hidden truths at pass 20:

| | A: nominal | B: calibrate-once | C: joint tracker | oracle |
|---|---|---|---|---|
| removal-map RMSE, median [IQR] µm | 2.24 [1.53–4.23] | 11.05 [9.36–13.72] | **0.053** [0.027–0.083] | 0.043 |
| 90% coverage (k_pad / λ / K / K0) | | | 89 / 86 / 90 / 86 of 100 | |
| abrasive-change error after pass 2, median (mean) passes | 3 (3.84) | no wear model | 1 (1.58) | |

### 4.1 Main run: one hidden truth

**How the draw was chosen.** A fixed rule that looks only at the sampled truths and never at estimator results: among the 100 robustness draws, take the one at the *median* standardised distance from the prior centre in (log k_pad, log K0, log λ). That is a typical draw rather than a tail draw or one that happens to match the nominal model.
- It is draw 15: `k_pad` = 0.2034 N/mm³, `K0` = 4.93e-05 mm²/N, `λ` = 0.000392 1/mm³.
- Its abrasive crosses 50% at pass 10; the nominal schedule says pass 12.
- An earlier version used an arbitrary draw id, which turned out to be an unusually fast-wearing truth. An intermediate rule (closest to the prior medians) picked a truth where the nominal model is almost exact. This rule avoids both. The robustness results in section 4.2 cover the full spread.

| Removal-map RMSE [µm] | A: nominal | B: calibrate-once | C: joint tracker | oracle |
|---|---|---|---|---|
| pass 1 (before any scan) | 3.51 | 3.51 | 2.30 | 0.00 |
| pass 2 | 4.14 | 0.97 | 0.28 | 0.12 |
| pass 20 | 1.93 | 9.35 | 0.091 | 0.066 |

* **B** is good right after its calibration, but never learns that the paper is dulling. At pass 20 it predicts a mean removal of 13.3 µm against a true 4.3 µm.
* **A** has the wear law but wrong parameters. The level error comes mostly from fresh effectiveness, the ripple difference from the pad:

  | Parameter | A uses | Truth |
  |---|---|---|
  | fresh effectiveness `K0` [mm²/N] | 6e-05 | 4.93e-05 |
  | pad stiffness `k_pad` [N/mm³] | 0.0561 | 0.203 |
  | wear rate `λ` [1/mm³] | 0.00025 | 0.000392 |

  | Mean removal [µm] | A predicts | True |
  |---|---|---|
  | pass 2 | 15.1 | 12.4 |
  | pass 20 | 5.8 | 4.3 |
* **C at pass 1** has seen no scan yet; its prediction is the prior-predictive average over all plausible pads.
* **C at pass 20.** Its posterior means and 90% intervals are:

  | Parameter | C: mean [90% interval] | Truth |
  |---|---|---|
  | `k_pad` [N/mm³] | 0.2034 [0.2027, 0.2041] | 0.2034 |
  | `λ` [1/mm³] | 0.000375 [0.000351, 0.000395] | 0.000392 |
  | `K` [mm²/N] | 1.584e-05 [1.572e-05, 1.597e-05] | 1.586e-05 |

  All four of C's 90% intervals (these three and `K0`) contain the truth at pass 20 (`results/main_run.json`, `C_final_inside_90`).
* **Crossing prediction.**

  | Prediction made after | Predicted crossing pass | Band |
  |---|---|---|
  | pass 1 | 15 | 9–29 |
  | pass 2 | 11 | 9–14 |
  | pass 3 | 10 | 9–11 |

  The truth is pass 10.

![Removal maps](figures/fig2_removal_maps.png)

![Parameter tracking](figures/fig4_parameter_tracking.png)

### 4.2 Robustness: 100 hidden truths

| | A: nominal | B: calibrate-once | C: joint tracker | oracle |
|---|---|---|---|---|
| final-pass RMSE, median [IQR] µm | 2.24 [1.53–4.23] | 11.05 [9.36–13.72] | **0.053** [0.027–0.083] | 0.043 [0.021–0.081] |
| RMSE averaged over the 20 passes, median µm | 2.51 | 5.81 | 0.261 | |
| draws where C is better at pass 20 | 100 / 100 | 100 / 100 | | |

For scale, the true mean removal is 8.0 µm in pass 1 and 5.92 µm in pass 20 (medians over draws). B's pass-20 error exceeds the true removal in 92 of 100 draws.

* **C against B, pass by pass** (draws where C's error is below B's):

  | Pass | Draws | Note |
  |---|---|---|
  | 1 | 52 | before any scan; B equals the nominal model |
  | 2 | 86 | the first pass after B's calibration; B's fresh one-scan fit is still ahead in the rest |
  | 3 to 20 | 100 at every pass | |

  C's 20-pass average is dominated by passes 1–2, before the wear rate is observable.
* **B against A.** By pass 20, B is worse than the uncalibrated nominal model A in 100 of 100 draws. A has a wear law, even a wrong one; B has none, and ignoring wear costs more than starting from wrong parameters. This does not mean calibration is useless: C is calibration that keeps going.
* **Parameter accuracy at pass 20.** C's posterior mean has a median absolute error of 0.15% for `k_pad`, 0.22% for the current `K`, and 2.4% for `λ`. B's one-scan stiffness estimate is off by a median 0.68%.

**Coverage of C's 90% credible intervals at pass 20:**

| `k_pad` | `λ` | `K_current` | `K0` | all of k_pad, λ, K inside |
|---|---|---|---|---|
| 89/100 | 86/100 | 90/100 | 86/100 | 69/100 (≈73 expected if the three were calibrated and independent) |

With 100 draws, a calibrated 90% interval gives a coverage estimate with a binomial standard error of about 3 percentage points.

- **`k_pad` and `K_current`** are on target.
- **`λ` and `K0`** are a little low: 86 and 86 against 90, with a standard error of about 3.

The per-pass records show where the shortfall comes from:

| Coverage of | pass 1 | pass 2 | pass 10 | pass 20 |
|---|---|---|---|---|
| `K0` | 88 | | 88 | 86 |
| `λ` | | 91 | 88 | 86 |

(out of 100)

- **`K0`'s** shortfall is already there after the first update and stays flat, so it comes from the first-scan posterior. One untested possibility is that the tempered update with roughening makes that posterior slightly too narrow.
- **`λ`'s** coverage declines over the passes. That fits the known weakness of particle filters with static parameters: repeated resampling and roughening narrow the interval a little more than the data justify.

Even with the filter matching the simulator (apart from the surrogate's 2.3e-04 interpolation error), coverage is not above nominal, so with real data the intervals would need to be wider.

### 4.3 Abrasive-change prediction (threshold: K < 50% of K0)

| Prediction made | draws | C: median \|error\| (mean) [passes] | C: within ±1 pass | C: band contains truth | A (fixed nominal schedule): median \|error\| (mean) |
|---|---|---|---|---|---|
| after pass 2 | 100 | 1 (1.58) | 59 | 98 | 3 (3.84) |
| after pass 5 | 99 | 1 (0.74) | 87 | 98 | 3 (3.81) |
| one pass before the crossing | 92 | 0 (0.16) | 92 | 92 | 3 (3.03) |

Only predictions made before the true crossing count, so rows can have fewer than 100 draws.

* **How early.** After two scans the tracker has seen one wear step, and already places the change point within ±1 pass in 59 of 100 runs.
* **Error against lead time** (figure 5, right), on a fixed population: the 65 draws that cross between pass 11 and 21, for which every lead from 1 to 10 passes is observable. C's mean error grows from 0.18 passes one pass ahead to 0.95 passes ten passes ahead. A's fixed schedule is off by 2.69 passes on average.
* **The band is conservative.** The crossing pass is an integer, so the 5–95% band (median width 6 passes after pass 2) holds more than 90% of the filter's own predictive probability. The median share inside the band is 95% after pass 2, 97% after pass 5 and 100% one pass before the crossing. That largely explains the empirical coverage: 98–100% at the decision points in the table, and 95–100% across the leads in figure 5.

![Abrasive change](figures/fig5_abrasive_change.png)

### 4.4 Identifiability ablation: constant force instead of alternating 20/40 N
The same 100 truths were rerun with 30 N on every pass, the same mean force. A *null control* reruns the alternating schedule with only the particle filter's random seed changed; it shows how much the comparison moves from Monte Carlo noise alone. Medians over draws at pass 20:

| | 90% width of `k_pad` / estimate | 90% width of `λ` | \|corr(log k_pad, log K)\| | C's RMSE / true removal | passes until `λ` width < 20% (never reached) |
|---|---|---|---|---|---|
| alternating 20/40 N | 0.74% | 13.0% | 0.032 | 0.85% | 5 (8) |
| constant 30 N | 0.82% | 13.7% | 0.028 | 0.90% | 5 (10) |
| null control (alternating, other filter seed) | 0.78% | 13.0% | 0.030 | 0.88% | 5 (9) |

Paired per-draw comparisons with the alternating run:

| compared with alternating | `k_pad` interval wider in | median width ratio | `λ` interval wider in | median width ratio |
|---|---|---|---|---|
| constant 30 N | 45 / 100 | 0.987 | 61 / 100 | 1.028 |
| null control | 55 / 100 | 1.011 | 45 / 100 | 0.988 |

Constant force against the null control directly: the `k_pad` interval is wider in 50 of 100 draws and the `λ` interval in 72 of 100.

**This mostly did not show what was expected.**

- **Stiffness.** Constant force did not widen the interval any more often than reseeding the filter does.
- **Wear rate.** Constant force gave a small but consistent loss of precision: the `λ` interval was slightly wider (median ratio 1.028). The narrower interval under alternating force did *not* come with better accuracy or coverage:

  | Schedule | median `λ` error at pass 20 | `λ` coverage |
  |---|---|---|
  | alternating | 2.37% | 86 |
  | constant | 2.22% | 90 |
  | null control | 2.21% | 85 |

- **Correlation between stiffness and current effectiveness.** At pass 20 they are essentially uncorrelated in the posterior under both schedules (median \|corr\| 0.032 alternating, 0.028 constant). After the first scan alone the correlation is larger: 0.28 after a 20 N first pass and 0.14 after a 30 N one.

The reason is in section 3: the commanded force is known, the contact shape depends only on `F/k_pad`, and the scale barely depends on `k_pad`, so one force level already identifies both parameters.

Two raw differences are not identifiability effects:

- **`K` interval width.** At pass 20 it is 1.11% (alternating) against 1.39% (constant), because pass 20 is a 40 N pass, which removes more material. At pass 19, a 20 N pass, the order reverses: 1.79% vs 1.36%.
- **Prediction error.** Normalised by the true removal, C's pass-20 error is 0.85% (alternating), 0.90% (constant) and 0.88% (control). Relative to the oracle it is 1.17×, 1.01× and 1.13×. Neither schedule gives a prediction advantage.

### 4.5 Scan-noise sensitivity

| scan noise σ | 1 µm | 2 µm | 5 µm | 10 µm |
|---|---|---|---|---|
| C final-pass RMSE, median [IQR] µm | 0.048 [0.026–0.085] | 0.053 [0.027–0.083] | 0.062 [0.031–0.090] | 0.078 [0.045–0.116] |
| 90% width of `k_pad` / estimate | 0.39% | 0.74% | 1.96% | 3.93% |
| 90% width of `λ` / estimate | 13.0% | 13.0% | 13.3% | 14.6% |
| coverage k_pad / λ / K (of 100) | 89 / 86 / 90 | 89 / 86 / 90 | 91 / 89 / 95 | 92 / 89 / 97 |

* **Stiffness interval:** grows roughly in proportion to σ.
* **Prediction error:** grows much more slowly, from 0.048 to 0.078 µm, against an oracle level of 0.043 µm. A scan has thousands of points, so even 10 µm noise pins the removal scale well. At 1–2 µm noise the remaining error is mostly the unpredictable wear fluctuation; at 10 µm, parameter uncertainty adds about as much again.
* **`λ` interval:** set mainly by that fluctuation, so it hardly changes.

![Ablation and noise](figures/fig6_ablation_and_noise.png)

## 5. Limitations

* **One abrasive.** There is a single grit and a single wear law. Grit changes, loading versus dulling, and recovery after cleaning are not modelled.
* **Winkler pad, not a full FE pad.** The independent-spring pad has no shear coupling, no bending of the backing plate, and a linear law even at the 4.1–6.8 mm indentations of the softest pads. The pad is held tangent at its centre even when it overhangs an edge, so it never tilts.
* **Contact uses the nominal (CAD) surface.** Micrometre-scale removal is not fed back into the contact. That is a good approximation when removal per pass is much smaller than the pad indentation, and weakest for the stiffest pads, whose indentation (~0.013 mm at 30 N) is comparable to the removal.
* **No heat, no surface roughness, no material response.** The model predicts removal depth only.
* **Wear is updated once per pass.** That is a coarse step for fast-wearing paper (section 3).
* **Simulated scans.** The noise is independent and Gaussian, with perfect registration and no outliers or missing data.
* **The wear law's form is assumed.** It is exponential in removed volume, and the tracker is told the form and its fluctuation level.
* **Best case for the tracker.** The truth and the tracker share physics, priors and noise levels (apart from the surrogate's 2.3e-04 interpolation error), so there is no model mismatch. Real performance would be worse, and real intervals would need to be wider.
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

Requires Python ≥ 3.11. The recorded run used Python 3.11.15 and NumPy 2.4.6; see `results/run_info.json`.

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib, pyyaml, pytest
python -m swt run-all                    # every experiment, every figure, every number in results/
python -m swt report                     # re-render README.md and SUMMARY.md from results/
python -m swt run --config configs/default.yaml    # main 20-pass experiment only
python -m pytest -q                      # tests
```

* **Runtime.** `run-all` uses `min(4, CPUs)` worker processes for the multi-draw experiments. In the recorded run, building the model and surrogate took 9 s and the experiments took 79 s, with 4 workers on a 4-CPU machine. `--workers 1` runs serially; results are identical for any worker count because every draw has its own seeds.
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
