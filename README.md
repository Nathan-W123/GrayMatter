# sanding-wear-tracker

A simulation study of tracking sanding-pad stiffness and abrasive wear together, so that a sanding process model could stay accurate as the paper dulls and say when to change it. The tracker is tested both inside its own model and in a "realistic" simulated world with five effects it does not model. All results are simulated; nothing has been tested on physical parts.

![Prediction error per pass](figures/fig3_rmse_per_pass.png)

## 1. Summary

A model that predicts how much material a robotic sanding pass removes needs two things it cannot measure directly: how the compliant pad presses on the part (set by a stiffness that is fixed but unknown) and how sharp the abrasive is (which drops every pass). This project simulates a 125 mm pad sanding a curved panel with hidden stiffness and wear parameters and compares four predictors of the next pass's removal map: a nominal model with fixed prior-mean parameters (A), a model calibrated once on the first scan (B), an engineering baseline that refits stiffness and effectiveness to every scan and extrapolates a fitted wear rate (D), and a particle filter that tracks stiffness, current effectiveness and a drifting wear rate with a robust scan model (C). In a *realistic* world that adds a foam pad that stiffens as it compresses, force-calibration error and line-to-line force ripple, a two-stage wear law and scanner artefacts, none of which any predictor models, C's median pass-20 error over 100 hidden truths was 0.171 µm, against 0.209 µm for D (C better in 72 of 100 draws), 2.23 µm for A, 10.93 µm for B and 0.142 µm for an oracle that knows the true physics and state (median true removal 5.14 µm); from the last pass before the abrasive fell below 50% of fresh effectiveness, C named that pass to within one pass in 97 of 100 draws (D: 79). In the tracker's own (matched) world C stays close to the oracle (0.056 vs 0.042 µm) and its 90% intervals are calibrated (92% predictive coverage; 92–100 of 100 for the parameters), but the realistic world exposes three weaknesses: its 90% predictive intervals for the next pass's mean removal cover only 73% (55% in passes 2–5, 79% from pass 11), its parameter estimates become effective values of its simplified model rather than the true ones, and abrasive-change forecasts made many passes ahead are no better than the nominal schedule because the fast break-in of the abrasive is extrapolated.

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
* **The part** is a height map on a 1 mm grid (30351 nodes): a convex cylindrical section, 200 mm along the straight x direction and 150 mm across the curved y direction, radius 300 mm (edges 9.5 mm below the crown).
* **One pass** is a serpentine raster: 11 lines along x, 15 mm apart; pad-centre stations every 5 mm (451 per pass); feed 25 mm/s; 0.25 s dwell at both ends of each line; 93.5 s per pass. The pad centre runs to the panel edge, so edge stations overhang the part.
* **Between passes** only the commanded normal force changes (alternating 20 N / 40 N). Spindle speed is part of the action but constant here.

### 3.2 Pad contact (`swt/pad.py`)
At each station the pad face is held tangent to the surface at the pad centre; a point under the pad lies a gap `g` below the pad plane (0 at the centre, 6.58 mm at the rim of the 125 mm pad). Pushing the pad in by `d` gives a Winkler pressure and a force balance:

```
p = k_pad · max(0, d − g)          [MPa]
Σ_i p_i · A_i = F                  (A_i = pad-face area over grid node i)
```

The balance is piecewise linear in `d` and is solved exactly (sorted gaps, one linear piece); a Brent solver gives the same root in the tests. A stiff pad touches a thin strip under its centre; a soft pad wraps around the curvature.

**Stiffness prior.** At 30 N, contact runs from **5.1% of the pad face for the stiffest pad to 97.3% for the softest**, reproducing the 5–97% range quoted in GrayMatter's post (their geometry is not published, so this calibrates the prior to the quoted range rather than reproducing their setup). At 20 N and 40 N the range is 5.1–89.0% and 7.1–100.0%; on a 0.25 mm grid the 30 N range is 5.8–97.0% (the stiff end is quantised by the 1 mm grid). The softest pads indent 4.1–6.8 mm at 20–40 N, beyond where a linear law holds for real foam; the realistic world's foam pad addresses this (section 3.6).

![Contact fraction vs stiffness](figures/fig1_contact_vs_stiffness.png)

### 3.3 Removal (`swt/process.py`): Preston's law with wear during the pass
```
dh = K · p · v · dt
```
`v` is the RMS sliding speed of a random-orbital sander (5 mm orbit at 8000 rpm plus free pad rotation at 2% of spindle speed), `v(r) = sqrt((π·D_orb·f)² + (2π·f·0.02·r)²)`, from 2094 mm/s at the pad centre to 2342 mm/s at the rim. With nominal parameters a fresh abrasive removes on average 8.1 µm at 20 N and 16.1 µm at 40 N.

**Wear.** `K = K0 · exp(−λ·W)`, with `W` the cumulative removed volume (by analogy with the grinding G-ratio; an assumption, untested here). Wear acts *during* the pass: within a pass `K(U) = K_start / (1 + λ·K_start·U)`, where `U` is the volume the pass would have removed at unit effectiveness so far. It is applied at the resolution of one raster line, using each line's exact average effectiveness, so the volume of the removal map is exactly the volume that drives the wear (a test checks this to 1e-9). Between passes a small multiplicative fluctuation stands in for abrasive variability:

```
K_{n+1} = K_n · exp(−λ·ΔV_n + σ_w·ξ_n),   ξ_n ~ N(0, 1),   σ_w = 0.01
```

### 3.4 Scan (`swt/scan.py`)
The scanner reports each pass's removal on a 2 mm sub-grid (7676 points, read as profiles along y, one every 2 mm in x) with independent Gaussian noise, σ = 2 µm. In the realistic world it also has artefacts (section 3.6).

### 3.5 Hidden truth and priors
| Parameter | Prior | Notes |
|---|---|---|
| `k_pad` | log-uniform on [0.00063, 5] N/mm³ | 97% to 5% contact at 30 N |
| `K0` (fresh effectiveness) | log-normal, median 6.0e-5 mm²/N, σ_log 0.25 | |
| `λ` (wear rate) | log-normal, median 2.5e-4 1/mm³, σ_log 0.35 | |
| wear fluctuation | σ_w = 0.01 per pass | the tracker's starting assumption |

Each draw id fixes `k_pad`, `K0`, `λ` (and the realistic world's force error) through its own random streams, so every world and schedule sees the same hidden parameters for the same draw. Draws 0–99 are used for the reported experiments and draws 200–219 only for tuning (section 3.8).

### 3.6 The realistic world: five effects no estimator models
| Effect | What the hidden process does | Size |
|---|---|---|
| **Foam pad** | `p = k·δ / (1 − δ/h)`, h = 15 mm: the pad stiffens as it densifies (same small-strain stiffness `k`) | softest pad at 40 N: peak strain 35%, peak pressure ×1.19, contact 96% instead of 100%; negligible for mid and stiff pads (×1.01 at the nominal pad) |
| **Force-calibration error** | actual force = gain × commanded, one gain per run | log-normal, σ_log = 0.04 |
| **Force-control ripple** | each raster line runs at its own force, AR(1) along the pass (first-order exposure model, checked against the exact model to < 0.1%) | σ = 3% per line, correlation 0.5 between lines |
| **Two-stage wear** | `K/K0 = (1−f)·e^{−λW} + f·e^{−rλW}`: a fast break-in of the sharpest grit tips plus slow dulling (f = 0.25, r = 6), integrated within the pass | `K` halves after 1766 mm³ instead of 2773 mm³ (nominal λ) |
| **Scanner artefacts** | per-scan misregistration of the whole map; a constant offset per scan profile; outliers; missing points (isolated and 8-point gaps along profiles) | registration σ = 0.3 mm per axis; profile offset σ = 0.5 µm; 0.5% outliers of σ = 15 µm; 2% missing |

The force error and ripple appear in the breakdown as one group ("force errors"). The **oracle** reference always uses the hidden process's own physics.

### 3.7 Estimators (`swt/estimators.py`)
All four see the commanded action of every pass and the scans, and nothing else; before each pass each predicts that pass's removal at the scan points.

**A: nominal.** Prior-mean parameters (geometric-mean stiffness 0.0561 N/mm³, prior medians of `K0` and `λ`), wear propagated with its own predicted volume, never updated.

**B: calibrate-once.** Least-squares `k_pad` and `K` from the first scan (`K` profiled out, 1-D search over log `k_pad`), then frozen, no wear model. The "learn the pad once" baseline.

**D: refit each pass** (the strong engineering baseline). The same least-squares fit on *every* scan; the wear rate is the slope of a straight line through log `K_i` against cumulative removed volume, and the next pass uses the latest `k_pad` and `K` extrapolated with that slope. Its abrasive-change forecast extrapolates the same line. It gives point predictions only.

**C: joint tracker.** A sequential Monte Carlo filter over log `k_pad` (static) and the paths of log `K` and log `λ` (one value per pass), 2000 particles:
* *Process model:* within-pass wear as in 3.3; between passes `log K` follows the wear law plus noise of s.d. `σ_w`, and `log λ` takes a random-walk step of s.d. `δ` (the *wear-rate drift*), so the wear rate can follow the recent decay when the true law is not exponential. `δ` is tuned on held-out draws (3.8); `δ = 0` is the constant-rate model.
* *Update:* the scan likelihood is applied in tempered stages that keep the effective sample size at 75% of N. After each stage the particles are resampled systematically and moved by Metropolis–Hastings steps that leave the exact path posterior invariant: a joint random walk on log `k_pad` and a common shift of the whole log `λ` path, local moves on log `λ` at the last two passes, and moves on log `K` at pass 1 (= `K0`) and at the current pass. Every past scan is kept as O(1) sufficient statistics, so the full path posterior is evaluated in every move and the static stiffness does not degenerate.
* *Robust measurement model:* (i) points more than 5 robust s.d. from C's own prediction for the pass are rejected (on pass 1, from the fitted map, with one redo); (ii) a random offset per scan profile is added to the noise model when the profile means vary significantly more than the noise explains (variance estimated from the residuals, folded exactly into the sufficient statistics via the Woodbury identity); (iii) the noise level is re-estimated from the residuals and inflated when they are spatially correlated. In the matched world they almost never act (median 0 rejected points per scan, 4 of the 100 × 20 updates redone); in the realistic world the median is 18 rejected points per scan and 484 updates were redone.
* *Adaptive process noise:* `σ_w` is replaced by the variance implied by C's own one-step forecast errors of log `K` (shrunk towards 0.01 with a weight of 4 passes, never below it), so the predictive spread widens when `K` changes from pass to pass more than the model expects.
* *Speed:* removal depends on stiffness only through `η = F/k_pad`; maps at the scan points are tabulated once on 512 log-spaced η values, with the within-pass wear expanded to third order. Against the exact model the relative RMS error is at most 1.7e-03 (median 3.4e-05, 90 checks). The hidden process and A, B use the exact model.
* *Outputs:* posterior median and 90% interval of `k_pad`, `K0`, current `K` and current `λ`; a 90% predictive interval of the next pass's mean removal; and the predicted abrasive-change pass with a ≥90% band (particles rolled forward with random future wear and wear-rate drift).

**Oracle (reference).** Knows the hidden process's physics and its true state after the previous pass, but not the wear fluctuation drawn since then, nor the coming pass's force ripple. Its error is a floor in expectation; on single passes C can beat it by luck.

### 3.8 Tuning on held-out draws
The wear-rate drift `δ` is the only tuned setting. `run-all` runs C with each `δ` of 0, 0.05, 0.1, 0.15 and 0.2 on held-out draws 200–219 in the matched and realistic worlds, and keeps the value with the lowest mean *interval score* of the 90% predictive interval of next-pass mean removal (a proper scoring rule: interval width plus 20× any miss; Gneiting & Raftery 2007). It chose **δ = 0.05**. Other settings (75% ESS target, 2 MH sweeps, outlier threshold 5, 4 passes of prior weight for the process noise) were fixed during development, which used held-out draws 200–239 and short runs on draws 0–3; no setting was changed after the full results on draws 0–99 had been seen.
## 4. Results

All numbers come from `results/` and were produced by `python -m swt run-all` with seed 2026. Unless a section says otherwise, every run has 20 passes, forces alternating 20/40 N, 2 µm scan noise and the 50%-of-K0 abrasive-change threshold. Errors are RMSE over the 7676 scan points against the simulator's noise-free removal (not against the scan).

**At a glance** (100 hidden truths per world, pass 20; RMSE median [IQR] in µm):

| | A: nominal | B: calibrate-once | D: refit each pass | C: joint tracker | oracle |
|---|---|---|---|---|---|
| matched world | 2.20 [1.52–4.24] | 10.62 [8.92–13.16] | 0.112 [0.091–0.144] | **0.056** [0.024–0.081] | 0.042 |
| realistic world | 2.23 [1.39–3.83] | 10.93 [9.16–13.77] | 0.209 [0.152–0.289] | **0.171** [0.123–0.241] | 0.142 |

| C's calibration (target 90%) | matched | realistic |
|---|---|---|
| next-pass mean removal inside the 90% predictive interval, passes 2–20 | 1752 / 1900 (92.2%) | 1384 / 1900 (72.8%) |
| … truth below / above the interval | 75 / 73 | 102 / 414 |
| … passes 2–5 / passes 11–20 | 93% / 92% | 55% / 79% |
| true parameter inside the 90% interval at pass 20: k_pad / λ / K / K0 (of 100) | 92 / 100 / 93 / 93 | 8 / 57 / 29 / 17 (effective parameters; see 4.2) |

With 100 draws, the binomial standard error of a calibrated 90% parameter coverage is 3 points. The λ interval is wider than needed in the matched world because C allows the wear rate to drift while the matched truth's rate is constant.

### 4.1 Main run: one hidden truth in both worlds

The draw is chosen by a rule that looks only at the sampled truths: among the 100 robustness draws, the one at the median standardised distance from the prior centre in (log k_pad, log K0, log λ). It is draw 15: `k_pad` = 0.2034 N/mm³, `K0` = 4.93e-05 mm²/N, `λ` = 0.000392 1/mm³, and in the realistic world a force gain of 0.979. Its abrasive crosses 50% at pass 11 in the matched world and at pass 7 in the realistic world (the break-in brings it forward); A's nominal model says pass 13.

| Removal-map RMSE [µm] | A | B | D | C | oracle |
|---|---|---|---|---|---|
| realistic, pass 2 | 5.02 | 1.68 | 1.10 | 0.43 | 0.37 |
| realistic, pass 20 | 2.35 | 8.23 | 0.192 | 0.160 | 0.106 |
| matched, pass 20 | 1.91 | 8.89 | 0.083 | 0.090 | 0.065 |

![Removal maps](figures/fig2_removal_maps.png)

* **B** is good right after its calibration but never learns that the paper is dulling, so its error grows with every pass.
* **A** has the wear law but the wrong parameters.
* **D and C** both follow the wear. On this one draw they are close, and D is slightly better at pass 20 in the matched world; single passes are noisy, and section 4.2 has the statistics over 100 draws.
* **Parameters at pass 20 (figure 4):**

  | | C: median [90% interval], matched world | C: median [90% interval], realistic world | truth |
  |---|---|---|---|
  | `k_pad` [N/mm³] | 0.2035 [0.2028, 0.2043] | 0.2087 [0.2077, 0.2098] | 0.2034 |
  | `λ` [1/mm³] | 0.000356 [0.000299, 0.00042] | 0.000403 [0.000335, 0.000488] | 0.000392 |
  | `K0` [mm²/N] | 4.95e-05 [4.92e-05, 4.98e-05] | 4.58e-05 [4.53e-05, 4.63e-05] | 4.93e-05 |

  In the realistic world the same tracker converges on an *effective* stiffness and an effective wear rate that is high during the break-in and falls later: the best linear-pad, exponential-wear description of a process that is neither, with the force gain absorbed into the effectiveness. Its map predictions remain good because they only need the effective values to reproduce the removal, not to equal the true parameters.

![Parameter tracking](figures/fig4_parameter_tracking.png)

### 4.2 Robustness: 100 hidden truths per world

* **Map accuracy.** C has the lowest median error of the four predictors at every pass after the first in the matched world, and at almost every pass in the realistic world, where D comes close (figure 3). At pass 20, D's error is 2.04× C's (median ratio) in the matched world and 1.18× in the realistic world; C is better than D in 94 and 72 of 100 draws at pass 20, and in 76 and 70 averaged over passes. In the realistic world C's pass-20 error is 1.28× the oracle's (median) and 3.6% of the true removal.
* **The advantage over D shrinks under mismatch.** D takes `k_pad` and `K` from the latest scan alone and fits a straight line for the wear; C pools every scan in one posterior and models wear within the pass. In the matched world that structure is exactly right; in the realistic world part of the advantage is lost to the effects neither models.
* **Calibration.** In the matched world the predictive intervals and parameter intervals are close to nominal. In the realistic world the predictive intervals under-cover, most of all in the first passes, before C's adaptive noise estimate has seen enough forecast errors, and the misses are mostly on one side (the truth above the interval: the abrasive keeps cutting better than an exponential law fitted through the break-in predicts). Parameter intervals in the realistic world describe effective parameters, so their "coverage" of the true foam-pad stiffness, gain-scaled `K` or two-stage `λ` is not expected to be 90% and is reported for completeness.

### 4.3 What each unmodelled effect costs

Each effect of the realistic world was also run alone on the first 40 draws (same draws in every world; figure 7, left).

| World (40 draws, pass 20) | C RMSE [µm] | oracle [µm] | C / oracle | D RMSE [µm] | predictive coverage | crossing error, last pass before (mean, passes) |
|---|---|---|---|---|---|---|
| matched | 0.060 | 0.044 | 1.26 | 0.109 | 92% | 0.14 |
| foam pad only | 0.077 | 0.044 | 1.46 | 0.132 | 91% | 0.16 |
| force errors only | 0.172 | 0.139 | 1.12 | 0.180 | 76% | 0.45 |
| two-stage wear only | 0.049 | 0.038 | 1.42 | 0.098 | 87% | 0.25 |
| scanner artefacts only | 0.084 | 0.044 | 1.69 | 0.159 | 85% | 0.27 |
| realistic (all five) | 0.171 | 0.123 | 1.47 | 0.216 | 73% | 0.53 |

* **Force errors** raise everyone's error, the oracle's included: the line-to-line ripple is unpredictable, and it is the largest single cause of C's under-coverage (the misses are on both sides: intervals too narrow until the adaptive process noise catches up).
* **Two-stage wear** biases the predictions (misses mostly on one side) and the long-range crossing forecasts (section 4.4).
* **Scanner artefacts:** C does not correct misregistration; per-profile offsets are absorbed by its offset model and spikes by its gating (the tests check both). Its misses in this world also lean towards the truth being above the interval (83 above, 33 below), a bias I have not traced.
* **The foam pad** changes the map noticeably only for soft pads and costs the least of the five; stiffness becomes an effective value.

![Mismatch and calibration](figures/fig7_mismatch_and_calibration.png)

### 4.4 Abrasive-change prediction (threshold: K < 50% of K0)

| Prediction made | world | n | C mean abs. error [passes] | D | A (nominal) | C within ±1 pass | C's ≥90% band contains the truth |
|---|---|---|---|---|---|---|---|
| after pass 2 | matched | 100 | 0.79 | 2.02 | 3.75 | 81 | 100% |
| last pass before the crossing | matched | 91 | 0.13 | 0.23 | 2.93 | 90 | 99% |
| after pass 2 | realistic | 100 | 2.06 | 2.12 | 4.01 | 46 | 56% |
| after pass 5 | realistic | 93 | 1.14 | 1.18 | 3.71 | 65 | 78% |
| last pass before the crossing | realistic | 100 | 0.41 | 0.95 | 4.01 | 97 | 92% |

Only predictions made before the true crossing count; the true crossing is simulated beyond pass 20 when needed. In the matched world C is more accurate than A at every lead time and than D at every lead but one, where they tie (figure 5, right). In the realistic world, early forecasts are biased early (mean signed error after pass 2: -1.24 passes): the break-in makes the abrasive look as if it wears faster than it will, and neither C (whose wear-rate drift lets λ fall only gradually) nor D's straight-line fit anticipates the slow-down. At long leads this is no better than the fixed nominal schedule (figure 5, right), and at intermediate leads D's straight-line forecast is slightly better than C's: on the 37 realistic truths whose crossing every lead can see, the mean error 5 passes ahead is 0.92 passes for C and 0.76 for D, against 0.46 and 1.38 one pass ahead. Close to the crossing, C is the most accurate and its band is calibrated.

![Abrasive change](figures/fig5_abrasive_change.png)

### 4.5 Tuning: what the drifting wear rate buys

| wear-rate drift δ (held-out draws 200–219) | 0 (constant rate) | chosen: 0.05 |
|---|---|---|
| predictive coverage, matched | 91.8% | 91.8% |
| predictive coverage, realistic | 70.5% | 76.3% |
| median relative width of the 90% interval, realistic | 7.41% | 6.90% |
| mean interval score (lower is better; both worlds) | 0.1104 | 0.0996 |

Larger drifts raise the realistic world's coverage further but widen the intervals more than they gain (figure 7, right), so the proper scoring rule stops at 0.05. The choice is made inside `run-all`, on draws not used anywhere else.

### 4.6 Identifiability ablation: constant 30 N instead of alternating 20/40 N

On the first 50 draws of the matched world, a constant 30 N schedule (same mean force) was compared with the alternating one, and a re-run of the alternating schedule with a different filter seed gives the spread expected from Monte Carlo noise alone.

| Paired over draws, pass 20 | constant ÷ alternating | other seed ÷ alternating |
|---|---|---|
| k_pad 90% width, median ratio (draws wider) | 0.968 (22 of 50) | 0.999 (25) |
| λ 90% width, median ratio (draws wider) | 1.076 (49) | 0.992 (21) |

**This did not show what was expected.** Alternating the force did not narrow the stiffness interval at all. Because the force is commanded and known, a single force level already pins `F / k_pad` from the shape of one removal map; a second force adds a second view of the same thing. Alternating did narrow the wear-rate interval slightly and consistently, because two forces remove different volumes and so trace the wear law at two rates. Varying the force would matter if something else also changed the contact shape (an unknown force offset, uncertain curvature, a drifting pad); with the realistic world's force-calibration error it could help separate gain from effectiveness, which was not tested.

![Ablation and noise](figures/fig6_ablation_and_noise.png)

### 4.7 Scan-noise sensitivity (realistic world, first 50 draws)

| scan noise σ [µm] | 1 | 2 | 5 | 10 |
|---|---|---|---|---|
| C pass-20 RMSE, median [µm] | 0.187 | 0.165 | 0.179 | 0.179 |
| oracle, median [µm] | 0.123 | 0.123 | 0.123 | 0.123 |
| predictive coverage | 70% | 73% | 73% | 72% |

In the realistic world C's error barely depends on scan noise up to 10 µm: with thousands of points per scan the noise averages out, and the error is dominated by what neither scans nor the oracle can predict (force ripple, fluctuations) and by model mismatch.

## 5. Limitations

* **Simulated only.** Nothing here has touched a physical part. The realistic world is a richer simulator, not reality: its five effects and their sizes were chosen by me, not measured, and real processes will contain effects that are in neither world (loading and clogging of the abrasive, heat, grit changes, pad wear, robot path errors).
* **Same author for the world and the tracker.** The scanner-offset model and outlier gating address artefacts that I also put into the realistic world; per-profile offsets and spikes are well-known line-scanner artefacts, but a tracker designed without knowing the test world would likely do worse.
* **Under-coverage under mismatch.** C's 90% predictive intervals cover 73% in the realistic world (55% in passes 2–5). Decisions taken on those intervals early in an abrasive's life would be over-confident.
* **Parameters are only meaningful in the matched world.** Under mismatch C's stiffness and wear rate are effective values; they should not be read as physical properties of the pad or the abrasive.
* **Long-range abrasive-change forecasts** are poor when the wear law has a break-in phase (section 4.4); only forecasts within a few passes of the crossing are reliable there.
* **Registration error is not corrected.** C treats misregistration as structured noise; it biases the effective stiffness.
* **Winkler-type pad.** Both pad laws are independent springs: no shear coupling, no bending of the backing plate, no tilt over an overhanging edge. The contact uses the nominal (CAD) surface; removal is not fed back into the contact.
* **One abrasive, removal depth only.** One grit and one wear law per world; no roughness or other finish metric.
* **Grid effects.** The 1 mm grid quantises the stiffest contact strips (section 3.2).
* **Surrogate.** C's table has a worst-case relative error of 1.7e-03 against the exact model, at the fastest-wearing draws (third-order expansion of the within-pass wear).
* **Not validated on physical coupons.**

## 6. Next steps that would make it real

1. **Fit the wear law to coupon data.** Scan coupons after every pass at several forces and grits, test whether removed volume or force × distance drives the decay, measure break-in and the real pass-to-pass fluctuation. This decides whether the tracker needs a break-in state.
2. **Add a transient per-pass removal term** (force ripple, local loading) to C's model, separate from the persistent wear, so that early-pass intervals are calibrated instead of waiting for the adaptive noise estimate.
3. **Register each scan to the prediction** (two shifts per scan as nuisance parameters) before updating.
4. **Replace the spring pad** with a finite-element or learned pad model, including tilt over edges.
5. **Run on GrayMatter-style scan data** with real registration error, missing data and noise.
6. **Use the tracker for decisions**: choose the next pass's force or dwell to hit a target removal, and the abrasive change from the predicted crossing distribution and the cost of a bad pass.

## 7. How to reproduce

Requires Python ≥ 3.11. The recorded run used Python 3.11.15 and NumPy 2.4.6; see `results/run_info.json`.

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib, pyyaml, pytest
python -m swt run-all                    # every experiment, every figure, every number in results/
python -m swt report                     # re-render README.md and SUMMARY.md from results/
python -m swt run --config configs/default.yaml    # main 20-pass experiment only (uses results/tuning.json if present)
python -m pytest -q                      # tests
```

* **Runtime.** `run-all` runs 610 episodes (tuning included) using `min(4, CPUs)` worker processes. In the recorded run, building the model and surrogate took 7 s and the experiments 427 s with 4 workers on a 4-CPU machine. `--workers 1` runs serially; results are identical for any worker count because every task has its own seeds.
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
