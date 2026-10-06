# sanding-wear-tracker

A simulation study of tracking sanding-pad stiffness and abrasive wear together, so that a sanding process model could stay accurate as the paper dulls and say when to change it. The tracker is tested inside its own model, in a "realistic" simulated world with effects it does not model, and in two further worlds that were fixed before their results were seen. All results are simulated; nothing has been tested on physical parts.

![Prediction error per pass](figures/fig3_rmse_per_pass.png)

## 1. Summary

A model that predicts how much material a robotic sanding pass removes needs two things it cannot measure directly: how the compliant pad presses on the part (set by a stiffness that is fixed but unknown) and how sharp the abrasive is (which drops every pass). This project simulates a 125 mm pad sanding a curved panel with hidden stiffness and wear parameters and compares four predictors of the next pass's removal map: a nominal model with prior-mean parameters (A), a model calibrated once on the first scan (B), joint least squares over all scans with the tracker's own model structure (D), and a particle filter (C) that tracks stiffness, force dependence, current effectiveness, a drifting wear rate and a per-pass gain, with a robust scan model. They are tested in the tracker's own (*matched*) world, in a *realistic* world with five effects that no predictor models (a foam pad, a Preston pressure exponent, force errors, a two-stage wear law, scanner artefacts), and in two further mismatch worlds that were committed to the repository before any of their results existed.

* **Accuracy.** A and B are far behind (median pass-20 errors of about 2 µm and 10 µm, against about 5 µm of removal). In the matched world C and D are equally accurate at pass 20 (0.054 vs 0.057 µm; oracle 0.042 µm), and over passes 2–20 D is slightly *more* accurate (median error ratio D/C 0.94). Under mismatch C is the more accurate: over passes 2–20 it beats D in 84 of 100 realistic draws and in 33 and 34 of 40 in the two pre-registered worlds (median D/C 1.08, 1.07, 1.06); at pass 20 in the realistic world C's median error is 0.247 µm, D's 0.271 µm and the oracle's 0.110 µm.
* **Calibration.** C's 90% predictive intervals for the next pass's mean removal cover 96% of passes 2–20 in the matched world (conservative), 85% in the realistic world and 80% and 90% in the pre-registered worlds. Checked block by block (16 blocks per map) they cover 96% in the matched world but only 77%, 48% and 54% under mismatch: the *shape* of the removal map is where the simplified model fails.
* **When to change the abrasive.** Scored as a sequential decision in which a pass on worn abrasive costs three times a pass of abrasive life thrown away, C's risk-based rule has the lowest mean cost in all four worlds (0.86 cost-weighted passes per run in the realistic world, against 1.33 for D, 1.77 for D with a one-pass safety margin and 1.94 for a model-free extrapolation of the scans). Judged by the error alone, C's median rule beats D in three of the four worlds, D in the milder pre-registered one; and as a point forecast of the crossing many passes ahead, D's is the more accurate.
* **Weak spots.** Under mismatch C's parameters are effective values, not the true ones; early abrasive-change forecasts are biased early when the abrasive has a break-in phase; and abrasive that wears ring by ring, which changes the map's shape, defeats C and D alike. Everything is simulated; nothing was tested on physical parts.

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

### 3.6 The realistic world: effects no estimator models
| Group | What the hidden process does | Size |
|---|---|---|
| **Foam pad** | `p = k·δ / (1 − δ/h)`, h = 15 mm: the pad stiffens as it densifies (same small-strain stiffness `k`) | softest pad at 40 N: peak strain 35%, peak pressure ×1.19, contact 96% instead of 100%; negligible for mid and stiff pads (×1.01 at the nominal pad) |
| **Preston exponent** | removal ∝ `p^α` instead of `p` (`dh = K·p_ref·(p/p_ref)^α·v·dt`, p_ref = 10 kPa): the force dependence and the pressure profile both change | α = 0.8; at 30 N a soft pad removes 1.73× the volume per unit `K` of a stiff pad (1.0 for α = 1) |
| **Force errors** | actual force = gain × commanded (one gain per run), and each raster line runs at its own force, AR(1) along the pass (first-order exposure model, checked against the exact model to < 0.1%) | gain: log-normal, σ_log = 0.04; ripple σ = 3% per line, correlation 0.5 between lines |
| **Two-stage wear** | `K/K0 = (1−f)·e^{−λW} + f·e^{−rλW}`: a fast break-in of the sharpest grit tips plus slow dulling (f = 0.25, r = 6), integrated within the pass | `K` halves after 1766 mm³ instead of 2773 mm³ (nominal λ) |
| **Scanner effects** | per-scan misregistration of the whole map; a constant offset per scan profile; outliers; missing points (isolated and 8-point gaps along profiles); and the removal is measured as the difference of the height scans before and after the pass, each pass leaving a new random surface texture (so each texture enters two consecutive scans with opposite signs) | registration σ = 0.3 mm per axis; profile offset σ = 0.5 µm; 0.5% outliers of σ = 15 µm; 2% missing; texture 1 µm RMS, correlation length 1 mm |

Each group is also run on its own ("only" worlds). Two further worlds:
* **Stress test: ring-wise abrasive wear** (`radial_wear`, not part of the realistic world). The pad face is split into 6 rings, each of which dulls with its own local work. A stiff pad touches only a strip through its centre, so its centre wears out first and the shape of the removal map changes over time, which neither C's nor D's model family can represent. With nominal parameters, a stiff pad's effective `K` after 20 passes is 33% of fresh against 37% with uniform wear (soft pad: 36% vs 37%).
* **Two pre-registered worlds** (labelled the second and third mismatch worlds in the figures). Their effects and sizes were committed to the repository before any result in them was computed, and they were never used for tuning or development (apart from the tiny end-to-end test configuration, whose outputs were not inspected).
  * `realistic_b` (commit `e05601c`), harsher than the realistic world: foam pad h = 10 mm, Preston exponent 0.7, force gain σ_log 0.06 and ripple 5% (correlation 0.7), two-stage wear with f = 0.4, r = 4 *and* ring-wise wear (6 rings), registration σ 0.5 mm, profile offsets 1 µm, 1% outliers, 5% missing, texture 1.5 µm. Only the world was frozen at that commit: C's force exponent, transient gain and D's current form were added afterwards (section 3.8).
  * `realistic_c` (commit `88ea220`), committed together with the final tracker C, whose code (`PFConfig` and `ParticleFilter`) has not changed since: foam pad h = 20 mm, Preston exponent 0.85, force gain σ_log 0.03 and ripple 4% (correlation 0.3), two-stage wear with f = 0.15, r = 8 and two-ring wear, registration σ 0.2 mm, profile offsets 0.3 µm, 1% outliers, 3% missing, texture 0.8 µm.

The **oracle** reference always uses the hidden process's own physics.

### 3.7 Estimators (`swt/estimators.py`)
All four see the commanded action of every pass and the scans, and nothing else; before each pass each predicts that pass's removal at the scan points.

**A: nominal.** Prior-mean parameters (geometric-mean stiffness 0.0561 N/mm³, prior medians of `K0` and `λ`), wear propagated with its own predicted volume, never updated.

**B: calibrate-once.** Least-squares `k_pad` and `K` from the first scan (`K` profiled out, 1-D search over log `k_pad`), then frozen, no wear model. The "learn the pad once" baseline.

**D: joint least squares** (the strongest point baseline: C's model structure, fitted by least squares instead of Bayesian inference). After every scan: points more than 5 robust s.d. from a fit of that scan alone are rejected; one stiffness is fitted to *all* scans so far (sum of the per-scan least-squares errors, each with its own `K` profiled out, including within-pass wear at the current wear rate); a least-squares fit of log `K_i` against cumulative removed volume, with a log-force term once two forces were scanned, gives the wear rate and the force exponent; the two steps are iterated twice. The next pass uses the pooled stiffness and the latest `K`, extrapolated through the pass just scanned and rescaled to the next force. Its abrasive-change forecast extrapolates the same fit, with `K0` = the first scan's `K`. Point predictions only: D has no uncertainty, no wear-rate drift, no transient gain and no scanner-offset model.

**C: joint tracker.** A sequential Monte Carlo filter with 2000 particles over log `k_pad`, a force exponent `β`, the wear-fluctuation level `σ_w`, a transient-gain level `σ_g` (all static) and the paths of log `K`, log `λ` and a per-pass gain `g` (one value per pass):
* *Process model:* removal of a pass with scale `K·e^g·F·s·(F/F_ref)^(β−1)` (`β` = 1 is Preston's law; prior N(1, 0.15²) on [0.5, 1.5], F_ref = 30 N) and within-pass wear as in 3.3. The gain `g ~ N(0, σ_g²)` belongs to one pass only (force ripple, local loading) and does not carry over, unlike `K`; `σ_g` has a log-uniform prior on [0.002, 0.05]; between passes `log K` follows the wear law plus noise of s.d. `σ_w`, and `log λ` takes a random-walk step of s.d. `δ` (the *wear-rate drift*), so the wear rate can follow the recent decay when the true law is not exponential. `σ_w` has a log-uniform prior on [0.005, 0.05] and is learned from how much `K` varies beyond the wear law; `δ` is tuned on held-out draws (3.8).
* *Update:* the scan likelihood is applied in tempered stages that keep the effective sample size at 75% of N. After each stage the particles are resampled systematically and moved by Metropolis–Hastings steps that leave the exact path posterior invariant: a joint random walk on log `k_pad`, `β` and a common shift of the whole log `λ` path; random walks on log `σ_w` and log `σ_g`; local moves on log `λ` and log `K` near the current pass; and "ridge" moves that shift log `K` up and `g` down by the same amount at a pass (the scan likelihood is unchanged; only the split between persistent and transient changes). After the last stage a full-path sweep also moves log `K`, log `λ` and `g` at every earlier pass. Every past scan is kept as O(1) sufficient statistics, so every move evaluates the full path posterior. The number of distinct particle values left at the most degenerate stored pass is recorded per pass (`C_min_path_diversity`).
* *Robust measurement model:* (i) points more than 5 robust s.d. from C's own prediction for the pass are rejected (on pass 1, from the fitted map, with one redo); (ii) a random offset per scan profile is added to the noise model when the profile means vary significantly more than the noise explains (variance estimated from the residuals, folded exactly into the sufficient statistics via the Woodbury identity); (iii) the noise level is re-estimated from the residuals and inflated when they are spatially correlated. In the matched world they almost never act (median 0 rejected points per scan, 4 of the 100 × 20 updates redone); in the realistic world the median is 15 rejected points per scan and 285 updates were redone.
* *Speed:* removal depends on stiffness only through `η = F/k_pad`; maps at the scan points are tabulated once on 512 log-spaced η values, with the within-pass wear expanded to third order. Against the exact model the relative RMS error is at most 1.7e-03 (median 3.4e-05, 90 checks). The hidden process and A, B use the exact model; D uses the same table as C.
* *Outputs:* posterior median and 90% interval of `k_pad`, `K0`, current `K`, current `λ` and `β`; 90% predictive intervals of the next pass's mean removal over the whole scan and over each of 16 blocks (4 × 4) of it; and the distribution of the abrasive-change pass (particles rolled forward with their own `σ_w` and `σ_g`, random future wear and wear-rate drift), from which C reports a median, a ≥90% band and the probability that the change is due by the next pass.

**Oracle (reference).** Knows the hidden process's physics and its true state after the previous pass, but not the wear fluctuation drawn since then, nor the coming pass's force ripple or surface texture. Its error is a floor in expectation; on single passes C can beat it by luck.

### 3.8 Tuning, development and what changed after the first full run
The wear-rate drift `δ` is the only setting tuned by the pipeline. `run-all` runs C with each `δ` of 0, 0.05, 0.1 and 0.15 on held-out draws 200–219 in the matched and realistic worlds, and keeps the value with the lowest mean *interval score* of the 90% predictive interval of next-pass mean removal (a proper scoring rule: interval width plus 20× any miss; Gneiting & Raftery 2007). It chose **δ = 0.1**.

**What changed after the first full run.** This README reports the third full run on draws 0–99. After each of the first two, an independent review pointed out weaknesses, and these changes were made: D gained within-pass wear, outlier rejection, a force term, a pooled stiffness and a first-scan `K0` (it is now C's model fitted by least squares); C's heuristic adaptive process noise was replaced by the learned `σ_w`, and a full-path sweep, the force exponent `β` and the transient gain `g` were added; the "mean over passes" comparison now excludes pass 1 (a prior prediction for every estimator, whose error dominated an arithmetic mean) and uses a geometric mean; the abrasive-change evaluation became a set of sequential decision rules, scored by error and by a cost that weights late changes three times early ones, against a model-free alternative; a block-level calibration check of the map shape was added; the realistic world gained the Preston exponent and the scan texture, and ring-wise wear was moved to a stress test after held-out checks showed that no estimator here can represent it. The checks for these changes used held-out draws 200–239 only. The `realistic_c` world and the final C were committed together (commit `88ea220`) before C was run in it. After that commit C was not changed; the baseline D, the decision rules and the metrics were finished on held-out draws before the first run in `realistic_c`, and after that run only run time was changed, to keep `run-all` under 10 minutes: the scan statistics are computed with fewer passes over the tables (equal to the old ones to rounding error, and checked against the direct formula by a test), the tuning runs share each hidden truth across the candidate drifts, the single-effect, noise and ablation experiments use fewer draws (20, 20 and 25 instead of 30, 30 and 40), and the tuning grid lost δ = 0.2, which had scored worse than 0.1 and 0.15 on the held-out draws in the previous full run. Other settings (75% ESS target, one final full-path sweep, outlier threshold 5, the `σ_w`, `σ_g` and `β` priors) were fixed by hand during development.

## 4. Results

All numbers come from `results/` and were produced by `python -m swt run-all` with seed 2026. Unless a section says otherwise, every run has 20 passes, forces alternating 20/40 N, 2 µm scan noise and the 50%-of-K0 abrasive-change threshold. Errors are RMSE over the 7676 scan points against the simulator's noise-free removal (not against the scan). Brackets after medians are IQRs unless marked as 90% bootstrap intervals (2000 resamples of the draws).

**At a glance: removal-map RMSE at pass 20, µm, median [IQR]**

| world | A: nominal | B: calibrate-once | D: joint least squares | C: joint tracker | oracle |
|---|---|---|---|---|---|
| matched (100 draws) | 2.20 [1.52–4.24] | 10.62 [8.92–13.16] | 0.057 [0.030–0.088] | 0.054 [0.027–0.086] | 0.042 |
| realistic (100 draws) | 2.13 [1.49–3.49] | 10.45 [8.54–12.53] | 0.271 [0.200–0.341] | 0.247 [0.180–0.324] | 0.110 |
| `realistic_b`, pre-registered (40 draws) | 2.69 [2.12–3.64] | 11.17 [8.27–13.56] | 0.534 [0.404–0.615] | 0.489 [0.379–0.573] | 0.129 |
| `realistic_c`, pre-registered (40 draws) | 2.26 [1.72–3.18] | 10.60 [8.79–12.61] | 0.435 [0.338–0.550] | 0.425 [0.345–0.529] | 0.132 |

**C against D** (paired over draws; a ratio above 1 means C is more accurate; 90% bootstrap intervals in brackets)

| world | D/C error ratio at pass 20, median | C better at pass 20 | D/C ratio of the geometric-mean error over passes 2–20, median | C better over passes 2–20 |
|---|---|---|---|---|
| matched | 1.02 [0.88–1.10] | 51 / 100 [42–59%] | 0.94 [0.92–0.96] | 34 / 100 [27–42%] |
| realistic | 1.03 [1.01–1.05] | 62 / 100 [54–70%] | 1.08 [1.05–1.09] | 84 / 100 [78–90%] |
| `realistic_b` | 1.04 [1.01–1.09] | 30 / 40 [65–85%] | 1.07 [1.05–1.08] | 33 / 40 [72–92%] |
| `realistic_c` | 1.01 [1.00–1.01] | 26 / 40 [52–78%] | 1.06 [1.04–1.06] | 34 / 40 [75–92%] |

**C's calibration** (target 90%)

| world | next-pass mean removal inside the 90% predictive interval, passes 2–20 | truth below / above | passes 2–5 / 11–20 | each of 16 map blocks inside its own 90% interval | true parameter inside the 90% interval at pass 20: k_pad / λ / K / K0 |
|---|---|---|---|---|---|
| matched | 1833 / 1900 (96.5%) | 29 / 38 | 99.5% / 94.2% | 96% | 91 / 100 / 97 / 98 of 100 |
| realistic | 1619 / 1900 (85.2%) | 30 / 251 | 83.5% / 86.0% | 77% | 1 / 91 / 8 / 5 of 100 |
| `realistic_b` | 610 / 760 (80.3%) | 33 / 117 | 66.9% / 86.2% | 48% | 4 / 26 / 6 / 5 of 40 |
| `realistic_c` | 682 / 760 (89.7%) | 19 / 59 | 83.1% / 91.8% | 54% | 1 / 31 / 5 / 3 of 40 |

The block column splits each scan into 4 × 4 blocks and checks each block's mean removal against C's 90% interval for that block, so it tests the predicted map shape as well as its level. With 100 draws the binomial standard error of a calibrated 90% parameter coverage is 3 points (5 with the 40 draws of the pre-registered worlds). In the mismatch worlds the parameter intervals describe *effective* parameters (section 4.1) and are not expected to cover the true values.

### 4.1 Main run: one hidden truth in both worlds

The draw is chosen by a rule that looks only at the sampled truths: among the 100 robustness draws, the one at the median standardised distance from the prior centre in (log k_pad, log K0, log λ). It is draw 15: `k_pad` = 0.2034 N/mm³, `K0` = 4.93e-05 mm²/N, `λ` = 0.000392 1/mm³, and in the realistic world a force gain of 0.979. Its abrasive crosses 50% at pass 11 in the matched world and at pass 9 in the realistic world (the break-in brings it forward); A's nominal model says pass 13.

| Removal-map RMSE [µm] | A | B | D | C | oracle |
|---|---|---|---|---|---|
| realistic, pass 2 | 6.08 | 2.01 | 1.41 | 0.87 | 0.27 |
| realistic, pass 20 | 2.58 | 7.34 | 0.199 | 0.248 | 0.083 |
| matched, pass 20 | 1.91 | 8.89 | 0.049 | 0.111 | 0.065 |

![Removal maps](figures/fig2_removal_maps.png)

* **B** is good right after its calibration but never learns that the paper is dulling, so its error grows with every pass.
* **A** has the wear law but the wrong parameters.
* **D and C** both follow the wear. On this draw D's pass-20 error is lower than C's in both worlds; single passes are noisy, and section 4.2 has the statistics over 100 draws.
* **Parameters at pass 20 (figure 4):**

  | | C: median [90% interval], matched world | C: median [90% interval], realistic world | truth |
  |---|---|---|---|
  | `k_pad` [N/mm³] | 0.2034 [0.2027, 0.2042] | 0.1817 [0.1804, 0.1831] | 0.2034 |
  | `λ` [1/mm³] | 0.000356 [0.000273, 0.000458] | 0.000352 [0.000257, 0.000488] | 0.000392 |
  | `K0` [mm²/N] | 4.94e-05 [4.88e-05, 5.01e-05] | 3.96e-05 [3.88e-05, 4.04e-05] | 4.93e-05 |

  In the realistic world the same tracker converges on an *effective* stiffness and an effective wear rate that is high during the break-in and falls later: the best linear-pad, exponential-wear description of a process that is neither, with the force gain absorbed into the effectiveness. Its map predictions remain good because they only need the effective values to reproduce the removal, not to equal the true parameters.

![Parameter tracking](figures/fig4_parameter_tracking.png)

### 4.2 Robustness: what the numbers say

* **Map accuracy.** A and B are one to two orders of magnitude worse than C and D throughout (figure 3). **In the matched world C and D are equally accurate at pass 20** (C better in 51 of 100 draws; the bootstrap interval of the median ratio contains 1) and **over passes 2–20 D is slightly more accurate** (D/C 0.94, C better in only 34 draws). With no mismatch, least squares on the right model is as good as the posterior, and C's extra freedom (a transient gain and a drifting wear rate it does not need here) probably costs it the few per cent. **In all three mismatch worlds C is more accurate than D over passes 2–20**, with the bootstrap interval of the median ratio above 1 in each; at pass 20 alone the margin is smaller (D/C 1.03, 1.04 and 1.01).
* **What C adds over D.** Both use the same model structure and pool every scan. C adds a model of the scanner (per-profile offsets, gating against its own forecast), a wear rate that may drift, a per-pass gain separate from the persistent wear, and a posterior instead of a point estimate. Under mismatch these buy a few per cent of map accuracy; the posterior is what the cost-aware abrasive-change rule of section 4.4 uses, and it is where C's advantage over D is largest.
* **Calibration of the mean removal.** In the matched world C's predictive intervals are conservative (99.5% in passes 2–5, while `σ_w` and `σ_g` are still uncertain; 94% from pass 11) and its parameter intervals cover the truth at or above the nominal rate. In the realistic world they cover 85%, and almost all misses have the truth above the interval (251 above, 30 below): the abrasive keeps cutting better than an exponential law fitted through the break-in predicts. In the milder pre-registered world coverage is close to nominal (90%); in the harsher one it is 80% (67% in passes 2–5).
* **Calibration of the map shape.** Block by block the intervals cover 96% in the matched world but 77%, 48% and 54% in the mismatch worlds. Under mismatch C gets the level of the next pass about right but not its distribution over the part: the foam pad, the Preston exponent and ring-wise wear change the map's shape in ways a linear-pad stiffness cannot follow, and C's uncertainty does not include that.

### 4.3 What each unmodelled effect costs

Each group of the realistic world was also run alone, on the first 20 draws (same draws in every world; figure 7, left), plus the ring-wise wear stress test.

| world (20 draws, pass 20) | C RMSE [µm] | D RMSE [µm] | oracle [µm] | C / oracle | predictive coverage (misses below↓ above↑) | crossing error, last pass before (mean, passes) |
|---|---|---|---|---|---|---|
| matched | 0.057 | 0.057 | 0.061 | 1.41 | 96% (5↓ 11↑) | 0.17 |
| foam pad only | 0.075 | 0.070 | 0.061 | 1.43 | 96% (6↓ 11↑) | 0.17 |
| Preston exponent only | 0.242 | 0.243 | 0.053 | 4.85 | 96% (5↓ 11↑) | 0.29 |
| force errors only | 0.161 | 0.161 | 0.139 | 0.98 | 86% (25↓ 30↑) | 0.68 |
| two-stage wear only | 0.049 | 0.096 | 0.052 | 1.19 | 96% (3↓ 12↑) | 0.30 |
| scanner effects only | 0.091 | 0.122 | 0.061 | 1.56 | 93% (2↓ 24↑) | 0.17 |
| realistic (all of the above) | 0.243 | 0.261 | 0.108 | 2.36 | 84% (8↓ 51↑) | 0.47 |
| stress test: ring-wise wear | 0.545 | 0.546 | 0.056 | 10.78 | 89% (39↓ 3↑) | 0.50 |

* **Preston exponent:** C and D both learn the force dependence (`β`), so the alternating 20/40 N schedule no longer produces alternating biases, but the flatter `p^0.8` pressure profile cannot be reproduced by any linear-pad stiffness: this is the largest single cost in map accuracy.
* **Force errors** raise everyone's error, the oracle's included (the line-to-line ripple is unpredictable). C and D are equally accurate here, and C's intervals under-cover with misses on both sides.
* **Two-stage wear** costs C little map accuracy, because its drifting wear rate follows the slowing decay; D, whose wear rate is one straight-line fit, has about twice C's error. The misses lean to one side, and long-range crossing forecasts are biased early (section 4.4).
* **Scanner effects** hurt D (no offset model, simple outlier rejection) more than C, whose per-profile offset model and innovation gating absorb most of them (the tests check both). C does not correct misregistration, and treats the scan texture, which is correlated between consecutive scans, as noise.
* **The foam pad** changes the map noticeably only for soft pads; stiffness becomes an effective value.
* **Ring-wise wear (stress test)** changes the *shape* of the removal map as the pad centre dulls. Neither model family can represent that, and C and D fail alike: their error is more than ten times the oracle's, and C's predictions are mostly too high. A tracker for such pads would need a ring-resolved effectiveness.

![Mismatch and calibration](figures/fig7_mismatch_and_calibration.png)

### 4.4 When to change the abrasive (threshold: K < 50% of K0)

**Sequential decision rules.** After each scanned pass, each rule decides whether the next pass runs on fresh abrasive. The ideal is to change at the true crossing pass c (the first pass that would start below 50% of fresh). Error = change pass − c: positive means passes run on worn abrasive (*late*), negative means abrasive life thrown away (*early*). Only draws whose crossing falls inside the run (c ≤ 21) are scored; a rule that has not fired by pass 20 counts as changing at pass 21.

* **C**: change when C's median crossing pass is the next pass or earlier (P(crossed by the next pass) ≥ 50%).
* **C_risk**: change when P(crossed by the next pass) ≥ 25%. If a pass run on worn abrasive costs 3 times a pass of abrasive life thrown away, this is the Bayes decision (threshold 1 / (1 + 3)). The ratio 3 is an assumption set by hand, not a measured cost; C_risk and D_margin were added in the last development round and checked only on held-out draws 200–239.
* **D**: change when D's point forecast is the next pass or earlier; **D_margin**: the same with a fixed one-pass safety margin, the usual way to make a point forecast cautious.
* **A**: the nominal model's crossing pass, fixed before the run.
* **R** (no process model): a least-squares fit of log(scanned mean removal per newton) against cumulative scanned removal, with a log-force term once two forces were seen, extrapolated to the start of the next pass; R changes when the extrapolated drop from fresh reaches 50%. This is the bar any model has to beat.

**Mean absolute error [passes] (draws within ±1 pass)**

| rule | matched | realistic | `realistic_b` | `realistic_c` |
|---|---|---|---|---|
| C: median | 0.15 (90) | 0.49 (90) | 0.80 (36) | 0.78 (37) |
| C_risk: 25% risk | 0.20 (90) | 0.57 (87) | 0.75 (35) | 0.68 (37) |
| D: forecast | 0.19 (90) | 0.60 (84) | 1.45 (27) | 0.65 (39) |
| D_margin: forecast − 1 | 0.96 (83) | 1.75 (53) | 3.52 (13) | 1.30 (25) |
| A: nominal schedule | 2.93 (24) | 3.64 (16) | 4.95 (3) | 3.55 (7) |
| R: model-free | 0.26 (90) | 0.73 (82) | 1.95 (20) | 1.00 (30) |
| draws scored | 91 | 96 | 40 | 40 |

**Mean cost when a late pass costs 3× an early one** (lower is better; draws late / early)

| rule | matched | realistic | `realistic_b` | `realistic_c` |
|---|---|---|---|---|
| C: median | 0.31 (7 / 6) | 1.18 (28 / 13) | 2.05 (21 / 5) | 1.93 (22 / 6) |
| C_risk: 25% risk | 0.26 (3 / 14) | 0.86 (13 / 31) | 1.35 (12 / 11) | 1.27 (12 / 11) |
| D: forecast | 0.36 (8 / 8) | 1.33 (29 / 15) | 2.40 (14 / 18) | 1.80 (22 / 3) |
| D_margin: forecast − 1 | 0.96 (0 / 78) | 1.77 (1 / 84) | 3.88 (4 / 32) | 1.35 (1 / 24) |
| A: nominal schedule | 6.16 (42 / 35) | 9.91 (70 / 19) | 14.75 (36 / 1) | 9.65 (29 / 6) |
| R: model-free | 0.57 (14 / 9) | 1.94 (46 / 7) | 2.90 (13 / 19) | 3.00 (27 / 0) |

**C_risk has the lowest cost in every world.** It trades early changes for fewer late ones, as the cost ratio asks: in the realistic world it is late in 13 of 96 draws, against 28 for C's median rule and 29 for D. A fixed one-pass margin on D's point forecast (D_margin) removes almost all late changes but is early in most draws, and costs more than C_risk everywhere: without a posterior, D cannot tell the draws where caution is needed from those where it is not. Judged by the error alone (first table), C's median rule has a smaller mean error than D in the matched, realistic and harsher pre-registered worlds, and D in the milder one. The model-free rule R is within one pass in most matched draws but is late more often under mismatch; C, C_risk and D all have a lower cost than R in every world. Draws whose crossing falls after the run (9 matched, 4 realistic, none in the pre-registered worlds) are not scored; on them the nominal schedule A changed prematurely in 9 and 4 draws, C_risk in 0 and 0.

**Forecast error by lead time** (figure 5, right; a fixed population of draws whose crossing every lead can see). As a *point* forecast of the crossing pass, C's median is no better than D's. In the matched world the two are within about a tenth of a pass of each other up to 8 passes ahead (5 passes ahead: C 0.35, D 0.32, A 2.28 passes). In the realistic world D's forecast is the more accurate from 1 to 8 passes ahead (5 passes ahead: C 0.96, D 0.56, A 1.81), and C's early forecasts are biased early (mean signed error after pass 2: -1.10 passes) because the break-in makes the abrasive look as if it wears faster than it will. What C adds is the spread: in the realistic world its ≥90% band contains the true crossing in 94% of draws one pass ahead and 98% five passes ahead (falling to 83% ten passes ahead), and C_risk turns that spread into the cheaper decision. D's lower error at fixed leads does not make its decisions better: its decision errors and costs are higher than those of C's median rule in the matched, realistic and harsher pre-registered worlds (tables above).

![Abrasive change](figures/fig5_abrasive_change.png)

### 4.5 Tuning: what the drifting wear rate buys

| wear-rate drift δ (held-out draws 200–219) | 0 (constant rate) | chosen: 0.1 |
|---|---|---|
| predictive coverage, matched | 95.8% | 95.0% |
| predictive coverage, realistic | 85.0% | 92.1% |
| median relative width of the 90% interval, realistic | 11.38% | 10.69% |
| mean interval score (lower is better; both worlds) | 0.1205 | 0.1097 |

Larger drifts raise the realistic world's coverage further but widen the intervals more than they gain (figure 7, right), so the proper scoring rule stops at 0.1. The choice is made inside `run-all`, on draws not used anywhere else.

### 4.6 Identifiability ablation: constant 30 N instead of alternating 20/40 N

On the first 25 draws of the matched world, a constant 30 N schedule (same mean force) was compared with the alternating one, and a re-run of the alternating schedule with a different filter seed gives the spread expected from Monte Carlo noise alone.

| Paired over draws, pass 20 | constant ÷ alternating | other seed ÷ alternating |
|---|---|---|
| k_pad 90% width, median ratio (draws wider) | 0.998 (12 of 25) | 1.010 (14) |
| λ 90% width, median ratio (draws wider) | 1.091 (24) | 0.998 (12) |

**This did not show what was expected.** Alternating the force did not narrow the stiffness interval at all. Because the force is commanded and known, a single force level already pins `F / k_pad` from the shape of one removal map; a second force adds a second view of the same thing. Alternating did narrow the wear-rate interval slightly and consistently, because two forces remove different volumes and so trace the wear law at two rates. Varying the force would matter if something else also changed the contact shape (an unknown force offset, uncertain curvature, a drifting pad); with the realistic world's force-calibration error it could help separate gain from effectiveness, which was not tested.

![Ablation and noise](figures/fig6_ablation_and_noise.png)

### 4.7 Scan-noise sensitivity (realistic world, first 20 draws)

| scan noise σ [µm] | 1 | 2 | 5 | 10 |
|---|---|---|---|---|
| C pass-20 RMSE, median [µm] | 0.246 | 0.243 | 0.247 | 0.249 |
| oracle, median [µm] | 0.108 | 0.108 | 0.108 | 0.108 |
| predictive coverage | 83% | 84% | 91% | 93% |

In the realistic world C's error barely depends on scan noise up to 10 µm: with thousands of points per scan the noise averages out, and the error is dominated by what neither scans nor the oracle can predict (force ripple, fluctuations) and by model mismatch. Coverage rises with the noise because noisier scans leave C less certain, which partly offsets the mismatch.

### 4.8 The pre-registered worlds

Both were committed before any of their results existed and were never used for tuning or development (section 3.6); the tables above include them. `realistic_c` was committed together with the final tracker C, so it is the only fully out-of-sample test of the final code.

| | `realistic_b`, pre-registered (harsher, ring-wise wear) | `realistic_c`, pre-registered (milder) |
|---|---|---|
| C pass-20 RMSE, median [µm] | 0.489 (D 0.534, oracle 0.129) | 0.425 (D 0.435, oracle 0.132) |
| true pass-20 mean removal, median [µm] | 3.59 | 4.65 |
| C better than D over passes 2–20 | 33 of 40 (D/C 1.07) | 34 of 40 (D/C 1.06) |
| C's 90% predictive coverage, passes 2–20 | 80% | 90% |
| abrasive-change cost (late ×3): C_risk / D / R | 1.35 / 2.40 / 2.90 | 1.27 / 1.80 / 3.00 |

In both pre-registered worlds C was more accurate than D over passes 2–20 in most draws and C_risk had the lowest abrasive-change cost, as in the realistic world. The milder `realistic_c` gives the best-calibrated mismatch result of the study (90% coverage of the mean removal), but its block-level coverage (54%) shows that the map shape is still wrong. The harsher `realistic_b`, with ring-wise wear, is where C is weakest: an error 3.7 times the oracle's at pass 20 (median over draws), 80% coverage of the mean removal and 48% of the map blocks inside their intervals.

## 5. Limitations

* **Simulated only.** Nothing here has touched a physical part. The realistic worlds are richer simulators, not reality: their effects and sizes were chosen by me, not measured, and real processes contain effects in none of them (loading and clogging of the abrasive, heat, grit changes, pad ageing, robot path errors).
* **Same author for the worlds and the tracker.** The scanner-offset model, the outlier gating, the force exponent and the transient gain address effects I also put into the realistic world. The two pre-registered worlds limit how much the tracker could be tuned to them, but they were designed by the same person, with the same kinds of effects.
* **Effects outside the model family.** Ring-wise abrasive wear and the pressure-profile part of a Preston exponent change the *shape* of the removal map in ways no linear-pad stiffness reproduces; C and D both degrade badly under ring-wise wear (section 4.3), and C's block-level intervals under-cover in every mismatch world.
* **Under-coverage under mismatch.** C's 90% predictive intervals for the mean removal cover 85% in the realistic world (84% in passes 2–5) and 80% in the harsher pre-registered world; block by block, 48–77%.
* **No advantage without mismatch.** In the tracker's own world, joint least squares (D) is as accurate as C at pass 20 and slightly more accurate over passes 2–20; C's benefit there is its calibrated uncertainty and the decisions built on it.
* **Assumed costs.** The 3:1 cost of a late against an early abrasive change, which C_risk is built for, is an assumption, not a measured cost.
* **Parameters are only meaningful in the matched world.** Under mismatch C's stiffness, wear rate and force exponent are effective values, not physical properties.
* **Long-range abrasive-change forecasts** are biased early when the wear law has a break-in phase (section 4.4).
* **Registration error is not corrected**; C treats misregistration as structured noise.
* **Winkler-type pad.** Both pad laws are independent springs: no shear coupling, no bending of the backing plate, no tilt over an overhanging edge. The contact uses the nominal (CAD) surface; removal is not fed back into the contact.
* **One abrasive, removal depth only.** One grit per run; no roughness or other finish metric (the scan texture is a measurement effect only).
* **Grid effects.** The 1 mm grid quantises the stiffest contact strips (section 3.2).
* **Surrogate.** C's table has a worst-case relative error of 1.7e-03 against the exact model, at the fastest-wearing draws (third-order expansion of the within-pass wear).
* **Not validated on physical coupons.**

## 6. Next steps that would make it real

1. **Fit the wear law to coupon data.** Scan coupons after every pass at several forces and grits; measure break-in, the force exponent, pass-to-pass fluctuation and whether the pad centre wears faster. This decides whether the tracker needs a break-in state and a ring-resolved effectiveness.
2. **Ring-resolved effectiveness in the tracker** (a few rings, with a smoothness prior), so that non-uniform wear changes the predicted map shape instead of biasing the stiffness.
3. **A break-in state in the wear model** (or a wear law learned from coupon data), so that early abrasive-change forecasts are not biased early.
4. **Register each scan to the prediction** (two shifts per scan as nuisance parameters) before updating.
5. **Replace the spring pad** with a finite-element or learned pad model, including tilt over edges.
6. **Run on GrayMatter-style scan data** with real registration error, missing data and noise.
7. **Close the loop**: choose the next pass's force or dwell from C's predictive distribution to hit a target removal, and set the abrasive-change cost ratio from real costs of a worn pass and of discarded abrasive.

## 7. How to reproduce

Requires Python ≥ 3.11. The recorded run used Python 3.11.15 and NumPy 2.4.6; see `results/run_info.json`.

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib, pyyaml, pytest
python -m swt run-all                    # every experiment, every figure, every number in results/
python -m swt report                     # re-render README.md and SUMMARY.md from results/
python -m swt run --config configs/default.yaml    # main 20-pass experiment only (uses results/tuning.json if present)
python -m pytest -q                      # tests
```

* **Runtime.** `run-all` runs 510 multi-draw episodes, plus the tuning runs and the main run, using `min(4, CPUs)` worker processes. In the recorded run, building the model and surrogate took 9 s and the experiments 548 s with 4 workers on a 4-CPU machine. `--workers 1` runs serially; results are identical for any worker count because every task has its own seeds.
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
