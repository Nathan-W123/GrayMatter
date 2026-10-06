# sanding-wear-tracker

A simulation study of tracking sanding-pad stiffness and abrasive wear together, so that a sanding process model could stay accurate as the paper dulls and say when to change it. The tracker is tested inside its own model, in a "realistic" simulated world with effects it does not model, in two further worlds that were fixed before their results were seen, and in stress tests. All results are simulated; nothing has been tested on physical parts.

![Prediction error per pass](figures/fig3_rmse_per_pass.png)

## 1. Summary

A model that predicts how much material a robotic sanding pass removes needs two things it cannot measure directly: how the compliant pad presses on the part (set by a stiffness that is fixed but unknown) and how sharp the abrasive is (which drops every pass). This project simulates a 125 mm pad sanding a curved panel with hidden stiffness and wear parameters and compares four predictors of the next pass's removal map:
* A, a nominal model with prior-mean parameters;
* B, a model calibrated once on the first scan;
* D, a least-squares fit over all scans of the tracker's own model, with the tracker's priors;
* C, a particle filter that tracks stiffness, force dependence, current effectiveness, a drifting wear rate and a per-pass gain, with a robust scan model.

They are tested in three kinds of world:
* the tracker's own (*matched*) world;
* a *realistic* world with effects no predictor models (a foam pad, a Preston pressure exponent, force errors, abrasive break-in and uneven wear across the pad face, scanner artefacts);
* two further mismatch worlds, committed to the repository before any of their results existed.

* **Map accuracy.** A and B are far behind (median pass-20 errors of about 2 µm and 11 µm, against about 4 µm of removal). C's first two predictions are better than D's in every world (median error ratio D/C 1.39, 1.77, 1.35 and 1.46 in the matched, realistic and two pre-registered worlds). But from the fourth pass on, C and D are equally accurate to within about 2% in all four worlds (D/C 0.98 matched, 1.00 realistic, 1.00 and 0.99 pre-registered). Under mismatch both stay well above the oracle that knows the true physics: at pass 20 in the realistic world C's median error is 0.413 µm, D's 0.429 µm and the oracle's 0.104 µm.
* **Calibration.** C's 90% predictive intervals for the next pass's mean removal cover these shares of passes 2–20:
  * 96% in the matched world (conservative);
  * 89% in the realistic world;
  * 80% and 90% in the pre-registered worlds.

  Checked block by block (16 blocks per map), they cover 96% in the matched world but only 52%, 48% and 54% under mismatch. The *shape* of the removal map is where the simplified model fails.
* **When to change the abrasive.** C's posterior gives the cost-optimal change rule for any cost of a late change relative to an early one, with no tuning. Point forecasts need a safety margin. With margins tuned on held-out draws for each cost ratio, D and R (a model-free extrapolation of the scans) are about as good as C:
  * C was significantly cheaper in the matched world, and in the harsher pre-registered world at a 3:1 cost;
  * the differences were not significant in most other cases;
  * the tuned D was significantly cheaper in one case (`realistic_c` at 19:1), and nominally in one more.

  Without tuned margins, point forecasts are far worse when late changes are expensive: at 19:1 in the realistic world, C's rule costs 1.53 cost-weighted passes per run, D's 9.89 and R's 16.55.
* **Weak spots.**
  * Under mismatch C's parameters are effective values, not the true ones.
  * Early abrasive-change forecasts are biased early when the abrasive has a break-in phase.
  * Three stress tests defeat C, with errors 7 to 38 times the oracle's: abrasive that wears ring by ring, a pad holder that tilts at the edges, and abrasive loading. The first two defeat D just as much; loading hurts C more than D.
  * Everything is simulated; nothing was tested on physical parts.

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
| **Abrasive** | (i) two-stage wear `K/K0 = (1−f)·e^{−λW} + f·e^{−rλW}`: a fast break-in of the sharpest grit tips plus slow dulling (f = 0.25, r = 6), integrated within the pass; (ii) the inner and outer halves of the pad face wear apart, each with its own local work (mild ring-wise wear) | `K` halves after 1766 mm³ instead of 2773 mm³ (nominal λ); after 20 nominal passes a stiff pad's effective `K` is 30% of fresh instead of 32% |
| **Scanner effects** | per-scan misregistration of the whole map; a constant offset per scan profile; outliers; missing points (isolated and 8-point gaps along profiles); and the removal is measured as the difference of the height scans before and after the pass, each pass leaving a new random surface texture (so each texture enters two consecutive scans with opposite signs) | registration σ = 0.3 mm per axis; profile offset σ = 0.5 µm; 0.5% outliers of σ = 15 µm; 2% missing; texture 1 µm RMS, correlation length 1 mm |

Each group is also run on its own ("only" worlds). Two further worlds:
* **Stress test: ring-wise abrasive wear** (`radial_wear`, not part of the realistic world). The pad face is split into 6 rings, each of which dulls with its own local work. A stiff pad touches only a strip through its centre, so its centre wears out first and the shape of the removal map changes over time, which neither C's nor D's model family can represent. With nominal parameters, a stiff pad's effective `K` after 20 passes is 33% of fresh against 37% with uniform wear (soft pad: 36% vs 37%).
* **Stress test: tilting pad holder** (`edge_tilt`, not part of the realistic world). The holder lets the pad tilt against a rotational spring (20000 N·mm/rad) until the moment of the pad pressure about the pad centre balances it; penetration and two tilt angles minimise the contact energy (damped Newton for all stations at once; force and moment balance are checked in the tests). Where the pad overhangs an edge this pushes pressure towards the edge. At 40 N the largest tilt is 4.2° for the softest pad, 2.1° for the nominal one and 0.7° for the stiffest, and the pass's exposure map changes by 37% RMS (nominal pad). Linear pad, otherwise the tracker's own model; C and D assume a rigid holder.
* **Stress test: abrasive loading** (`loading`, not part of the realistic world). Swarf clogs the abrasive while it cuts: each raster line's effectiveness is multiplied by (1 − L), where the loading L grows along the pass towards 15% with the removed volume (e-folding 50 mm³) and half of it is shed between passes. With nominal parameters it removes 11% of pass 1's volume and 5% of pass 20's. Loading is temporary, so the abrasive-change threshold still refers to the wear alone. It was first part of the realistic world. On the held-out tuning draws it broke C's calibration (predictive coverage about 55–65% in the realistic world instead of about 90%), so, like six-ring wear before it, it was moved to a stress test. That move was made after seeing those held-out results.
* **Two pre-registered worlds** (labelled the second and third mismatch worlds in the figures). Their effects and sizes were committed to the repository before any result in them was computed, and they were never used for tuning or development (apart from the tiny end-to-end test configuration, whose outputs were not inspected).
  * `realistic_b` (commit `e05601c`), harsher than the realistic world: foam pad h = 10 mm, Preston exponent 0.7, force gain σ_log 0.06 and ripple 5% (correlation 0.7), two-stage wear with f = 0.4, r = 4 *and* ring-wise wear (6 rings), registration σ 0.5 mm, profile offsets 1 µm, 1% outliers, 5% missing, texture 1.5 µm. Only the world was frozen at that commit: C's force exponent, transient gain and D's current form were added afterwards (section 3.8).
  * `realistic_c` (commit `88ea220`), committed together with the final tracker C, whose code (`PFConfig` and `ParticleFilter`) has not changed since: foam pad h = 20 mm, Preston exponent 0.85, force gain σ_log 0.03 and ripple 4% (correlation 0.3), two-stage wear with f = 0.15, r = 8 and two-ring wear, registration σ 0.2 mm, profile offsets 0.3 µm, 1% outliers, 3% missing, texture 0.8 µm.

The **oracle** reference always uses the hidden process's own physics.

### 3.7 Estimators (`swt/estimators.py`)
All four see the commanded action of every pass and the scans, and nothing else; before each pass each predicts that pass's removal at the scan points.

**A: nominal.** Prior-mean parameters (geometric-mean stiffness 0.0561 N/mm³, prior medians of `K0` and `λ`), wear propagated with its own predicted volume, never updated.

**B: calibrate-once.** Least-squares `k_pad` and `K` from the first scan (`K` profiled out, 1-D search over log `k_pad`), then frozen, no wear model. The "learn the pad once" baseline.

**D: joint least squares with C's priors** (the strongest point baseline: C's model structure and priors, fitted by penalised least squares instead of Bayesian inference). After every scan: points more than 5 robust s.d. from a fit of that scan alone are rejected; one stiffness is fitted to *all* scans so far (sum of the per-scan least-squares errors, each with its own `K` profiled out, including within-pass wear at the current wear rate); a fit of log `K_i` against cumulative removed volume and log force, penalised by C's priors on `λ` (log-normal, linearised at its median) and `β` (N(1, 0.15²)), with the log `K_i` noise set to C's prior median wear-fluctuation and transient levels, gives the wear rate and the force exponent (the MAP estimate, so one or two scans cannot give a wild wear rate); the two steps are iterated twice. The next pass uses the pooled stiffness and the latest `K`, extrapolated through the pass just scanned and rescaled to the next force. Its abrasive-change forecast extrapolates the same fit, with `K0` = the first scan's `K`; for decisions it reports its forecast of `K/K0` at the start of the next pass. Point predictions only: D has no uncertainty, no wear-rate drift, no transient gain and no scanner-offset model.

**C: joint tracker.** A sequential Monte Carlo filter with 2000 particles over log `k_pad`, a force exponent `β`, the wear-fluctuation level `σ_w`, a transient-gain level `σ_g` (all static) and the paths of log `K`, log `λ` and a per-pass gain `g` (one value per pass):
* *Process model:* removal of a pass with scale `K·e^g·F·s·(F/F_ref)^(β−1)` (`β` = 1 is Preston's law; prior N(1, 0.15²) on [0.5, 1.5], F_ref = 30 N) and within-pass wear as in 3.3. The gain `g ~ N(0, σ_g²)` belongs to one pass only (force ripple, local loading) and does not carry over, unlike `K`; `σ_g` has a log-uniform prior on [0.002, 0.05]; between passes `log K` follows the wear law plus noise of s.d. `σ_w`, and `log λ` takes a random-walk step of s.d. `δ` (the *wear-rate drift*), so the wear rate can follow the recent decay when the true law is not exponential. `σ_w` has a log-uniform prior on [0.005, 0.05] and is learned from how much `K` varies beyond the wear law; `δ` is tuned on held-out draws (3.8).
* *Update:* the scan likelihood is applied in tempered stages that keep the effective sample size at 75% of N. After each stage the particles are resampled systematically and moved by Metropolis–Hastings steps that leave the exact path posterior invariant: a joint random walk on log `k_pad`, `β` and a common shift of the whole log `λ` path; random walks on log `σ_w` and log `σ_g`; local moves on log `λ` and log `K` near the current pass; and "ridge" moves that shift log `K` up and `g` down by the same amount at a pass (the scan likelihood is unchanged; only the split between persistent and transient changes). After the last stage a full-path sweep also moves log `K`, log `λ` and `g` at every earlier pass. Every past scan is kept as O(1) sufficient statistics, so every move evaluates the full path posterior. The number of distinct particle values left at the most degenerate stored pass is recorded per pass (`C_min_path_diversity`).
* *Robust measurement model:* (i) points more than 5 robust s.d. from C's own prediction for the pass are rejected (on pass 1, from the fitted map, with one redo); (ii) a random offset per scan profile is added to the noise model when the profile means vary significantly more than the noise explains (variance estimated from the residuals, folded exactly into the sufficient statistics via the Woodbury identity); (iii) the noise level is re-estimated from the residuals and inflated when they are spatially correlated. In the matched world they almost never act (median 0 rejected points per scan, 4 of the 100 × 20 updates redone); in the realistic world the median is 15 rejected points per scan and 532 updates were redone.
* *Speed:* removal depends on stiffness only through `η = F/k_pad`; maps at the scan points are tabulated once on 512 log-spaced η values, with the within-pass wear expanded to third order. Against the exact model the relative RMS error is at most 1.7e-03 (median 3.4e-05, 90 checks). The hidden process and A, B use the exact model; D uses the same table as C.
* *Outputs:* posterior median and 90% interval of `k_pad`, `K0`, current `K`, current `λ` and `β`; 90% predictive intervals of the next pass's mean removal over the whole scan and over each of 16 blocks (4 × 4) of it; and the distribution of the abrasive-change pass (particles rolled forward with their own `σ_w` and `σ_g`, random future wear and wear-rate drift), from which C reports a median, a ≥90% band and the probability that the change is due by the next pass.

**Oracle (reference).** Knows the hidden process's physics and its true state after the previous pass, but not the wear fluctuation drawn since then, nor the coming pass's force ripple or surface texture. Its error is a floor in expectation; on single passes C can beat it by luck.

### 3.8 Tuning, development and what changed after the first full run
Two kinds of setting are tuned by the pipeline, both on held-out draws: C's wear-rate drift `δ`, and the safety margins of D's and R's abrasive-change rules (section 4.4; C's rules need none). `run-all` runs C with each `δ` of 0, 0.05, 0.1 and 0.15 on held-out draws 200–215 in the matched and realistic worlds, and keeps the value with the lowest mean *interval score* of the 90% predictive interval of next-pass mean removal (a proper scoring rule: interval width plus 20× any miss; Gneiting & Raftery 2007). It chose **δ = 0.1**.

**What changed after the first full run.** This README reports the fourth full run on draws 0–99. After each of the first three, an independent review pointed out weaknesses, and these changes were made. *Rounds 1–2:* D gained within-pass wear, outlier rejection, a force term, a pooled stiffness and a first-scan `K0`; C's heuristic adaptive process noise was replaced by the learned `σ_w`, and a full-path sweep, the force exponent `β` and the transient gain `g` were added; the "mean over passes" comparison excludes pass 1 (a prior prediction for every estimator) and uses a geometric mean; the realistic world gained the Preston exponent and the scan texture, and ring-wise wear (6 rings) was moved to a stress test. *Round 3:* the abrasive-change evaluation became a set of sequential decision rules scored by cost against a model-free rule, and a block-level calibration check was added. The `realistic_c` world and the final C were committed together (commit `88ea220`) before C was run in it; C has not changed since. *Round 4:* the third review found that C's accuracy edge over D under mismatch came almost entirely from passes 2–3, and that C's decision rule had been compared with an untuned D. So D now uses C's priors on `λ` and `β` (a MAP fit), the comparison is split into start-up (passes 2–3) and steady state (passes 4–20), and the decision rules of D and R get safety margins tuned on held-out draws for each cost ratio. The physics gained mild ring-wise wear (two rings) in the realistic world, and abrasive loading and a tilting pad holder as stress tests. The realistic world therefore changed after C was frozen (it became harder); the pre-registered worlds did not. *Run time:* to keep `run-all` under 10 minutes, the scan statistics are computed with fewer passes over the tables (equal to rounding error, checked against the direct formula by a test), tuning shares each hidden truth across the candidate drifts, the single-effect and stress worlds use 16 draws, the noise study 20 draws at 2, 5 and 10 µm, the ablation 25 draws, and the tuning 16 held-out draws per world over δ = 0, 0.05, 0.1 and 0.15 (δ = 0.2 had scored worse than 0.1 and 0.15 in the third run). All checks behind these changes used held-out draws 200–239 only. Other settings (75% ESS target, one final full-path sweep, outlier threshold 5, the `σ_w`, `σ_g` and `β` priors) were fixed by hand during development.

## 4. Results

All numbers come from `results/` and were produced by `python -m swt run-all` with seed 2026. Unless a section says otherwise, every run has 20 passes, forces alternating 20/40 N, 2 µm scan noise and the 50%-of-K0 abrasive-change threshold. Errors are RMSE over the 7676 scan points against the simulator's noise-free removal (not against the scan). Brackets after medians are IQRs unless marked as 90% bootstrap intervals (2000 resamples of the draws).

**At a glance: removal-map RMSE at pass 20, µm, median [IQR]**

| world | A: nominal | B: calibrate-once | D: joint least squares | C: joint tracker | oracle |
|---|---|---|---|---|---|
| matched (100 draws) | 2.20 [1.52–4.24] | 10.62 [8.92–13.16] | 0.057 [0.030–0.088] | 0.054 [0.027–0.086] | 0.042 |
| realistic (100 draws) | 2.27 [1.63–3.48] | 10.50 [8.66–12.44] | 0.429 [0.340–0.536] | 0.413 [0.324–0.524] | 0.104 |
| `realistic_b`, pre-registered (40 draws) | 2.69 [2.12–3.64] | 11.17 [8.27–13.56] | 0.532 [0.406–0.615] | 0.489 [0.379–0.573] | 0.129 |
| `realistic_c`, pre-registered (40 draws) | 2.26 [1.72–3.18] | 10.60 [8.79–12.61] | 0.435 [0.337–0.549] | 0.425 [0.345–0.529] | 0.132 |

**C against D** (paired over draws; D/C is the ratio of D's to C's error, so above 1 means C is more accurate; medians over draws with 90% bootstrap intervals; geometric means over the passes of each window)

| world | start-up, passes 2–3: D/C | C better | steady state, passes 4–20: D/C | C better | pass 20 (a 40 N pass): D/C | C better |
|---|---|---|---|---|---|---|
| matched | 1.39 [1.16–2.07] | 60 / 100 | 0.98 [0.95–1.01] | 44 / 100 | 1.02 [0.88–1.10] | 51 / 100 |
| realistic | 1.77 [1.48–1.93] | 87 / 100 | 1.00 [0.99–1.01] | 47 / 100 | 1.01 [1.01–1.02] | 64 / 100 |
| `realistic_b` | 1.35 [1.21–1.71] | 31 / 40 | 1.00 [0.98–1.02] | 18 / 40 | 1.04 [1.01–1.09] | 29 / 40 |
| `realistic_c` | 1.46 [1.15–1.68] | 31 / 40 | 0.99 [0.98–1.00] | 15 / 40 | 1.01 [1.00–1.01] | 25 / 40 |

**Start-up and steady state.** C's advantage over D is confined to the first two predictions. On passes 2–3, when D has one or two scans, C's error is smaller in every world (median D/C 1.39, 1.77, 1.35 and 1.46 in the matched, realistic, `realistic_b` and `realistic_c` worlds), probably because C averages its prediction over what one or two scans leave uncertain (stiffness, force exponent, wear rate) where D plugs in point estimates, and because of C's scanner model. From pass 4 on, C and D are equally accurate to within about 2% in all four worlds: the median D/C ratios are 0.979, 0.997, 0.996 and 0.991, and no 90% interval lies more than 2% from 1 (in `realistic_c` D is about 1% better, with an interval that just excludes 1). Pass 20 is a 40 N pass; averaged over passes 19 and 20 (one at each force), the realistic-world ratio is 1.010 [0.998–1.016], within noise of 1.

**C's calibration** (target 90%)

| world | next-pass mean removal inside the 90% predictive interval, passes 2–20 | truth below / above | passes 2–5 / 11–20 | each of 16 map blocks inside its own 90% interval | true parameter inside the 90% interval at pass 20: k_pad / λ / K / K0 |
|---|---|---|---|---|---|
| matched | 1833 / 1900 (96.5%) | 29 / 38 | 99.5% / 94.2% | 96% | 91 / 100 / 97 / 98 of 100 |
| realistic | 1682 / 1900 (88.5%) | 32 / 186 | 85.0% / 90.7% | 52% | 3 / 86 / 8 / 6 of 100 |
| `realistic_b` | 610 / 760 (80.3%) | 33 / 117 | 66.9% / 86.2% | 48% | 4 / 26 / 6 / 5 of 40 |
| `realistic_c` | 682 / 760 (89.7%) | 19 / 59 | 83.1% / 91.8% | 54% | 1 / 31 / 5 / 3 of 40 |

The block column splits each scan into 4 × 4 blocks and checks each block's mean removal against C's 90% interval for that block, so it tests the predicted map shape as well as its level. With 100 draws the binomial standard error of a calibrated 90% parameter coverage is 3 points (5 with the 40 draws of the pre-registered worlds). In the mismatch worlds the parameter intervals describe *effective* parameters (section 4.1) and are not expected to cover the true values.

### 4.1 Main run: one hidden truth in both worlds

The draw is chosen by a rule that looks only at the sampled truths: among the 100 robustness draws, the one at the median standardised distance from the prior centre in (log k_pad, log K0, log λ). It is draw 15: `k_pad` = 0.2034 N/mm³, `K0` = 4.93e-05 mm²/N, `λ` = 0.000392 1/mm³, and in the realistic world a force gain of 0.979. Its abrasive crosses 50% at pass 11 in the matched world and at pass 7 in the realistic world (the break-in and the faster-wearing pad centre bring it forward); A's nominal model says pass 13.

| Removal-map RMSE [µm] | A | B | D | C | oracle |
|---|---|---|---|---|---|
| realistic, pass 2 | 6.45 | 2.20 | 1.62 | 0.89 | 0.26 |
| realistic, pass 20 | 2.81 | 7.36 | 0.334 | 0.358 | 0.078 |
| matched, pass 20 | 1.91 | 8.89 | 0.049 | 0.111 | 0.065 |

![Removal maps](figures/fig2_removal_maps.png)

* **B** is good right after its calibration but never learns that the paper is dulling, so its error grows with every pass.
* **A** has the wear law but the wrong parameters.
* **D and C** both follow the wear. On this draw D's pass-20 error is lower than C's in both worlds; single passes are noisy, and section 4.2 has the statistics over 100 draws.
* **Parameters at pass 20 (figure 4):**

  | | C: median [90% interval], matched world | C: median [90% interval], realistic world | truth |
  |---|---|---|---|
  | `k_pad` [N/mm³] | 0.2034 [0.2027, 0.2042] | 0.1809 [0.1795, 0.1823] | 0.2034 |
  | `λ` [1/mm³] | 0.000356 [0.000273, 0.000458] | 0.000377 [0.000268, 0.000533] | 0.000392 |
  | `K0` [mm²/N] | 4.94e-05 [4.88e-05, 5.01e-05] | 3.92e-05 [3.84e-05, 4.02e-05] | 4.93e-05 |

  In the realistic world the same tracker converges on an *effective* stiffness and an effective wear rate that is high during the break-in and falls later: the best linear-pad, exponential-wear description of a process that is neither, with the force gain absorbed into the effectiveness. Its map predictions remain good because they only need the effective values to reproduce the removal, not to equal the true parameters.

![Parameter tracking](figures/fig4_parameter_tracking.png)

### 4.2 Robustness: what the numbers say

* **Map accuracy.** A and B are one to two orders of magnitude worse than C and D throughout (figure 3). C and D differ only on the first two predictions (table above). Under mismatch both stay well above the oracle: at pass 20 in the realistic world C's median error is 4.1 times the oracle's, mostly because of the two-ring wear and the Preston exponent (section 4.3), which change the map's shape.
* **What C adds over D.** Not steady-state map accuracy. It adds better first predictions, calibrated uncertainty, and with it abrasive-change decisions that need no tuning (section 4.4).
* **Calibration of the mean removal.** In the matched world C's predictive intervals are conservative (99.5% in passes 2–5, while `σ_w` and `σ_g` are still uncertain; 94% from pass 11) and its parameter intervals cover the truth at or above the nominal rate. In the realistic world they cover 89%, and most misses have the truth above the interval (186 above, 32 below): the abrasive keeps cutting better than an exponential law fitted through the break-in predicts. In the milder pre-registered world coverage is close to nominal (90%); in the harsher one it is 80% (67% in passes 2–5).
* **Calibration of the map shape.** Block by block the intervals cover 96% in the matched world but only 52%, 48% and 54% in the mismatch worlds. Under mismatch C gets the level of the next pass about right but not its distribution over the part: uneven wear across the pad face, the foam pad and the Preston exponent change the map's shape in ways a linear-pad, uniform-wear model cannot follow, and C's uncertainty does not include that.

### 4.3 What each unmodelled effect costs

Each group of the realistic world was also run alone, on the first 16 draws (same draws in every world; figure 7, left), plus the three stress tests.

| world (16 draws, pass 20) | C RMSE [µm] | D RMSE [µm] | oracle [µm] | C / oracle | predictive coverage (misses below↓ above↑) | crossing error, last pass before (mean, passes) |
|---|---|---|---|---|---|---|
| matched | 0.068 | 0.060 | 0.063 | 1.41 | 96% (4↓ 9↑) | 0.19 |
| foam pad only | 0.078 | 0.077 | 0.063 | 1.43 | 95% (6↓ 9↑) | 0.19 |
| Preston exponent only | 0.257 | 0.258 | 0.060 | 4.85 | 95% (5↓ 9↑) | 0.36 |
| force errors only | 0.174 | 0.166 | 0.157 | 0.98 | 85% (20↓ 25↑) | 0.62 |
| two-stage and two-ring wear only | 0.390 | 0.394 | 0.053 | 7.32 | 94% (14↓ 5↑) | 0.56 |
| scanner effects only | 0.083 | 0.152 | 0.063 | 1.35 | 93% (1↓ 19↑) | 0.12 |
| realistic (all of the above) | 0.390 | 0.413 | 0.103 | 3.74 | 87% (5↓ 35↑) | 0.62 |
| stress test: ring-wise wear | 0.600 | 0.601 | 0.060 | 10.00 | 89% (30↓ 3↑) | 0.38 |
| stress test: tilting pad holder | 2.570 | 2.548 | 0.067 | 38.22 | 34% (0↓ 201↑) | 0.19 |
| stress test: abrasive loading | 0.386 | 0.199 | 0.060 | 6.86 | 51% (0↓ 148↑) | 1.93 |

* **Preston exponent:** C and D both learn the force dependence (`β`), so the alternating 20/40 N schedule no longer produces alternating biases, but the flatter `p^0.8` pressure profile cannot be reproduced by any linear-pad stiffness: a large cost in map accuracy, second only to uneven wear.
* **Force errors** raise everyone's error, the oracle's included (the line-to-line ripple is unpredictable). D is slightly more accurate than C here, and C's intervals under-cover with misses on both sides.
* **Two-stage and two-ring wear** is the largest single cost: C and D are equally accurate and about 7 times worse than the oracle. When the centre of the pad face dulls faster than its rim, the shape of the removal map changes, which neither model represents; C's drifting wear rate follows the slowing decay of the two-stage law.
* **Scanner effects** hurt D (no offset model, simple outlier rejection) more than C, whose per-profile offset model and innovation gating absorb most of them (the tests check both). C does not correct misregistration, and treats the scan texture, which is correlated between consecutive scans, as noise.
* **The foam pad** changes the map noticeably only for soft pads; stiffness becomes an effective value.
* **Ring-wise wear with six rings (stress test)** makes uneven wear worse: C and D fail alike, at about 10 times the oracle's error, and C's predictions are mostly too high. A tracker for such pads would need a ring-resolved effectiveness.
* **Tilting holder (stress test)** defeats both trackers: their error is about 38 times the oracle's and C's intervals cover 34%, and in every miss the interval lies below the truth. Pressure moves towards the overhanging edges, which a rigid-holder model cannot represent; edge stations need the tilt in the model.
* **Abrasive loading (stress test)** hurts C more than D: C's error is 0.386 µm against D's 0.199 µm, its intervals cover 51%, with the interval below the truth in every miss, and its crossing forecasts are off by 1.9 passes on average. C apparently reads the loss of cutting as wear and predicts too little removal once the loading is shed. A loading state would have to be tracked separately.

![Mismatch and calibration](figures/fig7_mismatch_and_calibration.png)

### 4.4 When to change the abrasive (threshold: K < 50% of K0)

**The decision.** After each scanned pass, a rule decides whether the next pass runs on fresh abrasive. The ideal is to change at the true crossing pass c (the first pass that would start below 50% of fresh). Error = change pass − c: positive means passes run on worn abrasive (*late*), negative means abrasive life thrown away (*early*). With a cost ratio r, a late pass costs r and an early pass 1. Three ratios are reported: r = 1 (both errors equally bad), r = 3 and r = 19 (a pass on worn abrasive nearly as bad as a scrapped part); the real ratio depends on the shop and was not measured. Only draws whose crossing falls inside the run (c ≤ 21) are scored; a rule that has not fired by pass 20 counts as changing at pass 21.

**The rules.**
* **C_r**: change when C's probability that the crossing happens by the next pass is at least 1/(1 + r) (50%, 25% and 5%). This is the Bayes decision for cost ratio r; it uses C's crossing distribution as it is, with no tuning.
* **D_r** and **R_r**: change when the point forecast of `K/K0` at the start of the next pass is below the threshold raised by a safety margin, 0.5·(1 + m_r). D's forecast comes from its MAP fit; R needs no process model: a least-squares fit of log(scanned mean removal per newton) against cumulative scanned removal (with a log-force term once two forces were seen), extrapolated to the next pass. For each r the margins were tuned on the held-out draws 200–215 (matched and realistic worlds, the draws used for `δ`), over −30% to +50% in steps of 1%: D +0.02 / +0.02 / +0.06 and R +0.02 / +0.05 / +0.05 for r = 1 / 3 / 19. This gives each point forecast its best margin for that cost, chosen on data with known crossings.
* **D** and **R** without a margin, and **A**, the nominal model's crossing pass fixed before the run, are shown for reference.

**Mean cost per draw** (lower is better) and the paired difference to C's rule (mean over draws, 90% bootstrap interval; negative means C is cheaper)

| r | rule | matched | realistic | `realistic_b` | `realistic_c` |
|---|---|---|---|---|---|
| 1 | C_1 | 0.15 | 0.54 | 0.80 | 0.78 |
| 1 | D_1 | 0.43 | 0.44 | 0.93 | 0.65 |
| 1 | R_1 | 0.38 | 0.67 | 1.88 | 0.80 |
| 1 | C_1 − D_1 | -0.27 [-0.37, -0.18] | +0.10 [+0.00, +0.21] | -0.12 [-0.30, +0.05] | +0.12 [-0.05, +0.30] |
| 1 | C_1 − R_1 | -0.23 [-0.34, -0.13] | -0.13 [-0.28, +0.02] | -1.07 [-1.60, -0.62] | -0.03 [-0.23, +0.15] |
| 3 | C_3 | 0.26 | 0.96 | 1.35 | 1.27 |
| 3 | D_3 | 0.43 | 1.10 | 2.17 | 1.50 |
| 3 | R_3 | 1.07 | 0.97 | 2.55 | 0.90 |
| 3 | C_3 − D_3 | -0.16 [-0.29, -0.03] | -0.14 [-0.39, +0.09] | -0.82 [-1.43, -0.33] | -0.23 [-0.55, +0.10] |
| 3 | C_3 − R_3 | -0.80 [-0.95, -0.66] | -0.01 [-0.27, +0.25] | -1.20 [-1.98, -0.42] | +0.38 [+0.00, +0.75] |
| 19 | C_19 | 0.68 | 1.53 | 1.95 | 3.70 |
| 19 | D_19 | 1.23 | 1.67 | 3.90 | 1.85 |
| 19 | R_19 | 1.07 | 2.25 | 3.75 | 3.30 |
| 19 | C_19 − D_19 | -0.55 [-0.85, -0.10] | -0.14 [-0.98, +0.66] | -1.95 [-4.35, +0.15] | +1.85 [+0.02, +3.75] |
| 19 | C_19 − R_19 | -0.38 [-0.67, +0.07] | -0.72 [-1.65, +0.27] | -1.80 [-3.95, +0.00] | +0.40 [-1.12, +1.93] |
| 3 | D, no margin | 0.36 | 1.57 | 2.27 | 1.80 |
| 3 | R, no margin | 0.57 | 2.79 | 2.90 | 3.00 |
| 3 | A, nominal schedule | 6.16 | 12.09 | 14.75 | 9.65 |
| | draws scored | 91 | 100 | 40 | 40 |

**Mean absolute error [passes] (draws within ±1 pass)** of the rules without a cost weighting (r = 1)

| rule | matched | realistic | `realistic_b` | `realistic_c` |
|---|---|---|---|---|
| C_1 | 0.15 (90) | 0.54 (96) | 0.80 (36) | 0.78 (37) |
| D_1 | 0.43 (88) | 0.44 (97) | 0.93 (35) | 0.65 (38) |
| R_1 | 0.38 (87) | 0.67 (87) | 1.88 (23) | 0.80 (37) |
| D, no margin | 0.19 (90) | 0.53 (94) | 0.88 (35) | 0.65 (39) |
| R, no margin | 0.26 (90) | 1.07 (67) | 1.95 (20) | 1.00 (30) |
| A | 2.93 (24) | 4.25 (14) | 4.95 (3) | 3.55 (7) |

**What the tables say.** C's rules need no tuning: the same posterior gives the rule for every cost ratio. D and R need their margin re-tuned for each ratio, on draws whose crossing is known. Without a margin their cost grows quickly with the ratio: at r = 19 in the realistic world, D costs 9.89 and R 16.55, against C's 1.53. Against the *tuned* rules the picture is mixed:

* C_r is significantly cheaper in the matched world (against D_r at all three ratios, e.g. r = 3: -0.16 [-0.29, -0.03]) and in `realistic_b` at r = 3 (against D_3: -0.82 [-1.43, -0.33]; against R_3: -1.20 [-1.98, -0.42]).
* In the realistic world and in `realistic_c` most differences are not significant (realistic, r = 3: C_3 − D_3 = -0.14 [-0.39, +0.09], C_3 − R_3 = -0.01 [-0.27, +0.25]).
* One goes significantly against C: in `realistic_c` at r = 19 the tuned D is cheaper (+1.85 [+0.02, +3.75]). Two more lean that way without being significant: D_1 in the realistic world at r = 1 (+0.10 [+0.00, +0.21]) and R_3 in `realistic_c` at r = 3 (+0.38 [+0.00, +0.75]).

So, for these decisions, C's posterior is about as good as the best point forecast with a margin tuned for the cost at hand: sometimes better, occasionally worse. Its practical advantage is that it needs no labelled crossings to set that margin. The model-free rule R is a strong competitor once its margin is tuned; untuned, it is the most expensive rule after the nominal schedule in the three mismatch worlds.

**Forecast error by lead time** (figure 5, right; a fixed population of draws whose crossing every lead can see). As a *point* forecast of the crossing pass, C's median is no better than D's. In the matched world the two are within about a tenth of a pass of each other up to 8 passes ahead (5 passes ahead: C 0.35, D 0.31, A 2.28 passes), and C is better 9 and 10 passes ahead. In the realistic world D's forecast is the more accurate from 2 to 8 passes ahead (5 passes ahead: C 1.30, D 0.48, A 1.79), and C's early forecasts are biased early (mean signed error after pass 2: -1.02 passes): the break-in makes the abrasive look as if it wears faster than it will. What C adds is the spread: in the realistic world its ≥90% band contains the true crossing in 100% of draws one pass ahead and 94% five passes ahead, falling to 61% ten passes ahead. The C_r rules use that spread.

![Abrasive change](figures/fig5_abrasive_change.png)

### 4.5 Tuning: what the drifting wear rate buys

| wear-rate drift δ (held-out draws 200–215) | 0 (constant rate) | chosen: 0.1 |
|---|---|---|
| predictive coverage, matched | 95.4% | 93.8% |
| predictive coverage, realistic | 85.9% | 92.4% |
| median relative width of the 90% interval, realistic | 12.50% | 11.37% |
| mean interval score (lower is better; both worlds) | 0.1325 | 0.1157 |

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

| scan noise σ [µm] | 2 (default) | 5 | 10 |
|---|---|---|---|
| C pass-20 RMSE, median [µm] | 0.386 | 0.390 | 0.390 |
| oracle, median [µm] | 0.101 | 0.101 | 0.101 |
| predictive coverage | 88% | 91% | 93% |

In the realistic world C's error barely depends on scan noise up to 10 µm: with thousands of points per scan the noise averages out, and the error is dominated by what neither scans nor the oracle can predict (force ripple, fluctuations) and by model mismatch. Coverage rises with the noise because noisier scans leave C less certain, which partly offsets the mismatch.

### 4.8 The pre-registered worlds

Both were committed before any of their results existed and were never used for tuning or development (section 3.6); the tables above include them. `realistic_c` was committed together with the final tracker C, so it is the only fully out-of-sample test of the final code.

| | `realistic_b`, pre-registered (harsher, ring-wise wear) | `realistic_c`, pre-registered (milder) |
|---|---|---|
| C pass-20 RMSE, median [µm] | 0.489 (D 0.532, oracle 0.129) | 0.425 (D 0.435, oracle 0.132) |
| true pass-20 mean removal, median [µm] | 3.59 | 4.65 |
| C better than D, start-up (passes 2–3) / steady state (passes 4–20) | 31 / 18 of 40 | 31 / 15 of 40 |
| C's 90% predictive coverage, passes 2–20 | 80% | 90% |
| abrasive-change cost at r = 3: C_3 / D_3 / R_3 (tuned margins) | 1.35 / 2.17 / 2.55 | 1.27 / 1.50 / 0.90 |

In both pre-registered worlds C's first two predictions were better than D's, and from pass 4 on the two were equally accurate (steady-state D/C 0.996 and 0.991). At r = 3, C's untuned rule was significantly cheaper than the tuned D and R in `realistic_b`; in `realistic_c` the differences were not significant. The milder `realistic_c` gives the best-calibrated mismatch result of the study (90% coverage of the mean removal), but its block-level coverage (54%) shows that the map shape is still wrong. The harsher `realistic_b`, with six-ring wear, is where C is weakest: an error 3.7 times the oracle's at pass 20 (median over draws), 80% coverage of the mean removal and 48% of the map blocks inside their intervals.

## 5. Limitations

* **Simulated only.** Nothing here has touched a physical part. The realistic worlds are richer simulators, not reality: their effects and sizes were chosen by me, not measured, and real processes contain effects in none of them (loading and clogging of the abrasive, heat, grit changes, pad ageing, robot path errors).
* **Same author for the worlds and the tracker.** The scanner-offset model, the outlier gating, the force exponent and the transient gain address effects I also put into the realistic world. The two pre-registered worlds limit how much the tracker could be tuned to them, but they were designed by the same person, with the same kinds of effects.
* **No accuracy advantage after start-up.** From the fourth pass on, D (least squares on the same model, with the same priors) is as accurate as C in every world. C's advantages are its first two predictions, calibrated uncertainty, and the decisions built on it.
* **Effects outside the model family.** Uneven wear across the pad face, a tilting holder, abrasive loading and the pressure-profile part of a Preston exponent change the *shape* of the removal map in ways a linear-pad, uniform-wear model cannot reproduce. Two-ring wear is the largest single cost in the realistic world, and the stress tests defeat both trackers (section 4.3). C's block-level intervals under-cover in every mismatch world.
* **Under-coverage under mismatch.** For the mean removal, C's 90% predictive intervals cover 89% in the realistic world (85% in passes 2–5) and 80% in the harsher pre-registered world. Block by block, they cover 52%, 48% and 54%.
* **Assumed costs, small tuning set.** The late/early cost ratios (1, 3, 19) are assumptions, not measured costs. D's and R's margins were tuned on only 16 held-out draws per world.
* **Parameters are only meaningful in the matched world.** Under mismatch C's stiffness, wear rate and force exponent are effective values, not physical properties.
* **Long-range abrasive-change forecasts** are biased early when the wear law has a break-in phase (section 4.4).
* **Registration error is not corrected**; C treats misregistration as structured noise.
* **Winkler-type pad.** Both pad laws are independent springs: no shear coupling and no bending of the backing plate; holder tilt is simulated only in a stress test (with a linear pad), and its stiffness is assumed, not measured. The contact uses the nominal (CAD) surface; removal is not fed back into the contact.
* **One abrasive, removal depth only.** One grit per run; no roughness or other finish metric (the scan texture is a measurement effect only).
* **Grid effects.** The 1 mm grid quantises the stiffest contact strips (section 3.2).
* **Surrogate.** C's table has a worst-case relative error of 1.7e-03 against the exact model, at the fastest-wearing draws (third-order expansion of the within-pass wear).
* **Not validated on physical coupons.**

## 6. Next steps that would make it real

1. **Fit the wear law to coupon data.** Scan coupons after every pass at several forces and grits; measure break-in, the force exponent, pass-to-pass fluctuation and whether the pad centre wears faster. This decides whether the tracker needs a break-in state and a ring-resolved effectiveness.
2. **Ring-resolved effectiveness in the tracker** (a few rings, with a smoothness prior), so that non-uniform wear changes the predicted map shape instead of biasing the stiffness.
3. **Break-in and loading states in the abrasive model** (or a law learned from coupon data), so that early abrasive-change forecasts are not biased early and temporary clogging is not read as wear.
4. **Register each scan to the prediction** (two shifts per scan as nuisance parameters) before updating.
5. **Replace the spring pad** with a finite-element or learned pad model, and give the tracker the holder tilt (measured tilt stiffness) so that edge stations are predicted correctly.
6. **Run on GrayMatter-style scan data** with real registration error, missing data and noise.
7. **Close the loop**: choose the next pass's force or dwell from C's predictive distribution to hit a target removal, and set the abrasive-change cost ratio from real costs of a worn pass and of discarded abrasive.

## 7. How to reproduce

Requires Python ≥ 3.11. The recorded run used Python 3.11.15 and NumPy 2.4.6; see `results/run_info.json`.

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib, pyyaml, pytest
python -m swt run-all                    # every experiment, every figure, every number in results/
python -m swt run-all --full             # the same with 40-draw single-effect, stress, noise and ablation runs (~15 min)
python -m swt report                     # re-render README.md and SUMMARY.md from results/
python -m swt run --config configs/default.yaml    # main 20-pass experiment only (uses results/tuning.json if present)
python -m pytest -q                      # tests
```

* **Runtime.** `run-all` runs 498 multi-draw episodes, plus the tuning runs and the main run, using `min(4, CPUs)` worker processes. In the recorded run, building the model and surrogate took 8 s and the experiments 553 s with 4 workers on a 4-CPU machine. `--workers 1` runs serially; results are identical for any worker count because every task has its own seeds.
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
