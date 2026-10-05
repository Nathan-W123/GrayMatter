# sanding-wear-tracker: summary

This simulation study asks whether a sanding process model can keep track of two unknowns at once, a pad stiffness that is fixed but uncertain and an abrasive sharpness that drops every pass, and whether that still works when the simulated process contains effects the tracker does not model (a foam pad, force errors and ripple, a two-stage wear law, scanner artefacts). It was motivated by GrayMatter Robotics' post *[World Models for Manufacturing Processes](https://factory.graymatter-robotics.com/world-models-for-manufacturing-processes/)*, but it is independent work and uses no GrayMatter data or code. A particle filter that re-estimates stiffness, effectiveness and a drifting wear rate from each post-pass scan had a lower median next-pass prediction error than a nominal model, a model calibrated once, and a baseline that refits every scan, and it named the abrasive-change pass accurately close to the change; in the unmodelled world its 90% intervals were too narrow (73% coverage), its parameters became effective rather than true values, and its long-range change forecasts were no better than a fixed schedule. Nothing was tested on physical parts.

![Prediction error per pass](figures/fig3_rmse_per_pass.png)

**Key number:** in the world with unmodelled effects, over 100 simulated hidden truths, the median error of the pass-20 removal-map prediction was 0.171 µm for the particle filter, against 0.209 µm for refitting every scan, 10.93 µm for calibrating once and 0.142 µm for an oracle that knows the true physics and state (median true removal in that pass 5.14 µm).

Full model, results, weak spots and limitations: [README.md](README.md).
