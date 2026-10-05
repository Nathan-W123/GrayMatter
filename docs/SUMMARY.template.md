# sanding-wear-tracker: summary

This simulation study asks whether a sanding process model can keep track of two unknowns at once: a pad stiffness that is fixed but uncertain, and abrasive sharpness that drops every pass. It was motivated by GrayMatter Robotics' post *[World Models for Manufacturing Processes](https://factory.graymatter-robotics.com/world-models-for-manufacturing-processes/)*, but it is independent work and uses no GrayMatter data or code. In a simulation where a 125 mm compliant pad sands a curved panel with hidden parameters, a particle filter that re-estimates them from each post-pass scan reaches, after the first few passes, an error close to the floor set by random wear fluctuation, while a model calibrated once on the first scan drifts as the paper dulls; the filter also predicts when the abrasive falls below 50% of fresh effectiveness. These are best-case results: the filter shares the simulator's physics and noise levels, nothing was tested on physical parts, alternating the force between passes did not help separate stiffness from wear, and interval coverage was slightly low for two of four parameters.

![Prediction error per pass](figures/fig3_rmse_per_pass.png)

**Key number:** over {{robustness.n_draws:d}} simulated hidden truths, the median error of the pass-{{robustness.final_pass:d}} removal-map prediction was {{robustness.final_rmse_um.C.median:.3f}} µm for the particle filter, against {{robustness.final_rmse_um.B.median:.2f}} µm for the model calibrated once ({{robustness.final_rmse_um.A.median:.2f}} µm for a nominal model with a wear law; median true removal in that pass {{robustness.final_true_mean_removal_um.median:.2f}} µm).

Full model, results, weak spots and limitations: [README.md](README.md).
