# sanding-wear-tracker: summary

This simulation study asks whether a sanding model can track two unknowns at once, a fixed but unknown pad stiffness and an abrasive that dulls every pass, from one scan per pass, even when the process has effects the model leaves out. It was motivated by GrayMatter Robotics' post *[World Models for Manufacturing Processes](https://factory.graymatter-robotics.com/world-models-for-manufacturing-processes/)*, but it is independent work and uses no GrayMatter data or code. A particle filter predicted the next pass far better than a nominal or a calibrate-once model, and in most worlds its uncertainty alone chose when to change the abrasive about as well as a forecast with a tuned safety margin. But after a few scans a plain least-squares fit was at least as accurate, clogged abrasive misled the filter into changing it too early, and two extended trackers improved the maps but neither beat the least-squares fit's decisions under clogging; nothing was tested on physical parts.

![Prediction error per pass](figures/fig3_rmse_per_pass.png)

**Key number:** in a world fixed before any tracker was run in it, the filter's removal-map error from the fourth pass on was {{ext.robustness.worlds.realistic_d.C2.C_over_C2_steady.median:.2f}} times that of its extension that learns the missing map shape (90% interval {{ext.robustness.worlds.realistic_d.C2.C_over_C2_steady.ci90.0:.2f}}–{{ext.robustness.worlds.realistic_d.C2.C_over_C2_steady.ci90.1:.2f}}), which was more accurate in {{ext.robustness.worlds.realistic_d.C2.C_over_C2_steady.C2_better:d}} of {{ext.robustness.worlds.realistic_d.C2.C_over_C2_steady.n:d}} simulated cases.

Full model, results, weak spots and limitations: [README.md](README.md).
