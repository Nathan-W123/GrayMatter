"""Figures. Everything is drawn from the files in results/, never recomputed here."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated categorical slots (light mode) + recessive neutrals.
BLUE = "#2a78d6"      # slot 1: C, the joint tracker (the series that matters)
ORANGE = "#eb6834"    # slot 2: B, calibrate-once
GRAY = "#8c8b86"      # A, nominal (recessive)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("swt_blue", ["#f4f8fd"] + BLUE_RAMP)

EST = {"A": ("A: nominal", GRAY), "B": ("B: calibrate-once", ORANGE), "C": ("C: joint tracker", BLUE)}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "legend.fontsize": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2.0, "legend.frameon": False,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- fig 1


def fig_contact(res: Path, figdir: Path) -> None:
    sweep = _read_csv(res / "contact_sweep.csv")
    summ = json.loads((res / "contact_sweep.json").read_text())
    prof = _read_csv(res / "contact_profiles.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    fine = [r for r in sweep if r["grid_mm"] != summ["grid_mm"]]
    sweep = [r for r in sweep if r["grid_mm"] == summ["grid_mm"]]
    forces = sorted({r["force_N"] for r in sweep})
    shades = [BLUE_RAMP[2], BLUE_RAMP[4], BLUE_RAMP[6]][: len(forces)]
    F_ref = summ["reference_force_N"]
    for F, c in zip(forces, shades):
        rr = [r for r in sweep if r["force_N"] == F]
        ax.plot([r["k_pad"] for r in rr], [100 * r["contact_fraction"] for r in rr], color=c,
                lw=2.6 if F == F_ref else 1.8, label=f"{F:g} N")
    if fine:
        ax.plot([r["k_pad"] for r in fine], [100 * r["contact_fraction"] for r in fine], color=INK,
                lw=1.0, ls="--", label=f"{F_ref:g} N, {fine[0]['grid_mm']:g} mm grid")
    ax.axvspan(summ["k_low"], summ["k_high"], color="#f0efec", zorder=0, lw=0)
    b = summ["contact_fraction_at_prior_bounds"][f"{F_ref:g}N"]
    for k, frac, off in ((summ["k_low"], b["soft_k_low"], (10, -20)), (summ["k_high"], b["stiff_k_high"], (6, 22))):
        ax.plot([k], [100 * frac], "o", color=INK, ms=8, zorder=5)
        ax.annotate(f"{100 * frac:.1f}%", (k, 100 * frac), xytext=off, textcoords="offset points",
                    ha="left", fontsize=13, fontweight="bold")
    ax.text(np.sqrt(summ["k_low"] * summ["k_high"]), 104, "prior range of k_pad", ha="center", va="bottom",
            color=INK2, fontsize=12)
    ax.set_xscale("log")
    ax.set_ylim(0, 112)
    ax.set_xlabel("pad stiffness k_pad [N/mm³]")
    ax.set_ylabel("pad face in contact [%]")
    ax.set_title("Contact fraction vs. stiffness")
    ax.legend(title=f"force ({summ['grid_mm']:g} mm grid)", loc="lower left", fontsize=10.5, title_fontsize=10.5)
    labels = {"stiff": "stiff", "mid": "mid", "soft": "soft"}
    for name, c, ls in (("stiff", INK, "-"), ("mid", INK2, "--"), ("soft", GRAY, ":")):
        rr = [r for r in prof if r["pad"] == name]
        x = [r["offset_mm"] for r in rr]
        y = [r["pressure_kPa"] for r in rr]
        bx.plot(x, y, color=c, ls=ls, lw=2.2,
                label=f"k={rr[0]['k_pad']:.2g}: {100 * rr[0]['contact_fraction']:.0f}% contact")
    bx.set_yscale("log")
    bx.set_ylim(0.1, 3000)
    bx.set_xlabel("distance from pad centre, across curvature [mm]")
    bx.set_ylabel("pressure [kPa] (log)")
    bx.set_title(f"Pressure under the pad at {F_ref:g} N")
    bx.legend(loc="upper right", fontsize=11)
    fig.suptitle(f"Stiffness sets how much of the pad touches (R = {summ['radius_mm']:g} mm cylinder)",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, figdir / "fig1_contact_vs_stiffness.png")


# --------------------------------------------------------------------------- fig 2


def fig_maps(res: Path, figdir: Path) -> None:
    z = np.load(res / "main_run_maps.npz")
    main = json.loads((res / "main_run.json").read_text())
    passes = [main["early_pass"], main["final_pass"]]
    x, y = z["x"], z["y"]
    ext = [x[0], x[-1], y[0], y[-1]]
    cols = [("true", "True removal"), ("A", EST["A"][0]), ("B", EST["B"][0]), ("C", EST["C"][0])]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.3), constrained_layout=True)
    for i, p in enumerate(passes):
        true = z[f"pass{p}_true"]
        vmax = 1.5 * float(true.max())
        for j, (key, title) in enumerate(cols):
            ax = axes[i, j]
            m = z[f"pass{p}_{key}"]
            im = ax.imshow(m, origin="lower", extent=ext, cmap=SEQ, vmin=0, vmax=vmax, aspect="equal",
                           interpolation="nearest")
            ax.grid(False)
            note = f"mean {m.mean():.1f} µm"
            if key != "true":
                note += f"\nRMSE {np.sqrt(np.mean((m - true) ** 2)):.2f} µm"
            if np.mean(m > vmax) > 0.9:
                note += "\nabove colour-scale max\nalmost everywhere"
            ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", ha="left", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
            ax.set_title(title if i == 0 else "", fontsize=14)
            if j == 0:
                ax.set_ylabel(f"pass {p}\n" + "y (curved) [mm]", fontsize=13)
            else:
                ax.set_yticklabels([])
            if i == 1:
                ax.set_xlabel("x (straight) [mm]")
            else:
                ax.set_xticklabels([])
        cb = fig.colorbar(im, ax=axes[i, :], shrink=0.9, extend="max", pad=0.01)
        cb.set_label("removal depth [µm]")
    fig.suptitle(f"Removal maps, main run, pass {passes[0]} and pass {passes[1]} "
                 "(colour scale capped at 1.5× true max)", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    _save(fig, figdir / "fig2_removal_maps.png")


# --------------------------------------------------------------------------- fig 3


def fig_rmse(res: Path, figdir: Path) -> None:
    main = _read_csv(res / "main_run.csv")
    rob = _read_csv(res / "robustness_per_pass.csv")
    rsum = json.loads((res / "robustness_summary.json").read_text())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    passes = [r["pass"] for r in main]
    for k in ("A", "B", "C"):
        lab, c = EST[k]
        ys = [r[f"rmse_{k}_um"] for r in main]
        ax.plot(passes, ys, color=c, marker="o", ms=5, label=lab, lw=2.4 if k == "C" else 2.0)
    ax.plot(passes[1:], [r["rmse_oracle_um"] for r in main[1:]], color=INK2, lw=1.0, ls="--",
            label="reference: oracle (true k_pad, λ, last K)")
    ax.set_yscale("log")
    ax.set_xlabel("pass")
    ax.set_ylabel("removal-map RMSE [µm] (log)")
    ax.set_title("Main run")
    ax.set_xticks(range(0, len(passes) + 1, 5))
    n_draws = rsum["n_draws"]
    for k in ("A", "B", "C", "oracle"):
        med, lo, hi = [], [], []
        for p in passes:
            v = np.array([r[f"rmse_{k}_um"] for r in rob if r["pass"] == p])
            med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
        if k == "oracle":   # zero at pass 1 (the oracle knows K0), so start at pass 2
            bx.plot(passes[1:], med[1:], color=INK2, lw=1.0, ls="--")
            continue
        lab, c = EST[k]
        bx.fill_between(passes, lo, hi, color=c, alpha=0.18, lw=0)
        bx.plot(passes, med, color=c, lw=2.4 if k == "C" else 2.0)
        bx.annotate(lab.split(":")[0], (passes[-1], med[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=13, fontweight="bold")
    bx.set_yscale("log")
    bx.set_xlabel("pass")
    bx.set_title(f"{n_draws} hidden truths: median, IQR")
    bx.set_xticks(range(0, len(passes) + 1, 5))
    bx.set_xlim(0.5, len(passes) + 2.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("B drifts as the abrasive wears; C stays near the oracle",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, 0.0))
    _save(fig, figdir / "fig3_rmse_per_pass.png")


# --------------------------------------------------------------------------- fig 4


def fig_params(res: Path, figdir: Path) -> None:
    main = _read_csv(res / "main_run.csv")
    passes = np.array([r["pass"] for r in main])
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1]})
    specs = [("k_pad", "true_k_pad", "k_pad [N/mm³]", "B_k_pad", 1.0, "Stiffness k_pad (static)"),
             ("K", "true_K", "K [10⁻⁵ mm²/N]", "B_K", 1e5, "Abrasive effectiveness K (wears)"),
             ("lam", "true_lam", "λ [10⁻⁴ 1/mm³]", None, 1e4, "Wear rate λ (static)")]
    for j, (name, tkey, ylabel, bkey, s, title) in enumerate(specs):
        ax, ex = axes[0, j], axes[1, j]
        med = np.array([r[f"C_{name}_mean"] for r in main])
        lo = np.array([r[f"C_{name}_lo"] for r in main])
        hi = np.array([r[f"C_{name}_hi"] for r in main])
        true = np.array([r[tkey] for r in main])
        ax.fill_between(passes, s * lo, s * hi, color=BLUE, alpha=0.22, lw=0, label="C: 90% credible interval")
        ax.plot(passes, s * med, color=BLUE, lw=2.2, label="C: posterior mean")
        ax.plot(passes, s * true, color=INK, lw=1.4, marker="o", ms=4, label="true value")
        if bkey is not None:
            bvals = s * np.array([r[bkey] for r in main])
            ax.plot(passes[1:], bvals[1:], color=ORANGE, lw=2.0, label="B: frozen after pass 1")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if name == "k_pad":
            ax.set_ylim(s * true[0] * 0.9, s * true[0] * 1.1)
        elif name == "K":
            ax.set_ylim(0, max(true.max(), med.max()) * s * 1.15)
        else:
            ax.set_ylim(0, s * true[0] * 2.2)
        # relative error row
        ex.axhline(0, color=INK, lw=1.2)
        ex.fill_between(passes, 100 * (lo / true - 1), 100 * (hi / true - 1), color=BLUE, alpha=0.22, lw=0)
        ex.plot(passes, 100 * (med / true - 1), color=BLUE, lw=2.2)
        ex.set_ylabel("C error vs truth [%]")
        ex.set_xlabel("pass")
        ex.set_xticks(range(0, len(passes) + 1, 5))
        win = max(abs(100 * (lo[2:] / true[2:] - 1)).max(), abs(100 * (hi[2:] / true[2:] - 1)).max())
        ex.set_ylim(-1.3 * win, 1.3 * win)
        if j == 0:
            ex.text(0.03, 0.04, "truth inside ⇔ band covers 0", transform=ex.transAxes,
                    ha="left", va="bottom", fontsize=11, color=INK2)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.suptitle("Joint tracker C: estimates, 90% intervals (main run); bottom row from pass 3",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.0))
    _save(fig, figdir / "fig4_parameter_tracking.png")


# --------------------------------------------------------------------------- fig 5


def fig_change(res: Path, figdir: Path) -> None:
    main = _read_csv(res / "main_run.csv")
    rob = _read_csv(res / "robustness_per_pass.csv")
    n_passes = len(main)
    true_cross = int(main[0]["cross_true"])
    cross_A = main[0]["cross_A"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6))
    n_show = min(n_passes, true_cross + 2)
    rr = main[:n_show]
    p = np.array([r["pass"] for r in rr])
    med = np.array([r["cross_C_median"] for r in rr])
    lo = np.array([r["cross_C_lo"] for r in rr])
    hi = np.array([r["cross_C_hi"] for r in rr])
    ax.axvspan(true_cross - 0.5, n_show + 0.5, color="#f0efec", lw=0, zorder=0)
    ax.text(true_cross - 0.4, 0.6, "after the\ncrossing", color=INK2, fontsize=11, va="bottom")
    ax.errorbar(p, med, yerr=[med - lo, hi - med], color=BLUE, marker="o", ms=7, lw=2.2, capsize=5,
                label="C: predicted crossing (median, ≥90% band)")
    ax.axhline(true_cross, color=INK, lw=1.6, label=f"true crossing (pass {true_cross})")
    ax.axhline(cross_A, color=GRAY, lw=2.0, ls="--", label=f"A: nominal schedule (pass {cross_A:g})")
    ax.set_xlabel("prediction made after pass")
    thr = json.loads((res / "abrasive_change.json").read_text())["threshold_fraction"]
    ax.set_ylabel(f"predicted pass where K < {100 * thr:g}% of K0")
    ax.set_title("Main run")
    ax.set_xticks(p)
    ax.set_xlim(0.5, n_show + 0.5)
    ax.set_ylim(0, max(hi.max(), cross_A, true_cross) + 3)
    ax.legend(loc="upper right", fontsize=11)
    # robustness: error vs lead time on a fixed population (computed in experiments.py)
    ch = json.loads((res / "abrasive_change.json").read_text())
    lead = ch["by_lead"]
    rows = lead["rows"]
    if not rows:
        bx.text(0.5, 0.5, "no draw has every lead observable", ha="center", transform=bx.transAxes)
    else:
        leads = [r["lead_passes"] for r in rows]
        mC = [r["mean_abs_err_C"] for r in rows]
        mA = [r["mean_abs_err_A"] for r in rows]
        bx.plot(leads, mA, color=GRAY, lw=2.0, ls="--", label="A: nominal (same prediction at every lead)")
        bx.plot(leads, mC, color=BLUE, marker="o", ms=6, lw=2.4, label="C: joint tracker")
        bx.text(0.97, 0.52, f"truth inside C's ≥90% band:\n{100 * lead['band_coverage_min']:.0f}–"
                f"{100 * lead['band_coverage_max']:.0f}% of draws at every lead", transform=bx.transAxes,
                ha="right", va="center", fontsize=11.5, color=INK2)
        bx.set_xticks(leads)
        bx.set_ylim(0, max(mA) * 1.3)
        bx.legend(loc="upper left", fontsize=10.5)
    lo_c, hi_c = lead["crossing_range"]
    bx.set_xlabel("passes before the true crossing")
    bx.set_ylabel("mean |predicted − true| [passes]")
    bx.set_title(f"{lead['population']} truths crossing at pass {lo_c}–{hi_c}")
    fig.suptitle("When to change the abrasive: predicted vs. true crossing",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, figdir / "fig5_abrasive_change.png")


# --------------------------------------------------------------------------- fig 6


def fig_ablation_noise(res: Path, figdir: Path) -> None:
    ab = json.loads((res / "ablation_summary.json").read_text())
    nz = json.loads((res / "noise_sensitivity.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    sched = ab["by_schedule"]
    labels = {"alternating": (f"alternating {'/'.join(f'{f:g}' for f in ab['forces_alternating_N'])} N", BLUE),
              "constant": (f"constant {ab['force_constant_N']:g} N", ORANGE)}
    for ax, key, title in ((axes[0], "per_pass_relw_k_pad", "Ablation: k_pad interval"),
                           (axes[1], "per_pass_relw_lam", "Ablation: λ interval")):
        for name, (lab, c) in labels.items():
            v = sched[name][key]
            p = np.arange(1, len(v) + 1)
            med = 100 * np.array([x["median"] for x in v])
            lo = 100 * np.array([x["q25"] for x in v])
            hi = 100 * np.array([x["q75"] for x in v])
            ax.fill_between(p, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(p, med, color=c, lw=2.2, label=lab)
        if "alternating_reseeded" in sched:
            v = sched["alternating_reseeded"][key]
            ax.plot(np.arange(1, len(v) + 1), 100 * np.array([x["median"] for x in v]), color=INK2, lw=1.0,
                    ls="--", label="alternating, other filter seed")
        ax.set_yscale("log")
        ticks = [0.5, 1, 2, 3, 5] if "k_pad" in key else [10, 15, 20, 30, 50, 100]
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("pass")
        ax.set_ylabel("90% interval width [%] (log)")
        ax.set_title(title)
        ax.set_xticks(range(0, len(p) + 1, 5))
    axes[0].legend(loc="upper right", fontsize=11)
    lv = list(nz["levels"].values())
    sig = np.array([v["noise_um"] for v in lv])
    med = np.array([v["final_rmse_C_um"]["median"] for v in lv])
    lo = np.array([v["final_rmse_C_um"]["q25"] for v in lv])
    hi = np.array([v["final_rmse_C_um"]["q75"] for v in lv])
    axes[2].errorbar(sig, med, yerr=[med - lo, hi - med], color=BLUE, marker="o", ms=7, lw=2.2,
                     capsize=4, label="C (median, IQR)")
    floor = lv[0]["final_rmse_oracle_um"]["median"]   # the oracle does not use the scans
    axes[2].axhline(floor, color=INK2, lw=1.0, ls="--", label="reference: oracle (median)")
    axes[2].set_xscale("log")
    axes[2].set_xlim(0.8, 16)
    axes[2].set_xticks(sig)
    axes[2].set_xticklabels([f"{v['noise_um']:g}\n{100 * v['final_relative_width']['k_pad']['median']:.2f}%"
                             for v in lv])
    axes[2].set_xlabel("scan noise σ [µm]  /  k_pad 90% width")
    axes[2].set_ylabel("final-pass RMSE [µm]")
    axes[2].set_ylim(0, 1.5 * max(hi))
    axes[2].set_title("Scan noise")
    axes[2].legend(loc="upper left", fontsize=11)
    fig.suptitle(f"Constant-force ablation and scan-noise sensitivity ({ab['n_draws']} hidden truths)",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, figdir / "fig6_ablation_and_noise.png")


# --------------------------------------------------------------------------- drivers


def make_main(results: str | Path, figures: str | Path, ctx=None) -> None:
    res, figdir = Path(results), Path(figures)
    figdir.mkdir(parents=True, exist_ok=True)
    _style()
    fig_maps(res, figdir)
    fig_params(res, figdir)


def make_all(results: str | Path, figures: str | Path, ctx=None) -> None:
    res, figdir = Path(results), Path(figures)
    figdir.mkdir(parents=True, exist_ok=True)
    _style()
    print("[swt] drawing figures")
    fig_contact(res, figdir)
    fig_maps(res, figdir)
    fig_rmse(res, figdir)
    fig_params(res, figdir)
    fig_change(res, figdir)
    fig_ablation_noise(res, figdir)
