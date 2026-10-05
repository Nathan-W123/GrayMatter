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
GREEN = "#1f9e74"     # slot 3: D, refit each pass (the strong baseline)
GRAY = "#8c8b86"      # A, nominal (recessive)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("swt_blue", ["#f4f8fd"] + BLUE_RAMP)

EST = {"A": ("A: nominal", GRAY), "B": ("B: calibrate-once", ORANGE), "D": ("D: refit each pass", GREEN),
       "C": ("C: joint tracker", BLUE)}
WORLD_LABEL = {"matched": "matched world (tracker's own model)", "realistic": "realistic world (unmodelled effects)",
               "realistic_b": "second mismatch world (pre-registered)",
               "only_pad": "foam pad only", "only_removal": "Preston exponent only", "only_force": "force errors only",
               "only_wear": "abrasive wear only", "only_scan": "scanner effects only"}


def _truthy(v) -> bool:
    return v is True or v == 1.0 or str(v) == "True"


def _dodge(values: list[float], min_gap: float) -> list[float]:
    """Positions for end labels (log10 units): keep their order, at least min_gap apart."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if pos[b] - pos[a] < min_gap:
            pos[b] = pos[a] + min_gap
    return list(pos)


def _log_ticks(ax, candidates) -> None:
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([c for c in candidates if lo <= c <= hi]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _per_pass(rows: list[dict], key: str, passes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med, lo, hi = [], [], []
    for p in passes:
        v = np.array([r[key] for r in rows if r["pass"] == p], dtype=float)
        med.append(np.median(v))
        lo.append(np.percentile(v, 25))
        hi.append(np.percentile(v, 75))
    return np.array(med), np.array(lo), np.array(hi)


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
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.9), gridspec_kw={"width_ratios": [1.15, 1]})
    foam = [r for r in sweep if r.get("pad_law") == "foam"]
    fine = [r for r in sweep if r["grid_mm"] != summ["grid_mm"]]
    sweep = [r for r in sweep if r["grid_mm"] == summ["grid_mm"] and r.get("pad_law", "linear") == "linear"]
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
    if foam:
        h = (summ.get("foam_pad_at_prior_bounds") or {}).get("thickness_mm")
        ax.plot([r["k_pad"] for r in foam], [100 * r["contact_fraction"] for r in foam], color=GREEN,
                lw=1.8, ls="-.", label=f"{F_ref:g} N, foam pad ({h:g} mm, realistic world)")
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
    ax.legend(title=f"force ({summ['grid_mm']:g} mm grid unless stated)", loc="upper center",
              bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=10, title_fontsize=10)
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
    world = main["world"]
    s = main["worlds"][world]
    passes = [s["early_pass"], s["final_pass"]]
    x, y = z["x"], z["y"]
    ext = [x[0], x[-1], y[0], y[-1]]
    cols = [("true", "True removal"), ("scan", "Scan (noise, artefacts)"), ("A", EST["A"][0]),
            ("B", EST["B"][0]), ("D", EST["D"][0]), ("C", EST["C"][0])]
    fig, axes = plt.subplots(2, 6, figsize=(17, 5.6), constrained_layout=True)
    for i, p in enumerate(passes):
        true = z[f"pass{p}_true"]
        vmax = 1.5 * float(true.max())
        for j, (key, title) in enumerate(cols):
            ax = axes[i, j]
            m = z[f"pass{p}_{key}"]
            im = ax.imshow(np.ma.masked_invalid(m), origin="lower", extent=ext, cmap=SEQ, vmin=0, vmax=vmax,
                           aspect="equal", interpolation="nearest")
            ax.grid(False)
            note = f"mean {np.nanmean(m):.1f} µm"
            if key not in ("true", "scan"):
                note += f"\nRMSE {np.sqrt(np.mean((m - true) ** 2)):.2f} µm"
            if np.mean(m > vmax) > 0.9:
                note += "\nabove colour max"
            ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", ha="left", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
            ax.set_title(title if i == 0 else "", fontsize=13)
            if j == 0:
                ax.set_ylabel(f"pass {p}\n" + "y (curved) [mm]", fontsize=12)
            else:
                ax.set_yticklabels([])
            if i == 1:
                ax.set_xlabel("x [mm]")
            else:
                ax.set_xticklabels([])
        cb = fig.colorbar(im, ax=axes[i, :], shrink=0.9, extend="max", pad=0.01)
        cb.set_label("removal [µm]")
    fig.suptitle(f"Removal maps at the scan points, main run in the {world} world, pass {passes[0]} and "
                 f"pass {passes[1]} (colour scale capped at 1.5× true max)", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    _save(fig, figdir / "fig2_removal_maps.png")


# --------------------------------------------------------------------------- fig 3


def fig_rmse(res: Path, figdir: Path) -> None:
    rob = _read_csv(res / "robustness_per_pass.csv")
    rsum = json.loads((res / "robustness_summary.json").read_text())
    worlds = [w for w in ("matched", "realistic", "realistic_b") if w in rsum["worlds"]]
    fig, axes = plt.subplots(1, len(worlds), figsize=(5.6 * len(worlds), 4.9), sharey=True, squeeze=False)
    for ax, w in zip(axes[0], worlds):
        rows = [r for r in rob if r["world"] == w]
        passes = sorted({int(r["pass"]) for r in rows})
        med, _, _ = _per_pass(rows, "rmse_oracle_um", passes)
        ax.plot(passes[1:], med[1:], color=INK2, lw=1.2, ls="--", label="oracle (true physics and state)")
        ends = {}
        for k in ("A", "B", "D", "C"):
            lab, c = EST[k]
            med, lo, hi = _per_pass(rows, f"rmse_{k}_um", passes)
            ax.fill_between(passes, lo, hi, color=c, alpha=0.16, lw=0)
            ax.plot(passes, med, color=c, lw=2.6 if k == "C" else 1.9, label=lab)
            ends[k] = np.log10(med[-1])
        for k, y in zip(ends, _dodge(list(ends.values()), 0.13)):
            ax.annotate(k, (passes[-1], 10 ** y), xytext=(6, 0), textcoords="offset points", va="center",
                        color=INK, fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xlabel("pass")
        title = WORLD_LABEL[w].replace(" (", "\n(") if len(WORLD_LABEL[w]) > 30 else WORLD_LABEL[w]
        ax.set_title(f"{title}\n{rsum['worlds'][w]['n_draws']} hidden truths: median, IQR", fontsize=12.5)
        ax.set_xticks(range(0, len(passes) + 1, 5))
        ax.set_xlim(0.5, len(passes) + 2.0)
    axes[0, 0].set_ylabel("next-pass removal-map RMSE [µm] (log)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Next-pass prediction error per pass (lower is better)",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=11, bbox_to_anchor=(0.5, 0.0))
    _save(fig, figdir / "fig3_rmse_per_pass.png")


# --------------------------------------------------------------------------- fig 4


def fig_params(res: Path, figdir: Path) -> None:
    main = json.loads((res / "main_run.json").read_text())
    runs = {main["world"]: _read_csv(res / "main_run.csv")}
    for w in main["worlds"]:
        if w != main["world"]:
            runs[w] = _read_csv(res / f"main_run_{w}.csv")
    order = [w for w in ("matched", "realistic") if w in runs]
    specs = [("k_pad", "true_k_pad", "k_pad [N/mm³]", "D_k_pad", 1.0, "Stiffness k_pad"),
             ("K", "true_K", "K [10⁻⁵ mm²/N]", "D_K", 1e5, "Effectiveness K (at pass start)"),
             ("lam", "true_lam", "λ [10⁻⁴ 1/mm³]", "D_lam", 1e4, "Wear rate λ (current)")]
    fig, axes = plt.subplots(len(order), 3, figsize=(13, 3.7 * len(order) + 0.6), sharex=True, squeeze=False)
    for i, w in enumerate(order):
        rows = runs[w]
        passes = np.array([r["pass"] for r in rows])
        for j, (name, tkey, ylabel, dkey, s, title) in enumerate(specs):
            ax = axes[i, j]
            med = np.array([r[f"C_{name}_median"] for r in rows])
            lo = np.array([r[f"C_{name}_lo"] for r in rows])
            hi = np.array([r[f"C_{name}_hi"] for r in rows])
            true = np.array([r[tkey] for r in rows])
            ax.fill_between(passes, s * lo, s * hi, color=BLUE, alpha=0.22, lw=0, label="C: 90% interval")
            ax.plot(passes, s * med, color=BLUE, lw=2.2, label="C: posterior median")
            ax.plot(passes, s * true, color=INK, lw=1.4, marker="o", ms=3.5, label="true value")
            dv = s * np.array([r[dkey] for r in rows])
            ax.plot(passes, dv, color=GREEN, lw=1.6, ls="--", label="D: latest refit")
            if name == "k_pad":
                vals = np.concatenate([true, med[1:], dv[1:] / s])
                ax.set_ylim(s * vals.min() * 0.97, s * vals.max() * 1.03)
            else:
                ax.set_ylim(0, s * max(true.max(), hi[1:].max(), np.nanmax(dv[1:] / s)) * 1.15)
            ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(title)
            if i == len(order) - 1:
                ax.set_xlabel("pass")
                ax.set_xticks(range(0, len(passes) + 1, 5))
        axes[i, 0].text(0.02, 0.04, WORLD_LABEL[w], transform=axes[i, 0].transAxes, fontsize=11.5,
                        fontweight="bold", color=INK2, va="bottom")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.suptitle("Main run: tracked parameters. Matched world: the truth is recovered. Realistic world: "
                 "C's parameters\nbecome effective values of its simplified model (foam pad, force error, "
                 "two-stage wear)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.0))
    _save(fig, figdir / "fig4_parameter_tracking.png")


# --------------------------------------------------------------------------- fig 5


def fig_change(res: Path, figdir: Path) -> None:
    main_info = json.loads((res / "main_run.json").read_text())
    main = _read_csv(res / "main_run.csv")
    ch = json.loads((res / "abrasive_change.json").read_text())
    n_passes = len(main)
    true_cross = int(main[0]["cross_true"])
    cross_A = main[0]["cross_A"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8))
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
    ax.plot(p + 0.15, [r["cross_D"] for r in rr], "s", color=GREEN, ms=6, label="D: refit + fitted wear rate")
    ax.axhline(true_cross, color=INK, lw=1.6, label=f"true crossing (pass {true_cross})")
    ax.axhline(cross_A, color=GRAY, lw=2.0, ls="--", label=f"A: nominal model (pass {cross_A:g})")
    ax.set_xlabel("prediction made after pass")
    ax.set_ylabel(f"predicted pass where K < {100 * ch['threshold_fraction']:g}% of K0")
    ax.set_title(f"Main run, {main_info['world']} world")
    ax.set_xticks(p)
    ax.set_xlim(0.5, n_show + 0.5)
    ax.set_ylim(0, max(hi.max(), cross_A, true_cross, max(r["cross_D"] for r in rr)) + 3)
    ax.legend(loc="upper right", fontsize=10)
    ymax = 0.0
    for w, ls in (("matched", "--"), ("realistic", "-")):
        if w not in ch["worlds"]:
            continue
        lead = ch["worlds"][w]["by_lead"]
        rows = lead["rows"]
        if not rows:
            continue
        leads = [r["lead_passes"] for r in rows]
        for key, c, lab in (("mean_abs_err_A", GRAY, "A"), ("mean_abs_err_D", GREEN, "D"),
                            ("mean_abs_err_C", BLUE, "C")):
            v = [r[key] for r in rows]
            ymax = max(ymax, max(v))
            bx.plot(leads, v, color=c, ls=ls, lw=2.4 if lab == "C" else 1.8, marker="o" if lab == "C" else None,
                    ms=5, label=f"{lab}, {w} ({lead['population']} truths)")
        bx.set_xticks(leads)
    bx.set_ylim(0, ymax * 1.25 + 0.1)
    bx.set_xlabel("passes before the true crossing")
    bx.set_ylabel("mean |predicted − true| [passes]")
    bx.set_title("Error vs. lead time (fixed populations)")
    bx.legend(loc="upper left", fontsize=9.5, ncol=2)
    fig.suptitle("When to change the abrasive: predicted vs. true crossing", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, figdir / "fig5_abrasive_change.png")


# --------------------------------------------------------------------------- fig 6


def fig_ablation_noise(res: Path, figdir: Path) -> None:
    ab = json.loads((res / "ablation_summary.json").read_text())
    nz = json.loads((res / "noise_sensitivity.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
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
        _log_ticks(ax, [0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200])
        ax.set_xlabel("pass")
        ax.set_ylabel("90% interval width [%] (log)")
        ax.set_title(f"{title} ({ab['world']})")
        ax.set_xticks(range(0, len(p) + 1, 5))
    axes[0].legend(loc="upper right", fontsize=10.5)
    lv = list(nz["levels"].values())
    sig = np.array([v["noise_um"] for v in lv])
    med = np.array([v["final_rmse_C_um"]["median"] for v in lv])
    lo = np.array([v["final_rmse_C_um"]["q25"] for v in lv])
    hi = np.array([v["final_rmse_C_um"]["q75"] for v in lv])
    orc = np.array([v["final_rmse_oracle_um"]["median"] for v in lv])
    axes[2].errorbar(sig, med, yerr=[med - lo, hi - med], color=BLUE, marker="o", ms=7, lw=2.2,
                     capsize=4, label="C (median, IQR)")
    axes[2].plot(sig, orc, color=INK2, lw=1.0, ls="--", label="oracle (median)")
    axes[2].set_xscale("log")
    axes[2].set_xlim(0.8, 13)
    axes[2].set_xticks(sig)
    axes[2].set_xticklabels([f"{v['noise_um']:g}\n{100 * v['predictive_90']['passes_2_on']['fraction']:.0f}%"
                             for v in lv])
    axes[2].set_xlabel("scan noise σ [µm]  /  predictive coverage")
    axes[2].set_ylabel("final-pass RMSE [µm]")
    axes[2].set_ylim(0, 1.4 * max(hi.max(), orc.max()))
    axes[2].set_title(f"Scan noise ({nz['world']}, {nz['n_draws']} truths)")
    axes[2].legend(loc="upper left", fontsize=10.5)
    fig.suptitle(f"Constant-force ablation ({ab['n_draws']} truths) and scan-noise sensitivity",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, figdir / "fig6_ablation_and_noise.png")


# --------------------------------------------------------------------------- fig 7


def fig_mismatch(res: Path, figdir: Path) -> None:
    br = json.loads((res / "world_breakdown.json").read_text())
    rsum = json.loads((res / "robustness_summary.json").read_text())
    tun = json.loads((res / "tuning.json").read_text()) if (res / "tuning.json").exists() else None
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    worlds = list(br["worlds"])
    yy = np.arange(len(worlds))[::-1]
    for k, off in (("A", 0.27), ("D", 0.0), ("C", -0.27)):
        lab, c = EST[k]
        v = [br["worlds"][w]["final_rmse_um"][k]["median"] for w in worlds]
        ax.barh(yy + off, v, height=0.26, color=c, label=lab)
    orc = [br["worlds"][w]["final_rmse_um"]["oracle"]["median"] for w in worlds]
    ax.scatter(orc, yy - 0.27, marker="|", s=260, color=INK, zorder=5, label="oracle")
    ax.set_xscale("log")
    ax.set_yticks(yy)
    ax.set_yticklabels([WORLD_LABEL.get(w, w).split(" (")[0] for w in worlds])
    ax.set_xlabel("final-pass map RMSE [µm], median (log)")
    ax.set_title(f"Breakdown ({br['n_draws']} paired truths)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=4, fontsize=10)
    ax.grid(axis="y", visible=False)
    bx = axes[1]
    for w, c, ls in (("matched", INK2, "-"), ("realistic", BLUE, "-"), ("realistic_b", BLUE, "--")):
        if w not in rsum["worlds"]:
            continue
        pp = rsum["worlds"][w]["predictive_90"]["per_pass_fraction"]
        bx.plot(np.arange(2, len(pp) + 1), 100 * np.array(pp[1:]), color=c, lw=2.0, ls=ls, marker="o", ms=3.5,
                label=f"{w}: {100 * rsum['worlds'][w]['predictive_90']['passes_2_on']['fraction']:.0f}% overall")
    bx.axhline(90, color=INK, lw=1.0, ls=":")
    bx.set_ylim(0, 105)
    bx.set_xlabel("pass")
    bx.set_ylabel("truth inside C's 90% interval [%]")
    bx.set_title("Next-pass mean removal: calibration")
    bx.legend(loc="lower right", fontsize=10.5)
    cx = axes[2]
    if tun:
        vals = sorted(tun["by_value"], key=lambda v: v["rate_drift"])
        d = [v["rate_drift"] for v in vals]
        for w, c in (("matched", INK2), ("realistic", BLUE)):
            if w in tun["worlds"]:
                cx.plot(d, [100 * v["worlds"][w]["coverage"] for v in vals], color=c, lw=2.0, marker="o", ms=5,
                        label=f"coverage, {w}")
        cx.axhline(90, color=INK, lw=1.0, ls=":")
        cx.set_ylim(40, 100)
        cx.set_xlabel("wear-rate drift of the tracker (log λ s.d. per pass)")
        cx.set_ylabel("held-out predictive coverage [%]")
        cx2 = cx.twinx()
        cx2.plot(d, [v["mean_interval_score"] for v in vals], color=ORANGE, lw=1.8, ls="--",
                 label="interval score (lower = better)")
        cx2.axvline(tun["chosen_rate_drift"], color=ORANGE, lw=1.0, ls=":")
        cx2.set_ylabel("mean relative interval score", color=ORANGE)
        cx2.grid(False)
        cx.set_title(f"Tuning (held-out draws {tun['draws'][0]}–{tun['draws'][1] - 1})")
        h1, l1 = cx.get_legend_handles_labels()
        h2, l2 = cx2.get_legend_handles_labels()
        cx.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9.5, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=0.9)
    fig.suptitle("What the unmodelled effects cost, and how calibrated C stays", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, figdir / "fig7_mismatch_and_calibration.png")


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
    fig_mismatch(res, figdir)
