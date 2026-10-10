"""Summary statistics, paired tests and result files shared by the experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

UM = 1e3  # mm -> micrometres


# --------------------------------------------------------------------------- summaries


def rmse_um(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)) * UM)


def q(x, p) -> float:
    return float(np.percentile(np.asarray(x, dtype=float), p))


def iqr_summary(x) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"median": None, "q25": None, "q75": None, "n": 0, "min": None, "max": None}
    return {"median": q(x, 50), "q25": q(x, 25), "q75": q(x, 75), "n": int(x.size),
            "min": float(x.min()), "max": float(x.max())}


def gmean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x)))) if x.size and np.all(x > 0) else float("nan")


def mean_or_none(x) -> float | None:
    return float(np.mean(x)) if len(x) else None


def bootstrap_ci(x, stat=np.median, n_boot: int = 2000, level: float = 0.9, seed: int = 10) -> list[float]:
    """Percentile bootstrap interval of a statistic over draws (resampling draws)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[rng.integers(0, x.size, x.size)]) for _ in range(n_boot)])
    a = (1 - level) / 2
    return [float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))]


# --------------------------------------------------------------------------- paired tests


def paired_permutation_p(diff, n_perm: int = 20000, seed: int = 11) -> float:
    """Two-sided p-value for a zero mean of paired differences, by random sign flips
    (exact under the null hypothesis that each difference is symmetric about zero)."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or not np.any(d):
        return 1.0
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n_perm, 5000):
        k = min(5000, n_perm - start)
        signs = rng.integers(0, 2, size=(k, d.size)) * 2 - 1
        hits += int(np.sum(np.abs(signs @ d) / d.size >= obs - 1e-12))
    return float((1 + hits) / (n_perm + 1))


def holm_adjust(pvals: dict) -> dict:
    """Holm step-down adjusted p-values (family-wise error control) for a dict of p-values."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


# --------------------------------------------------------------------------- result files


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")   # nothing to report (e.g. a tiny configuration); keep the file
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
