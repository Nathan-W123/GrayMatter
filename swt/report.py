"""Render README.md and SUMMARY.md from templates whose numbers are lookups into results/.

A placeholder ``{{dotted.path:format}}`` is replaced by the value at that path in
results/summary.json (roots: contact_sweep, model_info, tuning, main_run,
robustness, ablation, noise_sensitivity, abrasive_change, world_breakdown), under
the root ``info`` in results/run_info.json, and under the roots ``ext`` and
``ext_info`` in results/extended/summary.json and run_info.json (the extended study). ``format`` is a Python format spec; ``:d`` rounds to an
integer and ``:pctN`` multiplies by 100 with N decimals. Rendering fails if any
placeholder cannot be resolved, so no number in the documents can be stale or typed by hand.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PLACEHOLDER = re.compile(r"\{\{([A-Za-z0-9_.]+)(?::([^}]*))?\}\}")


def load_values(results: str | Path) -> dict:
    res = Path(results)
    data = json.loads((res / "summary.json").read_text())
    data["info"] = json.loads((res / "run_info.json").read_text())
    ext = res / "extended"
    if (ext / "summary.json").exists():
        data["ext"] = json.loads((ext / "summary.json").read_text())
        data["ext_info"] = json.loads((ext / "run_info.json").read_text())
    return data


def lookup(data: dict, path: str):
    cur = data
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def render(template: str, data: dict) -> tuple[str, int]:
    count = 0

    def sub(m):
        nonlocal count
        value, spec = lookup(data, m.group(1)), m.group(2) or ""
        count += 1
        if spec == "d":
            return str(int(round(value)))
        if spec.startswith("pct"):
            return f"{100 * value:.{int(spec[3:] or 0)}f}"
        return format(value, spec)

    text = PLACEHOLDER.sub(sub, template)
    left = re.findall(r"\{\{[^}]*\}\}", text)
    if left:
        raise ValueError(f"unrendered placeholders: {left}")
    return text, count


def render_reports(results: str | Path = "results", docs: str | Path = "docs", out: str | Path = ".") -> dict:
    data = load_values(results)
    written = {}
    for name in ("README", "SUMMARY"):
        text, n = render((Path(docs) / f"{name}.template.md").read_text(), data)
        target = Path(out) / f"{name}.md"
        target.write_text(text)
        written[str(target)] = n
    return written
