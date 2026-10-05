from pathlib import Path

import pytest

from swt.report import load_values, render

ROOT = Path(__file__).resolve().parent.parent


def test_render_formats_and_rejects_unknown_keys():
    data = {"a": {"b": [0.12345, 7.0]}, "info": {}}
    text, n = render("x {{a.b.0:.2f}} y {{a.b.1:d}} z {{a.b.0:pct1}}%", data)
    assert text == "x 0.12 y 7 z 12.3%" and n == 3
    with pytest.raises(KeyError):
        render("{{a.missing}}", data)


@pytest.mark.skipif(not (ROOT / "results" / "summary.json").exists(), reason="run `python -m swt run-all` first")
def test_shipped_documents_match_the_results():
    """README.md and SUMMARY.md are exactly what the templates render from results/."""
    data = load_values(ROOT / "results")
    for name in ("README", "SUMMARY"):
        text, n = render((ROOT / "docs" / f"{name}.template.md").read_text(), data)
        assert n > 0
        assert text == (ROOT / f"{name}.md").read_text(), f"{name}.md is stale: run `python -m swt report`"
