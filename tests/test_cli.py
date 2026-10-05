import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_cli_help():
    out = subprocess.run([sys.executable, "-m", "swt", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0
    assert "run-all" in out.stdout


def test_cli_contact_sweep_small(tmp_path, small_cfg):
    cfg_path = tmp_path / "small.yaml"
    cfg_path.write_text(yaml.safe_dump(small_cfg))
    res = tmp_path / "res"
    out = subprocess.run([sys.executable, "-m", "swt", "contact-sweep", "--config", str(cfg_path),
                          "--results", str(res)], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    summary = json.loads((res / "contact_sweep.json").read_text())
    assert summary["k_low"] == small_cfg["priors"]["k_low"]
    assert (res / "contact_sweep.csv").exists()
