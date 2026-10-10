from swt import decisions


def test_decision_rules_score_change_passes():
    """Two draws crossing at pass 4. C's median rule changes on time, its 25% rule one pass
    early, its 5% rule two passes early; D's forecast K/K0 (0.55 after pass 3, 0.45 after
    pass 4) makes it one pass late without a margin and on time with a 20% margin."""
    rows = []
    d_ratio = [0.9, 0.7, 0.55, 0.45, 0.4, 0.35]
    for d in range(2):
        for p in range(1, 7):
            rows.append({"world": "w", "draw": d, "pass": p, "cross_true": 4, "cross_A": 6, "force_N": 20.0,
                         "scan_mean_um": 10.0 * 0.85 ** (p - 1), "D_next_ratio": d_ratio[p - 1],
                         "cross_C_median": 4 if p >= 3 else 9, "cross_C_risk": 3 if p >= 2 else 9,
                         "cross_C_lo": 2})
    dr = decisions.decision_rules(rows, 6, 0.5, {"r3": {"D": 0.2, "R": 0.0}}, cost_ratios=(1, 3, 19))
    rules = dr["rules"]
    assert rules["C_r1"]["exact"] == 2 and rules["C_r1"]["mean_abs_err"] == 0.0
    assert rules["C_r3"]["early"] == 2 and rules["C_r3"]["cost_r3"] == 1.0
    assert rules["C_r19"]["mean_passes_early"] == 2.0
    assert rules["D"]["late"] == 2 and rules["D"]["cost_r3"] == 3.0
    assert rules["D_r3"]["exact"] == 2
    assert rules["A"]["mean_abs_err"] == 2.0
    assert rules["R"]["n"] == 2
    assert dr["by_cost_ratio"]["r3"]["C_minus_D_r3"]["mean"] == 1.0
    assert dr["by_cost_ratio"]["r3"]["C_minus_D"]["mean"] == -2.0
    margins = decisions.tune_margins(rows, 6, 0.5, (3,))
    assert margins["r3"]["D"] == 0.11 and margins["r3"]["D_held_out_cost"] == 0.0
    # the same draw ids in two worlds are two episodes, not one
    rows2 = [dict(r, world="matched") for r in rows] + [dict(r, world="realistic") for r in rows]
    m2 = decisions.tune_margins(rows2, 6, 0.5, (3,))
    assert m2["r3"]["n_draws"] == 4 and m2["r3"]["D"] == 0.11


def test_model_free_rule_and_costs():
    """R needs two scans for a slope; without wear its forecast K/K0 is exactly 1. A rule fires
    at the first forecast below threshold * (1 + margin), for the next pass."""
    import numpy as np
    rows = [{"scan_mean_um": 4.0, "force_N": 20.0 + 20.0 * (p % 2)} for p in range(1, 6)]
    rows = [dict(r, scan_mean_um=r["force_N"] / 10.0) for r in rows]      # removal proportional to force
    ratios = decisions._model_free_ratios(rows)
    assert np.isnan(ratios[0]) and np.allclose(ratios[1:], 1.0)
    assert decisions._change_from_ratio([1, 2, 3], [0.9, 0.6, 0.4], 0.5, 0.0) == 4
    assert decisions._change_from_ratio([1, 2, 3], [0.9, 0.6, 0.4], 0.5, 0.25) == 3
    assert decisions._change_from_ratio([1, 2, 3], [0.9, 0.8, 0.7], 0.5, 0.0) is None
    assert list(decisions._cost(np.array([-2.0, 0.0, 3.0]), 19.0)) == [2.0, 0.0, 57.0]
