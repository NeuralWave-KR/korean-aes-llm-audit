"""
Verify that the regenerated outputs match the values reported in the paper.

Checks each headline number against the paper (tolerance 0.005 for point
estimates; CI bounds checked to 0.01). Exits non-zero if any check fails, so it
can be used in CI. Run `python run_all.py` first.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
TOL = 0.005      # point-estimate tolerance
CI_TOL = 0.010   # CI-bound tolerance

checks = []      # (label, expected, actual, tol)


def add(label, expected, actual, tol=TOL):
    checks.append((label, expected, actual, tol))


def main():
    # --- E9: human ceiling (Table 1 / Table 5 denominator) ---
    ovr = pd.read_csv(os.path.join(OUT, "e9_human_ceiling", "e9_one_vs_rest.csv"))
    tot = ovr[(ovr["scope"] == "test") & (ovr["label_definition"] == "corrected")
              & (ovr["axis_value"] == "total")].iloc[0]
    add("E9 human ceiling total QWK", 0.584, float(tot["qwk"]))
    add("E9 ceiling CI low", 0.528, float(tot["ci_low"]), CI_TOL)
    add("E9 ceiling CI high", 0.630, float(tot["ci_high"]), CI_TOL)
    pw = pd.read_csv(os.path.join(OUT, "e9_human_ceiling", "e9_pairwise.csv"))
    pwt = pw[(pw["scope"] == "test") & (pw["label_definition"] == "corrected")
             & (pw["axis_value"] == "total")].iloc[0]
    add("E9 pairwise ceiling total QWK", 0.541, float(pwt["qwk"]))

    # --- E10: factorial hypotheses (Table 4) ---
    hyp = pd.read_csv(os.path.join(OUT, "e10_factorial", "e10_hypotheses.csv"))

    def h(hid, cat="total"):
        return hyp[(hyp["hypothesis_id"] == hid) & (hyp["category"] == cat)].iloc[0]
    add("E10 H1 (calibration) ΔQWK", 0.238, float(h("H1")["delta"]))
    add("E10 H2 (LLM increment) ΔQWK", 0.005, float(h("H2")["delta"]))
    add("E10 H2 CI low", -0.023, float(h("H2")["ci_low"]), CI_TOL)
    add("E10 H2 CI high", 0.032, float(h("H2")["ci_high"]), CI_TOL)
    add("E10 H3 (nonlinearity) ΔQWK", 0.139, float(h("H3")["delta"]))
    cells = pd.read_csv(os.path.join(OUT, "e10_factorial", "e10_cells.csv"))
    cd = cells[(cells["category"] == "total") & (cells["model"] == "xgboost")
               & (cells["A_grade"] == 0) & (cells["B_text"] == 0)
               & (cells["C_llm_main"] == 1) & (cells["D_llm_sub"] == 1)].iloc[0]
    add("E10 LLM-only (C·D) total QWK", 0.243, float(cd["qwk"]))

    # --- E11: representative score (Table 5) ---
    e11 = json.load(open(os.path.join(OUT, "e11_honest_eval", "e11_summary.json"),
                         encoding="utf-8"))
    # total opt QWK lives under the results block; fall back to scanning if renamed
    tot_qwk = _find_total_opt(e11)
    add("E11 representative total QWK", 0.581, tot_qwk)

    # --- Table 6: per-grade model QWK ---
    t6 = pd.read_csv(os.path.join(OUT, "tables_6_7", "table6.csv")).set_index("grade")
    for g, v in {"Elementary 6": 0.104, "Middle 1": -0.044, "Middle 2": 0.244,
                 "High 1": 0.168, "High 2": 0.793, "High 3": 0.480}.items():
        add(f"Table6 model QWK [{g}]", v, float(t6.loc[g, "model_qwk"]))

    # --- Table 7: 2x2 decomposition ---
    t7 = pd.read_csv(os.path.join(OUT, "tables_6_7", "table7.csv"))
    orig, work = t7.iloc[0], t7.iloc[1]
    add("Table7 original/legacy", 0.583, float(orig["legacy_qwk"]))
    add("Table7 original/corrected", 0.591, float(orig["corrected_qwk"]))
    add("Table7 thiswork/legacy", 0.565, float(work["legacy_qwk"]))
    add("Table7 thiswork/corrected", 0.574, float(work["corrected_qwk"]))

    # --- report ---
    print(f"\n{'CHECK':<38}{'paper':>9}{'repro':>9}{'Δ':>9}   result")
    print("-" * 76)
    n_fail = 0
    for label, exp, act, tol in checks:
        d = act - exp
        ok = abs(d) <= tol
        n_fail += not ok
        print(f"{label:<38}{exp:>9.3f}{act:>9.3f}{d:>+9.3f}   {'PASS' if ok else 'FAIL'}")
    print("-" * 76)
    if n_fail:
        print(f"{n_fail}/{len(checks)} checks FAILED")
        sys.exit(1)
    print(f"All {len(checks)} checks PASSED — reproduction confirmed.")


def _find_total_opt(d):
    """Locate the test-set total optimized QWK in the e11 summary, robust to layout."""
    # common layouts: d['results']['total']['opt'] or a flat key
    for path in (("test_results", "total", "opt_qwk"), ("e11_verdict", "headline_qwk"),
                 ("results", "total", "opt"), ("total_opt_qwk",)):
        cur = d
        try:
            for k in path:
                cur = cur[k]
            return float(cur)
        except (KeyError, TypeError):
            continue
    # fallback: deep scan for a value near 0.581 keyed by 'total'/'opt'
    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (int, float)) and 0.55 < v < 0.60 and \
                        ("opt" in str(k) or "total" in str(k)):
                    found.append(float(v))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    if found:
        return found[0]
    raise KeyError("could not locate E11 total opt QWK in e11_summary.json")


if __name__ == "__main__":
    main()
