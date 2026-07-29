"""
Tables 6 & 7 of the paper.

Reproduces two derived tables from the predictions written by the core
experiments (no model re-training):

  Table 6 — per-grade total QWK (model vs. human ceiling), test set, with 95% CIs.
            Model per-grade QWK is computed here from e11 predictions; the human
            ceiling per-grade QWK+CI is read from the e9 output.

  Table 7 — decomposition of the 0.625 -> 0.581 representative-score gap into a
            label component (legacy vs. corrected) and a protocol component,
            holding the discretization fixed at standard 0.5-rounding. Uses the
            e7l (test-selected) and e11 (val-selected) predictions.

Inputs (produced by e09/e11/e7l — run those first, or use run_all.py):
  outputs/e11_honest_eval/e11_predictions.csv
  outputs/e7l_original/e7l_predictions.csv
  outputs/e9_human_ceiling/e9_by_grade.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import discretize, qwk_disc, boot_indices, boot_qwk_ci

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, "outputs", "tables_6_7")
os.makedirs(OUT, exist_ok=True)

BOOT_N, BOOT_SEED = 1000, 11          # same bootstrap settings as e11
GRADE_ORDER = ["초등_6학년", "중등_1학년", "중등_2학년",
               "고등_1학년", "고등_2학년", "고등_3학년"]
GRADE_EN = {"초등_6학년": "Elementary 6", "중등_1학년": "Middle 1",
            "중등_2학년": "Middle 2", "고등_1학년": "High 1",
            "고등_2학년": "High 2", "고등_3학년": "High 3"}


def qwk(y, p):
    return qwk_disc(discretize(y), discretize(p))


def ci(y, p):
    idx = boot_indices(len(y), BOOT_N, BOOT_SEED)
    return boot_qwk_ci(np.asarray(y), np.asarray(p), idx)


def main():
    e11 = pd.read_csv(os.path.join(BASE_DIR, "outputs", "e11_honest_eval",
                                   "e11_predictions.csv"))
    e7l = pd.read_csv(os.path.join(BASE_DIR, "outputs", "e7l_original",
                                   "e7l_predictions.csv"))
    ceil = pd.read_csv(os.path.join(BASE_DIR, "outputs", "e9_human_ceiling",
                                    "e9_by_grade.csv"))
    m = e11.merge(e7l[["essay_id", "total_pred"]], on="essay_id")

    # ---- Table 6: per-grade model QWK + ceiling ----
    ceil_t = ceil[(ceil["scope"] == "test") & (ceil["label_definition"] == "corrected")
                  & (ceil["category"] == "total")].set_index("grade")
    t6 = []
    for g in GRADE_ORDER:
        sub = m[m["grade_level"] == g]
        y, p = sub["total_score"].values, sub["pred_opt"].values
        mq = qwk(y, p)
        mlo, mhi = ci(y, p)
        cr = ceil_t.loc[g]
        t6.append({"grade": GRADE_EN[g], "n": len(sub),
                   "model_qwk": round(mq, 3),
                   "model_ci": f"[{mlo:.3f}, {mhi:.3f}]",
                   "ceiling_qwk": round(float(cr["qwk"]), 3),
                   "ceiling_ci": f"[{float(cr['ci_low']):.3f}, {float(cr['ci_high']):.3f}]"})
    t6 = pd.DataFrame(t6)
    t6.to_csv(os.path.join(OUT, "table6.csv"), index=False)

    # ---- Table 7: label x protocol (std discretization fixed) ----
    corr, leg = m["total_score"].values, m["total_score_legacy"].values
    rows = [("Original (train+val, test-selected)", m["total_pred"].values),
            ("This work (train, val-selected)", m["pred_std"].values)]
    t7 = []
    for name, pred in rows:
        ql, qc = qwk(leg, pred), qwk(corr, pred)
        cl, cc = ci(leg, pred), ci(corr, pred)
        t7.append({"predictions": name,
                   "legacy_qwk": round(ql, 3), "legacy_ci": f"[{cl[0]:.3f}, {cl[1]:.3f}]",
                   "corrected_qwk": round(qc, 3), "corrected_ci": f"[{cc[0]:.3f}, {cc[1]:.3f}]",
                   "label_delta": round(qc - ql, 4)})
    t7 = pd.DataFrame(t7)
    t7.to_csv(os.path.join(OUT, "table7.csv"), index=False)

    # ---- print ----
    print("\n=== Table 6: per-grade total QWK (test) ===")
    print(t6.to_string(index=False))
    print("\n=== Table 7: 0.625->0.581 gap decomposition (std QWK) ===")
    print(t7.to_string(index=False))
    prot = t7.iloc[1]["corrected_qwk"] - t7.iloc[0]["corrected_qwk"]
    print(f"\n  label effect (protocol fixed): "
          f"{t7.iloc[0]['label_delta']:+.4f} / {t7.iloc[1]['label_delta']:+.4f}")
    print(f"  protocol effect (label fixed, corrected col): {prot:+.4f}")
    print(f"\n  saved -> {OUT}")


if __name__ == "__main__":
    main()
