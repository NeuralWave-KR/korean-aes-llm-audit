#!/usr/bin/env python3
"""
E9b: per-grade score variance vs QWK (paper §5.4).

QWK is sensitive to the variance of the scores being compared, so grades whose
expert scores are tightly clustered yield unstable (near-zero, occasionally
negative) QWK. This stage quantifies that property: for each grade it computes the
SD of the expert total score on the test set and correlates it (Spearman) with the
per-grade total QWK of the model and of the human ceiling.

The per-grade model/ceiling QWK are read from the regenerated Table 6
(outputs/tables_6_7/table6.csv) — nothing is hardcoded — so the correlation
reproduces from the regenerated numbers. Reads only cached/regenerated data;
requires tables_6_7 to have run first (self-skips otherwise).

Reported (paper §5.4):
  SD vs human-ceiling QWK  Spearman rho ~ 0.94
  SD vs model QWK          Spearman rho ~ 0.71
"""
import json
import os

import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPLIT_TEST = os.path.join(ROOT, "data", "splits", "test.csv")
TABLE6 = os.path.join(ROOT, "outputs", "tables_6_7", "table6.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "e9b_grade_variance")

# test.csv grade_level (Korean) -> Table 6 grade label (English)
GMAP = {"초등_6학년": "Elementary 6", "중등_1학년": "Middle 1", "중등_2학년": "Middle 2",
        "고등_1학년": "High 1", "고등_2학년": "High 2", "고등_3학년": "High 3"}


def main():
    if not os.path.exists(TABLE6):
        print("E9b: outputs/tables_6_7/table6.csv not found — run tables_6_7 first; skipping.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    test = pd.read_csv(SPLIT_TEST)
    test["grade_en"] = test["grade_level"].map(GMAP)
    sd = test.groupby("grade_en")["total_score"].std(ddof=1)   # per-grade SD, indexed by English label

    t6 = pd.read_csv(TABLE6)
    t6["score_sd"] = t6["grade"].map(sd)
    t6["score_var"] = t6["score_sd"] ** 2

    s = t6["score_sd"].values
    m = t6["model_qwk"].values
    c = t6["ceiling_qwk"].values
    corr = {
        "sd_vs_model_spearman": round(float(spearmanr(s, m)[0]), 3),
        "sd_vs_model_pearson": round(float(pearsonr(s, m)[0]), 3),
        "sd_vs_ceiling_spearman": round(float(spearmanr(s, c)[0]), 3),
        "sd_vs_ceiling_pearson": round(float(pearsonr(s, c)[0]), 3),
    }

    out_csv = os.path.join(OUT_DIR, "e9b_grade_variance.csv")
    t6[["grade", "n", "score_sd", "score_var", "model_qwk", "ceiling_qwk"]].to_csv(out_csv, index=False)
    json.dump(corr, open(os.path.join(OUT_DIR, "e9b_correlations.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(t6[["grade", "n", "score_sd", "model_qwk", "ceiling_qwk"]].round(3).to_string(index=False))
    print("\nRank/linear correlation of per-grade score SD with per-grade QWK:")
    for k, v in corr.items():
        print(f"  {k}: {v}")
    print(f"\n-> {out_csv}")


if __name__ == "__main__":
    main()
