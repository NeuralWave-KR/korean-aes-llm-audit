#!/usr/bin/env python3
"""
E12 (supplementary): latest-model pilot — does the H2 finding hold with a newer LLM?

Offline analysis. A balanced sub-sample of the test set is re-scored with a recent
model (GPT-5.6) using the *same* 0-3 prompt/schema as the GPT-4o rater; this stage
then compares that model's agreement with expert scores against a trivial length
feature (log word count) and against GPT-4o — on the identical sub-sample.

Reads only cached data (no network / no API). Generating the pilot ratings is a
separate, key-gated step; see src/preprocessing/pilot_rater.py and the README
section "Supplementary: latest-model pilot". If the cached pilot ratings are not
present, this stage skips (it is supplementary to Tables 1-7).

Reported (paper §5.4):
  latest-model total vs expert total  Pearson 0.18
  < length 0.48 ; not above GPT-4o 0.20  -> surface features still dominate.

Caveats (documented in the paper): balanced sub-sample, so absolute QWK is NOT
comparable to the headline numbers (only the within-sample ordering is); a single
recent model; and GPT-5.6 does not support temperature=0 (generated at temp=1).
The cached ratings are the fixed reference — re-generation may differ slightly.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPLIT_TEST = os.path.join(ROOT, "data", "splits", "test.csv")
OLD_GPT = os.path.join(ROOT, "data", "gpt4o_ratings", "test_gpt4o_ratings.csv")
PILOT_DIR = os.path.join(ROOT, "data", "pilot_ratings")
PILOT_RATINGS = os.path.join(PILOT_DIR, "pilot_ratings.csv")
PILOT_SAMPLE = os.path.join(PILOT_DIR, "pilot_sample.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "e12_pilot")


def discretize(x):
    # paper convention: clip(round(x*2)/2, 0, 3) * 2 -> integers 0..6 (7 categories)
    return np.clip(np.round(np.asarray(x, float) * 2) / 2, 0, 3) * 2


def qwk(a, b):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(discretize(a).astype(int), discretize(b).astype(int),
                                   weights="quadratic", labels=list(range(7))))


def partial_r(x, y, z):
    """Partial correlation of x and y controlling for z."""
    from scipy.stats import pearsonr
    rxy = pearsonr(x, y)[0]
    rxz = pearsonr(x, z)[0]
    ryz = pearsonr(y, z)[0]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def main():
    if not (os.path.exists(PILOT_RATINGS) and os.path.exists(PILOT_SAMPLE)):
        print("E12 pilot: cached pilot ratings not present (data/pilot_ratings/) — "
              "supplementary stage, skipping. See README 'Supplementary: latest-model pilot'.")
        return

    from scipy.stats import pearsonr
    os.makedirs(OUT_DIR, exist_ok=True)

    test = pd.read_csv(SPLIT_TEST)[["essay_id", "essay_txt", "total_score"]]
    old = (pd.read_csv(OLD_GPT)[["essay_id", "llm_total_score"]]
           .rename(columns={"llm_total_score": "old_gpt_total"}))
    r = pd.read_csv(PILOT_RATINGS)
    new = (r[["essay_id", "llm_total_score", "model"]]
           .rename(columns={"llm_total_score": "new_gpt_total"})
           .dropna(subset=["new_gpt_total"]))
    sample = pd.read_csv(PILOT_SAMPLE)[["essay_id", "total_score"]]

    d = (sample.merge(new, on="essay_id")
         .merge(old, on="essay_id", how="left")
         .merge(test[["essay_id", "essay_txt"]], on="essay_id")
         .dropna(subset=["new_gpt_total", "old_gpt_total"]))
    d["logwc"] = np.log1p(d["essay_txt"].fillna("").str.split().str.len())

    res = {
        "model": str(new["model"].iloc[0]),
        "n_analyzed": int(len(d)),
        "new_gpt_vs_human_pearson": float(pearsonr(d["new_gpt_total"], d["total_score"])[0]),
        "old_gpt_vs_human_pearson": float(pearsonr(d["old_gpt_total"], d["total_score"])[0]),
        "logwc_vs_human_pearson": float(pearsonr(d["logwc"], d["total_score"])[0]),
        "new_gpt_vs_old_gpt_pearson": float(pearsonr(d["new_gpt_total"], d["old_gpt_total"])[0]),
        "new_gpt_vs_human_qwk": qwk(d["total_score"], d["new_gpt_total"]),
        "old_gpt_vs_human_qwk": qwk(d["total_score"], d["old_gpt_total"]),
        "new_gpt_partial_vs_human_given_logwc": float(partial_r(
            d["new_gpt_total"].values, d["total_score"].values, d["logwc"].values)),
    }

    out = os.path.join(OUT_DIR, "e12_pilot_analysis.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"E12 pilot ({res['model']}, n={res['n_analyzed']}):")
    print(f"  latest-model vs expert   Pearson {res['new_gpt_vs_human_pearson']:.3f}")
    print(f"  length (log wc) vs expert Pearson {res['logwc_vs_human_pearson']:.3f}")
    print(f"  GPT-4o vs expert         Pearson {res['old_gpt_vs_human_pearson']:.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
