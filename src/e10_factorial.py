"""
E10 — 요인설계 기여도 분해 (Factorial Attribution)

목적:
    "GPT-4o가 실제로 기여하는가"를, 정보원을 켜고 끄는 모든 조합(2^4)을 돌려
    증명하거나 반증한다. 논문 RQ2("최적 입력 특징 조합")의 정면 답.

설계:
    요인 A 학년(6) · B 텍스트통계(23) · C LLM총괄(7) · D LLM세부(11)
    wc_zscore(1)는 A와 B가 모두 켜진 셀에만 포함 -> 전체 셀 = 48피처 (E7l과 동일)
    2^4=16 조합 × 모델 2종(Ridge/XGBoost) × 시드 5 = 155회 학습 (상수셀은 1개)
    참조모델: TF-IDF+Ridge, 길이만

프로토콜 (엄수):
    학습 train 3,668건만 / 보고 val 786건 / **test 미사용** / 후처리 없음
    E10은 '추정' 전용 — 어떤 셀도 선택하지 않으므로 다중비교 문제가 없다.

명세: docs/e10-factorial.md
"""
import os
import sys
import json
import itertools
import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (BASE_DIR, TEXT_F, LOG_F, GPT4O_F, SUB_F, GRADE_F, EXTRA_F,
                       CATEGORIES, CAT_KR, EXPERT_COLS, load_splits,
                       metrics, boot_indices, boot_qwk_ci, boot_delta_ci, compute_qwk)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "e10_factorial")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]
N_BOOT = 1000
BOOT_SEED = 7
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# E7l의 deep_normal 설정 — 모든 셀에 동일 적용 (셀별 재탐색 금지)
XGB_PARAMS = dict(n_estimators=500, max_depth=5, learning_rate=0.02,
                  subsample=0.85, colsample_bytree=0.85,
                  reg_alpha=0.05, reg_lambda=0.5, verbosity=0, n_jobs=4)

FACTORS = {"A": ("grade", GRADE_F), "B": ("text", TEXT_F + LOG_F),
           "C": ("llm_main", GPT4O_F), "D": ("llm_sub", SUB_F)}


def cell_features(flags):
    """flags: dict(A=bool,B=bool,C=bool,D=bool) -> 피처 리스트"""
    f = []
    for k in ("A", "B", "C", "D"):
        if flags[k]:
            f += FACTORS[k][1]
    if flags["A"] and flags["B"]:
        f += EXTRA_F          # wc_zscore = 학년×텍스트 교호 특징
    return f


def ridge_alpha_by_oof(X, y, imp, seed=0):
    """train 내부 5-fold OOF로 alpha 선택. val을 보지 않는다."""
    Xi = np.nan_to_num(X, nan=imp)
    best, best_q = RIDGE_ALPHAS[0], -np.inf
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for a in RIDGE_ALPHAS:
        oof = np.zeros(len(y))
        for tr, te in kf.split(Xi):
            sc = StandardScaler().fit(Xi[tr])
            m = Ridge(alpha=a).fit(sc.transform(Xi[tr]), y[tr])
            oof[te] = np.clip(m.predict(sc.transform(Xi[te])), 0, 3)
        q = compute_qwk(y, oof)
        if q > best_q:
            best_q, best = q, a
    return best, best_q


def main():
    t0 = datetime.datetime.now()
    print("=" * 74)
    print("E10 — 요인설계 기여도 분해")
    print("=" * 74)

    # test 파일을 열지 않는다
    splits = load_splits(("train", "val"))
    tr, va = splits["train"], splits["val"]
    print(f"\n  train {len(tr)} / val {len(va)}  | test: 미사용")

    y_tr = {c: tr[EXPERT_COLS[i]].values for i, c in enumerate(CATEGORIES)}
    y_va = {c: va[EXPERT_COLS[i]].values for i, c in enumerate(CATEGORIES)}
    base_rmse = {c: float(np.sqrt(np.mean((y_va[c] - y_tr[c].mean()) ** 2)))
                 for c in CATEGORIES}
    print("  기준선 RMSE(상수=train평균):",
          {c: round(v, 4) for c, v in base_rmse.items()})

    combos = list(itertools.product([False, True], repeat=4))   # (A,B,C,D)
    boot_idx = boot_indices(len(va), N_BOOT, BOOT_SEED)

    cells, preds = {}, {}
    rows = []

    print(f"\n  셀 학습 ({len(combos)} 조합 × 2 모델)")
    for ci, (a, b, c_, d) in enumerate(combos, 1):
        flags = {"A": a, "B": b, "C": c_, "D": d}
        feats = cell_features(flags)
        tag = "".join(k for k in "ABCD" if flags[k]) or "none"

        models = ["constant"] if not feats else ["ridge", "xgboost"]
        for model in models:
            cid = f"cell{ci:02d}_{model}"
            res = {"factors": flags, "model": model, "n_features": len(feats),
                   "tag": tag, "val": {}}
            for cat in CATEGORIES:
                if model == "constant":
                    p = np.full(len(va), y_tr[cat].mean())
                    alpha = None
                else:
                    Xtr = tr[feats].values.astype(np.float32)
                    Xva = va[feats].values.astype(np.float32)
                    if model == "ridge":
                        imp = np.nanmedian(Xtr, axis=0)
                        imp = np.where(np.isfinite(imp), imp, 0.0)
                        alpha, _ = ridge_alpha_by_oof(Xtr, y_tr[cat], imp)
                        Xi = np.nan_to_num(Xtr, nan=imp)
                        sc = StandardScaler().fit(Xi)
                        m = Ridge(alpha=alpha).fit(sc.transform(Xi), y_tr[cat])
                        p = np.clip(m.predict(sc.transform(
                            np.nan_to_num(Xva, nan=imp))), 0, 3)
                    else:
                        alpha = None
                        ps = []
                        for s in SEEDS:
                            m = XGBRegressor(**XGB_PARAMS, random_state=s)
                            m.fit(Xtr, y_tr[cat])
                            ps.append(np.clip(m.predict(Xva), 0, 3))
                        p = np.mean(ps, axis=0)
                        res.setdefault("seed_std", {})[cat] = float(
                            np.std([compute_qwk(y_va[cat], x) for x in ps]))
                mt = metrics(y_va[cat], p, base_rmse[cat])
                mt["ci_low"], mt["ci_high"] = boot_qwk_ci(y_va[cat], p, boot_idx)
                if alpha is not None:
                    mt["ridge_alpha"] = alpha
                res["val"][cat] = mt
                preds[(cid, cat)] = p
                rows.append({"cell_id": cid, "A_grade": a, "B_text": b,
                             "C_llm_main": c_, "D_llm_sub": d, "model": model,
                             "n_features": len(feats), "category": cat,
                             **{k: mt.get(k) for k in
                                ("qwk", "ci_low", "ci_high", "rmse", "r2",
                                 "pearson_r", "within_one")},
                             "seed_std": res.get("seed_std", {}).get(cat)})
            cells[cid] = res
            q = res["val"]["total"]["qwk"]
            print(f"    {cid:20s} [{tag:4s}] f={len(feats):2d}  total QWK {q:.4f} "
                  f"[{res['val']['total']['ci_low']:.4f}, {res['val']['total']['ci_high']:.4f}]")

        # 예측 저장
        for model in models:
            cid = f"cell{ci:02d}_{model}"
            pd.DataFrame({"essay_id": va["essay_id"].values,
                          **{f"pred_{c}": preds[(cid, c)] for c in CATEGORIES}}).to_csv(
                os.path.join(OUTPUT_DIR, "predictions", f"{cid}.csv"), index=False)

    # ---------- 참조 모델 ----------
    print("\n  참조 모델")
    ref = {}
    # TF-IDF + Ridge (vectorizer도 train에서만 fit)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=5000,
                          min_df=3, sublinear_tf=True)
    Ttr = vec.fit_transform(tr["essay_txt"].fillna("")).toarray().astype(np.float32)
    Tva = vec.transform(va["essay_txt"].fillna("")).toarray().astype(np.float32)
    ref["tfidf_ridge"] = {"n_features": Ttr.shape[1], "val": {}}
    for cat in CATEGORIES:
        alpha, _ = ridge_alpha_by_oof(Ttr, y_tr[cat], np.zeros(Ttr.shape[1]))
        sc = StandardScaler(with_mean=False).fit(Ttr)
        m = Ridge(alpha=alpha).fit(sc.transform(Ttr), y_tr[cat])
        p = np.clip(m.predict(sc.transform(Tva)), 0, 3)
        mt = metrics(y_va[cat], p, base_rmse[cat])
        mt["ci_low"], mt["ci_high"] = boot_qwk_ci(y_va[cat], p, boot_idx)
        mt["ridge_alpha"] = alpha
        ref["tfidf_ridge"]["val"][cat] = mt
        preds[("tfidf_ridge", cat)] = p
    print(f"    tfidf_ridge   total QWK {ref['tfidf_ridge']['val']['total']['qwk']:.4f}")

    # 길이만
    ref["length_only"] = {"n_features": 1, "val": {}}
    for cat in CATEGORIES:
        Xtr = tr[["word_count"]].values.astype(np.float32)
        Xva = va[["word_count"]].values.astype(np.float32)
        alpha, _ = ridge_alpha_by_oof(Xtr, y_tr[cat], np.array([0.0]))
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=alpha).fit(sc.transform(Xtr), y_tr[cat])
        p = np.clip(m.predict(sc.transform(Xva)), 0, 3)
        mt = metrics(y_va[cat], p, base_rmse[cat])
        mt["ci_low"], mt["ci_high"] = boot_qwk_ci(y_va[cat], p, boot_idx)
        ref["length_only"]["val"][cat] = mt
        preds[("length_only", cat)] = p
    print(f"    length_only   total QWK {ref['length_only']['val']['total']['qwk']:.4f}")

    # GPT-4o 원점수 선형변환 (H1의 대조)
    ref["gpt4o_raw_linear"] = {"n_features": 1, "val": {}}
    raw_map = {"expression": "gpt4o_expr", "structure": "gpt4o_struct",
               "content": "gpt4o_cont", "total": "gpt4o_total"}
    for cat in CATEGORIES:
        r = va[raw_map[cat]].values.astype(float)
        rt = tr[raw_map[cat]].values.astype(float)
        # train 기준 최소-최대 선형 스케일 -> 0~3
        lo, hi = np.nanmin(rt), np.nanmax(rt)
        p = np.clip((r - lo) / (hi - lo) * 3 if hi > lo else np.zeros_like(r), 0, 3)
        p = np.nan_to_num(p, nan=float(y_tr[cat].mean()))
        mt = metrics(y_va[cat], p, base_rmse[cat])
        mt["ci_low"], mt["ci_high"] = boot_qwk_ci(y_va[cat], p, boot_idx)
        ref["gpt4o_raw_linear"]["val"][cat] = mt
        preds[("gpt4o_raw_linear", cat)] = p
    print(f"    gpt4o_raw     total QWK {ref['gpt4o_raw_linear']['val']['total']['qwk']:.4f}")

    # ---------- 요인 효과 ----------
    print("\n  요인 효과 (표준 대비 contrast)")
    combo_of = {}
    for ci, (a, b, c_, d) in enumerate(combos, 1):
        combo_of[(a, b, c_, d)] = ci

    def cid_for(a, b, c_, d, model):
        ci = combo_of[(a, b, c_, d)]
        feats = cell_features({"A": a, "B": b, "C": c_, "D": d})
        return f"cell{ci:02d}_{'constant' if not feats else model}"

    effects = {"main": {}, "interaction": {}}
    for model in ("ridge", "xgboost"):
        for cat in CATEGORIES:
            qs = {k: cells[cid_for(*k, model)]["val"][cat]["qwk"] for k in combo_of}
            for fi, fk in enumerate("ABCD"):
                on = [v for k, v in qs.items() if k[fi]]
                off = [v for k, v in qs.items() if not k[fi]]
                effects["main"].setdefault(f"{fk}_{FACTORS[fk][0]}", {}).setdefault(
                    model, {})[cat] = float(np.mean(on) - np.mean(off))
            for i, j in itertools.combinations(range(4), 2):
                nm = f"{'ABCD'[i]}_x_{'ABCD'[j]}"
                pos = [v for k, v in qs.items() if k[i] == k[j]]
                neg = [v for k, v in qs.items() if k[i] != k[j]]
                effects["interaction"].setdefault(nm, {}).setdefault(
                    model, {})[cat] = float(np.mean(pos) - np.mean(neg))
    for fk in "ABCD":
        e = effects["main"][f"{fk}_{FACTORS[fk][0]}"]["xgboost"]["total"]
        print(f"    주효과 {fk} ({FACTORS[fk][0]:9s}): {e:+.4f}")
    for nm, v in effects["interaction"].items():
        print(f"    상호작용 {nm}: {v['xgboost']['total']:+.4f}")

    # ---------- 가설 검정 ----------
    print("\n  가설 검정 (paired bootstrap, val 총점)")
    C16 = cid_for(True, True, True, True, "xgboost")     # 전체
    C6 = cid_for(True, True, False, False, "xgboost")    # LLM 없는 최강
    C12 = cid_for(True, True, True, False, "xgboost")    # 세부항목 없음
    C3 = cid_for(False, True, False, False, "xgboost")   # 텍스트만

    hyp = {}
    for hid, (defn, pa, pb) in {
        "H1": ("cell(ABCD) - gpt4o_raw_linear", C16, "gpt4o_raw_linear"),
        "H2": ("cell(ABCD) - cell(AB)  [LLM 증분]", C16, C6),
        "H2b": ("cell(ABCD) - cell(ABC) [세부항목 증분]", C16, C12),
        "vs_tfidf": ("cell(ABCD) - tfidf_ridge", C16, "tfidf_ridge"),
        "vs_length": ("cell(B) - length_only", C3, "length_only"),
    }.items():
        hyp[hid] = {"definition": defn, "by_category": {}}
        for cat in CATEGORIES:
            r = boot_delta_ci(y_va[cat], preds[(pa, cat)], preds[(pb, cat)], boot_idx)
            hyp[hid]["by_category"][cat] = r
        t = hyp[hid]["by_category"]["total"]
        band = ("large" if t["delta"] >= 0.10 else "moderate" if t["delta"] >= 0.05
                else "small" if t["delta"] >= 0.02 else "none")
        hyp[hid]["supported"] = bool(t["excludes_zero"] and t["delta"] > 0)
        hyp[hid]["effect_size_band"] = band
        print(f"    {hid:10s} Δ={t['delta']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] "
              f"{'지지' if hyp[hid]['supported'] else '기각/불명'} ({band})")

    # H2c: B×C 상호작용
    h2c = effects["interaction"]["B_x_C"]["xgboost"]["total"]
    hyp["H2c"] = {"definition": "interaction B x C (val total, xgboost)",
                  "value": h2c,
                  "supported": bool(h2c >= -0.01),
                  "interpretation": ("독립 기여" if h2c >= -0.01 else
                                     "부분적 대체" if h2c >= -0.05 else "강한 대체 관계")}
    print(f"    H2c        B×C={h2c:+.4f}  -> {hyp['H2c']['interpretation']}")

    # H3: 모델 주효과
    d3 = boot_delta_ci(y_va["total"], preds[(C16, "total")],
                       preds[(cid_for(True, True, True, True, "ridge"), "total")], boot_idx)
    hyp["H3"] = {"definition": "xgboost - ridge (full cell, val total)", **d3,
                 "supported": bool(d3["excludes_zero"] and d3["delta"] > 0)}
    print(f"    H3         Δ={d3['delta']:+.4f} [{d3['ci_low']:+.4f}, {d3['ci_high']:+.4f}]")

    # 진단: 예측값 상관
    pc = float(np.corrcoef(preds[(C6, "total")], preds[(C16, "total")])[0, 1])
    print(f"\n  진단: pred_corr(LLM없음, 전체) = {pc:.4f}")

    # ---------- 기록 ----------
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "e10_cells.csv"), index=False)

    eff_rows = []
    for et in ("main", "interaction"):
        for nm, bym in effects[et].items():
            for model, byc in bym.items():
                for cat, v in byc.items():
                    eff_rows.append({"effect_type": et, "effect_name": nm,
                                     "model": model, "category": cat, "value": v})
    pd.DataFrame(eff_rows).to_csv(os.path.join(OUTPUT_DIR, "e10_effects.csv"), index=False)

    hyp_rows = []
    for hid, h in hyp.items():
        if "by_category" in h:
            for cat, r in h["by_category"].items():
                hyp_rows.append({"hypothesis_id": hid, "definition": h["definition"],
                                 "category": cat, **r,
                                 "supported": h.get("supported"),
                                 "effect_size_band": h.get("effect_size_band")})
        else:
            hyp_rows.append({"hypothesis_id": hid, "definition": h["definition"],
                             "category": "total",
                             "delta": h.get("value", h.get("delta")),
                             "ci_low": h.get("ci_low"), "ci_high": h.get("ci_high"),
                             "excludes_zero": h.get("excludes_zero"),
                             "supported": h.get("supported"),
                             "effect_size_band": None})
    pd.DataFrame(hyp_rows).to_csv(os.path.join(OUTPUT_DIR, "e10_hypotheses.csv"), index=False)

    ref_rows = [{"model": k, "n_features": v["n_features"], "category": cat, **v["val"][cat]}
                for k, v in ref.items() for cat in CATEGORIES]
    pd.DataFrame(ref_rows).to_csv(os.path.join(OUTPUT_DIR, "e10_reference.csv"), index=False)

    out = {
        "timestamp": t0.isoformat(),
        "experiment": "e10_factorial",
        "label_definition": "corrected",
        "design": {
            "type": "2^4 full factorial x 2 model classes",
            "factors": {f"{k}_{v[0]}": {"n": len(v[1]), "features": v[1]}
                        for k, v in FACTORS.items()},
            "extra_feature_rule": "wc_zscore included only when A and B are both ON",
            "n_combinations": 16, "models": ["ridge", "xgboost"],
            "n_cells": len(cells), "seeds": SEEDS,
            "full_cell_n_features": 48,
            "xgb_params": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
            "ridge_alphas": RIDGE_ALPHAS,
        },
        "protocol": {
            "train_only": True, "train_size": int(len(tr)), "val_size": int(len(va)),
            "test_used": False, "postprocessing": False,
            "hyperparam_selection": "ridge alpha via train 5-fold OOF only",
            "bootstrap_n": N_BOOT, "ci_level": 0.95, "bootstrap_seed": BOOT_SEED,
            "statistical_role": "estimation only (no cell is selected)",
        },
        "baseline_rmse_val": base_rmse,
        "cells": cells,
        "effects": effects,
        "reference_models": ref,
        "hypotheses": hyp,
        "diagnostics": {"pred_corr_noLLM_vs_full_total": pc},
        "e10_verdict": {
            "h2_delta": hyp["H2"]["by_category"]["total"]["delta"],
            "h2_ci": [hyp["H2"]["by_category"]["total"]["ci_low"],
                      hyp["H2"]["by_category"]["total"]["ci_high"]],
            "h2_supported": hyp["H2"]["supported"],
            "h2_band": hyp["H2"]["effect_size_band"],
            "h2b_supported": hyp["H2b"]["supported"],
            "h2c_interpretation": hyp["H2c"]["interpretation"],
            "beats_tfidf": hyp["vs_tfidf"]["supported"],
        },
    }
    with open(os.path.join(OUTPUT_DIR, "e10_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)

    make_figures(df, effects, ref, cells, cid_for)

    print("\n" + "=" * 74)
    print(f"E10 완료 — 소요 {(datetime.datetime.now() - t0).total_seconds():.0f}초")
    print(f"산출물 -> {OUTPUT_DIR}")
    print("=" * 74)


def make_figures(df, effects, ref, cells, cid_for):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = df[(df.category == "total")].copy()
    t["label"] = t.apply(lambda r: "".join(
        k for k, on in zip("ABCD", [r.A_grade, r.B_text, r.C_llm_main, r.D_llm_sub]) if on)
        or "none", axis=1) + " / " + t["model"].str[:3]
    t = t.sort_values("qwk")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 8),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(t))
    err = [t.qwk - t.ci_low, t.ci_high - t.qwk]
    cols = ["#264653" if m == "xgboost" else "#8ab17d" if m == "ridge" else "#adb5bd"
            for m in t.model]
    ax.barh(y, t.qwk, xerr=err, color=cols,
            error_kw={"ecolor": "#333", "capsize": 2, "lw": .8})
    ax.set_yticks(y)
    ax.set_yticklabels(t.label, fontsize=7.5, family="monospace")
    for nm, c, ls in (("tfidf_ridge", "#e63946", "--"), ("length_only", "#f4a261", ":"),
                      ("gpt4o_raw_linear", "#9d4edd", "-.")):
        ax.axvline(ref[nm]["val"]["total"]["qwk"], color=c, ls=ls, lw=1.4, label=nm)
    ax.set_xlabel("val total QWK (95% bootstrap CI)")
    ax.set_title("E10 — All 2^4 factorial cells\nA=grade B=text C=LLM-main D=LLM-sub",
                 fontsize=10)
    ax.grid(axis="x", alpha=.3)
    ax.legend(fontsize=7.5, loc="lower right")

    names, vals = [], []
    for fk in "ABCD":
        k = [x for x in effects["main"] if x.startswith(fk + "_")][0]
        names.append(f"main {k}")
        vals.append(effects["main"][k]["xgboost"]["total"])
    for nm, v in effects["interaction"].items():
        names.append(f"int {nm}")
        vals.append(v["xgboost"]["total"])
    o = np.argsort(vals)
    ax2.barh(np.arange(len(vals)), [vals[i] for i in o],
             color=["#2a9d8f" if vals[i] >= 0 else "#e76f51" for i in o])
    ax2.set_yticks(np.arange(len(vals)))
    ax2.set_yticklabels([names[i] for i in o], fontsize=8)
    ax2.axvline(0, color="k", lw=1)
    ax2.set_xlabel("Effect on val total QWK (XGBoost)")
    ax2.set_title("Main effects & 2-way interactions", fontsize=10)
    ax2.grid(axis="x", alpha=.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_factorial_qwk.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
