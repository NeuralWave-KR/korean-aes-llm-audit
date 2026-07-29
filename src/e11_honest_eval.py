"""
E11 — 정직한 재평가 (Honest Re-evaluation)

목적:
    논문의 대표 수치를 만든 E7l을, 논문 3장이 원래 지시한 프로토콜대로 다시 돌린다.
    새 방법이 아니라 **정정**이다. (3장 E6-1: "최적 λ는 Val 세트에서의 그리드 탐색으로 결정")

E7l 대비 변경 (모델 구성은 그대로 — 프로토콜만 바꿔야 비교가 성립):
    1. 학습        train+val 4,454  ->  train 3,668만
    2. 앙상블 가중치  test QWK      ->  val QWK
    3. 스무딩 λ      test QWK      ->  val QWK
    4. 등급컷 6개    test 라벨 피팅 ->  val 라벨 피팅
    5. 분포보정      test 배치 순위 ->  train OOF 분포로 고정 (귀납적, 답안 1건 처리 가능)
    6. test 사용    1,155 스크리닝 ->  최종 1회
    7. OOF          없음          ->  적용
    8. 라벨          legacy        ->  corrected (legacy 병기)
    9. CI           없음          ->  부트스트랩 95%

명세: docs/e11-honest-eval.md
"""
import os
import sys
import json
import itertools
import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (BASE_DIR, TEXT_F, LOG_F, GPT4O_F, SUB_F, GRADE_F, EXTRA_F,
                       CATEGORIES, CAT_KR, EXPERT_COLS, EXPERT_COLS_LEGACY,
                       load_splits, metrics, boot_indices, boot_qwk_ci,
                       boot_delta_ci, compute_qwk, discretize, qwk_disc)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "e11_honest_eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_F = TEXT_F + LOG_F + GPT4O_F + SUB_F + GRADE_F + EXTRA_F   # 48개, E7l과 동일

# E7l의 5개 설정 — 그대로
CONFIGS = {
    "deep_normal": dict(n_estimators=500, max_depth=5, learning_rate=0.02,
                        subsample=0.85, colsample_bytree=0.85,
                        reg_alpha=0.05, reg_lambda=0.5),
    "deep_strong": dict(n_estimators=500, max_depth=4, learning_rate=0.02,
                        subsample=0.7, colsample_bytree=0.7,
                        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5),
    "deep_mid": dict(n_estimators=500, max_depth=5, learning_rate=0.02,
                     subsample=0.75, colsample_bytree=0.75,
                     reg_alpha=0.3, reg_lambda=2.0, min_child_weight=3),
    "med_normal": dict(n_estimators=800, max_depth=3, learning_rate=0.01,
                       subsample=0.75, colsample_bytree=0.7,
                       reg_alpha=0.5, reg_lambda=2.0),
    "med_strong": dict(n_estimators=800, max_depth=3, learning_rate=0.01,
                       subsample=0.65, colsample_bytree=0.6,
                       reg_alpha=2.0, reg_lambda=8.0, min_child_weight=5),
}
SEEDS = [42, 123, 456, 789, 2024]
SMOOTHINGS = np.arange(0.0, 0.41, 0.04)        # 11단계 — E7l과 동일
TOP_N = 30                                      # E7l과 동일
N_BOOT = 1000
BOOT_SEED = 11
N_FOLDS = 5


# ============================================================
# 후처리 — 귀납적 분포보정 (E7l의 transductive 버전을 정정)
# ============================================================
def fit_dist_cal(ref_pred, ref_label):
    """train OOF 예측 분포와 train 정답 분포로 고정 변환표를 만든다."""
    return {"pred_sorted": np.sort(np.asarray(ref_pred, float)),
            "ref_sorted": np.sort(np.asarray(ref_label, float))}


def apply_dist_cal(pred, m, smoothing=0.0):
    """
    새 예측값에 고정 변환표를 적용한다. 배치 순위를 쓰지 않으므로
    **답안 1건에도 적용 가능**하다 (E7l은 불가능했음).
    """
    pred = np.asarray(pred, float)
    ps, rs = m["pred_sorted"], m["ref_sorted"]
    n, nr = len(ps), len(rs)
    q = np.interp(pred, ps, np.linspace(1 / (n + 1), n / (n + 1), n),
                  left=0.0, right=1.0)
    ref_ext = np.concatenate([[0.0], rs, [3.0]])
    perc_ext = np.concatenate([[0.0], np.linspace(0, 1, nr), [1.0]])
    cal = np.interp(q, perc_ext, ref_ext)
    if smoothing > 0:
        cal = (1 - smoothing) * cal + smoothing * pred
    return np.clip(cal, 0, 3)


def apply_thresholds(pred, thresholds):
    d = np.zeros(len(pred), dtype=int)
    for i, t in enumerate(sorted(thresholds)):
        d[np.asarray(pred, float) >= t] = i + 1
    return d


def qwk_with_thresholds(y_true_d, pred, thresholds):
    return qwk_disc(y_true_d, apply_thresholds(pred, thresholds))


def optimize_thresholds(y_true, y_pred):
    """E7l과 동일 알고리즘: grid -> 3-pass fine-tune -> Nelder-Mead. 입력만 val."""
    td = discretize(y_true)
    init_t = np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    best_t, best_q = init_t.copy(), -1.0
    for off in np.arange(-0.30, 0.31, 0.015):
        t = np.clip(init_t + off, 0.01, 2.99)
        q = qwk_with_thresholds(td, y_pred, t)
        if q > best_q:
            best_q, best_t = q, t.copy()
    for _ in range(3):
        for i in range(6):
            for dlt in np.arange(-0.25, 0.26, 0.003):
                tn = best_t.copy()
                tn[i] = np.clip(tn[i] + dlt, 0.01, 2.99)
                tn = np.sort(tn)
                q = qwk_with_thresholds(td, y_pred, tn)
                if q > best_q:
                    best_q, best_t = q, tn.copy()
    r = minimize(lambda t: -qwk_with_thresholds(td, y_pred, np.sort(np.clip(t, .01, 2.99))),
                 best_t, method="Nelder-Mead",
                 options={"maxiter": 8000, "xatol": 5e-4, "fatol": 5e-5})
    ft = np.sort(np.clip(r.x, 0.01, 2.99))
    fq = qwk_with_thresholds(td, y_pred, ft)
    return (ft, fq) if fq > best_q else (best_t, best_q)


# ============================================================
def main():
    t0 = datetime.datetime.now()
    print("=" * 74)
    print("E11 — 정직한 재평가 (E7l 정정판)")
    print("=" * 74)

    splits = load_splits(("train", "val", "test"))
    tr, va, te = splits["train"], splits["val"], splits["test"]
    print(f"\n  train {len(tr)} (학습 전용) / val {len(va)} (선택 전용) / test {len(te)} (최종 1회)")
    print(f"  특징 {len(ALL_F)}개 | 설정 {len(CONFIGS)} | 시드 {len(SEEDS)}")

    Xtr = tr[ALL_F].values.astype(np.float32)
    Xva = va[ALL_F].values.astype(np.float32)
    Xte = te[ALL_F].values.astype(np.float32)

    y_tr = {c: tr[EXPERT_COLS[i]].values for i, c in enumerate(CATEGORIES)}
    y_va = {c: va[EXPERT_COLS[i]].values for i, c in enumerate(CATEGORIES)}
    y_te = {c: te[EXPERT_COLS[i]].values for i, c in enumerate(CATEGORIES)}
    y_te_legacy = {c: te[EXPERT_COLS_LEGACY[i]].values for i, c in enumerate(CATEGORIES)}
    base_rmse = {c: float(np.sqrt(np.mean((y_te[c] - y_tr[c].mean()) ** 2)))
                 for c in CATEGORIES}
    print(f"  기준선 RMSE(test, 상수=train평균): "
          f"{ {c: round(v, 4) for c, v in base_rmse.items()} }")

    # ---------- 1) 학습: 전체 train 적합 + OOF ----------
    print(f"\n[1/5] 학습 — 설정 {len(CONFIGS)} × 영역 4 × 시드 {len(SEEDS)} "
          f"(+ {N_FOLDS}-fold OOF)")
    P_va, P_te, P_oof = {}, {}, {}
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(kf.split(Xtr))
    for cname, params in CONFIGS.items():
        for ci, cat in enumerate(CATEGORIES):
            y = y_tr[cat]
            pv, pt = [], []
            for s in SEEDS:
                m = XGBRegressor(**params, random_state=s, verbosity=0, n_jobs=4)
                m.fit(Xtr, y)
                pv.append(np.clip(m.predict(Xva), 0, 3))
                pt.append(np.clip(m.predict(Xte), 0, 3))
            P_va[(cname, cat)] = np.mean(pv, axis=0)
            P_te[(cname, cat)] = np.mean(pt, axis=0)
            # OOF (분포보정 참조분포용) — 시드 1개로 충분(분포 추정 목적)
            oof = np.zeros(len(y))
            for tr_i, te_i in folds:
                m = XGBRegressor(**params, random_state=SEEDS[0], verbosity=0, n_jobs=4)
                m.fit(Xtr[tr_i], y[tr_i])
                oof[te_i] = np.clip(m.predict(Xtr[te_i]), 0, 3)
            P_oof[(cname, cat)] = oof
        print(f"    {cname:12s} 완료  (val 총점 raw QWK "
              f"{compute_qwk(y_va['total'], P_va[(cname, 'total')]):.4f})")

    # ---------- 2) 후보 생성 (E7l과 동일 구조: 105 가중치 × 11 λ) ----------
    names = list(CONFIGS)
    weight_cands = [(c, {c: 1.0}) for c in names]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for w in np.arange(0.1, 1.0, 0.1):
                weight_cands.append((f"{a}*{w:.1f}+{b}*{1 - w:.1f}", {a: w, b: 1 - w}))
    for a, b, c in itertools.combinations(names, 3):
        weight_cands.append((f"eq({a},{b},{c})", {a: 1 / 3, b: 1 / 3, c: 1 / 3}))
    n_cand = len(weight_cands) * len(SMOOTHINGS)
    print(f"\n[2/5] 후보 {len(weight_cands)} 가중조합 × {len(SMOOTHINGS)} λ = {n_cand}"
          f"  ← E7l과 동일 규모, 채점자만 val")

    # ---------- 3) val에서 선택 ----------
    print(f"\n[3/5] val 선택 (상위 {TOP_N} 정밀 임계값 최적화)")
    selected, sel_log = {}, {}
    for cat in CATEGORIES:
        scored = []
        for cname, w in weight_cands:
            bo = sum(wt * P_oof[(c, cat)] for c, wt in w.items())
            bv = sum(wt * P_va[(c, cat)] for c, wt in w.items())
            dmap = fit_dist_cal(bo, y_tr[cat])
            for s in SMOOTHINGS:
                cal_v = apply_dist_cal(bv, dmap, s)
                scored.append((compute_qwk(y_va[cat], cal_v), cname, w, float(s)))
        scored.sort(key=lambda x: -x[0])
        top = scored[:TOP_N]

        best = None
        for std_q, cname, w, s in top:
            bo = sum(wt * P_oof[(c, cat)] for c, wt in w.items())
            bv = sum(wt * P_va[(c, cat)] for c, wt in w.items())
            dmap = fit_dist_cal(bo, y_tr[cat])
            cal_v = apply_dist_cal(bv, dmap, s)
            th, oq = optimize_thresholds(y_va[cat], cal_v)
            if best is None or oq > best["val_opt_qwk"]:
                best = {"blend": cname, "weights": w, "smoothing": s,
                        "thresholds": th.tolist(),
                        "val_std_qwk": float(std_q), "val_opt_qwk": float(oq)}
        selected[cat] = best
        sel_log[cat] = {
            "n_weight_candidates": len(weight_cands),
            "n_smoothing_levels": len(SMOOTHINGS),
            "n_total_candidates_scored_on_val": n_cand,
            "n_threshold_optimizations": TOP_N,
            "val_std_qwk_range": [float(scored[-1][0]), float(scored[0][0])],
            "selected": {"blend": best["blend"], "smoothing": best["smoothing"]},
            "val_std_qwk_of_selected": best["val_std_qwk"],
            "val_opt_qwk_of_selected": best["val_opt_qwk"],
        }
        print(f"    {CAT_KR[cat]:4s} {best['blend'][:38]:38s} λ={best['smoothing']:.2f}"
              f"  val std={best['val_std_qwk']:.4f} opt={best['val_opt_qwk']:.4f}")

    # ---------- 4) test 최종 1회 ----------
    print("\n[4/5] test 최종 평가 (1회)")
    boot_idx = boot_indices(len(te), N_BOOT, BOOT_SEED)
    test_results, test_pred, cal_map_out = {}, {}, {}
    for cat in CATEGORIES:
        b = selected[cat]
        bo = sum(wt * P_oof[(c, cat)] for c, wt in b["weights"].items())
        bt = sum(wt * P_te[(c, cat)] for c, wt in b["weights"].items())
        dmap = fit_dist_cal(bo, y_tr[cat])
        cal_t = apply_dist_cal(bt, dmap, b["smoothing"])
        th = np.array(b["thresholds"])
        opt_pred = apply_thresholds(cal_t, th) / 2.0        # 0..6 -> 0~3

        mt = metrics(y_te[cat], cal_t, base_rmse[cat])
        mt["std_qwk"] = mt.pop("qwk")
        mt["opt_qwk"] = qwk_disc(discretize(y_te[cat]), apply_thresholds(cal_t, th))
        mt["ci_low"], mt["ci_high"] = boot_qwk_ci(y_te[cat], opt_pred, boot_idx)
        mt["std_ci_low"], mt["std_ci_high"] = boot_qwk_ci(y_te[cat], cal_t, boot_idx)
        mt["legacy_label_opt_qwk"] = qwk_disc(discretize(y_te_legacy[cat]),
                                              apply_thresholds(cal_t, th))
        test_results[cat] = mt
        test_pred[cat] = {"std": cal_t, "opt": opt_pred}
        cal_map_out[cat] = {
            "train_pred_quantiles": np.percentile(dmap["pred_sorted"],
                                                  np.arange(0, 101, 5)).tolist(),
            "train_label_quantiles": np.percentile(dmap["ref_sorted"],
                                                   np.arange(0, 101, 5)).tolist(),
            "smoothing": b["smoothing"], "thresholds": b["thresholds"],
            "blend": b["blend"], "weights": b["weights"],
        }
        print(f"    {CAT_KR[cat]:4s} std={mt['std_qwk']:.4f} opt={mt['opt_qwk']:.4f} "
              f"[{mt['ci_low']:.4f},{mt['ci_high']:.4f}] RMSE={mt['rmse']:.4f} R²={mt['r2']:.4f}")

    # H4: 후처리 효과 (등급컷 적용 전/후)
    h4 = boot_delta_ci(y_te["total"], test_pred["total"]["opt"],
                       test_pred["total"]["std"], boot_idx)
    h4["supported"] = bool(h4["excludes_zero"] and h4["delta"] > 0)
    print(f"\n    H4 후처리 효과 Δ={h4['delta']:+.4f} "
          f"[{h4['ci_low']:+.4f},{h4['ci_high']:+.4f}] "
          f"{'지지' if h4['supported'] else '기각/불명'}")

    # ---------- 5) 비교 및 기록 ----------
    print("\n[5/5] 비교 · 기록")
    e9 = json.load(open(os.path.join(BASE_DIR, "outputs", "e9_human_ceiling",
                                     "e9_summary.json"), encoding="utf-8"))
    e10 = json.load(open(os.path.join(BASE_DIR, "outputs", "e10_factorial",
                                      "e10_summary.json"), encoding="utf-8"))
    ceil = {c: e9["ceiling_test"]["one_vs_rest"][c]["qwk"] for c in CATEGORIES}

    vs_ceiling = []
    for cat in CATEGORIES:
        q = test_results[cat]["opt_qwk"]
        vs_ceiling.append({"category": cat, "e11_qwk": q,
                           "e11_ci_low": test_results[cat]["ci_low"],
                           "e11_ci_high": test_results[cat]["ci_high"],
                           "human_ceiling": ceil[cat],
                           "ratio_pct": round(q / ceil[cat] * 100, 1) if ceil[cat] else None,
                           "exceeds_ceiling": bool(q > ceil[cat])})
        print(f"    {CAT_KR[cat]:4s} E11 {q:.4f} vs 인간상한 {ceil[cat]:.4f} "
              f"-> {q / ceil[cat] * 100:5.1f}%")
    pd.DataFrame(vs_ceiling).to_csv(os.path.join(OUTPUT_DIR, "e11_vs_ceiling.csv"),
                                    index=False)

    e7l = {"total_opt_qwk": 0.6246, "total_std_qwk": 0.5916, "total_rmse": 0.2679}
    changes = [
        (1, "학습 데이터", "train+val 4,454", "train 3,668", "3.7.1 데이터 누수 통제"),
        (2, "앙상블 가중치", "test QWK 기준", "val QWK 기준", "3장 E6-1"),
        (3, "스무딩 λ", "test QWK 기준", "val QWK 기준", "3장 E6-1 명시"),
        (4, "등급컷 6개", "test 라벨 피팅", "val 라벨 피팅", "3.7.1"),
        (5, "분포보정 참조", "test 배치 순위(전이적)", "train OOF 분포(귀납적)", "3장 E6-1"),
        (6, "test 사용", "1,155 스크리닝+30회 컷최적화", "최종 1회", "3.7.1"),
        (7, "OOF", "없음", "적용", "3장 E2-2"),
        (8, "라벨 정의", "legacy", "corrected", "E9 §4.2"),
        (9, "신뢰구간", "없음", "부트스트랩 95%", "사전등록 원칙"),
    ]
    cmp_rows = [{"change_id": i, "item": a, "e7l_setting": b, "e11_setting": c,
                 "rationale_in_thesis": d} for i, a, b, c, d in changes]
    tt = test_results["total"]
    for k, ev, nv in (("total_opt_qwk", e7l["total_opt_qwk"], tt["opt_qwk"]),
                      ("total_std_qwk", e7l["total_std_qwk"], tt["std_qwk"]),
                      ("total_rmse", e7l["total_rmse"], tt["rmse"])):
        cmp_rows.append({"change_id": None, "item": k, "e7l_setting": ev,
                         "e11_setting": round(nv, 4),
                         "rationale_in_thesis": f"delta={nv - ev:+.4f}"})
    pd.DataFrame(cmp_rows).to_csv(os.path.join(OUTPUT_DIR, "e11_comparison_e7l.csv"),
                                  index=False)

    c16 = e10["cells"]["cell16_xgboost"]["val"]["total"]["qwk"]
    c13 = e10["cells"]["cell13_xgboost"]["val"]["total"]["qwk"]
    pd.DataFrame([
        {"reference": "e10_cell16_ABCD_val", "qwk": c16, "note": "후처리·앙상블 없음"},
        {"reference": "e10_cell13_AB_val(LLM없음)", "qwk": c13, "note": "LLM 없는 최강 대조군"},
        {"reference": "e11_test_opt", "qwk": tt["opt_qwk"], "note": "본 실험"},
        {"reference": "e7d_saved_json", "qwk": 0.5818, "note": "E7 계열 중 유일하게 깨끗"},
        {"reference": "e7l_leaked", "qwk": 0.6246, "note": "누수 있음 — 인용 금지"},
    ]).to_csv(os.path.join(OUTPUT_DIR, "e11_vs_e10.csv"), index=False)

    pd.DataFrame({"essay_id": te["essay_id"].values,
                  "grade_level": te["grade_level"].values,
                  "total_score": y_te["total"],
                  "total_score_legacy": y_te_legacy["total"],
                  "pred_std": test_pred["total"]["std"],
                  "pred_opt": test_pred["total"]["opt"]}).to_csv(
        os.path.join(OUTPUT_DIR, "e11_predictions.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "e11_calibration_map.json"), "w",
              encoding="utf-8") as f:
        json.dump({"method": "fixed empirical CDF from train OOF predictions",
                   "note": "새 답안 1건에도 적용 가능 (test 배치 순위 불필요)",
                   **cal_map_out}, f, ensure_ascii=False, indent=2)

    sel_log["test_evaluation_count"] = 1
    sel_log["settings_changed_after_seeing_test"] = 0
    with open(os.path.join(OUTPUT_DIR, "e11_selection_log.json"), "w",
              encoding="utf-8") as f:
        json.dump(sel_log, f, ensure_ascii=False, indent=2)

    by_grade = {}
    for g in sorted(te["grade_level"].unique()):
        m = (te["grade_level"] == g).values
        if m.sum() < 10:
            continue
        by_grade[g] = {"qwk": qwk_disc(discretize(y_te["total"][m]),
                                       discretize(test_pred["total"]["opt"][m])),
                       "n": int(m.sum()),
                       "human_ceiling": e9["by_grade"].get(f"{g}|total", {}).get("qwk")}

    out = {
        "timestamp": t0.isoformat(),
        "experiment": "e11_honest_eval",
        "label_definition": "corrected",
        "corrects": "e7l_optimized",
        "protocol": {
            "train_size": int(len(tr)), "val_size": int(len(va)), "test_size": int(len(te)),
            "val_in_training": False, "selection_set": "val",
            "test_evaluations": 1,
            "n_candidates_screened_on_val": n_cand,
            "threshold_optimized_on": "val",
            "dist_cal_reference": "train OOF (fixed, inductive)",
            "oof_used": True, "n_folds": N_FOLDS,
            "bootstrap_n": N_BOOT, "ci_level": 0.95, "bootstrap_seed": BOOT_SEED,
        },
        "model": {"n_configs": len(CONFIGS), "configs": CONFIGS, "seeds": SEEDS,
                  "n_features": len(ALL_F),
                  "n_models_trained": len(CONFIGS) * len(CATEGORIES) * len(SEEDS),
                  "n_oof_models": len(CONFIGS) * len(CATEGORIES) * N_FOLDS},
        "baseline_rmse_test": base_rmse,
        "selected_on_val": selected,
        "test_results": test_results,
        "by_grade": by_grade,
        "hypotheses": {"H4": {"definition": "opt_qwk - std_qwk (test total)", **h4}},
        "comparison": {
            "e7l": {**e7l, "protocol": "test-fitted",
                    "delta_opt_from_e11": round(tt["opt_qwk"] - e7l["total_opt_qwk"], 4),
                    "note": "누수 있음 — 인용 금지"},
            "e7d": {"total_opt_qwk_saved": 0.5818, "total_opt_qwk_cited": 0.5926,
                    "protocol": "OOF 기반 (E7 계열 중 유일하게 깨끗)",
                    "note": "저장값을 정본으로 채택. 인용값 출처 실행 미보존"},
            "e10_cell16_ABCD_val": c16,
            "e10_cell13_AB_val": c13,
            "e9_ceiling_test": ceil,
        },
        "vs_ceiling": vs_ceiling,
        "e11_verdict": {
            "headline_qwk": round(tt["opt_qwk"], 4),
            "headline_ci": [round(tt["ci_low"], 4), round(tt["ci_high"], 4)],
            "headline_r2": round(tt["r2"], 4),
            "vs_e7l": f"{e7l['total_opt_qwk']:.4f} -> {tt['opt_qwk']:.4f} "
                      f"({tt['opt_qwk'] - e7l['total_opt_qwk']:+.4f})",
            "vs_ceiling_pct": round(tt["opt_qwk"] / ceil["total"] * 100, 1),
            "sanity_band": ("정상 (0.55~0.62)" if 0.55 <= tt["opt_qwk"] <= 0.62 else
                            "경보: E7d보다 높음 — 누수 잔존 의심" if tt["opt_qwk"] > 0.62 else
                            "낮음 — 구현 확인 필요" if tt["opt_qwk"] >= 0.45 else
                            "구현 오류 의심"),
            "rmse_target_legacy": 0.80,
            "rmse_target_status": "폐기 — 상수 예측도 통과하므로 무의미",
        },
    }
    with open(os.path.join(OUTPUT_DIR, "e11_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)

    make_figures(test_results, ceil, e7l, c13, c16)

    print("\n" + "=" * 74)
    print("E11 최종")
    print("=" * 74)
    print(f"  대표 수치(총점 opt QWK): {tt['opt_qwk']:.4f} "
          f"[{tt['ci_low']:.4f}, {tt['ci_high']:.4f}]")
    print(f"  E7l(누수) 0.6246 -> E11 {tt['opt_qwk']:.4f} "
          f"({tt['opt_qwk'] - 0.6246:+.4f})")
    print(f"  인간 상한 {ceil['total']:.4f} 대비 {tt['opt_qwk'] / ceil['total'] * 100:.1f}%")
    print(f"  R² {tt['r2']:.4f} | RMSE {tt['rmse']:.4f} (기준선 {base_rmse['total']:.4f})")
    print(f"  정상성: {out['e11_verdict']['sanity_band']}")
    print(f"\n  소요 {(datetime.datetime.now() - t0).total_seconds():.0f}초 -> {OUTPUT_DIR}")


def make_figures(test_results, ceil, e7l, c13, c16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    en = {"expression": "Expression", "structure": "Structure",
          "content": "Content", "total": "TOTAL"}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(4)
    e11 = [test_results[c]["opt_qwk"] for c in CATEGORIES]
    err = [[test_results[c]["opt_qwk"] - test_results[c]["ci_low"] for c in CATEGORIES],
           [test_results[c]["ci_high"] - test_results[c]["opt_qwk"] for c in CATEGORIES]]
    cl = [ceil[c] for c in CATEGORIES]
    a1.bar(x - 0.2, e11, 0.4, yerr=err, label="E11 (honest)", color="#264653",
           error_kw={"ecolor": "#333", "capsize": 4})
    a1.bar(x + 0.2, cl, 0.4, label="Human ceiling (E9)", color="#e9c46a")
    for i, (v, c) in enumerate(zip(e11, cl)):
        a1.text(i, max(v, c) + .03, f"{v / c * 100:.0f}%", ha="center", fontsize=9)
    a1.set_xticks(x); a1.set_xticklabels([en[c] for c in CATEGORIES])
    a1.set_ylabel("QWK"); a1.set_ylim(0, 1)
    a1.set_title("E11 vs human ceiling (test n=786)", fontsize=11)
    a1.legend(fontsize=9); a1.grid(axis="y", alpha=.3)

    bars = {"E7l\n(leaked)": e7l["total_opt_qwk"],
            "E11\n(honest)": test_results["total"]["opt_qwk"],
            "E10 ABCD\n(no postproc)": c16,
            "E10 AB\n(no LLM)": c13,
            "Human\nceiling": ceil["total"]}
    cols = ["#e63946", "#264653", "#457b9d", "#8ab17d", "#e9c46a"]
    a2.bar(range(len(bars)), list(bars.values()), color=cols)
    a2.errorbar(1, test_results["total"]["opt_qwk"],
                yerr=[[test_results["total"]["opt_qwk"] - test_results["total"]["ci_low"]],
                      [test_results["total"]["ci_high"] - test_results["total"]["opt_qwk"]]],
                fmt="none", ecolor="k", capsize=5)
    for i, v in enumerate(bars.values()):
        a2.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=9)
    a2.set_xticks(range(len(bars))); a2.set_xticklabels(bars.keys(), fontsize=8.5)
    a2.set_ylabel("total QWK"); a2.set_ylim(0, .75)
    a2.set_title("Where the headline number lands", fontsize=11)
    a2.grid(axis="y", alpha=.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_e11_vs_ceiling.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
