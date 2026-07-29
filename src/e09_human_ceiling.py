"""
E9 — 인간 평가자 상한 측정 (Human Ceiling)

목적:
    이 데이터에서 사람 채점자끼리 얼마나 일치하는지를 측정하여,
    자동채점 모델이 넘을 수 있는 현실적 천장(ceiling)을 확정한다.

배경:
    논문 RQ1은 "인간 평가자 수준의 채점 일치도"를 목표로 하나, 정작
    인간 평가자 간 일치도를 한 번도 계산하지 않았다. 기존 E1-ceiling의
    0.9748은 전문가 세부항목을 입력으로 전문가 총점을 예측한 값(거의 항등식)
    이므로 인간 상한이 아니다.

산출:
    results/e9_human_ceiling/
        e9_summary.json
        e9_one_vs_rest.csv      ★ 주 기준
        e9_pairwise.csv
        e9_by_grade.csv
        e9_label_definition_diff.csv
        fig_ceiling_by_item.png

명세: docs/e9-human-ceiling.md
"""
import os
import sys
import json
import glob
import zipfile
import datetime
import itertools

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_GLOB = os.path.join(BASE_DIR, "data", "raw", "aihub-essay", "**", "*글짓기.zip")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "e9_human_ceiling")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BOOT = 1000
CI_LEVEL = 0.95
RNG_SEED = 42

# ============================================================
# 세부항목 매핑 (docs/e9-human-ceiling.md §4.1)
#   (원본 배열, 인덱스, 항목명)
#   내용 인덱스 2(con_prompt)는 전수 0 / 가중치 0 -> 제외
# ============================================================
SUBITEMS = [
    ("essay_scoreT_exp", 0, "grammar"),
    ("essay_scoreT_exp", 1, "word_choice"),
    ("essay_scoreT_exp", 2, "sentence_expression"),
    ("essay_scoreT_org", 0, "paragraph_connection"),
    ("essay_scoreT_org", 1, "paragraph_structure"),
    ("essay_scoreT_org", 2, "consistency"),
    ("essay_scoreT_org", 3, "length"),
    ("essay_scoreT_cont", 0, "topic_clarity"),
    ("essay_scoreT_cont", 1, "creativity"),
    ("essay_scoreT_cont", 3, "prompt_comprehension"),
]
EXCLUDED = [("essay_scoreT_cont", 2, "con_prompt")]

CAT_SUBITEMS = {
    "expression": ["grammar", "word_choice", "sentence_expression"],
    # corrected: 원본 4개 그대로. legacy: overall_structure(=consistency 복제) 추가
    "structure_corrected": ["paragraph_connection", "paragraph_structure",
                            "consistency", "length"],
    "structure_legacy": ["paragraph_connection", "paragraph_structure",
                         "consistency", "length", "consistency"],
    "content": ["topic_clarity", "creativity", "prompt_comprehension"],
}
CATEGORIES = ["expression", "structure", "content", "total"]
CAT_KR = {"expression": "표현", "structure": "구성", "content": "내용", "total": "총점"}


# ============================================================
# 지표 (experiments/e7l_optimized.py:44-50 과 동일 규약)
# ============================================================
def discretize(x):
    """0~3 연속값 -> 0..6 정수 (0.5단위 7범주). e7l compute_qwk와 동일."""
    return (np.clip(np.round(np.asarray(x, dtype=float) * 2) / 2, 0, 3) * 2).astype(int)


_K = 7
_IDX = np.arange(_K)
_W = ((_IDX[:, None] - _IDX[None, :]) ** 2) / ((_K - 1) ** 2)


def qwk_disc(a_d, b_d):
    """이산화된 0..6 정수 배열 두 개의 QWK. sklearn(quadratic, labels=0..6)과 동일."""
    n = len(a_d)
    if n < 5:
        return 0.0
    O = np.bincount(a_d * _K + b_d, minlength=_K * _K).reshape(_K, _K).astype(float)
    ha = np.bincount(a_d, minlength=_K).astype(float)
    hb = np.bincount(b_d, minlength=_K).astype(float)
    E = np.outer(ha, hb) / n
    denom = (_W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (_W * O).sum() / denom


def agreement_stats(a, b):
    """연속값 두 배열 -> QWK / within-0.5 / within-1.0 / pearson r"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    ad, bd = discretize(a), discretize(b)
    d = np.abs(ad - bd)
    r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    return {
        "qwk": qwk_disc(ad, bd),
        "within_half": float((d <= 1).mean()),   # 0.5점 이내
        "within_one": float((d <= 2).mean()),    # 1.0점 이내 (문헌 관행)
        "pearson_r": float(r) if np.isfinite(r) else None,
        "n": int(len(a)),
    }


def bootstrap_qwk_ci(pairs, rng, n_boot=N_BOOT, level=CI_LEVEL):
    """
    pairs: [(a_vec, b_vec), ...]  — 여러 쌍(1vs나머지 3회 등)을 평균하는 통계량
    에세이 단위 리샘플링. 모든 쌍에 동일 인덱스를 적용(paired).
    """
    n = len(pairs[0][0])
    disc = [(discretize(a), discretize(b)) for a, b in pairs]
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = np.mean([qwk_disc(ad[idx], bd[idx]) for ad, bd in disc])
    lo = float(np.percentile(vals, (1 - level) / 2 * 100))
    hi = float(np.percentile(vals, (1 + level) / 2 * 100))
    return lo, hi


# ============================================================
# 1. 원본 파싱
# ============================================================
def load_raw():
    zips = sorted(p for p in glob.glob(RAW_GLOB, recursive=True)
                  if os.path.basename(p).startswith(("TL_", "VL_")))
    print(f"  라벨링 zip: {[os.path.basename(p) for p in zips]}")
    recs = []
    bad_rater, bad_shape = 0, 0
    conprompt_nonzero = 0
    for zp in zips:
        z = zipfile.ZipFile(zp)
        for name in z.namelist():
            j = json.loads(z.read(name).decode("utf-8"))
            info = j.get("info") or {}
            eid = info.get("essay_id")
            if not eid:
                continue
            det = j["score"]["essay_scoreT_detail"]
            # 평가자 수 검증
            ns = {k: len(det[k]) for k in ("essay_scoreT_exp", "essay_scoreT_org",
                                           "essay_scoreT_cont")}
            if set(ns.values()) != {3}:
                bad_rater += 1
                continue
            # 항목 수 검증
            if (len(det["essay_scoreT_exp"][0]), len(det["essay_scoreT_org"][0]),
                    len(det["essay_scoreT_cont"][0])) != (3, 4, 4):
                bad_shape += 1
                continue
            for arr, i, _n in EXCLUDED:
                for rat in det[arr]:
                    if rat[i] != 0:
                        conprompt_nonzero += 1
            row = {"essay_id": eid,
                   "grade_level": (j.get("student") or {}).get("student_grade", "")}
            for arr, i, nm in SUBITEMS:
                for r in range(3):
                    row[f"{nm}__r{r}"] = float(det[arr][r][i])
            recs.append(row)
    df = pd.DataFrame(recs)
    print(f"  파싱 {len(df)}편 | 평가자수 이상 {bad_rater} | 항목수 이상 {bad_shape}")
    print(f"  con_prompt 비영(非零) 관측: {conprompt_nonzero}")
    return df, {"n_parsed": len(df), "bad_rater": bad_rater,
                "bad_shape": bad_shape, "conprompt_nonzero": conprompt_nonzero,
                "zips": [os.path.basename(p) for p in zips]}


# ============================================================
# 2. 평가자별 카테고리 점수
# ============================================================
def rater_category_scores(df, label_def):
    """label_def: 'corrected' | 'legacy'  -> {cat: (n,3) ndarray}"""
    skey = f"structure_{label_def}"
    out = {}
    for cat, cols in (("expression", CAT_SUBITEMS["expression"]),
                      ("structure", CAT_SUBITEMS[skey]),
                      ("content", CAT_SUBITEMS["content"])):
        m = np.stack([np.mean([df[f"{c}__r{r}"].values for c in cols], axis=0)
                      for r in range(3)], axis=1)
        out[cat] = m
    out["total"] = np.mean([out["expression"], out["structure"], out["content"]], axis=0)
    return out


def subitem_scores(df):
    return {nm: np.stack([df[f"{nm}__r{r}"].values for r in range(3)], axis=1)
            for _, _, nm in SUBITEMS}


# ============================================================
# 3. 두 가지 상한
# ============================================================
def one_vs_rest_pairs(M):
    """M: (n,3) -> [(rater_i, mean_of_other_two), x3]"""
    return [(M[:, i], np.mean(np.delete(M, i, axis=1), axis=1)) for i in range(3)]


def pairwise_pairs(M):
    return [(M[:, i], M[:, j]) for i, j in itertools.combinations(range(3), 2)]


def ceiling(M, rng, method):
    pairs = one_vs_rest_pairs(M) if method == "one_vs_rest" else pairwise_pairs(M)
    stats = [agreement_stats(a, b) for a, b in pairs]
    res = {k: float(np.mean([s[k] for s in stats]))
           for k in ("qwk", "within_half", "within_one")}
    rs = [s["pearson_r"] for s in stats if s["pearson_r"] is not None]
    res["pearson_r"] = float(np.mean(rs)) if rs else None
    res["n"] = int(len(M))
    res["ci_low"], res["ci_high"] = bootstrap_qwk_ci(pairs, rng)
    return res


# ============================================================
# 3-B. 선행연구(서경숙 2026) 대조용 비가중 지표: Fleiss κ / Cohen κ / ICC(3,k)
#      정수 세부항목 점수(0..3)에 그대로 적용. QWK와 달리 비가중.
# ============================================================
_CATS4 = [0, 1, 2, 3]


def fleiss_kappa(M):
    """M: (N,3) 정수 0..3. 세 채점자 명목 일치도(비가중)."""
    N, n = M.shape
    counts = np.array([[np.sum(M[i] == c) for c in _CATS4] for i in range(N)])
    Pi = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))
    Pbar = Pi.mean()
    pj = counts.sum(0) / (N * n)
    Pe = np.sum(pj ** 2)
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else 0.0


def cohen_kappa_unweighted(a, b):
    """두 채점자 비가중 Cohen κ."""
    n = len(a)
    O = np.array([[np.sum((a == i) & (b == j)) for j in _CATS4] for i in _CATS4]) / n
    r, c = O.sum(1), O.sum(0)
    Pe, Po = float(np.sum(r * c)), float(np.trace(O))
    return (Po - Pe) / (1 - Pe) if Pe < 1 else 0.0


def cohen_pairwise_mean(M):
    return float(np.mean([cohen_kappa_unweighted(M[:, i], M[:, j])
                          for i, j in itertools.combinations(range(3), 2)]))


def icc_3k(M):
    """ICC(3,k): two-way mixed, consistency, average measures. M:(N,k)."""
    N, k = M.shape
    gm = M.mean()
    SSR = k * np.sum((M.mean(1) - gm) ** 2)
    SSC = N * np.sum((M.mean(0) - gm) ** 2)
    SSE = np.sum((M - gm) ** 2) - SSR - SSC
    MSR = SSR / (N - 1)
    MSE = SSE / ((N - 1) * (k - 1))
    return float((MSR - MSE) / MSR) if MSR > 0 else 0.0


# ============================================================
# main
# ============================================================
def main():
    t0 = datetime.datetime.now()
    print("=" * 70)
    print("E9 — 인간 평가자 상한 측정")
    print("=" * 70)

    print("\n[1/6] 원본 파싱")
    raw, parse_meta = load_raw()

    print("\n[2/6] splits 대조")
    splits = {s: pd.read_csv(os.path.join(SPLITS_DIR, f"{s}.csv"))
              for s in ("train", "val", "test")}
    split_ids = {s: set(d["essay_id"]) for s, d in splits.items()}
    all_ids = set().union(*split_ids.values())
    matched = all_ids & set(raw["essay_id"])
    print(f"  splits 총 {len(all_ids)} | 원본 매칭 {len(matched)} | 미매칭 {len(all_ids - matched)}")
    assert len(all_ids - matched) == 0, "essay_id 미매칭 발생"

    raw = raw[raw["essay_id"].isin(all_ids)].reset_index(drop=True)
    raw["_scope_test"] = raw["essay_id"].isin(split_ids["test"])

    # --- legacy 라벨 재현 검증 (원 파이프라인과 동일해야 함) ---
    print("\n[3/6] legacy 라벨 재현 검증")
    leg = rater_category_scores(raw, "legacy")
    recon = pd.DataFrame({
        "essay_id": raw["essay_id"],
        "expression_score": leg["expression"].mean(axis=1),
        "structure_score": leg["structure"].mean(axis=1),
        "content_score": leg["content"].mean(axis=1),
        "total_score": leg["total"].mean(axis=1),
    })
    ref = pd.concat(splits.values(), ignore_index=True)[
        ["essay_id", "expression_score", "structure_score", "content_score", "total_score"]]
    mrg = recon.merge(ref, on="essay_id", suffixes=("_recon", "_ref"))
    recon_check = {}
    for c in ("expression_score", "structure_score", "content_score", "total_score"):
        d = np.abs(mrg[f"{c}_recon"] - mrg[f"{c}_ref"]).max()
        recon_check[c] = float(d)
        print(f"    {c:20s} 최대 절대차 {d:.2e}  {'OK' if d < 1e-9 else '★불일치'}")
    assert max(recon_check.values()) < 1e-9, "legacy 재현 실패 — 파싱 로직이 원 파이프라인과 다름"

    # --- corrected vs legacy 차이 ---
    cor = rater_category_scores(raw, "corrected")
    diff_rows = []
    for cat in ("structure", "total"):
        a = cor[cat].mean(axis=1)
        b = leg[cat].mean(axis=1)
        for scope, m in (("full", np.ones(len(raw), bool)), ("test", raw["_scope_test"].values)):
            diff_rows.append({
                "scope": scope, "category": cat, "n": int(m.sum()),
                "mean_corrected": float(a[m].mean()), "mean_legacy": float(b[m].mean()),
                "max_abs_diff": float(np.abs(a[m] - b[m]).max()),
                "pearson_r": float(np.corrcoef(a[m], b[m])[0, 1]),
                "n_grade_changed": int((discretize(a[m]) != discretize(b[m])).sum()),
            })
    pd.DataFrame(diff_rows).to_csv(
        os.path.join(OUTPUT_DIR, "e9_label_definition_diff.csv"), index=False)
    for r in diff_rows:
        if r["scope"] == "test":
            print(f"    [{r['category']}] test 등급변경 {r['n_grade_changed']}/{r['n']} "
                  f"({r['n_grade_changed']/r['n']*100:.1f}%) r={r['pearson_r']:.4f}")

    # --- 상한 계산 ---
    print(f"\n[4/6] 상한 계산 (부트스트랩 {N_BOOT}회)")
    rng = np.random.default_rng(RNG_SEED)
    subs = subitem_scores(raw)
    rows = []
    summary = {}

    for label_def in ("corrected", "legacy"):
        cats = rater_category_scores(raw, label_def)
        for scope, mask in (("test", raw["_scope_test"].values),
                            ("full", np.ones(len(raw), bool))):
            for method in ("one_vs_rest", "pairwise"):
                # 카테고리
                for cat in CATEGORIES:
                    res = ceiling(cats[cat][mask], rng, method)
                    rows.append(dict(scope=scope, label_definition=label_def,
                                     method=method, axis="category",
                                     axis_value=cat, **res))
                    summary.setdefault(label_def, {}).setdefault(
                        f"ceiling_{scope}", {}).setdefault(method, {})[cat] = res
                # 세부항목 (라벨 정의와 무관 -> corrected 루프에서만)
                if label_def == "corrected":
                    for _, _, nm in SUBITEMS:
                        res = ceiling(subs[nm][mask], rng, method)
                        rows.append(dict(scope=scope, label_definition=label_def,
                                         method=method, axis="subitem",
                                         axis_value=nm, **res))
            print(f"    [{label_def}/{scope}] 완료")

    # 학년별 (corrected, one_vs_rest 기준)
    print("\n[5/6] 학년별 분해")
    cats_c = rater_category_scores(raw, "corrected")
    grade_rows = []
    for g in sorted(raw["grade_level"].unique()):
        gm = (raw["grade_level"] == g).values
        for scope, sm in (("test", raw["_scope_test"].values & gm), ("full", gm)):
            if sm.sum() < 30:
                continue
            for cat in CATEGORIES:
                res = ceiling(cats_c[cat][sm], rng, "one_vs_rest")
                grade_rows.append(dict(scope=scope, label_definition="corrected",
                                       method="one_vs_rest", axis="grade",
                                       axis_value=f"{g}|{cat}", grade=g,
                                       category=cat, **res))
    pd.DataFrame(grade_rows).to_csv(os.path.join(OUTPUT_DIR, "e9_by_grade.csv"), index=False)

    # --- 3-B. 다중 신뢰도 지표 (서경숙 2026 대조용): Fleiss/Cohen/ICC + QWK ---
    # 전체 5,240편 기준(라벨 신뢰도 분석은 서경숙의 전체 데이터 접근과 맞춤).
    # 세부항목(0..3 정수) 수준만 — 서경숙의 루브릭 항목별 보고와 직접 대응.
    # 카테고리 점수는 항목 평균(연속값)이라 명목 지표(Fleiss/Cohen) 부적합하여 제외.
    print("\n[5.4/6] 다중 신뢰도 지표 (Fleiss/Cohen/ICC, 전체 5,240편)")

    def _qwk_ovr_full(M):  # 전체 데이터 one-vs-rest QWK 직접 산출
        return float(np.mean([qwk_disc(discretize(a), discretize(b))
                              for a, b in one_vs_rest_pairs(M)]))

    multimetric_rows = []
    for _, _, nm in SUBITEMS:
        M = subs[nm]  # (5240, 3) 전체
        multimetric_rows.append({
            "axis": "subitem", "item": nm, "n": int(M.shape[0]),
            "fleiss_kappa": fleiss_kappa(M),
            "cohen_kappa_pairwise": cohen_pairwise_mean(M),
            "icc_3k": icc_3k(M),
            "qwk_one_vs_rest": _qwk_ovr_full(M),
        })
    multi_df = pd.DataFrame(multimetric_rows)
    multi_df.to_csv(os.path.join(OUTPUT_DIR, "e9_reliability_multimetric.csv"), index=False)
    print("    " + " | ".join(f"{r['item'][:8]} F={r['fleiss_kappa']:.3f} "
                              f"ICC={r['icc_3k']:.3f}" for r in multimetric_rows[:3]) + " ...")

    # --- 분산 진단: QWK의 prevalence 민감성 (kappa paradox) ---
    # 점수 분산이 작으면 원 일치율이 높아도 QWK가 낮게 나온다. 항목 간 QWK 비교 시
    # 반드시 함께 봐야 하는 진단이다. (EDM 2023 kappa paradox 논의 참조)
    print("\n[5.5/6] 분산 진단 (kappa paradox)")
    df_all = pd.DataFrame(rows)
    var_rows = []
    tm = raw["_scope_test"].values
    for _, _, nm in SUBITEMS:
        M = subs[nm][tm]
        exact3 = float(((M[:, 0] == M[:, 1]) & (M[:, 1] == M[:, 2])).mean())
        q = float(df_all[(df_all.axis == "subitem") & (df_all.axis_value == nm) &
                         (df_all.scope == "test") &
                         (df_all.method == "one_vs_rest")]["qwk"].iloc[0])
        var_rows.append({"axis": "subitem", "item": nm, "n": int(M.shape[0]),
                         "mean_score": float(M.mean()), "sd_score": float(M.std()),
                         "exact_agree_3raters": exact3, "qwk_one_vs_rest": q})
    for cat in CATEGORIES:
        M = cats_c[cat][tm]
        q = float(df_all[(df_all.axis == "category") & (df_all.axis_value == cat) &
                         (df_all.scope == "test") & (df_all.label_definition == "corrected") &
                         (df_all.method == "one_vs_rest")]["qwk"].iloc[0])
        var_rows.append({"axis": "category", "item": cat, "n": int(M.shape[0]),
                         "mean_score": float(M.mean()), "sd_score": float(M.std()),
                         "exact_agree_3raters": float(
                             ((discretize(M[:, 0]) == discretize(M[:, 1])) &
                              (discretize(M[:, 1]) == discretize(M[:, 2]))).mean()),
                         "qwk_one_vs_rest": q})
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(os.path.join(OUTPUT_DIR, "e9_variance_diagnostic.csv"), index=False)

    # 지렛대점 민감도: 단일 이상항목이 상관을 만들어내는지 반드시 확인한다.
    _s = var_df[var_df.axis == "subitem"]
    _out = _s.loc[_s["sd_score"].idxmax(), "item"]
    _s2 = _s[_s["item"] != _out]
    sd_qwk = {
        "pearson_all": float(np.corrcoef(_s["sd_score"], _s["qwk_one_vs_rest"])[0, 1]),
        "pearson_excl_outlier": float(np.corrcoef(_s2["sd_score"], _s2["qwk_one_vs_rest"])[0, 1]),
        "spearman_all": float(pd.Series(_s["sd_score"].values).corr(
            pd.Series(_s["qwk_one_vs_rest"].values), method="spearman")),
        "outlier_item": _out,
        "n_items": int(len(_s)),
    }
    sd_qwk["robust"] = bool(
        np.sign(sd_qwk["pearson_all"]) == np.sign(sd_qwk["pearson_excl_outlier"])
        and abs(sd_qwk["spearman_all"]) >= 0.5)
    print(f"    SD ~ QWK  Pearson(전체) {sd_qwk['pearson_all']:+.4f} | "
          f"Pearson({_out} 제외) {sd_qwk['pearson_excl_outlier']:+.4f} | "
          f"Spearman {sd_qwk['spearman_all']:+.4f}")
    print(f"    -> 견고성: {'견고' if sd_qwk['robust'] else '★비견고 (단일 지렛대점 주도)'}")

    cols = ["scope", "label_definition", "axis", "axis_value", "n",
            "qwk", "ci_low", "ci_high", "within_half", "within_one", "pearson_r"]
    df_all[df_all["method"] == "one_vs_rest"][cols].to_csv(
        os.path.join(OUTPUT_DIR, "e9_one_vs_rest.csv"), index=False)
    df_all[df_all["method"] == "pairwise"][cols].to_csv(
        os.path.join(OUTPUT_DIR, "e9_pairwise.csv"), index=False)

    # --- 정상성 점검 ---
    ovr = summary["corrected"]["ceiling_test"]["one_vs_rest"]
    pw = summary["corrected"]["ceiling_test"]["pairwise"]
    direction_ok = all(ovr[c]["qwk"] >= pw[c]["qwk"] for c in CATEGORIES)

    # --- 요약 JSON ---
    print("\n[6/6] 산출물 기록")
    sub_tbl = {nm: {
        "one_vs_rest_qwk": float(df_all[(df_all.axis == "subitem") & (df_all.axis_value == nm) &
                                        (df_all.scope == "test") &
                                        (df_all.method == "one_vs_rest")]["qwk"].iloc[0]),
        "ci_low": float(df_all[(df_all.axis == "subitem") & (df_all.axis_value == nm) &
                               (df_all.scope == "test") &
                               (df_all.method == "one_vs_rest")]["ci_low"].iloc[0]),
        "ci_high": float(df_all[(df_all.axis == "subitem") & (df_all.axis_value == nm) &
                                (df_all.scope == "test") &
                                (df_all.method == "one_vs_rest")]["ci_high"].iloc[0]),
        "within_one": float(df_all[(df_all.axis == "subitem") & (df_all.axis_value == nm) &
                                   (df_all.scope == "test") &
                                   (df_all.method == "one_vs_rest")]["within_one"].iloc[0]),
    } for _, _, nm in SUBITEMS}

    tot = ovr["total"]["qwk"]
    verdict_band = ("천장 높음" if tot >= 0.70 else "천장 중간" if tot >= 0.55
                    else "천장 낮음" if tot >= 0.45 else "경보: 라벨 품질 재검토")
    best_sub = max(sub_tbl.items(), key=lambda kv: kv[1]["one_vs_rest_qwk"])
    worst_sub = min(sub_tbl.items(), key=lambda kv: kv[1]["one_vs_rest_qwk"])
    gap = best_sub[1]["one_vs_rest_qwk"] - worst_sub[1]["one_vs_rest_qwk"]

    out = {
        "timestamp": t0.isoformat(),
        "experiment": "e9_human_ceiling",
        "label_definition": "corrected",
        "data": {
            "source_zips": parse_meta["zips"],
            "n_essays_total": int(len(raw)),
            "n_essays_test": int(raw["_scope_test"].sum()),
            "n_raters": 3,
            "n_subitems_used": len(SUBITEMS),
            "n_subitems_excluded": len(EXCLUDED),
            "excluded_subitems": ["con_prompt (전수 0, 루브릭 가중치 0)"],
            "parse_anomalies": {"bad_rater_count": parse_meta["bad_rater"],
                                "bad_shape_count": parse_meta["bad_shape"],
                                "conprompt_nonzero": parse_meta["conprompt_nonzero"]},
        },
        "method": {
            "discretization": "clip(round(x*2)/2, 0, 3) * 2 -> 7 categories",
            "qwk": "quadratic weighted kappa, labels=0..6",
            "bootstrap_n": N_BOOT, "ci_level": CI_LEVEL,
            "bootstrap_unit": "essay", "rng_seed": RNG_SEED,
            "primary_statistic": "one_vs_rest",
            "within_half": "|diff| <= 0.5점", "within_one": "|diff| <= 1.0점",
        },
        "validation": {
            "legacy_label_reconstruction_max_abs_diff": recon_check,
            "essay_id_match": "5240/5240",
            "one_vs_rest_ge_pairwise": bool(direction_ok),
        },
        "ceiling_test": summary["corrected"]["ceiling_test"],
        "ceiling_full": summary["corrected"]["ceiling_full"],
        "by_subitem": sub_tbl,
        "reliability_multimetric": {
            "note": ("서경숙(2026) 논술형 데이터 보고 지표와 직접 대조용. "
                     "세부항목(0..3 정수) 수준. Fleiss/Cohen은 비가중 명목, "
                     "ICC(3,k)는 평균측정 급내상관, QWK는 2차가중."),
            "scope": "full (5,240)",
            "items": {r["item"]: {k: round(r[k], 4) for k in
                                  ("fleiss_kappa", "cohen_kappa_pairwise",
                                   "icc_3k", "qwk_one_vs_rest")}
                      for r in multimetric_rows},
        },
        "by_grade": {
            f"{r['grade']}|{r['category']}": {"qwk": r["qwk"], "ci_low": r["ci_low"],
                                              "ci_high": r["ci_high"], "n": r["n"]}
            for r in grade_rows if r["scope"] == "test"
        },
        "sensitivity_legacy_label": {
            "ceiling_test_legacy": summary["legacy"]["ceiling_test"],
            "label_diff": diff_rows,
        },
        "variance_diagnostic": {
            "note": ("QWK는 우연 일치를 차감하므로 점수 분산의 영향을 받는다(kappa paradox). "
                     "다만 아래 견고성 검정 결과를 반드시 함께 볼 것 — "
                     "단일 지렛대점이 상관을 만들어낼 수 있다."),
            "sd_vs_qwk": sd_qwk,
            "interpretation": (
                f"{sd_qwk['outlier_item']}은 양극(0/3) 분포로 SD가 홀로 크고 QWK도 홀로 높다. "
                f"이 1개 항목을 빼면 상관 부호가 뒤집히므로(전체 {sd_qwk['pearson_all']:+.3f} "
                f"-> 제외 {sd_qwk['pearson_excl_outlier']:+.3f}), "
                "'분산이 QWK를 결정한다'로 일반화하면 안 된다. "
                "나머지 9개 항목은 SD가 사실상 동일한데도 QWK가 0.30~0.50으로 흩어진다."
                if not sd_qwk["robust"] else
                "분산과 QWK의 관계가 지렛대점 제거 후에도 유지된다."),
            "items": var_rows,
        },
        "e9_verdict": {
            "total_ceiling_one_vs_rest": round(tot, 4),
            "total_ceiling_ci": [round(ovr["total"]["ci_low"], 4),
                                 round(ovr["total"]["ci_high"], 4)],
            "band": verdict_band,
            "highest_subitem": f"{best_sub[0]} ({best_sub[1]['one_vs_rest_qwk']:.4f})",
            "lowest_subitem": f"{worst_sub[0]} ({worst_sub[1]['one_vs_rest_qwk']:.4f})",
            "subitem_gap": round(gap, 4),
            "implication_for_h5": ("항목별 상한 격차 0.2 이상 -> H5는 항목별로만 검정. 평균 비교 금지"
                                   if gap >= 0.2 else
                                   "항목 간 격차 작음 -> 평균 비교도 보조적으로 허용 가능"),
        },
    }
    with open(os.path.join(OUTPUT_DIR, "e9_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # --- 그림 ---
    make_figure(var_df, sub_tbl, ovr)

    # --- 콘솔 요약 ---
    print("\n" + "=" * 70)
    print("E9 결과 요약 (test 786건, corrected 라벨, one-vs-rest)")
    print("=" * 70)
    for c in CATEGORIES:
        r = ovr[c]
        print(f"  {CAT_KR[c]:4s}  QWK {r['qwk']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
              f"   within-1.0 {r['within_one']*100:5.1f}%   r={r['pearson_r']:.4f}")
    print(f"\n  최고 항목: {best_sub[0]} {best_sub[1]['one_vs_rest_qwk']:.4f}")
    print(f"  최저 항목: {worst_sub[0]} {worst_sub[1]['one_vs_rest_qwk']:.4f}   (격차 {gap:.4f})")
    print(f"\n  판정: {verdict_band}")
    print(f"  방향성 점검(one_vs_rest >= pairwise): {'OK' if direction_ok else '★실패'}")
    print(f"\n  소요 {(datetime.datetime.now() - t0).total_seconds():.1f}초")
    print(f"  산출물 -> {OUTPUT_DIR}")


def make_figure(var_df, sub_tbl, ovr):
    """
    좌: 항목별 인간 상한 (SD로 음영 — kappa paradox 가시화)
    우: 점수 SD 대 QWK 산점도
    한글 폰트가 보장되지 않으므로 영문 라벨을 쓴다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_en = {
        "grammar": "Grammar", "word_choice": "Word choice",
        "sentence_expression": "Sentence expr.", "paragraph_connection": "Para. connection",
        "paragraph_structure": "Para. structure", "consistency": "Consistency",
        "length": "Length", "topic_clarity": "Topic clarity",
        "creativity": "Creativity", "prompt_comprehension": "Prompt comprehension",
        "expression": "[CAT] Expression", "structure": "[CAT] Structure",
        "content": "[CAT] Content", "total": "[CAT] TOTAL",
    }
    sd_of = dict(zip(var_df["item"], var_df["sd_score"]))

    items = [(k, labels_en[k], v["one_vs_rest_qwk"], v["ci_low"], v["ci_high"], False)
             for k, v in sub_tbl.items()]
    items += [(c, labels_en[c], ovr[c]["qwk"], ovr[c]["ci_low"], ovr[c]["ci_high"], True)
              for c in CATEGORIES]
    items.sort(key=lambda x: x[2])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.5),
                                  gridspec_kw={"width_ratios": [1.6, 1]})
    cmap = plt.get_cmap("viridis")
    sds = [sd_of[i[0]] for i in items]
    norm = plt.Normalize(min(sds), max(sds))
    y = np.arange(len(items))
    vals = [i[2] for i in items]
    err = [[i[2] - i[3] for i in items], [i[4] - i[2] for i in items]]
    bars = ax.barh(y, vals, xerr=err, color=[cmap(norm(s)) for s in sds],
                   edgecolor=["black" if i[5] else "none" for i in items],
                   linewidth=[1.6 if i[5] else 0 for i in items],
                   error_kw={"ecolor": "#333", "capsize": 3, "lw": 1})
    for yi, (it, sd) in enumerate(zip(items, sds)):
        ax.text(1.01, yi, f"SD {sd:.2f}", va="center", fontsize=8, color="#555")
    ax.set_yticks(y)
    ax.set_yticklabels([i[1] for i in items], fontsize=9)
    ax.set_xlabel("Human inter-rater QWK (one-vs-rest), test n=786")
    ax.set_title("E9 — Human ceiling by rubric item (95% bootstrap CI)\n"
                 "bold outline = category; color = score SD", fontsize=10)
    ax.set_xlim(0, 1.18)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="x", alpha=0.3)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="score SD", fraction=0.03, pad=0.10)

    sub = var_df[var_df.axis == "subitem"]
    out_item = sub.loc[sub["sd_score"].idxmax(), "item"]
    sub2 = sub[sub["item"] != out_item]
    ax2.scatter(sub["sd_score"], sub["qwk_one_vs_rest"], s=70,
                c=sub["sd_score"], cmap=cmap, norm=norm, edgecolor="k", zorder=3)
    for _, r in sub.iterrows():
        ax2.annotate(labels_en[r["item"]], (r["sd_score"], r["qwk_one_vs_rest"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    r_all = np.corrcoef(sub["sd_score"], sub["qwk_one_vs_rest"])[0, 1]
    r_ex = np.corrcoef(sub2["sd_score"], sub2["qwk_one_vs_rest"])[0, 1]
    rho = pd.Series(sub["sd_score"].values).corr(
        pd.Series(sub["qwk_one_vs_rest"].values), method="spearman")
    # 지렛대점 강조
    ax2.scatter(sub.loc[sub["item"] == out_item, "sd_score"],
                sub.loc[sub["item"] == out_item, "qwk_one_vs_rest"],
                s=260, facecolor="none", edgecolor="crimson", lw=2, zorder=4)
    ax2.annotate("leverage point", (sub.loc[sub["item"] == out_item, "sd_score"].iloc[0],
                                    sub.loc[sub["item"] == out_item, "qwk_one_vs_rest"].iloc[0]),
                 textcoords="offset points", xytext=(-18, -26), fontsize=8, color="crimson")
    ax2.set_xlabel("Score SD (across all raters)")
    ax2.set_ylabel("Human ceiling QWK")
    ax2.set_title("Variance does NOT explain the QWK spread\n"
                  f"Pearson all={r_all:+.3f} | excl. leverage pt={r_ex:+.3f} | "
                  f"Spearman={rho:+.3f}", fontsize=10)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_ceiling_by_item.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
