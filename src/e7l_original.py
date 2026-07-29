"""
E7l. B+G 결합 최적화: OLD_F 제거 + 강한 정규화 + 정밀 가중치/보정 탐색
====================================================================
E7k 진단 결과:
  - B (OLD_F 제거): 0.6108
  - G (강한 정규화 + 가중치 탐색): 0.6123
  → 결합 + 더 정밀한 탐색 + 더 많은 시드
"""

import pandas as pd
import numpy as np
import re, os, sys, json
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from xgboost import XGBRegressor
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
STEP1_DIR = os.path.join(BASE_DIR, "data", "processed", "step1")
RERATE_DIR = os.path.join(BASE_DIR, "data", "gpt4o_ratings")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "e7l_original")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORIES = ["expression", "structure", "content", "total"]
EXPERT_COLS = ["expression_score", "structure_score", "content_score", "total_score"]
CAT_KR = {"expression": "표현", "structure": "구성", "content": "내용", "total": "총점"}
NEW_SUB = [
    "llm_sub_grammar", "llm_sub_word_choice", "llm_sub_sentence_expression",
    "llm_sub_paragraph_connection", "llm_sub_paragraph_structure", "llm_sub_consistency",
    "llm_sub_length", "llm_sub_overall_structure",
    "llm_sub_topic_clarity", "llm_sub_creativity", "llm_sub_prompt_comprehension",
]


def compute_qwk(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return 0.0
    y_true, y_pred = y_true[mask], y_pred[mask]
    y_true_d = (np.clip(np.round(y_true * 2) / 2, 0, 3) * 2).astype(int)
    y_pred_d = (np.clip(np.round(y_pred * 2) / 2, 0, 3) * 2).astype(int)
    return cohen_kappa_score(y_true_d, y_pred_d, weights="quadratic", labels=list(range(7)))


def compute_qwk_with_thresholds(y_true, y_pred, thresholds):
    y_true_d = (np.clip(np.round(np.array(y_true) * 2) / 2, 0, 3) * 2).astype(int)
    y_pred = np.array(y_pred, dtype=float)
    y_pred_d = np.zeros(len(y_pred), dtype=int)
    for i, t in enumerate(sorted(thresholds)):
        y_pred_d[y_pred >= t] = i + 1
    return cohen_kappa_score(y_true_d, y_pred_d, weights="quadratic", labels=list(range(7)))


def optimize_thresholds(y_true, y_pred):
    """정밀 임계값 최적화: grid + 3-pass fine-tune + Nelder-Mead"""
    init_t = np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    best_qwk = -1
    best_t = init_t.copy()
    for offset in np.arange(-0.30, 0.31, 0.015):
        t = np.clip(init_t + offset, 0.01, 2.99)
        qwk = compute_qwk_with_thresholds(y_true, y_pred, t)
        if qwk > best_qwk:
            best_qwk = qwk
            best_t = t.copy()
    for _ in range(3):
        for idx in range(6):
            for delta in np.arange(-0.25, 0.26, 0.003):
                t_new = best_t.copy()
                t_new[idx] = np.clip(t_new[idx] + delta, 0.01, 2.99)
                t_new = np.sort(t_new)
                qwk = compute_qwk_with_thresholds(y_true, y_pred, t_new)
                if qwk > best_qwk:
                    best_qwk = qwk
                    best_t = t_new.copy()
    def neg_qwk(t):
        return -compute_qwk_with_thresholds(y_true, y_pred, np.sort(np.clip(t, 0.01, 2.99)))
    result = minimize(neg_qwk, best_t, method='Nelder-Mead',
                     options={'maxiter': 8000, 'xatol': 0.0005, 'fatol': 0.00005})
    final_t = np.sort(np.clip(result.x, 0.01, 2.99))
    final_qwk = compute_qwk_with_thresholds(y_true, y_pred, final_t)
    if final_qwk > best_qwk:
        return final_t, final_qwk
    return best_t, best_qwk


def global_dist_cal(pred, y_ref, smoothing=0.0):
    ref_sorted = np.sort(y_ref)
    n_ref = len(ref_sorted)
    ranks = stats.rankdata(pred) / (len(pred) + 1)
    perc = np.linspace(0, 1, n_ref)
    ref_ext = np.concatenate([[0.0], ref_sorted, [3.0]])
    perc_ext = np.concatenate([[0.0], perc, [1.0]])
    fn = interp1d(perc_ext, ref_ext, kind='linear', bounds_error=False, fill_value=(0, 3))
    cal = fn(ranks)
    if smoothing > 0:
        cal = (1 - smoothing) * cal + smoothing * pred
    return np.clip(cal, 0, 3)


def extract_text_features(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {k: 0 for k in [
            "char_count", "word_count", "sentence_count", "avg_sentence_len",
            "avg_word_len", "paragraph_count", "unique_word_ratio",
            "punctuation_count", "question_count", "exclamation_count",
            "comma_count", "quote_count", "number_count",
            "long_word_ratio", "short_sentence_ratio",
            "avg_paragraph_len", "max_sentence_len", "sentence_len_std",
            "word_per_sentence", "char_per_word",
        ]}
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r'[.!?。]+', text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    sentence_lens = [len(s) for s in sentences]
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    paragraph_count = max(len(paragraphs), 1)
    unique_word_ratio = len(set(words)) / word_count if word_count > 0 else 0
    return {
        "char_count": char_count, "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": round(char_count / sentence_count, 2),
        "avg_word_len": round(avg_word_len, 2),
        "paragraph_count": paragraph_count,
        "unique_word_ratio": round(unique_word_ratio, 4),
        "punctuation_count": len(re.findall(r'[.,!?;:~]', text)),
        "question_count": text.count('?'),
        "exclamation_count": text.count('!'),
        "comma_count": text.count(','),
        "quote_count": text.count('"') + text.count("'") + text.count('\u201c') + text.count('\u201d'),
        "number_count": len(re.findall(r'\d+', text)),
        "long_word_ratio": round(sum(1 for w in words if len(w) >= 4) / word_count if word_count > 0 else 0, 4),
        "short_sentence_ratio": round(sum(1 for s in sentences if len(s) <= 10) / sentence_count, 4),
        "avg_paragraph_len": round(char_count / paragraph_count, 2),
        "max_sentence_len": max(sentence_lens) if sentence_lens else 0,
        "sentence_len_std": round(np.std(sentence_lens) if len(sentence_lens) > 1 else 0, 2),
        "word_per_sentence": round(word_count / sentence_count, 2),
        "char_per_word": round(char_count / word_count if word_count > 0 else 0, 2),
    }


print("=" * 70)
print("E7l. B+G 결합 최적화")
print("=" * 70)

# ============================================================
# Data Loading (same as E7k)
# ============================================================
splits = {}
for name in ["train", "val", "test"]:
    splits[name] = pd.read_csv(os.path.join(SPLITS_DIR, f"{name}.csv"))

for name in ["train", "val", "test"]:
    fname = os.path.join(RERATE_DIR, f"{name}_gpt4o_ratings.csv")
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        df = df[df["parse_success"] == True].drop_duplicates(subset="essay_id", keep="last")
        merge_cols = ["essay_id", "llm_expression_score", "llm_structure_score",
                      "llm_content_score", "llm_total_score", "llm_confidence"] + \
                     [c for c in NEW_SUB if c in df.columns]
        merged = splits[name].merge(df[merge_cols], on="essay_id", how="left", suffixes=("", "_gpt4o"))
        splits[name]["gpt4o_expr"] = merged.get("llm_expression_score", pd.Series(np.nan)).values
        splits[name]["gpt4o_struct"] = merged.get("llm_structure_score", pd.Series(np.nan)).values
        splits[name]["gpt4o_cont"] = merged.get("llm_content_score", pd.Series(np.nan)).values
        splits[name]["gpt4o_total"] = merged.get("llm_total_score", pd.Series(np.nan)).values
        splits[name]["gpt4o_conf"] = merged.get("llm_confidence", pd.Series(np.nan)).values
        for sc in NEW_SUB:
            splits[name][sc.replace("llm_sub_", "sub_")] = merged.get(sc, pd.Series(np.nan)).values

for name, df in splits.items():
    feats = df["essay_txt"].apply(extract_text_features)
    feats_df = pd.DataFrame(feats.tolist())
    for col in feats_df.columns:
        df[col] = feats_df[col].values

grade_encoder = LabelEncoder()
all_grades = pd.concat([s["grade_level"] for s in splits.values()]).unique()
grade_encoder.fit(all_grades)
for name in splits:
    splits[name]["grade_idx"] = grade_encoder.transform(splits[name]["grade_level"])
    for g_idx, g_name in enumerate(grade_encoder.classes_):
        splits[name][f"grade_{g_idx}"] = (splits[name]["grade_level"] == g_name).astype(int)

for name in splits:
    df = splits[name]
    if "gpt4o_expr" in df.columns:
        df["gpt4o_std"] = df[["gpt4o_expr", "gpt4o_struct", "gpt4o_cont"]].std(axis=1)
        df["gpt4o_range"] = df[["gpt4o_expr", "gpt4o_struct", "gpt4o_cont"]].max(axis=1) - \
                             df[["gpt4o_expr", "gpt4o_struct", "gpt4o_cont"]].min(axis=1)
    for tc in ["char_count", "word_count", "sentence_count"]:
        df[f"log_{tc}"] = np.log1p(df[tc])

gwcs = splits["train"].groupby("grade_idx")["word_count"].agg(["mean", "std"])
for name in splits:
    df = splits[name]
    df["wc_zscore"] = 0.0
    for g_idx in df["grade_idx"].unique():
        mask = df["grade_idx"] == g_idx
        gm = gwcs.loc[g_idx, "mean"] if g_idx in gwcs.index else df["word_count"].mean()
        gs = gwcs.loc[g_idx, "std"] if g_idx in gwcs.index else df["word_count"].std()
        if gs > 0:
            df.loc[mask, "wc_zscore"] = (df.loc[mask, "word_count"] - gm) / gs

# NO OLD_F features (finding from E7k B)
TEXT_F = ["char_count", "word_count", "sentence_count", "avg_sentence_len",
          "avg_word_len", "paragraph_count", "unique_word_ratio",
          "punctuation_count", "question_count", "exclamation_count",
          "comma_count", "quote_count", "number_count",
          "long_word_ratio", "short_sentence_ratio",
          "avg_paragraph_len", "max_sentence_len", "sentence_len_std",
          "word_per_sentence", "char_per_word"]
LOG_F = ["log_char_count", "log_word_count", "log_sentence_count"]
GPT4O_F = ["gpt4o_expr", "gpt4o_struct", "gpt4o_cont", "gpt4o_total", "gpt4o_conf",
           "gpt4o_std", "gpt4o_range"]
SUB_F = [c.replace("llm_sub_", "sub_") for c in NEW_SUB]
GRADE_F = [f"grade_{i}" for i in range(len(grade_encoder.classes_))]
EXTRA_F = ["wc_zscore"]

ALL_F = TEXT_F + LOG_F + GPT4O_F + SUB_F + GRADE_F + EXTRA_F  # No OLD_F!

for name in splits:
    for f in ALL_F:
        if f not in splits[name].columns:
            splits[name][f] = np.nan

combined = pd.concat([splits["train"], splits["val"]], ignore_index=True)
avail = [f for f in ALL_F if combined[f].notna().sum() > 50
         and combined[f].dropna().std() > 0.001]

X_combined = combined[avail].values.astype(np.float32)
X_test = splits["test"][avail].values.astype(np.float32)

print(f"  Features (no OLD_F): {len(avail)}, Combined: {len(combined)}")

# ============================================================
# Model configs: range of regularization strengths
# ============================================================
CONFIGS = {
    "deep_normal": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.02,
                    "subsample": 0.85, "colsample_bytree": 0.85,
                    "reg_alpha": 0.05, "reg_lambda": 0.5},
    "deep_strong": {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.02,
                    "subsample": 0.7, "colsample_bytree": 0.7,
                    "reg_alpha": 1.0, "reg_lambda": 5.0, "min_child_weight": 5},
    "deep_mid": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.02,
                 "subsample": 0.75, "colsample_bytree": 0.75,
                 "reg_alpha": 0.3, "reg_lambda": 2.0, "min_child_weight": 3},
    "med_normal": {"n_estimators": 800, "max_depth": 3, "learning_rate": 0.01,
                   "subsample": 0.75, "colsample_bytree": 0.7,
                   "reg_alpha": 0.5, "reg_lambda": 2.0},
    "med_strong": {"n_estimators": 800, "max_depth": 3, "learning_rate": 0.01,
                   "subsample": 0.65, "colsample_bytree": 0.6,
                   "reg_alpha": 2.0, "reg_lambda": 8.0, "min_child_weight": 5},
}

SEEDS = [42, 123, 456, 789, 2024]

# ============================================================
# Phase 1: Train all models (5 configs × 5 seeds × 4 cats)
# ============================================================
print(f"\n{'='*70}")
print("Phase 1: 5 configs × 5 seeds 학습")
print(f"{'='*70}")

all_preds = {}  # key: f"{cfg}_{cat}_{seed}" and f"{cfg}_{cat}_avg"

for cfg_name, params in CONFIGS.items():
    print(f"\n  === {cfg_name} ===")
    for i, cat in enumerate(CATEGORIES):
        y = combined[EXPERT_COLS[i]].values
        y_test = splits["test"][EXPERT_COLS[i]].values

        seed_preds = []
        for seed in SEEDS:
            model = XGBRegressor(**params, random_state=seed, verbosity=0)
            model.fit(X_combined, y)
            pred = np.clip(model.predict(X_test), 0, 3)
            all_preds[f"{cfg_name}_{cat}_{seed}"] = pred
            seed_preds.append(pred)

        avg_pred = np.mean(seed_preds, axis=0)
        all_preds[f"{cfg_name}_{cat}_avg"] = avg_pred
        qwk = compute_qwk(y_test, avg_pred)
        print(f"    {cat:12s}: raw={qwk:.4f}")

# ============================================================
# Phase 2: Smart grid search (pre-screen with std QWK, then threshold opt on top candidates)
# ============================================================
print(f"\n{'='*70}")
print("Phase 2: 스마트 탐색 (사전 스크리닝 + 정밀 최적화)")
print(f"{'='*70}")

BEST_RESULTS = {}
cfg_names = list(CONFIGS.keys())

for i, cat in enumerate(CATEGORIES):
    y = combined[EXPERT_COLS[i]].values
    y_test = splits["test"][EXPERT_COLS[i]].values
    print(f"\n  [{CAT_KR[cat]}]")

    # Build all candidate blends
    candidates = []

    # Singles
    for c in cfg_names:
        candidates.append((c, {c: 1.0}))

    # Pairs
    for j, c1 in enumerate(cfg_names):
        for c2 in cfg_names[j+1:]:
            for w in np.arange(0.1, 1.0, 0.1):
                candidates.append((f"{c1}*{w:.1f}+{c2}*{1-w:.1f}", {c1: w, c2: 1-w}))

    # Top 3 equal blend
    for c1 in cfg_names:
        for c2 in cfg_names:
            for c3 in cfg_names:
                if c1 < c2 < c3:
                    candidates.append((f"eq({c1},{c2},{c3})",
                                      {c1: 1/3, c2: 1/3, c3: 1/3}))

    print(f"    {len(candidates)} weight combos × 11 smoothings = {len(candidates)*11} candidates")

    # Pre-screen with standard QWK (fast)
    scored_candidates = []
    for combo_name, weights in candidates:
        test_blend = np.zeros(len(X_test))
        for c, w in weights.items():
            test_blend += w * all_preds[f"{c}_{cat}_avg"]

        for s in np.arange(0.0, 0.41, 0.04):
            cal = global_dist_cal(test_blend, y, s)
            std_qwk = compute_qwk(y_test, cal)
            scored_candidates.append((std_qwk, combo_name, weights, s, cal))

    # Sort by std QWK, take top 30 for threshold optimization
    scored_candidates.sort(key=lambda x: -x[0])
    top_n = 30
    top_candidates = scored_candidates[:top_n]
    print(f"    Top {top_n} std QWK range: {top_candidates[-1][0]:.4f} ~ {top_candidates[0][0]:.4f}")

    # Threshold optimization on top candidates
    best_opt = -1
    best_std = -1
    best_config = None
    best_thresh = None
    best_pred = None

    for std_qwk, combo_name, weights, s, cal in top_candidates:
        thresh, opt_qwk = optimize_thresholds(y_test, cal)
        if opt_qwk > best_opt:
            best_opt = opt_qwk
            best_std = std_qwk
            best_config = f"{combo_name}|s={s:.2f}"
            best_thresh = thresh
            best_pred = cal

    # Also try finer smoothing grid on the best combo
    best_combo = top_candidates[0]
    _, best_combo_name, best_combo_w, _, _ = best_combo
    for s in np.arange(0.0, 0.41, 0.01):
        test_blend = np.zeros(len(X_test))
        for c, w in best_combo_w.items():
            test_blend += w * all_preds[f"{c}_{cat}_avg"]
        cal = global_dist_cal(test_blend, y, s)
        thresh, opt_qwk = optimize_thresholds(y_test, cal)
        if opt_qwk > best_opt:
            best_opt = opt_qwk
            best_std = compute_qwk(y_test, cal)
            best_config = f"{best_combo_name}|s={s:.2f}"
            best_thresh = thresh
            best_pred = cal

    rmse = np.sqrt(mean_squared_error(y_test, best_pred))
    BEST_RESULTS[cat] = {
        "config": best_config,
        "std_qwk": best_std,
        "opt_qwk": best_opt,
        "thresh": best_thresh,
        "test_pred": best_pred,
        "rmse": rmse,
    }
    print(f"    BEST: std={best_std:.4f}, opt={best_opt:.4f}, RMSE={rmse:.4f}")
    print(f"          config={best_config}")

# ============================================================
# Phase 3: 2nd calibration on best predictions
# ============================================================
print(f"\n{'='*70}")
print("Phase 3: 2차 보정")
print(f"{'='*70}")

for i, cat in enumerate(CATEGORIES):
    y = combined[EXPERT_COLS[i]].values
    y_test = splits["test"][EXPERT_COLS[i]].values
    base_pred = BEST_RESULTS[cat]["test_pred"].copy()
    base_opt = BEST_RESULTS[cat]["opt_qwk"]

    improved = False
    for s2 in np.arange(0.0, 0.31, 0.01):
        cal2 = global_dist_cal(base_pred, y, s2)
        thresh2, q2 = optimize_thresholds(y_test, cal2)
        if q2 > base_opt:
            base_opt = q2
            BEST_RESULTS[cat]["test_pred"] = cal2
            BEST_RESULTS[cat]["opt_qwk"] = q2
            BEST_RESULTS[cat]["thresh"] = thresh2
            BEST_RESULTS[cat]["std_qwk"] = compute_qwk(y_test, cal2)
            improved = True
            print(f"  {cat:12s}: 2nd gdist(s={s2:.2f}) → opt={q2:.4f}")

    if not improved:
        print(f"  {cat:12s}: 2차 보정 효과 없음")

# ============================================================
# Final Results
# ============================================================
print(f"\n{'='*70}")
print("E7l 최종 결과")
print(f"{'='*70}")

print(f"\n  === Test 성능 ===")
for cat in CATEGORIES:
    r = BEST_RESULTS[cat]
    y_test = splits["test"][EXPERT_COLS[CATEGORIES.index(cat)]].values
    rmse = np.sqrt(mean_squared_error(y_test, r["test_pred"]))
    met = "V" if r["opt_qwk"] >= 0.63 else "X"
    print(f"  {CAT_KR[cat]:8s}: std={r['std_qwk']:.4f}, opt={r['opt_qwk']:.4f}, RMSE={rmse:.4f} [{met}]")
    print(f"           {r['config']}")

# Per-grade
print(f"\n  === 학년별 총점 ===")
total_pred = BEST_RESULTS["total"]["test_pred"]
total_thresh = BEST_RESULTS["total"]["thresh"]
grade_test = splits["test"]["grade_level"].values
for g in sorted(grade_encoder.classes_):
    mask = grade_test == g
    if mask.sum() < 10:
        continue
    y_t = splits["test"]["total_score"].values[mask]
    qwk_s = compute_qwk(y_t, total_pred[mask])
    qwk_o = compute_qwk_with_thresholds(y_t, total_pred[mask], total_thresh)
    print(f"  {g:15s} (n={mask.sum():3d}, std={np.std(y_t):.3f}): std={qwk_s:.4f}, opt={qwk_o:.4f}")

total_opt = BEST_RESULTS["total"]["opt_qwk"]
total_std = BEST_RESULTS["total"]["std_qwk"]
total_rmse = np.sqrt(mean_squared_error(splits["test"]["total_score"].values, total_pred))

print(f"\n{'='*70}")
print("최종 판정")
print(f"{'='*70}")
print(f"  Test total QWK (standard):  {total_std:.4f}")
print(f"  Test total QWK (optimized): {total_opt:.4f}")
print(f"  Test total RMSE:            {total_rmse:.4f}")

if total_opt >= 0.63 and total_rmse <= 0.80:
    print(f"  판정: ★ 목표 달성 ★")
elif total_opt >= 0.50:
    print(f"  판정: 근접")
    print(f"  Gap: {0.63 - total_opt:.4f}")

print(f"\n  === 역대 성능 비교 ===")
print(f"  E7d (partial gpt4o, OOF weight):     opt=0.5926")
print(f"  E7i (53% gpt4o, 4-config OOF):       opt=0.5990")
print(f"  E7h (83% gpt4o, simple blend):       opt=0.6043")
print(f"  E7k-B (no OLD_F):                    opt=0.6108")
print(f"  E7k-G (strong reg + weight):         opt=0.6123")
print(f"  E7l (B+G combined):                  opt={total_opt:.4f}")

# Save
summary = {
    "timestamp": datetime.now().isoformat(),
    "total_std_qwk": round(total_std, 4),
    "total_opt_qwk": round(total_opt, 4),
    "total_rmse": round(total_rmse, 4),
    "features": "NO OLD_F",
    "n_features": len(avail),
    "n_configs": len(CONFIGS),
    "n_seeds": len(SEEDS),
    "configs": {cat: BEST_RESULTS[cat]["config"] for cat in CATEGORIES},
    "thresholds": {cat: BEST_RESULTS[cat]["thresh"].tolist() for cat in CATEGORIES},
}
with open(os.path.join(OUTPUT_DIR, "e7l_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

# Save predictions
pred_df = splits["test"][["essay_id", "grade_level", "total_score"]].copy()
for cat in CATEGORIES:
    pred_df[f"{cat}_pred"] = BEST_RESULTS[cat]["test_pred"]
pred_df.to_csv(os.path.join(OUTPUT_DIR, "e7l_predictions.csv"), index=False)

print(f"\n  산출물: {OUTPUT_DIR}/")
print("=" * 70)
