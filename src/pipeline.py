"""
E9~E11 공용 파이프라인 모듈 (실험 스크립트 아님 — 접두사 _ 로 구분)

목적:
    E10(요인설계)과 E11(정직한 재평가)이 **완전히 동일한** 피처 생성·라벨 정의·
    지표 계산을 쓰도록 보장한다. 복사-붙여넣기로 두 실험이 갈라지면 비교가 무의미해진다.

피처 생성 로직은 experiments/e7l_optimized.py 를 그대로 옮긴 것이다.
(E11이 E7l과 동일 구성이어야 "프로토콜만 바꿨다"는 논증이 성립하므로)

라벨 정의:
    corrected — overall_structure(=consistency 복제) 제거. **주 지표**
    legacy    — 기존 splits/*.csv 그대로. 민감도 분석용
    상세: docs/e9-human-ceiling.md §4.2
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
RERATE_DIR = os.path.join(BASE_DIR, "data", "gpt4o_ratings")

NEW_SUB = [
    "llm_sub_grammar", "llm_sub_word_choice", "llm_sub_sentence_expression",
    "llm_sub_paragraph_connection", "llm_sub_paragraph_structure", "llm_sub_consistency",
    "llm_sub_length", "llm_sub_overall_structure",
    "llm_sub_topic_clarity", "llm_sub_creativity", "llm_sub_prompt_comprehension",
]

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
GRADE_F = [f"grade_{i}" for i in range(6)]
EXTRA_F = ["wc_zscore"]

CATEGORIES = ["expression", "structure", "content", "total"]
CAT_KR = {"expression": "표현", "structure": "구성", "content": "내용", "total": "총점"}
EXPERT_COLS_LEGACY = ["expression_score", "structure_score", "content_score", "total_score"]
EXPERT_COLS = ["expression_score", "structure_score_c", "content_score", "total_score_c"]


# ============================================================
# 텍스트 특징 (e7l_optimized.py 와 동일)
# ============================================================
_ZERO_FEATS = {k: 0 for k in TEXT_F}


def extract_text_features(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return dict(_ZERO_FEATS)
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?。]+", text) if s.strip()]
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
        "punctuation_count": len(re.findall(r"[.,!?;:~]", text)),
        "question_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "comma_count": text.count(","),
        "quote_count": (text.count('"') + text.count("'")
                        + text.count("“") + text.count("”")),
        "number_count": len(re.findall(r"\d+", text)),
        "long_word_ratio": round(sum(1 for w in words if len(w) >= 4) / word_count
                                 if word_count > 0 else 0, 4),
        "short_sentence_ratio": round(sum(1 for s in sentences if len(s) <= 10)
                                      / sentence_count, 4),
        "avg_paragraph_len": round(char_count / paragraph_count, 2),
        "max_sentence_len": max(sentence_lens) if sentence_lens else 0,
        "sentence_len_std": round(np.std(sentence_lens) if len(sentence_lens) > 1 else 0, 2),
        "word_per_sentence": round(word_count / sentence_count, 2),
        "char_per_word": round(char_count / word_count if word_count > 0 else 0, 2),
    }


# ============================================================
# 데이터 적재
# ============================================================
def load_splits(names=("train", "val")):
    """
    names 에 포함된 분할만 읽는다.
    E10은 ("train","val") — test 파일을 아예 열지 않는다.
    """
    splits = {n: pd.read_csv(os.path.join(SPLITS_DIR, f"{n}.csv")) for n in names}

    # GPT-4o 재채점 병합
    for n in names:
        f = os.path.join(RERATE_DIR, f"{n}_gpt4o_ratings.csv")
        df = pd.read_csv(f)
        df = df[df["parse_success"] == True].drop_duplicates(subset="essay_id", keep="last")
        cols = (["essay_id", "llm_expression_score", "llm_structure_score",
                 "llm_content_score", "llm_total_score", "llm_confidence"]
                + [c for c in NEW_SUB if c in df.columns])
        m = splits[n].merge(df[cols], on="essay_id", how="left", suffixes=("", "_g"))
        splits[n]["gpt4o_expr"] = m["llm_expression_score"].values
        splits[n]["gpt4o_struct"] = m["llm_structure_score"].values
        splits[n]["gpt4o_cont"] = m["llm_content_score"].values
        splits[n]["gpt4o_total"] = m["llm_total_score"].values
        splits[n]["gpt4o_conf"] = m["llm_confidence"].values
        for sc in NEW_SUB:
            splits[n][sc.replace("llm_sub_", "sub_")] = m[sc].values

    # 텍스트 특징
    for n in names:
        fe = pd.DataFrame(splits[n]["essay_txt"].apply(extract_text_features).tolist())
        for c in fe.columns:
            splits[n][c] = fe[c].values

    # 학년 원핫 (라벨이 아니므로 전 분할로 인코더를 맞춰도 누수 아님)
    enc = LabelEncoder()
    enc.fit(pd.concat([s["grade_level"] for s in splits.values()]).unique())
    for n in names:
        splits[n]["grade_idx"] = enc.transform(splits[n]["grade_level"])
        for i, g in enumerate(enc.classes_):
            splits[n][f"grade_{i}"] = (splits[n]["grade_level"] == g).astype(int)

    # 파생
    for n in names:
        d = splits[n]
        tri = d[["gpt4o_expr", "gpt4o_struct", "gpt4o_cont"]]
        d["gpt4o_std"] = tri.std(axis=1)
        d["gpt4o_range"] = tri.max(axis=1) - tri.min(axis=1)
        for tc in ("char_count", "word_count", "sentence_count"):
            d[f"log_{tc}"] = np.log1p(d[tc])

    # 학년별 단어수 z점수 — 통계는 train에서만 산출 (누수 방지)
    g = splits["train"].groupby("grade_idx")["word_count"].agg(["mean", "std"])
    for n in names:
        d = splits[n]
        d["wc_zscore"] = 0.0
        for gi in d["grade_idx"].unique():
            msk = d["grade_idx"] == gi
            gm = g.loc[gi, "mean"] if gi in g.index else splits["train"]["word_count"].mean()
            gs = g.loc[gi, "std"] if gi in g.index else splits["train"]["word_count"].std()
            if gs and gs > 0:
                d.loc[msk, "wc_zscore"] = (d.loc[msk, "word_count"] - gm) / gs

    # 라벨: corrected 정의 추가 (overall_structure 중복 제거)
    for n in names:
        d = splits[n]
        d["structure_score_c"] = d[["paragraph_connection", "paragraph_structure",
                                    "consistency", "length"]].mean(axis=1)
        d["total_score_c"] = d[["expression_score", "structure_score_c",
                                "content_score"]].mean(axis=1)

    return splits


# ============================================================
# 지표 (e7l_optimized.py:44-50 과 동일 규약)
# ============================================================
_K = 7
_IDX = np.arange(_K)
_W = ((_IDX[:, None] - _IDX[None, :]) ** 2) / ((_K - 1) ** 2)


def discretize(x):
    return (np.clip(np.round(np.asarray(x, dtype=float) * 2) / 2, 0, 3) * 2).astype(int)


def qwk_disc(a_d, b_d):
    n = len(a_d)
    if n < 5:
        return 0.0
    O = np.bincount(a_d * _K + b_d, minlength=_K * _K).reshape(_K, _K).astype(float)
    E = np.outer(np.bincount(a_d, minlength=_K), np.bincount(b_d, minlength=_K)) / n
    den = (_W * E).sum()
    return 0.0 if den == 0 else 1.0 - (_W * O).sum() / den


def compute_qwk(y_true, y_pred):
    return qwk_disc(discretize(y_true), discretize(y_pred))


def metrics(y_true, y_pred, baseline_rmse=None):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    base = baseline_rmse if baseline_rmse is not None else float(y_true.std())
    ad, bd = discretize(y_true), discretize(y_pred)
    return {
        "qwk": qwk_disc(ad, bd),
        "rmse": rmse,
        "r2": float(1 - (rmse / base) ** 2) if base > 0 else None,
        "pearson_r": (float(np.corrcoef(y_true, y_pred)[0, 1])
                      if y_true.std() > 0 and y_pred.std() > 0 else None),
        "within_one": float((np.abs(ad - bd) <= 2).mean()),
        "n": int(len(y_true)),
    }


# ============================================================
# 부트스트랩
# ============================================================
def boot_indices(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, (n_boot, n))


def boot_qwk_ci(y_true, y_pred, idx, level=0.95):
    ad, bd = discretize(y_true), discretize(y_pred)
    v = np.array([qwk_disc(ad[i], bd[i]) for i in idx])
    return (float(np.percentile(v, (1 - level) / 2 * 100)),
            float(np.percentile(v, (1 + level) / 2 * 100)))


def boot_delta_ci(y_true, pred_a, pred_b, idx, level=0.95):
    """paired bootstrap: QWK(a) - QWK(b). 동일 리샘플 인덱스 사용."""
    td = discretize(y_true)
    ad, bd = discretize(pred_a), discretize(pred_b)
    v = np.array([qwk_disc(td[i], ad[i]) - qwk_disc(td[i], bd[i]) for i in idx])
    lo = float(np.percentile(v, (1 - level) / 2 * 100))
    hi = float(np.percentile(v, (1 + level) / 2 * 100))
    return {"delta": float(np.mean(v)), "ci_low": lo, "ci_high": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}
