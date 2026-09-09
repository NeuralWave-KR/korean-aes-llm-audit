#!/usr/bin/env python3
"""
Generation step for the supplementary latest-model pilot (analyzed by
src/e12_pilot_latest_model.py).

Re-scores a balanced sub-sample of the test set with a recent model (GPT-5.6)
using the SAME 0-3 rubric prompt and JSON schema as the GPT-4o rater
(src/preprocessing/openai_rater.py / prompts.py): only the model changes.

Reproducibility settings and honest caveat:
- Balanced sub-sample: n-per-grade essays drawn per grade with a fixed seed
  (random_state=42), so the sample is deterministic and re-runs identically.
- Seed 42 is passed to the API where accepted. GPT-5.6 does NOT support
  temperature=0, so scoring runs at the model default (temperature=1); unlike
  the GPT-4o rater (temp=0) this is NOT fully deterministic. The cached ratings
  committed for analysis are therefore the fixed reference, and a re-generation
  may differ slightly.

Requires a live API key (env OPENAI_API_KEY, or a local .env next to this repo
root). No key is stored in this file. Anonymized for double-blind review.

Usage:
  python src/preprocessing/pilot_rater.py --list-models
  python src/preprocessing/pilot_rater.py --model gpt-5.6-sol --n-per-grade 25
Output: data/pilot_ratings/pilot_ratings.csv  and  data/pilot_ratings/pilot_sample.csv
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # src/preprocessing/ -> repo root
SPLIT_TEST = os.path.join(ROOT, "data", "splits", "test.csv")
OUT_DIR = os.path.join(ROOT, "data", "pilot_ratings")
os.makedirs(OUT_DIR, exist_ok=True)
RATINGS = os.path.join(OUT_DIR, "pilot_ratings.csv")
SAMPLE = os.path.join(OUT_DIR, "pilot_sample.csv")


def load_env():
    """Load OPENAI_API_KEY from a local .env at the repo root, if present."""
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def sha_ids(ids):
    return hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()


# Prompt — identical to the GPT-4o rater (0-3 scale, 11 sub-items, model-reported total).
PROMPT_SYSTEM = """You are an expert essay evaluator for Korean students (grades 6-12).
You will evaluate essays using a 0-3 point rubric scale with 11 detailed sub-rubric items.
Scores: 0 = Very poor, 1 = Below average, 2 = Average, 3 = Excellent.
Scores can include decimals (e.g., 1.5, 2.3).
Return your evaluation as a JSON object."""

PROMPT_USER = """Please evaluate the following Korean student essay on a 0-3 scale.

Essay topic: {topic}
Student grade level: {grade_level}

Essay:
{essay_text}

Evaluate on 11 detailed sub-rubric items grouped into 3 dimensions:

**Expression:** grammar (문법), word_choice (어휘선택), sentence_expression (문장표현)
**Structure:** paragraph_connection (문단연결), paragraph_structure (문단구조), consistency (일관성), length (분량), overall_structure (전체구성)
**Content:** topic_clarity (주제명료성), creativity (창의성), prompt_comprehension (과제이해도)

Return a JSON object:
{{
  "expression": {{"score": <0.0-3.0>, "sub_items": {{"grammar": <0-3>, "word_choice": <0-3>, "sentence_expression": <0-3>}}, "reasoning": "<Korean>"}},
  "structure": {{"score": <0.0-3.0>, "sub_items": {{"paragraph_connection": <0-3>, "paragraph_structure": <0-3>, "consistency": <0-3>, "length": <0-3>, "overall_structure": <0-3>}}, "reasoning": "<Korean>"}},
  "content": {{"score": <0.0-3.0>, "sub_items": {{"topic_clarity": <0-3>, "creativity": <0-3>, "prompt_comprehension": <0-3>}}, "reasoning": "<Korean>"}},
  "total": {{"score": <0.0-3.0>, "reasoning": "<Korean>"}},
  "confidence": <0.0-1.0>
}}"""

# GPT-5.x rejects temperature=0 -> seed+json first, then progressively drop params.
_PARAM_SETS = [
    {"seed": 42, "response_format": {"type": "json_object"}},
    {"response_format": {"type": "json_object"}},
    {"seed": 42},
    {},
]


def call_api(client, essay_text, topic, grade_level, model):
    prompt = PROMPT_USER.format(topic=topic or "", grade_level=grade_level or "",
                                essay_text=str(essay_text)[:3000])
    msgs = [{"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": prompt}]
    for attempt in range(4):
        params = _PARAM_SETS[min(attempt, len(_PARAM_SETS) - 1)]
        try:
            resp = client.chat.completions.create(model=model, messages=msgs, **params)
            raw = resp.choices[0].message.content
            if raw and "{" in raw:
                raw = raw[raw.index("{"): raw.rindex("}") + 1]
            parsed = json.loads(raw)
            out = {"parse_success": True, "model": resp.model,
                   "fingerprint": resp.system_fingerprint}
            for dim in ["expression", "structure", "content", "total"]:
                if dim in parsed and isinstance(parsed[dim], dict):
                    out[f"llm_{dim}_score"] = parsed[dim].get("score")
                    for k, v in (parsed[dim].get("sub_items") or {}).items():
                        out[f"llm_sub_{k}"] = v
            return out
        except Exception as e:            # noqa: BLE001 — log + retry with fewer params
            print(f"    API error (attempt {attempt + 1}/4): {e}", flush=True)
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
    return {"parse_success": False}


def build_sample(n_per_grade, seed=42):
    df = pd.read_csv(SPLIT_TEST)
    parts = [sub.sample(n=min(n_per_grade, len(sub)), random_state=seed)
             for _, sub in df.groupby("grade_level")]
    return pd.concat(parts).sort_values("essay_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--n-per-grade", type=int, default=25)
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    load_env()
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if args.list_models:
        for m in sorted(x.id for x in client.models.list().data):
            print(m)
        return

    sample = build_sample(args.n_per_grade)
    print(f"sample n={len(sample)}  sha(ids)={sha_ids(sample['essay_id'])[:16]}  "
          f"grades={sample['grade_level'].value_counts().to_dict()}", flush=True)
    sample[["essay_id", "grade_level", "total_score"]].to_csv(SAMPLE, index=False)

    done = set()
    if os.path.exists(RATINGS):
        done = set(pd.read_csv(RATINGS)["essay_id"])
    rows = []
    todo = sample[~sample["essay_id"].isin(done)]
    for i, (_, e) in enumerate(todo.iterrows(), 1):
        r = call_api(client, e.get("essay_txt", ""), e.get("topic", ""),
                     e.get("grade_level", ""), args.model)
        r["essay_id"] = e["essay_id"]
        rows.append(r)
        if i % 20 == 0:
            _flush(rows)
            rows = []
            print(f"  progress {i}/{len(todo)}", flush=True)
    _flush(rows)
    final = pd.read_csv(RATINGS)
    print(f"generated: {len(final)} rows, parse_success {int(final['parse_success'].sum())} "
          f"| ratings sha={sha256_file(RATINGS)[:16]}", flush=True)


def _flush(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(RATINGS):
        df = pd.concat([pd.read_csv(RATINGS), df], ignore_index=True)
    df.to_csv(RATINGS, index=False)


if __name__ == "__main__":
    main()
