"""
OpenAI GPT-4o 기반 에세이 자동 평가

재현성을 위한 설정:
- temperature=0.0 (deterministic greedy decoding)
- seed=42 (고정 시드)
- system_fingerprint 기록 (API 버전 추적)
"""

import os
import json
import time
from typing import Dict, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from pathlib import Path as _Path
# 레포 루트 (src/preprocessing/<file>.py -> parents[2])
REPO_ROOT = _Path(__file__).resolve().parents[2]


class OpenAIRater:
    """GPT-4o 기반 에세이 평가기"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        seed: int = 42,
        temperature: float = 0.0
    ):
        """
        Args:
            model: OpenAI 모델 이름 (gpt-4o-mini, gpt-4o 등)
            api_key: OpenAI API 키 (None이면 환경변수에서 읽음)
            seed: 재현성을 위한 시드값
            temperature: 샘플링 온도 (0.0 = deterministic)
        """
        self.model = model
        self.seed = seed
        self.temperature = temperature

        # API 키 설정
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. "
                    "Set OPENAI_API_KEY environment variable or pass api_key parameter."
                )

        self.client = OpenAI(api_key=api_key)

        print("=" * 80)
        print("OpenAI Rater Initialized")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Temperature: {self.temperature}")
        print(f"Seed: {self.seed}")
        print(f"Reproducibility: Best-effort (seed + temp=0)")
        print("=" * 80)

    def generate_evaluation(
        self,
        prompt: str,
        max_tokens: int = 512
    ) -> tuple[str, str]:
        """
        OpenAI API로 평가 생성

        Args:
            prompt: 평가 프롬프트
            max_tokens: 최대 토큰 수

        Returns:
            tuple[str, str]: (생성된 텍스트, system_fingerprint)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 중고등학생의 에세이를 평가하는 전문 교육자입니다. JSON 형식으로만 정확하게 응답하세요."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}  # JSON mode
            )

            generated_text = response.choices[0].message.content
            system_fingerprint = response.system_fingerprint

            return generated_text, system_fingerprint

        except Exception as e:
            print(f"Error during API call: {e}")
            return "", ""

    def parse_json_response(self, response: str) -> Optional[Dict]:
        """
        JSON 응답 파싱

        Args:
            response: LLM 생성 텍스트

        Returns:
            Dict: 파싱된 평가 결과, 실패 시 None
        """
        try:
            parsed = json.loads(response)

            # 필수 필드 확인
            required_keys = ['expression', 'structure', 'content', 'total']
            if all(k in parsed for k in required_keys):
                # 각 카테고리가 score를 가지고 있는지 확인
                if all(isinstance(parsed[k], dict) and 'score' in parsed[k] for k in required_keys):
                    return parsed
        except json.JSONDecodeError:
            pass

        return None

    def evaluate_essay(
        self,
        essay_text: str,
        topic: str = "",
        grade_level: str = "",
        retry_on_failure: int = 2
    ) -> Dict:
        """
        에세이 평가 (프롬프트 → 생성 → 파싱)

        Args:
            essay_text: 에세이 텍스트
            topic: 주제
            grade_level: 학년
            retry_on_failure: 파싱 실패 시 재시도 횟수

        Returns:
            Dict: 평가 결과
        """
        # 프롬프트 임포트
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from prompts import format_prompt

        prompt = format_prompt(essay_text, topic, grade_level)

        system_fingerprint = None

        for attempt in range(retry_on_failure + 1):
            response, fingerprint = self.generate_evaluation(prompt)
            system_fingerprint = fingerprint

            parsed = self.parse_json_response(response)

            if parsed:
                # 성공
                return {
                    'llm_expression_score': float(parsed['expression']['score']),
                    'llm_structure_score': float(parsed['structure']['score']),
                    'llm_content_score': float(parsed['content']['score']),
                    'llm_total_score': float(parsed['total']['score']),
                    'llm_expression_reasoning': str(parsed['expression'].get('reasoning', '')),
                    'llm_structure_reasoning': str(parsed['structure'].get('reasoning', '')),
                    'llm_content_reasoning': str(parsed['content'].get('reasoning', '')),
                    'llm_total_reasoning': str(parsed['total'].get('reasoning', '')),
                    'llm_confidence': float(parsed.get('confidence', 0.5)),
                    'llm_raw_response': response,
                    'llm_parse_success': True,
                    'llm_system_fingerprint': system_fingerprint,
                    'llm_model': self.model,
                    'llm_seed': self.seed,
                    'llm_temperature': self.temperature
                }

            # 재시도
            if attempt < retry_on_failure:
                print(f"  Warning: JSON parsing failed (attempt {attempt+1}/{retry_on_failure+1})")
                time.sleep(0.5)  # API rate limit 방지

        # 모든 시도 실패
        return {
            'llm_expression_score': None,
            'llm_structure_score': None,
            'llm_content_score': None,
            'llm_total_score': None,
            'llm_expression_reasoning': '',
            'llm_structure_reasoning': '',
            'llm_content_reasoning': '',
            'llm_total_reasoning': '',
            'llm_confidence': 0.0,
            'llm_raw_response': response,
            'llm_parse_success': False,
            'llm_system_fingerprint': system_fingerprint,
            'llm_model': self.model,
            'llm_seed': self.seed,
            'llm_temperature': self.temperature
        }

    def batch_evaluate(
        self,
        essays_df: pd.DataFrame,
        batch_size: int = 1,
        save_interval: int = 50,
        output_prefix: str = str(REPO_ROOT / "data/processed/ratings/openai_ratings")
    ) -> pd.DataFrame:
        """
        배치 평가

        Args:
            essays_df: 에세이 데이터프레임
            batch_size: 배치 크기 (현재는 1만 지원)
            save_interval: 중간 저장 간격
            output_prefix: 출력 파일 프리픽스

        Returns:
            pd.DataFrame: 평가 결과가 추가된 데이터프레임
        """
        results = []

        print(f"Starting batch evaluation of {len(essays_df)} essays...")
        print(f"Checkpoint will be saved every {save_interval} essays")
        print(f"Model: {self.model}, Seed: {self.seed}, Temperature: {self.temperature}")

        start_time = time.time()

        for idx, row in tqdm(essays_df.iterrows(), total=len(essays_df), desc="Evaluating"):
            eval_result = self.evaluate_essay(
                essay_text=str(row.get('essay_txt', '')),
                topic=str(row.get('topic', '')),
                grade_level=str(row.get('grade_level', ''))
            )

            # 원본 데이터 + 평가 결과
            result_row = {**row.to_dict(), **eval_result}
            results.append(result_row)

            # 중간 저장
            if (idx + 1) % save_interval == 0:
                temp_df = pd.DataFrame(results)
                checkpoint_path = f"{output_prefix}_checkpoint_{idx+1}.csv"
                temp_df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')

                elapsed = time.time() - start_time
                speed = len(results) / elapsed

                print(f"\n✓ Checkpoint saved: {checkpoint_path}")
                print(f"  Essays evaluated: {len(results)}")
                print(f"  Parse success rate: {temp_df['llm_parse_success'].mean() * 100:.1f}%")
                print(f"  Speed: {speed:.2f} essays/sec")
                print(f"  Elapsed time: {elapsed/60:.1f} min")

        results_df = pd.DataFrame(results)

        total_elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print("Batch evaluation complete!")
        print("=" * 80)
        print(f"Total essays: {len(results_df)}")
        print(f"Parse success: {results_df['llm_parse_success'].sum()} / {len(results_df)}")
        print(f"Success rate: {results_df['llm_parse_success'].mean() * 100:.1f}%")
        print(f"Total time: {total_elapsed/60:.1f} minutes")
        print(f"Average speed: {len(results_df)/total_elapsed:.2f} essays/sec")

        return results_df


def main():
    """OpenAI Rater 파일럿 테스트"""
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI API Pilot Test")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (gpt-4o-mini, gpt-4o, etc.)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(REPO_ROOT / "data/processed/splits/train.csv"),
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=100,
        help="Number of essays for pilot test"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # 데이터 로드
    print("=" * 80)
    print("Loading data...")
    print("=" * 80)

    train_df = pd.read_csv(args.data_path)
    pilot_df = train_df.sample(n=min(args.pilot_size, len(train_df)), random_state=args.seed)

    print(f"Pilot test with {len(pilot_df)} essays")
    print("=" * 80)

    # OpenAI Rater 초기화
    rater = OpenAIRater(
        model=args.model,
        seed=args.seed
    )

    # 배치 평가
    results_df = rater.batch_evaluate(
        pilot_df,
        batch_size=1,
        save_interval=20,
        output_prefix=str(REPO_ROOT / "data/processed/ratings/openai_pilot_ratings")
    )

    # 최종 결과 저장
    output_path = str(REPO_ROOT / "data/processed/ratings/openai_pilot_ratings.csv")
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n✓ Pilot evaluation saved: {output_path}")

    # 통계 출력
    print("\n" + "=" * 80)
    print("LLM Raw Scores Statistics")
    print("=" * 80)

    valid_df = results_df[results_df['llm_parse_success'] == True]

    if len(valid_df) > 0:
        score_cols = ['llm_expression_score', 'llm_structure_score', 'llm_content_score', 'llm_total_score']
        print(valid_df[score_cols].describe())

        # 상관관계 분석 (LLM vs Expert)
        if 'total_score' in valid_df.columns:
            from scipy.stats import pearsonr

            print("\n" + "=" * 80)
            print("Correlation with Expert Scores")
            print("=" * 80)

            for cat in ['expression', 'structure', 'content', 'total']:
                if f'{cat}_score' in valid_df.columns and f'llm_{cat}_score' in valid_df.columns:
                    # 결측치 제거
                    valid_pairs = valid_df[[f'{cat}_score', f'llm_{cat}_score']].dropna()

                    if len(valid_pairs) > 1:
                        r, p = pearsonr(valid_pairs[f'{cat}_score'], valid_pairs[f'llm_{cat}_score'])
                        print(f"{cat.capitalize()}: Pearson r = {r:.3f} (p={p:.4f})")
    else:
        print("No valid evaluations found. Please check the API key and settings.")

    print("\n" + "=" * 80)
    print("Pilot test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
