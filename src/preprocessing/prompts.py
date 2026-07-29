"""
에세이 평가 프롬프트 템플릿

LLM-as-a-Judge 방식으로 에세이를 평가하는 프롬프트
"""

ESSAY_EVALUATION_PROMPT = """당신은 중고등학생의 에세이를 평가하는 전문 교육자입니다.
다음 에세이를 읽고, 아래의 평가 기준에 따라 정확하게 점수를 매겨주세요.

[평가 기준]
에세이는 3개의 주요 카테고리로 평가됩니다. 각 카테고리는 0~10점 척도입니다.

1. **표현 점수 (Expression)**: 문법의 정확성, 단어 선택의 적절성, 문장 표현의 자연스러움
   - 0-3점: 문법 오류 다수, 단어 선택 부적절, 문장 구성 어색
   - 4-6점: 일부 문법 오류, 단어 선택 보통, 문장 구성 평범
   - 7-8점: 문법 정확, 단어 선택 적절, 문장 구성 자연스러움
   - 9-10점: 문법 완벽, 단어 선택 탁월, 문장 구성 우수

2. **구성 점수 (Structure)**: 문단 간 연결, 문단 내 구성, 일관성, 분량의 적절성, 전체 구성
   - 0-3점: 문단 구성 미흡, 논리적 흐름 부족
   - 4-6점: 기본적인 문단 구성, 일부 논리적 연결
   - 7-8점: 명확한 문단 구성, 논리적 흐름 우수
   - 9-10점: 완벽한 문단 구성, 논리 전개 탁월

3. **내용 점수 (Content)**: 주제의 명료성, 내용의 참신성, 프롬프트 이해도
   - 0-3점: 주제 불명확, 내용 빈약, 프롬프트 이해 부족
   - 4-6점: 주제 기본적, 내용 평범, 프롬프트 이해 보통
   - 7-8점: 주제 명확, 내용 충실, 프롬프트 정확히 이해
   - 9-10점: 주제 탁월, 내용 창의적, 프롬프트 완벽 이해

[평가할 에세이]
제목: {topic}
학년: {grade_level}
에세이:
{essay_text}

[출력 형식]
반드시 아래의 JSON 형식으로만 응답하세요. 다른 설명은 포함하지 마세요.

{{
  "expression": {{
    "score": 0.0,
    "reasoning": "표현 점수에 대한 간단한 근거 (1-2문장)"
  }},
  "structure": {{
    "score": 0.0,
    "reasoning": "구성 점수에 대한 간단한 근거 (1-2문장)"
  }},
  "content": {{
    "score": 0.0,
    "reasoning": "내용 점수에 대한 간단한 근거 (1-2문장)"
  }},
  "total": {{
    "score": 0.0,
    "reasoning": "전체 평가 요약 (2-3문장)"
  }},
  "confidence": 0.0
}}

confidence는 0.0~1.0 사이의 값으로, 평가의 확신도를 나타냅니다.
"""


def format_prompt(essay_text: str, topic: str = "", grade_level: str = "") -> str:
    """
    프롬프트 포매팅

    Args:
        essay_text: 에세이 텍스트
        topic: 에세이 주제 (선택)
        grade_level: 학년 (선택)

    Returns:
        str: 포맷된 프롬프트
    """
    return ESSAY_EVALUATION_PROMPT.format(
        essay_text=essay_text,
        topic=topic or "제공되지 않음",
        grade_level=grade_level or "제공되지 않음"
    )


# Few-shot 예제 (필요 시 사용)
FEW_SHOT_EXAMPLES = [
    {
        "essay": "오늘은 날씨가 좋았다. 학교에 갔다. 친구를 만났다. 점심을 먹었다.",
        "evaluation": {
            "expression": {"score": 4.0, "reasoning": "문법은 정확하나 단순한 문장 구조"},
            "structure": {"score": 3.0, "reasoning": "문단 구성 없음, 단순 나열"},
            "content": {"score": 3.0, "reasoning": "주제가 불명확하고 내용이 빈약함"},
            "total": {"score": 3.5, "reasoning": "기초적인 수준의 에세이"},
            "confidence": 0.9
        }
    },
    {
        "essay": """환경 보호의 중요성

현대 사회에서 환경 오염은 심각한 문제로 대두되고 있다. 대기 오염, 수질 오염, 토양 오염 등 다양한 형태의 환경 파괴가 진행되고 있으며, 이는 인류의 생존을 위협하고 있다.

환경을 보호하기 위해서는 개인과 사회 모두의 노력이 필요하다. 첫째, 일회용품 사용을 줄이고 재활용을 실천해야 한다. 둘째, 대중교통을 이용하여 탄소 배출을 감소시켜야 한다. 셋째, 기업들은 친환경 제품을 개발하고 생산 과정에서 오염을 최소화해야 한다.

결론적으로, 환경 보호는 선택이 아닌 필수이다. 우리 모두가 작은 실천부터 시작한다면, 더 나은 미래를 만들 수 있을 것이다.""",
        "evaluation": {
            "expression": {"score": 8.0, "reasoning": "문법이 정확하고 문장 표현이 자연스러움"},
            "structure": {"score": 8.5, "reasoning": "서론-본론-결론의 명확한 구성, 논리적 흐름 우수"},
            "content": {"score": 7.5, "reasoning": "주제가 명확하고 구체적인 방안 제시"},
            "total": {"score": 8.0, "reasoning": "전반적으로 우수한 에세이, 논리적이고 설득력 있음"},
            "confidence": 0.85
        }
    }
]


def format_few_shot_prompt(essay_text: str, topic: str = "", grade_level: str = "", num_examples: int = 1) -> str:
    """
    Few-shot 예제를 포함한 프롬프트 생성

    Args:
        essay_text: 평가할 에세이 텍스트
        topic: 주제
        grade_level: 학년
        num_examples: 포함할 예제 수 (0-2)

    Returns:
        str: Few-shot 프롬프트
    """
    if num_examples == 0:
        return format_prompt(essay_text, topic, grade_level)

    # 예제 추가
    examples_text = "\n\n[평가 예시]\n"

    for i, example in enumerate(FEW_SHOT_EXAMPLES[:num_examples]):
        examples_text += f"\n예시 {i+1}:\n"
        examples_text += f"에세이: {example['essay']}\n"
        examples_text += f"평가: {json.dumps(example['evaluation'], ensure_ascii=False, indent=2)}\n"

    # 기본 프롬프트에 예제 추가
    base_prompt = format_prompt(essay_text, topic, grade_level)

    # [평가 기준] 섹션 뒤에 예제 삽입
    parts = base_prompt.split("[평가할 에세이]")
    return parts[0] + examples_text + "\n\n[평가할 에세이]" + parts[1]


if __name__ == "__main__":
    # 테스트
    test_essay = "오늘은 학교에서 재미있는 일이 있었다. 친구들과 함께 축구를 했는데, 우리 팀이 이겼다."

    print("=" * 80)
    print("Standard Prompt:")
    print("=" * 80)
    print(format_prompt(test_essay, "나의 하루", "중학교 1학년"))

    print("\n" + "=" * 80)
    print("Few-shot Prompt (1 example):")
    print("=" * 80)
    print(format_few_shot_prompt(test_essay, "나의 하루", "중학교 1학년", num_examples=1)[:500] + "...")
