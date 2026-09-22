import sys

import httpx


BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/chat"
TIMEOUT = 180.0

TEST_CASES = [
    {
        "name": "점검 주기와 충전량",
        "question": "ESS 배터리의 정기 점검 주기와 최대 충전량을 알려줘.",
        "expected_source_terms": [
            "정기 점검 주기는 30일",
            "최대 충전량은 85%"
        ],
        "required_answer_terms": ["30일", "85%"],
        "forbidden_answer_terms": []
    },
    {
        "name": "저온 충전 제한",
        "question": "ESS 배터리 온도가 0도 미만이면 어떻게 해야 하나요?",
        "expected_source_terms": [
            "0도 미만이면 충전을 제한"
        ],
        "required_answer_terms": ["충전", "제한"],
        "forbidden_answer_terms": []
    },
    {
        "name": "충전 주의사항 중복 제거",
        "question": "ESS 충전 관련 주의사항",
        "expected_source_terms": [
            "0도 미만이면 충전을 제한",
            "최대 충전량은 85%"
        ],
        "required_answer_terms": ["충전", "85%"],
        "forbidden_answer_terms": [],
        "expected_source_count": 2
    },
    {
        "name": "문서에 없는 질문",
        "question": "ESS 배터리의 제조사 보증 기간은 얼마인가요?",
        "expected_source_terms": [],
        "required_answer_terms": ["찾을 수 없습니다"],
        "forbidden_answer_terms": ["년", "개월"]
    }
]


def normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def validate_response(test_case: dict, response_body: dict) -> list[str]:
    errors = []
    answer = normalize_text(response_body.get("answer", ""))
    sources = response_body.get("sources", [])
    source_ids = {
        source.get("id")
        for source in sources
        if source.get("id") is not None
    }

    missing_terms = [
        term
        for term in test_case["required_answer_terms"]
        if term not in answer
    ]

    if missing_terms:
        errors.append(f"필수 답변 내용 누락: {missing_terms}")

    forbidden_terms = [
        term
        for term in test_case["forbidden_answer_terms"]
        if term in answer
    ]

    if forbidden_terms:
        errors.append(f"금지 답변 내용 포함: {forbidden_terms}")

    source_contents = [
        normalize_text(source.get("content", ""))
        for source in sources
    ]

    missing_source_terms = [
        term
        for term in test_case["expected_source_terms"]
        if not any(term in content for content in source_contents)
    ]

    if missing_source_terms:
        errors.append(f"기대 출처 내용 누락: {missing_source_terms}")

    expected_source_count = test_case.get("expected_source_count")

    if expected_source_count is not None and len(sources) != expected_source_count:
        errors.append(
            f"출처 개수 불일치: 기대 {expected_source_count}, 실제 {len(sources)}"
        )

    if not test_case["expected_source_terms"] and sources:
        errors.append(f"문서에 없는 질문인데 출처 반환: {sorted(source_ids)}")

    return errors


def run_quality_tests() -> list[dict]:
    results = []

    with httpx.Client(timeout=TIMEOUT) as client:
        for test_case in TEST_CASES:
            try:
                response = client.post(
                    f"{BASE_URL}{ENDPOINT}",
                    json={"question": test_case["question"]}
                )
                response_body = response.json()

                errors = []

                if response.status_code != 200:
                    errors.append(f"HTTP 상태 코드: {response.status_code}")
                else:
                    errors.extend(validate_response(test_case, response_body))

                results.append({
                    "name": test_case["name"],
                    "status_code": response.status_code,
                    "passed": not errors,
                    "errors": errors,
                    "answer": response_body.get("answer", ""),
                    "source_ids": [
                        source.get("id")
                        for source in response_body.get("sources", [])
                    ]
                })
            except (httpx.RequestError, ValueError) as error:
                results.append({
                    "name": test_case["name"],
                    "status_code": None,
                    "passed": False,
                    "errors": [f"요청 실패: {error}"],
                    "answer": "",
                    "source_ids": []
                })

    return results


def print_results(results: list[dict]) -> None:
    print(f"대상 API: {ENDPOINT}")
    print(f"테스트 수: {len(results)}")
    print()

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['name']}")
        print(f"  HTTP: {result['status_code']}")
        print(f"  source IDs: {result['source_ids']}")
        print(f"  answer: {result['answer']}")

        for error in result["errors"]:
            print(f"  error: {error}")

        print()


if __name__ == "__main__":
    quality_results = run_quality_tests()
    print_results(quality_results)

    if any(not result["passed"] for result in quality_results):
        sys.exit(1)
