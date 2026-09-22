import time
from statistics import mean, median, pstdev

import httpx


BASE_URL = "http://127.0.0.1:8000"
QUESTION = "ESS 배터리의 정기 점검 주기와 최대 충전량을 알려줘."
RUN_COUNT = 5

ENDPOINTS = [
    "/chat",
    "/test/chat/basic",
    "/test/chat/filtered",
    "/test/chat/reranked",
    "/test/chat/deduplicated",
    "/test/chat/deduplicated-db",
    "/test/chat/semantic-reranked"
]


def count_duplicate_contents(sources: list[dict]) -> int:
    contents = [
        source.get("content", "")
        for source in sources
    ]

    return len(contents) - len(set(contents))


def send_request(client: httpx.Client, endpoint: str, run_number: int) -> dict:
    request_body = {
        "question": QUESTION
    }

    started_at = time.perf_counter()

    try:
        response = client.post(
            f"{BASE_URL}{endpoint}",
            json=request_body
        )
        elapsed_time = time.perf_counter() - started_at

        try:
            response_body = response.json()
        except ValueError:
            response_body = {
                "error": response.text[:500]
            }

        sources = response_body.get("sources", [])
        source_ids = [
            source.get("id")
            for source in sources
        ]

        answer = response_body.get(
            "answer",
            response_body.get("error", "")
        )

        if response.status_code >= 400:
            answer = f"HTTP {response.status_code}: {answer}"

        return {
            "run_number": run_number,
            "status_code": response.status_code,
            "elapsed_time": elapsed_time,
            "process_time": response.headers.get("X-Process-Time"),
            "source_count": len(sources),
            "duplicate_count": count_duplicate_contents(sources),
            "source_ids": source_ids,
            "answer": answer
        }

    except httpx.RequestError as error:
        return {
            "run_number": run_number,
            "status_code": None,
            "elapsed_time": None,
            "process_time": None,
            "source_count": 0,
            "duplicate_count": 0,
            "source_ids": [],
            "answer": f"요청 실패: {error}"
        }


def run_benchmark() -> list[dict]:
    results = []

    with httpx.Client(timeout=180.0) as client:
        for endpoint in ENDPOINTS:
            endpoint_results = [
                send_request(client, endpoint, run_number)
                for run_number in range(1, RUN_COUNT + 1)
            ]

            elapsed_times = [
                result["elapsed_time"]
                for result in endpoint_results
                if result["elapsed_time"] is not None
            ]

            latest_result = endpoint_results[-1]

            if elapsed_times:
                average_time = mean(elapsed_times)
                median_time = median(elapsed_times)
                standard_deviation = pstdev(elapsed_times)
                minimum_time = min(elapsed_times)
                maximum_time = max(elapsed_times)
            else:
                average_time = None
                median_time = None
                standard_deviation = None
                minimum_time = None
                maximum_time = None

            results.append({
                "endpoint": endpoint,
                "status_code": latest_result["status_code"],
                "average_time": average_time,
                "median_time": median_time,
                "standard_deviation": standard_deviation,
                "minimum_time": minimum_time,
                "maximum_time": maximum_time,
                "process_time": latest_result["process_time"],
                "source_count": latest_result["source_count"],
                "duplicate_count": latest_result["duplicate_count"],
                "source_ids": latest_result["source_ids"],
                "answer": latest_result["answer"]
            })

    return results


def format_time(value: float | None) -> str:
    if value is None:
        return "-"

    return f"{value:.3f}s"


def print_results(results: list[dict]) -> None:
    print(f"질문: {QUESTION}")
    print(f"반복 횟수: {RUN_COUNT}")
    print()
    print("=" * 125)
    print(
        f"{'Endpoint':<25}"
        f"{'Status':>8}"
        f"{'Average':>12}"
        f"{'Median':>12}"
        f"{'StdDev':>12}"
        f"{'Minimum':>12}"
        f"{'Maximum':>12}"
        f"{'Header':>12}"
        f"{'Sources':>10}"
        f"{'Duplicate':>12}"
        f"  Source IDs"
    )
    print("=" * 125)

    for result in results:
        print(
            f"{result['endpoint']:<25}"
            f"{str(result['status_code']):>8}"
            f"{format_time(result['average_time']):>12}"
            f"{format_time(result['median_time']):>12}"
            f"{format_time(result['standard_deviation']):>12}"
            f"{format_time(result['minimum_time']):>12}"
            f"{format_time(result['maximum_time']):>12}"
            f"{(result['process_time'] or '-'):>12}"
            f"{result['source_count']:>10}"
            f"{result['duplicate_count']:>12}"
            f"  {result['source_ids']}"
        )

    print("=" * 125)
    print()

    for result in results:
        print(f"[{result['endpoint']}]")
        print(result["answer"])
        print()


if __name__ == "__main__":
    benchmark_results = run_benchmark()
    print_results(benchmark_results)
