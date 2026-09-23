import os
import time
from pathlib import Path
from statistics import mean, median, pstdev

import httpx


BASE_URL = "http://127.0.0.1:8000"
QUESTION = "ESS 배터리의 정기 점검 주기와 최대 충전량을 알려줘."
RUN_COUNT = 5
LOCK_FILE = Path(__file__).with_name(".search_benchmark.lock")

ENDPOINTS = [
    "/api/chat",
    "/api/test/chat/basic",
    "/api/test/chat/filtered",
    "/api/test/chat/reranked",
    "/api/test/chat/deduplicated",
    "/api/test/chat/deduplicated-db",
    "/api/test/chat/semantic-deduplicated",
    "/api/test/chat/semantic-reranked"
]


def acquire_benchmark_lock() -> None:
    try:
        lock_handle = os.open(
            LOCK_FILE,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
    except FileExistsError:
        raise SystemExit(
            "이미 벤치마크가 실행 중입니다. "
            f"완료 후 {LOCK_FILE.name}이 자동으로 삭제됩니다."
        )

    with os.fdopen(lock_handle, "w", encoding="utf-8") as lock_file:
        lock_file.write(str(os.getpid()))


def release_benchmark_lock() -> None:
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass


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
            "embedding_time": response.headers.get("X-Embedding-Time"),
            "db_time": response.headers.get("X-DB-Time"),
            "rerank_time": response.headers.get("X-Rerank-Time"),
            "answer_time": response.headers.get("X-Answer-Time"),
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
            "embedding_time": None,
            "db_time": None,
            "rerank_time": None,
            "answer_time": None,
            "source_count": 0,
            "duplicate_count": 0,
            "source_ids": [],
            "answer": f"요청 실패: {error}"
        }


def run_benchmark() -> list[dict]:
    rs = []

    with httpx.Client(timeout=180.0) as client:
        for endpoint in ENDPOINTS:
            print(
                f"벤치마크 진행 중: {endpoint}",
                flush=True
            )

            endpoint_rs = [
                send_request(client, endpoint, run_number)
                for run_number in range(1, RUN_COUNT + 1)
            ]

            elapsed_times = [
                item_rs["elapsed_time"]
                for item_rs in endpoint_rs
                if item_rs["elapsed_time"] is not None
            ]

            latest_rs = endpoint_rs[-1]

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

            rs.append({
                "endpoint": endpoint,
                "status_code": latest_rs["status_code"],
                "average_time": average_time,
                "median_time": median_time,
                "standard_deviation": standard_deviation,
                "minimum_time": minimum_time,
                "maximum_time": maximum_time,
                "process_time": latest_rs["process_time"],
                "embedding_time": latest_rs["embedding_time"],
                "db_time": latest_rs["db_time"],
                "rerank_time": latest_rs["rerank_time"],
                "answer_time": latest_rs["answer_time"],
                "source_count": latest_rs["source_count"],
                "duplicate_count": latest_rs["duplicate_count"],
                "source_ids": latest_rs["source_ids"],
                "answer": latest_rs["answer"]
            })

            print(
                f"벤치마크 완료: {endpoint}",
                flush=True
            )

    return rs


def format_time(value: float | None) -> str:
    if value is None:
        return "-"

    return f"{value:.3f}s"


def print_rs(rs: list[dict]) -> None:
    print(f"질문: {QUESTION}")
    print(f"반복 횟수: {RUN_COUNT}")
    print()
    print("=" * 175)
    print(
        f"{'Endpoint':<33}"
        f"{'Status':>8}"
        f"{'Average':>12}"
        f"{'Median':>12}"
        f"{'StdDev':>12}"
        f"{'Minimum':>12}"
        f"{'Maximum':>12}"
        f"{'Total':>10}"
        f"{'Embed':>10}"
        f"{'DB':>10}"
        f"{'Rerank':>10}"
        f"{'Answer':>10}"
        f"{'Sources':>10}"
        f"{'Duplicate':>12}"
        f"  Source IDs"
    )
    print("=" * 175)

    for item_rs in rs:
        print(
            f"{item_rs['endpoint']:<33}"
            f"{str(item_rs['status_code']):>8}"
            f"{format_time(item_rs['average_time']):>12}"
            f"{format_time(item_rs['median_time']):>12}"
            f"{format_time(item_rs['standard_deviation']):>12}"
            f"{format_time(item_rs['minimum_time']):>12}"
            f"{format_time(item_rs['maximum_time']):>12}"
            f"{(item_rs['process_time'] or '-'):>10}"
            f"{(item_rs['embedding_time'] or '-'):>10}"
            f"{(item_rs['db_time'] or '-'):>10}"
            f"{(item_rs['rerank_time'] or '-'):>10}"
            f"{(item_rs['answer_time'] or '-'):>10}"
            f"{item_rs['source_count']:>10}"
            f"{item_rs['duplicate_count']:>12}"
            f"  {item_rs['source_ids']}"
        )

    print("=" * 175)
    print()

    for item_rs in rs:
        print(f"[{item_rs['endpoint']}]")
        print(item_rs["answer"])
        print()


if __name__ == "__main__":
    acquire_benchmark_lock()

    try:
        benchmark_rs = run_benchmark()
        print_rs(benchmark_rs)
    finally:
        release_benchmark_lock()
