import asyncio
import sys
import time

import httpx


BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/api/chat"
TIMEOUT = 180.0
CONCURRENT_REQUESTS = 3
QUESTION = "ESS 배터리의 정기 점검 주기와 최대 충전량을 알려줘."


async def request_chat(client: httpx.AsyncClient) -> dict:
    started_at = time.perf_counter()
    response = await client.post(
        ENDPOINT,
        json={"question": QUESTION}
    )
    elapsed = time.perf_counter() - started_at

    return {
        "status_code": response.status_code,
        "elapsed": elapsed,
        "process_time": response.headers.get("X-Process-Time"),
        "source_count": len(response.json().get("sources", []))
    }


async def run_concurrency_test() -> list[dict]:
    async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=TIMEOUT
    ) as client:
        tasks = [
            request_chat(client)
            for _ in range(CONCURRENT_REQUESTS)
        ]
        return await asyncio.gather(*tasks)


def validate_rs(rs: list[dict]) -> None:
    if len(rs) != CONCURRENT_REQUESTS:
        raise AssertionError("동시 요청 결과 수가 일치하지 않습니다.")

    failed_statuses = [
        item_rs["status_code"]
        for item_rs in rs
        if item_rs["status_code"] != 200
    ]

    if failed_statuses:
        raise AssertionError(f"실패한 HTTP 상태 코드: {failed_statuses}")

    missing_process_time = [
        item_rs
        for item_rs in rs
        if item_rs["process_time"] is None
    ]

    if missing_process_time:
        raise AssertionError("동시 요청 응답에 X-Process-Time이 없습니다.")

    invalid_sources = [
        item_rs
        for item_rs in rs
        if item_rs["source_count"] == 0
    ]

    if invalid_sources:
        raise AssertionError("동시 요청 응답에 출처가 없습니다.")


if __name__ == "__main__":
    try:
        concurrency_rs = asyncio.run(run_concurrency_test())
        validate_rs(concurrency_rs)
        print("[PASS] 동시성 API 테스트")

        for index, item_rs in enumerate(concurrency_rs, start=1):
            print(
                f"요청 {index}: HTTP {item_rs['status_code']}, "
                f"실제 {item_rs['elapsed']:.3f}초, "
                f"서버 {item_rs['process_time']}초, "
                f"출처 {item_rs['source_count']}개"
            )
    except (AssertionError, httpx.RequestError) as error:
        print("[FAIL] 동시성 API 테스트")
        print(f"error: {error}")
        sys.exit(1)
