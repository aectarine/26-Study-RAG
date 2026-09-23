import sys

import httpx


BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30.0


def check_status(response: httpx.Response, expected_status: int) -> None:
    if response.status_code != expected_status:
        raise AssertionError(
            f"HTTP {response.status_code} 응답: {response.text}"
        )


def run_error_api_tests() -> list[str]:
    rs = []

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        response = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "invalid.pdf",
                    b"not a txt file",
                    "application/pdf"
                )
            }
        )
        check_status(response, 400)
        rs.append("잘못된 확장자 업로드")

        response = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "empty.txt",
                    b"",
                    "text/plain"
                )
            }
        )
        check_status(response, 400)
        rs.append("빈 파일 업로드")

        response = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "invalid-encoding.txt",
                    b"\xff\xfe\xfd",
                    "text/plain"
                )
            }
        )
        check_status(response, 400)
        rs.append("UTF-8가 아닌 파일 업로드")

        response = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "../unsafe.txt",
                    "경로 테스트".encode("utf-8"),
                    "text/plain"
                )
            }
        )
        check_status(response, 400)
        rs.append("경로가 포함된 파일명 업로드")

        response = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "too-large.txt",
                    b"a" * (5 * 1024 * 1024 + 1),
                    "text/plain"
                )
            }
        )
        check_status(response, 413)
        rs.append("5MB 초과 파일 업로드")

        missing_id = 999999999

        response = client.get(f"/api/documents/{missing_id}")
        check_status(response, 404)
        rs.append("존재하지 않는 업로드 문서 조회")

        response = client.put(
            f"/api/documents/{missing_id}",
            files={
                "file": (
                    "missing.txt",
                    "교체 테스트".encode("utf-8"),
                    "text/plain"
                )
            }
        )
        check_status(response, 404)
        rs.append("존재하지 않는 업로드 문서 교체")

        response = client.delete(f"/api/documents/{missing_id}")
        check_status(response, 404)
        rs.append("존재하지 않는 업로드 문서 삭제")

    return rs


if __name__ == "__main__":
    try:
        completed_tests = run_error_api_tests()
        print("[PASS] 오류 응답 API 테스트")
        print(f"검증 항목: {', '.join(completed_tests)}")
    except (AssertionError, httpx.RequestError) as error:
        print("[FAIL] 오류 응답 API 테스트")
        print(f"error: {error}")
        sys.exit(1)
